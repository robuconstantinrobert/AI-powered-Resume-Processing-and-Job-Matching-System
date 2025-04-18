#!/usr/bin/env python
"""
app.py – offline résumé analyser for 16 GB laptops
-------------------------------------------------
python app.py --pdf CV.pdf --model tinyllama
python app.py --pdf CV.pdf --gguf models/phi-2.Q4_K_M.gguf
"""
# ──────────────────────────────────  imports  ──────────────────────────────────
import argparse, json, pickle, hashlib, re, os, sys
from pathlib import Path
import fitz                           # PyMuPDF
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, logging as hf_logging)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# silence HF warnings
hf_logging.set_verbosity_error()

# ────────────────────────────────  configuration  ─────────────────────────────
LOCAL_MODELS = {
    "gptneo":    "./gpt-neo-1.3B",
    "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
    "phi2":      "./phi-2"
}

QUANT_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CACHE_DIR = Path(".cache");  CACHE_DIR.mkdir(exist_ok=True)

torch.set_num_threads(min(4, os.cpu_count() or 4))   # pin CPU threads

# ─────────────────────────────  text‑cleaning utils  ──────────────────────────
SECTION_HEADINGS = {
    "contact","about me","projects","work experience","education",
    "certifications","languages","skills","qualities"
}
def _is_heading(line: str) -> bool:
    t = line.strip().lower()
    return t.isupper() and any(t.startswith(h) for h in SECTION_HEADINGS)

def clean_text(inp: str) -> str:
    """Return cleaned, lower‑cased text (accepts PDF path or raw string)."""
    if Path(inp).is_file():
        raw = "\n".join(p.get_text() for p in fitz.open(inp))
    else:
        raw = inp
    txt = re.sub(r"\b\d{7,}\b|\S+@\S+|https?://\S+|www\.\S+", " ", raw)
    keep = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or _is_heading(ln) or (ln.isupper() and len(ln.split()) <= 2):
            continue
        keep.append(ln)
    txt = " ".join(keep)
    txt = re.sub(r"[^A-Za-z0-9.,:/\\+& -]", " ", txt)   # ← hyphen safe at end
    txt = re.sub(r"\s{2,}", " ", txt).lower().strip()
    return txt

# ──────────────────────────────  keyword entities  ────────────────────────────
JOB_KW  = {"developer","engineer","scientist","programmer","specialist",
           "architect","analyst","manager","technician","lead"}
TOOL_KW = {"python","java","c++","javascript","react","nodejs","spring",
           "linux","docker","kubernetes","aws","azure","git","mysql",
           "sql","vscode","gitlab","c#","php","html","css"}
def extract_entities(text: str) -> dict:
    return {"job_titles": sorted(k for k in JOB_KW  if k in text),
            "tools":      sorted(k for k in TOOL_KW if k in text)}

# ───────────────────────────────  embed with cache  ───────────────────────────
def _sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest()
def embed(texts):
    vecs = []
    for t in texts:
        f = CACHE_DIR / f"{_sha1(t)}.pkl"
        if f.exists():
            vecs.append(pickle.loads(f.read_bytes()))
        else:
            v = EMB_MODEL.encode(t)
            f.write_bytes(pickle.dumps(v))
            vecs.append(v)
    return vecs

# ───────────────────────────────  llama‑cpp loader  ───────────────────────────
def load_llm_gguf(gguf_path: str):
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path,
                n_ctx=2048,
                n_threads=min(4, os.cpu_count() or 4))
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    class Wrapper:
        def __init__(self, l): self.l = l
        def generate(self, input_ids=None, max_new_tokens=128,
                     temperature=0.2, do_sample=False, **_):
            prompt = tok.decode(input_ids[0])
            out = self.l(prompt, max_tokens=max_new_tokens,
                         temperature=temperature)["choices"][0]["text"]
            full = tok(prompt + out, return_tensors="pt").input_ids
            return full  # mimic HF output shape
        @property
        def device(self): return torch.device("cpu")
    return tok, Wrapper(llm)

# ───────────────────────────────  HF loader (auto‑CPU)  ───────────────────────
def load_llm_hf(local_dir: str):
    """
    • If CUDA is present → try 4‑bit bits‑and‑bytes.
    • Otherwise load on CPU, using bf16 if the runtime supports it;
      if the bf16 capability flag is missing, default to fp32.
    """
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    # ---------- 4‑bit path ---------------------------------------------------
    if torch.cuda.is_available():
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                local_dir,
                quantization_config=QUANT_CFG,
                device_map="auto",
                trust_remote_code=True,
            )
            return tok, mdl
        except Exception as e:
            print("[warn] 4‑bit load failed → fall back to CPU fp32:", e)

    # ---------- CPU path -----------------------------------------------------
    # Is bf16 available on *this* build?  Check safely.
    bf16_ok = (
        hasattr(torch, "bfloat16") and
        hasattr(torch, "utils") and
        callable(getattr(torch, "tensor", None))
    )
    dtype = torch.bfloat16 if bf16_ok else torch.float32

    mdl = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map={"": "cpu"},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return tok, mdl


# ───────────────────────────  JSON extract via LLM  ───────────────────────────
def llm_json_extract(llm, tok, text: str, max_new=128):
    prompt = ("You are an HR assistant.\n"
              "Return STRICT JSON with keys job_titles, skills, suggested_roles.\n"
              "### RESUME\n" + text + "\n### JSON:\n")
    ids = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**ids, max_new_tokens=max_new,
                       temperature=0.2, do_sample=False)
    resp = tok.decode(out[0], skip_special_tokens=True)
    try:
        return json.loads(resp.split("### JSON:")[-1].strip())
    except json.JSONDecodeError:
        return {}

# ───────────────────────────────────  main  ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to résumé PDF")
    ap.add_argument("--model", choices=list(LOCAL_MODELS))
    ap.add_argument("--gguf", help="Path to .gguf 4‑bit model (overrides --model)")
    ap.add_argument("--jobs", nargs="+", default=[
        "Software Engineer with expertise in Python, JavaScript, and machine learning.",
        "Data Scientist specializing in AI and data analysis.",
        "Web Developer with experience in React, Node.js, and database management.",
        "Android Developer with proficiency in Java and Android Studio.",
        "Cloud Engineer with experience in AWS, Docker, Kubernetes, and Python."
    ])
    args = ap.parse_args()

    # ---- cleaning --------------------------------------------------
    cv_clean  = clean_text(args.pdf)
    job_clean = [clean_text(j) for j in args.jobs]

    # ---- entities --------------------------------------------------
    kw_entities = extract_entities(cv_clean)

    # ---- embeddings ------------------------------------------------
    resume_vec = embed([cv_clean])[0]
    job_vecs   = embed(job_clean)
    sims = cosine_similarity([resume_vec], job_vecs)[0]

    # ---- LLM load --------------------------------------------------
    if args.gguf:
        tok, llm = load_llm_gguf(args.gguf)
    else:
        tok, llm = load_llm_hf(LOCAL_MODELS[args.model])

    llm_info = llm_json_extract(llm, tok, cv_clean)

    # ---- output ----------------------------------------------------
    print("\n== Keyword entities ==")
    print(json.dumps(kw_entities, indent=2))
    print("\n== LLM extracted info ==")
    print(json.dumps(llm_info, indent=2))
    print("\n== Cosine similarity ==")
    for i, (desc, sc) in enumerate(zip(args.jobs, sims), 1):
        print(f"{i}. {desc[:65]:65s} → {sc:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
