# #!/usr/bin/env python
# """
# app.py – offline résumé analyser for 16 GB laptops
# -------------------------------------------------
# python app.py --pdf CV.pdf --model tinyllama
# python app.py --pdf CV.pdf --gguf models/phi-2.Q4_K_M.gguf
# """
# # ──────────────────────────────────  imports  ──────────────────────────────────
# import argparse, json, pickle, hashlib, re, os, sys
# from pathlib import Path
# import fitz                           # PyMuPDF
# import torch
# from transformers import (AutoTokenizer, AutoModelForCausalLM,
#                           BitsAndBytesConfig, logging as hf_logging)
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import psycopg2, psycopg2.extras


# # silence HF warnings
# hf_logging.set_verbosity_error()

# # ────────────────────────────────  configuration  ─────────────────────────────
# LOCAL_MODELS = {
#     "gptneo":    "./gpt-neo-1.3B",
#     "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
#     "phi2":      "./phi-2"
# }

# LOCAL_EMB = {
#     "minilm": "./all-MiniLM-L6-v2",
#     "mpnet":  "./all-mpnet-base-v2",
#     "gtr":    "./gtr-t5-base",
# }

# EMB_MODEL = None

# QUANT_CFG = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# CACHE_DIR = Path(".cache");  CACHE_DIR.mkdir(exist_ok=True)

# torch.set_num_threads(min(4, os.cpu_count() or 4))   # pin CPU threads

# # ─────────────────────────────  text‑cleaning utils  ──────────────────────────
# SECTION_HEADINGS = {
#     "contact","about me","projects","work experience","education",
#     "certifications","languages","skills","qualities"
# }
# def _is_heading(line: str) -> bool:
#     t = line.strip().lower()
#     return t.isupper() and any(t.startswith(h) for h in SECTION_HEADINGS)

# def clean_text(inp: str) -> str:
#     """Return cleaned, lower‑cased text (accepts PDF path or raw string)."""
#     if Path(inp).is_file():
#         raw = "\n".join(p.get_text() for p in fitz.open(inp))
#     else:
#         raw = inp
#     txt = re.sub(r"\b\d{7,}\b|\S+@\S+|https?://\S+|www\.\S+", " ", raw)
#     keep = []
#     for ln in txt.splitlines():
#         ln = ln.strip()
#         if not ln or _is_heading(ln) or (ln.isupper() and len(ln.split()) <= 2):
#             continue
#         keep.append(ln)
#     txt = " ".join(keep)
#     txt = re.sub(r"[^A-Za-z0-9.,:/\\+& -]", " ", txt)   # ← hyphen safe at end
#     txt = re.sub(r"\s{2,}", " ", txt).lower().strip()
#     return txt

# # ──────────────────────────────  keyword entities  ────────────────────────────
# def extract_entities(text: str, skill_set) -> dict:
#     tokens = set(text.split())
#     skills = sorted(k for k in skill_set if k in text)
#     return {"skills": skills}

# def get_pg_conn(args):
#     return psycopg2.connect(
#         dbname=args.db, user=args.user, password=args.password,
#         host=args.host, port=args.port
#     )

# def load_occupation_vectors(cur):
#     cur.execute("SELECT concepturi, preferredlabel, description "
#                 "FROM occupation")
#     occ_rows = cur.fetchall()   # ~3 000 rows
#     texts  = [f"{r[1]} {r[2] or ''}" for r in occ_rows]
#     ids    = [r[0] for r in occ_rows]

#     vecs = embed(texts)   # cached MiniLM vectors
#     return ids, texts, vecs

# def skills_for_occs(cur, occ_uris):
#     sql = """
#     SELECT DISTINCT lower(s.preferredlabel)
#     FROM occupation_skill_rel r
#     JOIN skill s ON s.concepturi = r.skilluri
#     WHERE r.occupationuri = ANY(%s)
#     """
#     cur.execute(sql, (occ_uris,))
#     return {row[0] for row in cur.fetchall()}

# # ───────────────────────────────  embed with cache  ───────────────────────────
# def _sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest()
# def embed(texts):
#     vecs = []
#     for t in texts:
#         f = CACHE_DIR / f"{_sha1(t)}.pkl"
#         if f.exists():
#             vecs.append(pickle.loads(f.read_bytes()))
#         else:
#             v = EMB_MODEL.encode(t)
#             f.write_bytes(pickle.dumps(v))
#             vecs.append(v)
#     return vecs

# # ───────────────────────────────  llama‑cpp loader  ───────────────────────────
# def load_llm_gguf(gguf_path: str):
#     from llama_cpp import Llama
#     llm = Llama(model_path=gguf_path,
#                 n_ctx=2048,
#                 n_threads=min(4, os.cpu_count() or 4))
#     tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

#     class Wrapper:
#         def __init__(self, l): self.l = l
#         def generate(self, input_ids=None, max_new_tokens=128,
#                      temperature=0.2, do_sample=False, **_):
#             prompt = tok.decode(input_ids[0])
#             out = self.l(prompt, max_tokens=max_new_tokens,
#                          temperature=temperature)["choices"][0]["text"]
#             full = tok(prompt + out, return_tensors="pt").input_ids
#             return full  # mimic HF output shape
#         @property
#         def device(self): return torch.device("cpu")
#     return tok, Wrapper(llm)

# # ───────────────────────────────  HF loader (auto‑CPU)  ───────────────────────
# def load_llm_hf(local_dir: str):
#     """
#     • If CUDA is present → try 4‑bit bits‑and‑bytes.
#     • Otherwise load on CPU, using bf16 if the runtime supports it;
#       if the bf16 capability flag is missing, default to fp32.
#     """
#     tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

#     # ---------- 4‑bit path ---------------------------------------------------
#     if torch.cuda.is_available():
#         try:
#             mdl = AutoModelForCausalLM.from_pretrained(
#                 local_dir,
#                 quantization_config=QUANT_CFG,
#                 device_map="auto",
#                 trust_remote_code=True,
#             )
#             mdl.name_or_path = local_dir
#             return tok, mdl
#         except Exception as e:
#             print("[warn] 4‑bit load failed → fall back to CPU fp32:", e)

#     # ---------- CPU path -----------------------------------------------------
#     # Is bf16 available on *this* build?  Check safely.
#     bf16_ok = (
#         hasattr(torch, "bfloat16") and
#         hasattr(torch, "utils") and
#         callable(getattr(torch, "tensor", None))
#     )
#     dtype = torch.bfloat16 if bf16_ok else torch.float32

#     mdl = AutoModelForCausalLM.from_pretrained(
#         local_dir,
#         device_map={"": "cpu"},
#         torch_dtype=dtype,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
#     return tok, mdl


# # ───────────────────────────  JSON extract via LLM  ───────────────────────────
# # def llm_json_extract(llm, tok, text: str, max_new=400):
# #     is_gptneo = isinstance(llm, torch.nn.Module) and "gptneo" in llm.name_or_path.lower()
    
# #     if is_gptneo:
# #         prompt = (
# #             "Extract this information in JSON format with keys: job_titles (max 3), "
# #             "skills (max 10), suggested_roles (exact 3).\nText:\n" + text + "\n\nJSON:\n"
# #         )
# #     else:
# #         prompt = (
# #             "You are an HR assistant.\n"
# #             "Return **ONLY** valid JSON, no markdown, no explanations.\n"
# #             "The JSON object must have keys:\n"
# #             "  • job_titles (max‑3)\n"
# #             "  • skills (max‑10)\n"
# #             "  • suggested_roles (exact‑3)\n"
# #             "Resume:\n" + text + "\n\nJSON:\n"
# #         )

# #     inputs = tok(prompt, return_tensors="pt").to(llm.device)
# #     out = llm.generate(
# #         **inputs,
# #         max_new_tokens=max_new,
# #         do_sample=False,
# #         temperature=0.0,
# #         eos_token_id=tok.eos_token_id,
# #     )
# #     raw = tok.decode(out[0], skip_special_tokens=True)

# #     print("[debug] raw output:\n", raw)  # debug temporar

# #     # Extrage JSON-ul din text
# #     start = raw.find("{")
# #     end = raw.rfind("}")
# #     if start == -1 or end == -1:
# #         print("[warn] no JSON found – raw output:\n", raw)
# #         return {}
# #     try:
# #         return json.loads(raw[start : end+1])
# #     except json.JSONDecodeError as e:
# #         print("[error] JSON decode failed:", e)
# #         print("[json candidate]:", raw[start:end+1])
# #         return {}

# def llm_json_extract(llm, tok, text: str, max_new=400):
#     # identificăm dacă e gpt-neo
#     is_gptneo = hasattr(llm, "name_or_path") and "gpt-neo" in llm.name_or_path.lower()

#     if is_gptneo:
#         prompt = (
#             "Give me this information in JSON format only:\n"
#             "{\n"
#             "  \"job_titles\": [... up to 3 job titles],\n"
#             "  \"skills\": [... up to 10 relevant skills],\n"
#             "  \"suggested_roles\": [... exactly 3 roles you think match the candidate]\n"
#             "}\n"
#             "Resume:\n" + text + "\n\nJSON:\n"
#         )
#     else:
#         prompt = (
#             "You are an HR assistant.\n"
#             "Return **ONLY** valid JSON, no markdown, no explanations.\n"
#             "The JSON object must have keys:\n"
#             "  • job_titles (max‑3)\n"
#             "  • skills (max‑10)\n"
#             "  • suggested_roles (exact‑3)\n"
#             "Resume:\n" + text + "\n\nJSON:\n"
#         )

#     inputs = tok(prompt, return_tensors="pt").to(llm.device)
#     out = llm.generate(
#         **inputs,
#         max_new_tokens=max_new,
#         do_sample=True,        # GPT-Neo are nevoie de sampling #  False pentru mpnet
#         temperature=0.7, # 0.0,
#         eos_token_id=tok.eos_token_id,
#     )

#     raw = tok.decode(out[0], skip_special_tokens=True)
#     print("[debug] raw output:\n", raw)

#     start = raw.find("{")
#     end = raw.rfind("}")
#     if start == -1 or end == -1:
#         print("[warn] no JSON found – raw output:\n", raw)
#         return {}
#     try:
#         return json.loads(raw[start : end+1])
#     except json.JSONDecodeError as e:
#         print("[error] JSON decode failed:", e)
#         print("[json candidate]:", raw[start:end+1])
#         return {}




# # ───────────────────────────────────  main  ───────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pdf", required=True, help="Path to résumé PDF")
#     ap.add_argument("--emb", choices=list(LOCAL_EMB), 
#                     default="minilm", 
#                     help="which local embedding model to use")
#     ap.add_argument("--model", choices=list(LOCAL_MODELS),
#                     default="tinyllama")
#     ap.add_argument("--gguf", help="Path to .gguf 4‑bit model")
#     # --- PostgreSQL creds ---
#     ap.add_argument("--db",   default="vector_database")
#     ap.add_argument("--user", default="postgres")
#     ap.add_argument("--password", default=os.getenv("PGPASSWORD","password"))
#     ap.add_argument("--host", default="localhost")
#     ap.add_argument("--port", type=int, default=5432)
#     ap.add_argument("--top", type=int, default=3,
#                     help="how many best‑matching occupations to show")
#     args = ap.parse_args()

#     global EMB_MODEL
#     EMB_MODEL = SentenceTransformer(
#         LOCAL_EMB[args.emb],
#         device="cpu",
#         trust_remote_code=True,
#     )

#     # 1. Clean résumé text -----------------------------------------
#     cv_clean = clean_text(args.pdf)
#     cv_vec   = embed([cv_clean])[0]

#     # 2. Pull occupations + vectors --------------------------------
#     pg = get_pg_conn(args)
#     cur = pg.cursor()
#     occ_ids, occ_texts, occ_vecs = load_occupation_vectors(cur)

#     # 3. Cosine similarity résumé ↔ occupations --------------------
#     sims = cosine_similarity([cv_vec], occ_vecs)[0]
#     top_idx = sims.argsort()[-args.top:][::-1]
#     top_occ_ids  = [occ_ids[i]  for i in top_idx]
#     top_occ_text = [occ_texts[i] for i in top_idx]
#     top_sims     = [sims[i] for i in top_idx]

#     # 4. Get skills linked to these occupations --------------------
#     skill_set = skills_for_occs(cur, top_occ_ids)

#     # 5. Keyword extraction using live skill_set -------------------
#     kw_entities = extract_entities(cv_clean, skill_set)

#     # 6. LLM load --------------------------------------------------
#     tok, llm = (load_llm_gguf(args.gguf) if args.gguf
#                 else load_llm_hf(LOCAL_MODELS[args.model]))

#     #llm_info = llm_json_extract(llm, tok, cv_clean)
#     def tail(text, n=1200):
#         return text[-n:] if len(text) > n else text

#     llm_info = llm_json_extract(llm, tok, tail(cv_clean))



#     # 7. Output ----------------------------------------------------
#     print("\n== Best‑matching occupations ==")
#     for uri, txt, sc in zip(top_occ_ids, top_occ_text, top_sims):
#         print(f"{sc: .4f}  {txt[:80]}  ({uri})")

#     print("\n== Skills matched by keyword ==")
#     print(", ".join(sorted(kw_entities["skills"])))

#     print("\n== LLM extracted info ==")
#     print(json.dumps(llm_info, indent=2))


# # ──────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
"""
app.py – offline résumé analyser for 16 GB laptops
-------------------------------------------------
Examples
--------
# default: MiniLM embeddings  + TinyLlama chat
python app.py --pdf CV.pdf

# MPNet embeddings + Phi-2 chat (HF folder)
python app.py --pdf CV.pdf --emb mpnet --model phi2

# GTR-T5 embeddings + Phi-2 chat (GGUF quant)
python app.py --pdf CV.pdf --emb gtr  --gguf models/phi-2.Q4_K_M.gguf
"""
# ───────────────────────────────  imports  ────────────────────────────────
import argparse, json, pickle, hashlib, re, os, sys
from pathlib import Path

import fitz                           # PyMuPDF
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, logging as hf_logging)
from sentence_transformers import SentenceTransformer

# silence HF warnings
hf_logging.set_verbosity_error()

# ─────────────────────────  local model folders  ──────────────────────────
LOCAL_CHAT = {
    "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
    "phi2":      "./phi-2",
}

LOCAL_EMB = {
    "minilm": "./all-MiniLM-L6-v2",
    "mpnet":  "./all-mpnet-base-v2",
    "gtr":    "./gtr-t5-base",
}

# embedding model will be initialised in main()
EMB_MODEL = None

# quant-config for 4-bit GPU loading (if you have CUDA)
QUANT_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

CACHE_DIR = Path(".cache");  CACHE_DIR.mkdir(exist_ok=True)
torch.set_num_threads(min(4, os.cpu_count() or 4))

# ─────────────────────── text-cleaning util ───────────────────────────────
SECTION_HEADINGS = {
    "contact","about me","projects","work experience","education",
    "certifications","languages","skills","qualities"
}
def _is_heading(line: str) -> bool:
    t = line.strip().lower()
    return t.isupper() and any(t.startswith(h) for h in SECTION_HEADINGS)

def clean_text(inp: str) -> str:
    """Return cleaned, lower-cased text (accepts PDF path or raw string)."""
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
    txt = re.sub(r"[^A-Za-z0-9.,:/\\+& -]", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).lower().strip()
    return txt

# ───────────────────────── embedding helper ───────────────────────────────
def _sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest()

def embed(texts, emb_key: str):
    """Encode list of texts with caching on disk."""
    vecs = []
    for t in texts:
        f = CACHE_DIR / f"{emb_key}_{_sha1(t)}.pkl"
        if f.exists():
            vecs.append(pickle.loads(f.read_bytes()))
        else:
            v = EMB_MODEL.encode(t)
            f.write_bytes(pickle.dumps(v))
            vecs.append(v)
    return vecs

# ───────────────────────── LLM loaders ────────────────────────────────────
def load_llm_gguf(gguf_path: str):
    """Wrap a llama.cpp GGUF quant model so it looks like HF generate()."""
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path,
                n_ctx=2048,
                n_threads=min(4, os.cpu_count() or 4))
    tok  = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    class Wrapper:
        def __init__(self, l): self.l = l
        def generate(self, input_ids=None, max_new_tokens=128,
                     temperature=0.2, do_sample=False, **_):
            prompt = tok.decode(input_ids[0])
            out = self.l(prompt, max_tokens=max_new_tokens,
                         temperature=temperature)["choices"][0]["text"]
            full = tok(prompt + out, return_tensors="pt").input_ids
            return full
        @property
        def device(self): return torch.device("cpu")
    return tok, Wrapper(llm)

def load_llm_hf(local_dir: str):
    """Load HF chat model (CPU-only fallback if no CUDA)."""
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    if torch.cuda.is_available():
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                local_dir, device_map="auto",
                quantization_config=QUANT_CFG, trust_remote_code=True)
            mdl.name_or_path = local_dir
            return tok, mdl
        except Exception as e:
            print("[warn] 4-bit load failed – CPU fp32:", e)

    bf16_ok = getattr(torch, "bfloat16", None) is not None
    dtype   = torch.bfloat16 if bf16_ok else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        local_dir, device_map={"": "cpu"},
        torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True)
    return tok, mdl

# ───────────────────────  LLM JSON extractor  ─────────────────────────────
def llm_json_extract(llm, tok, text: str, max_new=400):
    prompt = (
        "You are an HR assistant.\n"
        "Return **ONLY** valid JSON, no markdown, no explanations.\n"
        "Keys required:\n"
        "  - job_titles        (max-3)\n"
        "  - skills            (max-10)\n"
        "  - suggested_roles   (exact-3)\n\n"
        "Résumé:\n" + text + "\n\nJSON:\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
    )
    raw = tok.decode(out[0], skip_special_tokens=True)

    # grab first balanced {...}
    depth = 0; start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blob = raw[start:i+1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    break
    print("[warn] no valid JSON – raw output follows\n", raw)
    return {}

# ──────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",   required=True, help="Path to résumé PDF")
    ap.add_argument("--emb",   choices=list(LOCAL_EMB),
                    default="minilm",
                    help="embedding encoder (minilm | mpnet | gtr)")
    ap.add_argument("--model", choices=list(LOCAL_CHAT),
                    default="tinyllama",
                    help="chat model (tinyllama | phi2)")
    ap.add_argument("--gguf",  help="Path to .gguf 4-bit model for phi2")
    args = ap.parse_args()

    # 1. Load embedding model
    global EMB_MODEL
    EMB_MODEL = SentenceTransformer(
        LOCAL_EMB[args.emb], device="cpu", trust_remote_code=True)

    # 2. Clean résumé
    cv_clean = clean_text(args.pdf)

    # 3. Embed (cached) – not yet used downstream but kept for future RAG
    _ = embed([cv_clean], args.emb)[0]

    # 4. Load chat-completion LLM
    tok, llm = (load_llm_gguf(args.gguf) if args.gguf
                else load_llm_hf(LOCAL_CHAT[args.model]))

    # 5. Extract JSON
    llm_info = llm_json_extract(llm, tok, cv_clean)

    # 6. Output
    print(json.dumps(llm_info, indent=2))

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
