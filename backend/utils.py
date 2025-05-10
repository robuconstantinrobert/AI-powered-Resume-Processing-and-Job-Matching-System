import re, fitz, hashlib, pickle, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)

LOCAL_CHAT = {
    "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
    "zephyr": "./stablelm-zephyr-3b",
    "qwen": "./Qwen2.5-1.5B-Instruct",
}

QUANT_CFG = None

def clean_text(file):
    pdf = fitz.open(stream=file.read(), filetype='pdf')
    raw = "\n".join(page.get_text() for page in pdf)
    txt = re.sub(r"\b\d{7,}\b|\S+@\S+|https?://\S+|www\.\S+", " ", raw)
    keep = []
    SECTION_HEADINGS = {
        "contact","about me","projects","work experience","education",
        "certifications","languages","skills","qualities"
    }
    def _is_heading(line): 
        t = line.strip().lower()
        return t.isupper() and any(t.startswith(h) for h in SECTION_HEADINGS)
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or _is_heading(ln) or (ln.isupper() and len(ln.split()) <= 2):
            continue
        keep.append(ln)
    txt = " ".join(keep)
    txt = re.sub(r"[^A-Za-z0-9.,:/\\+& -]", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).lower().strip()
    return txt

def _sha1(s): return hashlib.sha1(s.encode()).hexdigest()

def embed(texts, emb_key):
    from services import EMB_MODEL
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

def load_llm_hf(local_dir):
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto")
    return tok, model

def load_llm_gguf(gguf_path):
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=min(4, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    class Wrapper:
        def __init__(self, l): self.l = l
        def generate(self, input_ids=None, max_new_tokens=128, temperature=0.2, do_sample=False, **_):
            prompt = tok.decode(input_ids[0])
            out = self.l(prompt, max_tokens=max_new_tokens, temperature=temperature)["choices"][0]["text"]
            full = tok(prompt + out, return_tensors="pt").input_ids
            return full
        @property
        def device(self): return torch.device("cpu")
    return tok, Wrapper(llm)

def build_prompt(model_key, resume, tok):
    if model_key == "tinyllama":
        return (
            "You are an HR assistant.\n"
            "Reply ONLY with valid JSON.\n\n"
            "Required keys & limits:\n"
            "  job_titles        (max 3)\n"
            "  skills            (max 10)\n"
            "  suggested_roles   (exact 3)\n\n"
            "Résumé:\n" + resume + "\n\nJSON:\n"
        )
    elif model_key == "zephyr":
        return (
            "<|system|>You are a helpful HR assistant.<|end|>\n"
            "<|user|>Extract structured data from the résumé below.\n"
            "Return ONLY valid JSON with keys:\n"
            "  job_titles (max 3)\n"
            "  skills (max 10)\n"
            "  suggested_roles (exact 3)\n\n"
            "Résumé:\n" + resume + "<|end|>\n"
            "<|assistant|>"
        )
    else:
        sys = "<|im_start|>system\nYou are an HR assistant.<|im_end|>\n"
        usr = (
            "<|im_start|>user\n"
            "Extract job_titles (max 3), skills (max 10) and suggested_roles "
            "(exact 3) from the résumé below and reply with ONLY JSON.\n\n"
            + resume + "<|im_end|>\n"
        )
        assistant = "<|im_start|>assistant\n"
        return sys + usr + assistant

def llm_json_extract(llm, tok, prompt, decoding_kwargs):
    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, eos_token_id=tok.eos_token_id, **decoding_kwargs)
    raw = tok.decode(out[0], skip_special_tokens=True)
    depth = 0; start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    break
    return {"error": "Failed to extract JSON"}
