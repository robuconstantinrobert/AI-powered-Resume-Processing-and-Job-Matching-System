import re, fitz, hashlib, pickle, json, torch, os
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, logging as hf_logging)
from postgres import save_embedding_pg, fetch_embedding_pg
import numpy as np

CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)

LOCAL_CHAT = {
    "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
    "zephyr":    "./stablelm-zephyr-3b",
    "qwen":      "./Qwen2.5-1.5B-Instruct",
}

LOCAL_EMB = {
    "minilm": "./all-MiniLM-L6-v2", 
    "mpnet":  "./all-mpnet-base-v2",
    "gtr":    "./gtr-t5-base",
}

DECODE=dict(tinyllama=dict(max_new_tokens=512,do_sample=True,temperature=.7),
            zephyr    =dict(max_new_tokens=512,do_sample=True,temperature=.7),
            qwen      =dict(max_new_tokens=512,do_sample=True,temperature=.7))

QUANT_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

torch.set_num_threads(min(4, os.cpu_count() or 4))

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
        embedding = fetch_embedding_pg(_sha1(t), emb_key)
        if embedding is None:
            v = EMB_MODEL.encode(t)
            save_embedding_pg(_sha1(t), emb_key, v)
            vecs.append(v)
        else:
            vecs.append(embedding)
    return vecs

def load_llm_hf(local_dir):
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_dir, device_map="auto",
                quantization_config=QUANT_CFG, trust_remote_code=True)
            return tok, model
        except Exception as e:
            print("[warn] 4-bit GPU load failed – falling back to CPU:", e)
    
    dtype = torch.bfloat16 if getattr(torch, "bfloat16", None) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, device_map={"": "cpu"},
        torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True)
    return tok, model

def load_llm_gguf(gguf_path):
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=min(4, os.cpu_count() or 4))
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

    print("[DEBUG] Raw LLM output:\n", raw)

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


def pg_conn():
    import psycopg2
    return psycopg2.connect(dbname='vector_database', user='postgres', password='password', host='localhost', port=5432)

def fetch_occupations(cur):
    cur.execute("SELECT concepturi, preferredlabel, COALESCE(description, '') FROM occupation")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [f"{r[1]} {r[2]}" for r in rows]
    return ids, texts

def fetch_skills(cur):
    cur.execute("SELECT concepturi, preferredlabel, COALESCE(description, '') FROM skill")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [f"{r[1]} {r[2]}" for r in rows]
    labels = [r[1] for r in rows]
    return ids, texts, labels

def embed_texts(model, texts, key_prefix):
    vecs = []
    for t in texts:
        f = CACHE_DIR / f"{key_prefix}_{_sha1(t)}.pkl"
        if f.exists():
            vecs.append(pickle.loads(f.read_bytes()))
        else:
            v = model.encode(t, normalize_embeddings=True)
            f.write_bytes(pickle.dumps(v))
            vecs.append(v)
    return np.vstack(vecs)

def prompt_for(model_key,resume,occ_txt,skill_txt):
    occ_block="\n".join(f"- {t}" for t in occ_txt)
    skill_block=", ".join(skill_txt)
    if model_key=="tinyllama":
        return (
          "You are an HR assistant.\n"
          "Reply ONLY with valid JSON.\n\n"
          "Required keys & limits:\n"
          "  job_titles        (max 3)\n"
          "  skills            (max 10)\n"
          "  suggested_roles   (exact 3)\n\n"
          f"Résumé:\n {resume} \n\n Occupations:\n{occ_block}\n\n"
          f"Skills:\n{skill_block}\n\nJSON:\n")
    if model_key=="zephyr":
        return (
          "<|system|>You are a helpful HR assistant.<|end|>\n"
          "<|user|>Extract structured data from the résumé below.\n"
          "Return ONLY valid JSON with keys:\n"
            "  job_titles (max 3)\n"
            "  skills (max 10)\n"
            "  suggested_roles (exact 3)\n\n"
          f"Résumé:\n{resume}\n\n Occupations:\n{occ_block}\n\n"
          f"Skills:\n{skill_block}<|end|>\n<|assistant|>")
    # qwen
    return (
      "<|im_start|>system\nYou are an HR assistant.<|im_end|>\n"
      "<|im_start|>user\nPlease extract the information as JSON "
      "(job_titles max 3, skills max 10, suggested_roles 3) "
      "using the résumé and the context below.\n\n"
      f"Résumé:\n{resume}\n\nOccupations:\n{occ_block}\n\n"
      f"Skills:\n{skill_block}<|im_end|>\n<|im_start|>assistant\n")


def extract_json(llm,tok,prompt,dec):
    ids=tok(prompt,return_tensors="pt").to(llm.device)
    out=llm.generate(**ids,eos_token_id=tok.eos_token_id,**dec)
    txt=tok.decode(out[0],skip_special_tokens=True)
    s=txt.find("{"); e=txt.rfind("}")
    if s!=-1 and e!=-1:
        try: return json.loads(txt[s:e+1])
        except: pass
    print("[warn] bad JSON\n",txt); return {}

def load_llm(local_dir):
    tok=AutoTokenizer.from_pretrained(local_dir,trust_remote_code=True)
    if torch.cuda.is_available():
        try:
            mdl=AutoModelForCausalLM.from_pretrained(
                 local_dir,device_map="auto",
                 quantization_config=QUANT_CFG,trust_remote_code=True)
            return tok,mdl
        except Exception: pass
    mdl=AutoModelForCausalLM.from_pretrained(
         local_dir,device_map={"": "cpu"},
         torch_dtype=torch.float32,low_cpu_mem_usage=True,
         trust_remote_code=True)
    return tok,mdl

def extract_json_fixed(llm, tok, prompt, decoding_kwargs):
    import re

    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    decoding_kwargs.setdefault('max_new_tokens', 1024)

    out = llm.generate(**inputs, eos_token_id=tok.eos_token_id, **decoding_kwargs)
    raw = tok.decode(out[0], skip_special_tokens=True)

    print("[DEBUG] Raw LLM output:\n", raw)
    print("[DEBUG] Output length:", len(raw))

    # Caută toate blocurile JSON posibile
    json_matches = re.findall(r"\{[\s\S]*?\}", raw)

    for candidate in json_matches:
        try:
            parsed = json.loads(candidate)

            # Normalizare chei pentru Mongo
            return {
                "job_titles": parsed.get("job_titles", parsed.get("jobTitles", [])),
                "skills": parsed.get("skills", []),
                "suggested_roles": parsed.get("suggested_roles", parsed.get("suggestedRoles", []))
            }

        except json.JSONDecodeError:
            continue

    return {"error": "Failed to extract JSON"}

