#!/usr/bin/env python
"""
app.py – résumé analyser + ESCO-based job/skill recommender
-----------------------------------------------------------
• COSINE retrieval over ~3 000 ESCO occupations (Postgres)
• semantic skill retrieval (14 k ESCO skills, cached vectors)
• three local embedding encoders   : minilm | mpnet | gtr
• three local chat models          : tinyllama | zephyr | qwen
"""
# ──────────────────────── imports ─────────────────────────
import argparse, json, pickle, hashlib, re, os, sys, sqlite3
from pathlib import Path
import fitz, torch, numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, logging as hf_logging)
from sentence_transformers import SentenceTransformer

try:
    from peft import PeftModel; PEFT=True
except ModuleNotFoundError:
    PEFT=False

hf_logging.set_verbosity_error()

# ─────────── local folders for chat models ───────────────
LOCAL_CHAT = {
    "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
    "zephyr":    "./stablelm-zephyr-3b",
    "qwen":      "./Qwen2.5-1.5B-Instruct",
}

# ─────────── local folders for embedding encoders ────────
LOCAL_EMB = {
    "minilm": "./all-MiniLM-L6-v2",
    "mpnet":  "./all-mpnet-base-v2",
    "gtr":    "./gtr-t5-base",
}

# optional LoRA adapters
LORA = {
    "tinyllama": "./lora_tinyllama",
    "zephyr":    "./lora_zephyr",
    "qwen":      "./lora_qwen",
}

QUANT_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

CACHE_DIR = Path(".cache");  CACHE_DIR.mkdir(exist_ok=True)
torch.set_num_threads(min(4, os.cpu_count() or 4))

# ─────────── utils ───────────
def _sha1(s:str)->str: return hashlib.sha1(s.encode()).hexdigest()

def clean_text(path_or_str: str) -> str:
    txt = "\n".join(p.get_text() for p in fitz.open(path_or_str)) \
          if Path(path_or_str).is_file() else path_or_str
    txt = re.sub(r"\b\d{7,}\b|\S+@\S+|https?://\S+|www\.\S+", " ", txt)
    keep=[]
    for ln in txt.splitlines():
        t=ln.strip()
        if not t: continue
        if t.isupper() and len(t.split())<=2: continue
        keep.append(t)
    txt=" ".join(keep)
    txt=re.sub(r"[^A-Za-z0-9.,:/\\+& -]"," ",txt)
    return re.sub(r"\s{2,}"," ",txt).lower().strip()

# ─────────── embedding with cache ───────────
def embed_texts(model, texts, key_prefix):
    vecs=[]
    for t in texts:
        f=CACHE_DIR/f"{key_prefix}_{_sha1(t)}.pkl"
        if f.exists(): vecs.append(pickle.loads(f.read_bytes())); continue
        v=model.encode(t,normalize_embeddings=True)
        f.write_bytes(pickle.dumps(v)); vecs.append(v)
    return np.vstack(vecs)

# ─────────── database helpers ───────────
def pg_conn(args):
    import psycopg2
    return psycopg2.connect(
        dbname=args.db,user=args.user,password=args.password,
        host=args.host,port=args.port)

def fetch_occupations(cur):
    cur.execute("select concepturi,preferredlabel,coalesce(description,'') "
                "from occupation")
    rows=cur.fetchall()
    ids=[r[0] for r in rows]
    texts=[f"{r[1]} {r[2]}" for r in rows]
    return ids,texts

def fetch_skills(cur):
    cur.execute("select concepturi,preferredlabel,coalesce(description,'') "
                "from skill")
    rows=cur.fetchall()
    ids=[r[0] for r in rows]
    texts=[f"{r[1]} {r[2]}" for r in rows]
    labels=[r[1] for r in rows]
    return ids,texts,labels

# ─────────── chat prompts ───────────
def prompt_for(model_key,resume,occ_txt,skill_txt):
    occ_block="\n".join(f"- {t}" for t in occ_txt)
    skill_block=", ".join(skill_txt)
    if model_key=="tinyllama":
        return (
          "You are an HR assistant.\n"
          "Below is a candidate résumé, similar ESCO occupations, "
          "and skills extracted via semantic search.\n"
          "Return ONLY valid JSON with keys:\n"
          "  job_titles (max 3) • skills (max 10) • suggested_roles (exact 3)\n\n"
          f"Résumé:\n{resume}\n\nOccupations:\n{occ_block}\n\n"
          f"Skills:\n{skill_block}\n\nJSON:\n")
    if model_key=="zephyr":
        return (
          "<|system|>You are a helpful HR assistant.<|end|>\n"
          "<|user|>Extract structured info from the résumé and context.\n"
          "Return ONLY JSON with keys job_titles, skills, suggested_roles.\n"
          f"Résumé:\n{resume}\n\nOccupations:\n{occ_block}\n\n"
          f"Skills:\n{skill_block}<|end|>\n<|assistant|>")
    # qwen
    return (
      "<|im_start|>system\nYou are an HR assistant.<|im_end|>\n"
      "<|im_start|>user\nPlease extract the information as JSON "
      "(job_titles max 3, skills max 10, suggested_roles 3) "
      "using the résumé and the context below.\n\n"
      f"Résumé:\n{resume}\n\nOccupations:\n{occ_block}\n\n"
      f"Skills:\n{skill_block}<|im_end|>\n<|im_start|>assistant\n")

DECODE=dict(tinyllama=dict(max_new_tokens=512,do_sample=True,temperature=.7),
            zephyr    =dict(max_new_tokens=512,do_sample=True,temperature=.7),
            qwen      =dict(max_new_tokens=512,do_sample=True,temperature=.7))

# ─────────── LLM loaders (same as previous answer) ───────────
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

# ─────────── JSON extractor ───────────
def extract_json(llm,tok,prompt,dec):
    ids=tok(prompt,return_tensors="pt").to(llm.device)
    out=llm.generate(**ids,eos_token_id=tok.eos_token_id,**dec)
    txt=tok.decode(out[0],skip_special_tokens=True)
    s=txt.find("{"); e=txt.rfind("}")
    if s!=-1 and e!=-1:
        try: return json.loads(txt[s:e+1])
        except: pass
    print("[warn] bad JSON\n",txt); return {}

# ─────────── main ───────────
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pdf",required=True)
    p.add_argument("--emb",choices=LOCAL_EMB,default="minilm")
    p.add_argument("--model",choices=LOCAL_CHAT,default="tinyllama")
    p.add_argument("--db",default="vector_database");p.add_argument("--user",default="postgres")
    p.add_argument("--password",default=os.getenv("PGPASSWORD","password"))
    p.add_argument("--host",default="localhost");p.add_argument("--port",type=int,default=5432)
    p.add_argument("--top",type=int,default=3,help="top-N occupations & skills")
    args=p.parse_args()

    # 1 embedding model
    emb=SentenceTransformer(LOCAL_EMB[args.emb],device="cpu",trust_remote_code=True)
    resume=clean_text(args.pdf); cv_vec=emb.encode(resume,normalize_embeddings=True)

    # 2 connect PG + get ESCO data (embed & cache)
    pg=pg_conn(args); cur=pg.cursor()
    occ_ids,occ_txt=fetch_occupations(cur)
    occ_vecs=embed_texts(emb,occ_txt,args.emb+"_occ")
    skill_ids,skill_txt,skill_labels=fetch_skills(cur)
    skill_vecs=embed_texts(emb,skill_txt,args.emb+"_skill")

    # 3 similarity search
    occ_sims = occ_vecs @ cv_vec
    top_occ_idx=occ_sims.argsort()[-args.top:][::-1]
    top_occ_txt=[occ_txt[i] for i in top_occ_idx]

    skill_sims = skill_vecs @ cv_vec
    top_skill_idx=skill_sims.argsort()[-args.top:][::-1]
    top_skill_lbl=[skill_labels[i] for i in top_skill_idx]

    # 4 chat model
    tok,llm=load_llm(LOCAL_CHAT[args.model])
    if PEFT and Path(LORA[args.model]).exists():
        llm=PeftModel.from_pretrained(llm,LORA[args.model],device_map=llm.device)

    prompt=prompt_for(args.model,resume,top_occ_txt,top_skill_lbl)
    info=extract_json(llm,tok,prompt,DECODE[args.model])

    print("\n== Top occupations ==")
    for i,idx in enumerate(top_occ_idx): print(f"{i+1}. {occ_txt[idx][:80]}")
    print("\n== Top skills =="); print(", ".join(top_skill_lbl))
    print("\n== LLM output =="); print(json.dumps(info,indent=2))

if __name__=="__main__":
    main()




# #####################################################################################
# ########################### WORKING 3-EMBEDDERS 3-CHAT COMPLETION ###################
# #####################################################################################
# #!/usr/bin/env python
# """
# app.py – offline résumé analyser for 16 GB laptops
# -------------------------------------------------
# usage examples
# --------------

# # MiniLM + TinyLlama  (default)
# python app.py --pdf CV.pdf

# # MPNet + Zephyr-3B
# python app.py --pdf CV.pdf --emb mpnet --model zephyr

# # GTR-T5 + Qwen-1.5B
# python app.py --pdf CV.pdf --emb gtr --model qwen
# """
# # ─────────────────────────────  imports  ─────────────────────────────
# import argparse, json, pickle, hashlib, re, os, sys
# from pathlib import Path

# import fitz, torch
# from transformers import (AutoTokenizer, AutoModelForCausalLM,
#                           BitsAndBytesConfig, logging as hf_logging)
# from sentence_transformers import SentenceTransformer

# try:
#     from peft import PeftModel                # optional LoRA
#     PEFT = True
# except ModuleNotFoundError:
#     PEFT = False

# hf_logging.set_verbosity_error()

# # ───────────── local folders (chat) ─────────────
# LOCAL_CHAT = {
#     "tinyllama": "./TinyLlama-1.1B-Chat-v1.0",
#     "zephyr":    "./stablelm-zephyr-3b",
#     "qwen":      "./Qwen2.5-1.5B-Instruct",
# }

# # ───────────── local folders (embeddings) ───────
# LOCAL_EMB = {
#     "minilm": "./all-MiniLM-L6-v2",
#     "mpnet":  "./all-mpnet-base-v2",
#     "gtr":    "./gtr-t5-base",
# }

# # optional LoRA adapters – put them here if you train any
# LORA_ADAPTER = {
#     "tinyllama": "./lora_tinyllama",
#     "zephyr":    "./lora_zephyr",
#     "qwen":      "./lora_qwen",
# }

# EMB_MODEL = None

# QUANT_CFG = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16)

# CACHE_DIR = Path(".cache");  CACHE_DIR.mkdir(exist_ok=True)
# torch.set_num_threads(min(4, os.cpu_count() or 4))

# # ───────────── text-cleaning ─────────────
# SECTION_HEADINGS = {
#     "contact","about me","projects","work experience","education",
#     "certifications","languages","skills","qualities"
# }
# def _is_heading(line: str) -> bool:
#     t = line.strip().lower()
#     return t.isupper() and any(t.startswith(h) for h in SECTION_HEADINGS)

# def clean_text(inp: str) -> str:
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
#     txt = re.sub(r"[^A-Za-z0-9.,:/\\+& -]", " ", txt)
#     txt = re.sub(r"\s{2,}", " ", txt).lower().strip()
#     return txt

# # ───────────── embeddings ─────────────
# def _sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest()

# def embed(texts, emb_key: str):
#     vecs = []
#     for t in texts:
#         f = CACHE_DIR / f"{emb_key}_{_sha1(t)}.pkl"
#         if f.exists():
#             vecs.append(pickle.loads(f.read_bytes()))
#         else:
#             v = EMB_MODEL.encode(t)
#             f.write_bytes(pickle.dumps(v))
#             vecs.append(v)
#     return vecs

# # ───────────── LLM loaders ─────────────
# def load_llm_gguf(gguf_path: str):
#     from llama_cpp import Llama
#     llm = Llama(model_path=gguf_path,
#                 n_ctx=2048, n_threads=min(4, os.cpu_count() or 4))
#     tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
#     class Wrapper:
#         def __init__(self, l): self.l = l
#         def generate(self, input_ids=None, max_new_tokens=128,
#                      temperature=0.2, do_sample=False, **_):
#             prompt = tok.decode(input_ids[0])
#             out = self.l(prompt, max_tokens=max_new_tokens,
#                          temperature=temperature)["choices"][0]["text"]
#             full = tok(prompt + out, return_tensors="pt").input_ids
#             return full
#         @property
#         def device(self): return torch.device("cpu")
#     return tok, Wrapper(llm)

# def load_llm_hf(local_dir: str):
#     tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
#     if torch.cuda.is_available():
#         try:
#             mdl = AutoModelForCausalLM.from_pretrained(
#                 local_dir, device_map="auto",
#                 quantization_config=QUANT_CFG, trust_remote_code=True)
#             return tok, mdl
#         except Exception as e:
#             print("[warn] 4-bit GPU load failed – falling back to CPU:", e)

#     dtype = torch.bfloat16 if getattr(torch, "bfloat16", None) else torch.float32
#     mdl = AutoModelForCausalLM.from_pretrained(
#         local_dir, device_map={"": "cpu"},
#         torch_dtype=dtype, low_cpu_mem_usage=True,
#         trust_remote_code=True)
#     return tok, mdl

# # ───────────── model-specific prompts ─────────────
# def build_prompt(model_key: str, resume: str, tok: AutoTokenizer) -> str:
#     if model_key == "tinyllama":
#         return (
#             "You are an HR assistant.\n"
#             "Reply ONLY with valid JSON.\n\n"
#             "Required keys & limits:\n"
#             "  job_titles        (max 3)\n"
#             "  skills            (max 10)\n"
#             "  suggested_roles   (exact 3)\n\n"
#             "Résumé:\n" + resume + "\n\nJSON:\n"
#         )

#     elif model_key == "zephyr":
#         # StableLM-Zephyr follows the ChatML "<|system|>" tags
#         return (
#             "<|system|>You are a helpful HR assistant.<|end|>\n"
#             "<|user|>Extract structured data from the résumé below.\n"
#             "Return ONLY valid JSON with keys:\n"
#             "  job_titles (max 3)\n"
#             "  skills (max 10)\n"
#             "  suggested_roles (exact 3)\n\n"
#             "Résumé:\n" + resume + "<|end|>\n"
#             "<|assistant|>"
#         )

#     else:  # qwen
#         # Qwen-2 chat template: <|im_start|>{role}\n{content}<|im_end|>
#         sys = "<|im_start|>system\nYou are an HR assistant.<|im_end|>\n"
#         usr = (
#             "<|im_start|>user\n"
#             "Extract job_titles (max 3), skills (max 10) and suggested_roles "
#             "(exact 3) from the résumé below and reply with ONLY JSON.\n\n"
#             + resume + "<|im_end|>\n"
#         )
#         assistant = "<|im_start|>assistant\n"
#         return sys + usr + assistant

# # ───────────── decoding defaults ─────────────
# DECODING = {
#     "tinyllama": dict(max_new_tokens=512, do_sample=True, temperature=0.7),
#     "zephyr":    dict(max_new_tokens=512, do_sample=True, temperature=0.7),
#     "qwen":      dict(max_new_tokens=512, do_sample=True, temperature=0.7),
# }

# # ───────────── JSON extractor ─────────────
# def llm_json_extract(llm, tok, prompt: str, decoding_kwargs):
#     inputs = tok(prompt, return_tensors="pt").to(llm.device)
#     out = llm.generate(**inputs,
#                        eos_token_id=tok.eos_token_id,
#                        **decoding_kwargs)
#     raw = tok.decode(out[0], skip_special_tokens=True)

#     depth = 0; start = None
#     for i, ch in enumerate(raw):
#         if ch == "{":
#             if depth == 0: start = i
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0 and start is not None:
#                 try: return json.loads(raw[start:i+1])
#                 except json.JSONDecodeError: break
#     print("[warn] no valid JSON – raw output follows\n", raw)
#     return {}

# # ───────────── main ─────────────
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pdf",   required=True, help="Path to résumé PDF")
#     ap.add_argument("--emb",   choices=list(LOCAL_EMB), default="minilm")
#     ap.add_argument("--model", choices=list(LOCAL_CHAT), default="tinyllama")
#     ap.add_argument("--gguf",  help="Path to optional GGUF quant model")
#     args = ap.parse_args()

#     # 1 embedding
#     global EMB_MODEL
#     EMB_MODEL = SentenceTransformer(LOCAL_EMB[args.emb],
#                                     device="cpu", trust_remote_code=True)
#     resume_text = clean_text(args.pdf)
#     _ = embed([resume_text], args.emb)[0]   # cache only

#     # 2 chat model
#     tok, llm = (load_llm_gguf(args.gguf) if args.gguf
#                 else load_llm_hf(LOCAL_CHAT[args.model]))

#     # 2a optional LoRA
#     if PEFT and Path(LORA_ADAPTER[args.model]).exists():
#         llm = PeftModel.from_pretrained(llm, LORA_ADAPTER[args.model],
#                                         device_map=llm.device)

#     # 3 prompt & generate
#     prompt = build_prompt(args.model, resume_text, tok)
#     llm_info = llm_json_extract(llm, tok, prompt, DECODING[args.model])

#     print(json.dumps(llm_info, indent=2))

# # ─────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     main()

