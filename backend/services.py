import json
from utils import clean_text, embed, load_llm_hf, load_llm_gguf, build_prompt, llm_json_extract
from sentence_transformers import SentenceTransformer
from utils import LOCAL_EMB, LOCAL_CHAT
import numpy as np
from utils import pg_conn, fetch_occupations, fetch_skills, embed_texts, _sha1, CACHE_DIR, prompt_for, extract_json
import psycopg2


EMB_MODEL = None

def process_cv_service(file, emb_key, model_key, gguf_path):
    global EMB_MODEL

    # Obține calea locală din mapări
    emb_path = LOCAL_EMB.get(emb_key, emb_key)
    model_path = LOCAL_CHAT.get(model_key, model_key)

    # Încarcă modelul de embedding
    EMB_MODEL = SentenceTransformer(emb_path, device='cpu', trust_remote_code=True)

    # Curăță textul din PDF
    resume_text = clean_text(file)
    _ = embed([resume_text], emb_key)[0]  # doar cache

    # Încarcă modelul LLM (din Hugging Face sau gguf)
    tok, llm = (load_llm_gguf(gguf_path) if gguf_path else load_llm_hf(model_path))

    # Construiește promptul
    prompt = build_prompt(model_key, resume_text, tok)

    # Rulează completarea și extrage JSON-ul
    result = llm_json_extract(llm, tok, prompt, {
        'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7
    })

    return result


def process_cv_with_esco_service(file, emb_key, model_key, top_n):

    # Încarcă model embedding
    emb_model = SentenceTransformer(LOCAL_EMB[emb_key], device='cpu', trust_remote_code=True)
    model_path = LOCAL_CHAT.get(model_key, model_key)

    # Citește textul din fișier
    resume_text = clean_text(file)
    cv_vec = emb_model.encode(resume_text, normalize_embeddings=True)

    # Conectare la Postgres
    conn = pg_conn()
    cur = conn.cursor()

    # Extrage ocupații și skilluri din DB
    occ_ids, occ_txt = fetch_occupations(cur)
    occ_vecs = embed_texts(emb_model, occ_txt, emb_key + "_occ")

    skill_ids, skill_txt, skill_labels = fetch_skills(cur)
    skill_vecs = embed_texts(emb_model, skill_txt, emb_key + "_skill")

    # Similaritate cosine
    occ_sims = occ_vecs @ cv_vec
    top_occ_idx = occ_sims.argsort()[-top_n:][::-1]
    top_occ_txt = [occ_txt[i] for i in top_occ_idx]

    skill_sims = skill_vecs @ cv_vec
    top_skill_idx = skill_sims.argsort()[-top_n:][::-1]
    top_skill_lbl = [skill_labels[i] for i in top_skill_idx]

    # Încarcă model LLM
    tok, llm = load_llm_hf(model_path)

    # Construiește prompt + extrage JSON
    prompt = prompt_for(model_key, resume_text, top_occ_txt, top_skill_lbl)
    result = llm_json_extract(llm, tok, prompt, {'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7})

    cur.close()
    conn.close()
    return result
