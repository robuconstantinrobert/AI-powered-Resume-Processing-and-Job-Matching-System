import json
from utils import clean_text, embed, load_llm_hf, load_llm_gguf, build_prompt, llm_json_extract
from sentence_transformers import SentenceTransformer
from utils import LOCAL_EMB, LOCAL_CHAT, DECODE
import numpy as np
from utils import pg_conn, fetch_occupations, fetch_skills, embed_texts, _sha1, CACHE_DIR, prompt_for, extract_json, load_llm, extract_json_fixed
import psycopg2
from mongo import get_documents_collection
from bson.objectid import ObjectId
from datetime import datetime
import os
import re

import torch
import gc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EMB_MODEL = None

def process_cv_service(file, emb_key, model_key, gguf_path, user_id, file_name):
    global EMB_MODEL

    emb_path = LOCAL_EMB.get(emb_key, emb_key)
    model_path = LOCAL_CHAT.get(model_key, model_key)

    EMB_MODEL = SentenceTransformer(emb_path, device=DEVICE, trust_remote_code=True)

    resume_text = clean_text(file)
    cv_vec = embed([resume_text], emb_key)[0]

    tok, llm = (load_llm_gguf(gguf_path) if gguf_path else load_llm_hf(model_path))

    prompt = build_prompt(model_key, resume_text, tok)

    result = llm_json_extract(llm, tok, prompt, {
        'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7
    })

    save_processed_document(
        user_id = ObjectId(user_id),
        file_name = file_name,
        raw_text=resume_text,
        vector=cv_vec,
        extracted_data=result
    )

    return result


def process_cv_with_esco_service(file, emb_key, model_key, top_n, user_id, file_name):
    emb_model = SentenceTransformer(LOCAL_EMB[emb_key], device='cpu', trust_remote_code=True)
    model_path = LOCAL_CHAT.get(model_key, model_key)

    resume_text = clean_text(file)
    cv_vec = emb_model.encode(resume_text, normalize_embeddings=True)

    conn = pg_conn()
    cur = conn.cursor()

    occ_ids, occ_txt = fetch_occupations(cur)
    occ_vecs = embed_texts(emb_model, occ_txt, emb_key + "_occ")

    skill_ids, skill_txt, skill_labels = fetch_skills(cur)
    skill_vecs = embed_texts(emb_model, skill_txt, emb_key + "_skill")

    occ_sims = occ_vecs @ cv_vec
    top_occ_idx = occ_sims.argsort()[-top_n:][::-1]
    top_occ_txt = [occ_txt[i] for i in top_occ_idx]

    skill_sims = skill_vecs @ cv_vec
    top_skill_idx = skill_sims.argsort()[-top_n:][::-1]
    top_skill_lbl = [skill_labels[i] for i in top_skill_idx]

    tok, llm = load_llm(LOCAL_CHAT[model_key])

    prompt = prompt_for(model_key, resume_text, top_occ_txt, top_skill_lbl)
    result = extract_json_fixed(llm,tok,prompt,DECODE[model_key])

    save_processed_document(
        user_id = ObjectId(user_id),
        file_name = file_name,
        raw_text=resume_text,
        vector=cv_vec,
        extracted_data=result
    )

    cur.close()
    conn.close()
    return result


def save_processed_document(user_id, raw_text, vector, extracted_data, file_name):
    collection = get_documents_collection()

    base, ext = os.path.splitext(file_name)
    esc_base = re.escape(base)
    esc_ext  = re.escape(ext)
    pattern = f"^{esc_base}(\\(\\d+\\))?{esc_ext}$"

    existing_count = collection.count_documents({
        "utilizator_id": ObjectId(user_id),
        "file_name":     {"$regex": pattern}
    })

    if existing_count:
        stored_name = f"{base}({existing_count+1}){ext}"
    else:
        stored_name = file_name

    doc = {
        "utilizator_id": ObjectId(user_id),
        "file_name": stored_name,
        "continut_text": raw_text,
        "continut_vector": vector.tolist(),
        "data_upload": datetime.utcnow(),
        "date_extrase": {
            "competente": extracted_data.get("skills", []),
            "job_titles": extracted_data.get("job_titles", []),
            "suggested_roles": extracted_data.get("suggested_roles", []),
            "work_experience": extracted_data.get("seniority_level", [])
        }
    }

    result = collection.insert_one(doc)
    return str(result.inserted_id)


def get_documents_by_user(user_id):
    collection = get_documents_collection()
    return list(collection.find({"utilizator_id": ObjectId(user_id)}))
