import json
from utils import clean_text, embed, load_llm_hf, load_llm_gguf, build_prompt, llm_json_extract
from sentence_transformers import SentenceTransformer

EMB_MODEL = None

def process_cv_service(file, emb_key, model_key, gguf_path):
    global EMB_MODEL
    EMB_MODEL = SentenceTransformer(f"./{emb_key}", device='cpu', trust_remote_code=True)
    
    # extrage textul din PDF
    resume_text = clean_text(file)
    _ = embed([resume_text], emb_key)[0]  # doar cache
    
    # încarcă modelul LLM
    tok, llm = (load_llm_gguf(gguf_path) if gguf_path else load_llm_hf(f"./{model_key}"))
    
    # pregătește promptul
    prompt = build_prompt(model_key, resume_text, tok)
    
    # execută completarea
    result = llm_json_extract(llm, tok, prompt, {
        'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7
    })
    
    return result
