import re, fitz, hashlib, pickle, json, torch, os
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, logging as hf_logging)
from postgres import save_embedding_pg, fetch_embedding_pg
import numpy as np
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import json
import os
import urllib
from linkedin_scraper.job_search import JobSearch as OriginalJobSearch
from linkedin_scraper.jobs import Job
import jwt
from functools import wraps

SECRET_KEY = "cheia_mea_secreta"

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

SECTION_KEYWORDS = {
    "contact", "about", "summary", "projects", "experience", "education",
    "certifications", "languages", "skills", "qualities", "publications",
    "interests", "awards", "volunteer"
}

PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s\-.])?"   
    r"\(?\d{2,4}\)?[\s\-.]?"   
    r"\d{2,4}[\s\-.]?\d{2,4}",
)
EMAIL_URL_RE = re.compile(r"\S+@\S+|https?://\S+|www\.\S+", re.IGNORECASE)

def clean_text(file) -> str:
    """
    1) Extracts each line’s text + bbox via fitz.
    2) Learns the header/footer y-zones (top/bottom 10% by average).
    3) Filters out any line within those zones.
    4) Groups remaining lines into “paragraphs” by vertical proximity.
    5) Strips phones, emails, URLs, non-essential punctuation, extra spaces.
    6) Drops section headers (all-caps + startswith keyword) and tiny labels.
    """
    if isinstance(file, str) and not Path(file).is_file():
        raw = file
        lines_with_bbox = []
    else:
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        lines_with_bbox = [] 
        header_zones = []
        footer_zones = []

        for page in doc:
            H = page.rect.height
            header_zones.append((0, H * 0.1))
            footer_zones.append((H * 0.9, H))

            txt_dict = page.get_text("dict")
            for block in txt_dict["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = line["bbox"]
                    lines_with_bbox.append((text, y0, y1))

        doc.close()

    if not lines_with_bbox:
        txt = raw
    else:
        h_min = sum(lo for lo, _ in header_zones) / len(header_zones)
        h_max = sum(hi for _, hi in header_zones) / len(header_zones)
        f_min = sum(lo for lo, _ in footer_zones) / len(footer_zones)
        f_max = sum(hi for _, hi in footer_zones) / len(footer_zones)

        kept = [
            (t, y0, y1)
            for t, y0, y1 in lines_with_bbox
            if not (h_min <= y1 <= h_max or f_min <= y0 <= f_max)
        ]

        kept.sort(key=lambda x: x[1])
        paras = []
        cur_lines = [kept[0][0]]
        last_bot = kept[0][2]

        for text, y0, y1 in kept[1:]:
            if abs(y0 - last_bot) < 5:
                cur_lines.append(text)
            else:
                paras.append(" ".join(cur_lines))
                cur_lines = [text]
            last_bot = y1
        paras.append(" ".join(cur_lines))

        txt = "\n\n".join(paras)

    txt = PHONE_RE.sub(" ", txt)
    txt = EMAIL_URL_RE.sub(" ", txt)
    txt = re.sub(r"[^\w\s\.\,\-\/&]", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()

    final_lines = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if ln.isupper() and any(low.startswith(k) for k in SECTION_KEYWORDS):
            continue
        if ln.isupper() and len(ln.split()) <= 3:
            continue
        final_lines.append(ln)

    return " ".join(final_lines)


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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,       
            quantization_config=QUANT_CFG, 
            trust_remote_code=True,
        )
    except Exception as e:
        print("[warn] Could not load on GPU, falling back to CPU:", e)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
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
            "You are given a Résumé.\n" + resume + "\n\n"
            "Reply ONLY with valid JSON and find possible suggested roles, skills and a seniority level based on the given resume\n\n"
            "Required keys & limits:\n"
            "  job_titles        (maximum of 3)\n"
            "  skills            (maximum of 5)\n"
            "  suggested_roles   (exactly 3)\n"
            "  seniority_level   (only one example from: \"Junior\", \"Mid\", \"Senior\", or \"Not enough data\")\n\n"
            "Return a valid JSON with the required keys, JSON:\n"
        )
    elif model_key == "zephyr":
        return (
            "<|system|>You are given a Résumé.\n" + resume + "\n<|end|>\n"
            "<|user|>Extract structured data from the résumé below.Reply ONLY with valid JSON and find possible suggested roles, skills and a seniority level based on the given resume\n"
            "  job_titles (maximum of 3)\n"
            "  skills (maximum of 5)\n"
            "  suggested_roles (exactly 3)\n"
            "  seniority_level (only one example from: \"Junior\", \"Mid\", \"Senior\", or \"Not enough data\")\n\n"
            "<|assistant|> Return a valid JSON with the required keys, JSON:\n"
        )
    else:
        sys = "<|im_start|>system\nYou are given a Résumé.\n" + resume + "\n<|im_end|>\n"
        usr = (
            "<|im_start|>user\n"
            "Reply ONLY with valid JSON and find possible suggested roles, skills and a seniority level based on the given resume. Extract job_titles (maximum of 3), skills (maximum of 5), suggested_roles "
            "(exactly 3) and seniority_level (only one example from: \"Junior\", \"Mid\", \"Senior\", or \"Not enough data\") from the résumé below and reply with ONLY JSON.\n\n"
            "Return a valid JSON with the required keys, JSON: <|im_end|>\n"
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
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="cuda",
            quantization_config=QUANT_CFG,
            trust_remote_code=True,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    return tok, mdl


def extract_json_fixed(llm, tok, prompt, decoding_kwargs):
    import re

    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    decoding_kwargs.setdefault('max_new_tokens', 1024)

    out = llm.generate(**inputs, eos_token_id=tok.eos_token_id, **decoding_kwargs)
    raw = tok.decode(out[0], skip_special_tokens=True)

    print("[DEBUG] Raw LLM output:\n", raw)
    print("[DEBUG] Output length:", len(raw))

    json_matches = re.findall(r"\{[\s\S]*?\}", raw)

    for candidate in json_matches:
        try:
            parsed = json.loads(candidate)

            return {
                "job_titles": parsed.get("job_titles", parsed.get("jobTitles", [])),
                "skills": parsed.get("skills", []),
                "suggested_roles": parsed.get("suggested_roles", parsed.get("suggestedRoles", []))
            }

        except json.JSONDecodeError:
            continue

    return {"error": "Failed to extract JSON"}


def load_linkedin_cookies(driver, cookie_file):
    driver.get("https://www.linkedin.com")
    time.sleep(3)
    with open(cookie_file, 'r') as f:
        cookies = json.load(f)
    for cookie in cookies:
        cookie.pop('sameSite', None)
        cookie.pop('expiry', None)
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"Cookie error: {e}")
    driver.get("https://www.linkedin.com/feed")
    time.sleep(3)

class FixedJobSearch(OriginalJobSearch):
    def scrape_job_card(self, base_element):
        try:
            title_elem = None
            for selector in [
                "a.job-card-list__title-link",
                "a.job-card-container__link",
                "a.jobs-unified-top-card__job-title-link",
                "a.job-card-list__title",
                "a.job-card-list__title--link",
                "job-card-list__title--link"
            ]:
                try:
                    title_elem = base_element.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not title_elem:
                print("Could not find job title element")
                return None

            job_title = title_elem.get_attribute("aria-label") or title_elem.text.strip()
            linkedin_url = title_elem.get_attribute("href")

            company = ""
            for selector in [
                ".artdeco-entity-lockup__subtitle span",
                ".job-card-container__company-name",   
                ".jobs-unified-top-card__company-name a" 
            ]:
                try:
                    company = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            location = ""
            for selector in [
                ".job-card-container__metadata-wrapper li:first-child",
                ".job-card-container__metadata-item",                   
                ".jobs-unified-top-card__primary-description span" 
            ]:
                try:
                    location = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            salary = None
            for selector in [
                ".job-card-container__metadata-wrapper li:nth-child(2)",
                ".job-salary",                         
                ".job-card-container__salary-info"        
            ]:
                try:
                    salary = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            job = Job(
                linkedin_url=linkedin_url,
                job_title=job_title,
                company=company,
                location=location,
                salary=salary,
                scrape=False,
                driver=self.driver
            )
            return job
            
        except Exception as e:
            print(f"Error scraping job card: {e}")
            return None


    def search(self, search_term: str):
        url = f"{self.base_url}search/?keywords={urllib.parse.quote(search_term)}"
        
        try:
            self.driver.get(url)
            print("Page loaded successfully")
        except Exception as e:
            print(f"Failed to load page: {e}")
            return []

        time.sleep(random.uniform(3, 6))

        try:
            if "authwall" in self.driver.current_url:
                print("Hit a login wall, trying to login again")
                #actions.login(self.driver, email, password)
                load_linkedin_cookies(self.driver)
                time.sleep(random.uniform(2, 4))
                self.driver.get(url)
                time.sleep(random.uniform(3, 5))
        except:
            pass

        # Scroll to load more jobs
        # for _ in range(10):
        #     self.scroll_to_bottom()
        #     time.sleep(random.uniform(1, 2))
        scroll_button_value = ""
        try:
            # Wait for the buttons with specific text to appear
            jump_buttons = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH,
                    "//button[contains(., 'Jump to active job details') or contains(., 'Jump to active search result')]"))
            )

            # Once we have the button(s), traverse up to their common ancestor <div>
            for button in jump_buttons:
                ancestor_div = button.find_element(By.XPATH, "./ancestor::div[1]")
                dynamic_class = ancestor_div.get_attribute("class")
                if dynamic_class:
                    print(f"[INFO] Found dynamic container class: {dynamic_class}")
                    scroll_button_value = dynamic_class
                    break
        except Exception as e:
            print(f"[ERROR] Could not find dynamic container: {e}")
            return []

        #self.scroll_to_bottom(pause_time=2, max_attempts=30, min_job_count=25)
        if scroll_button_value:
            self.scroll_to_bottom(
                pause_time=2,
                max_attempts=10,
                min_job_count=25,
                scroll_button_value=scroll_button_value
            )
        else:
            print("[ERROR] Cannot scroll, scroll_button_value not found")
            return []


        #Find job listings container with multiple fallbacks
        container = None
        container_selectors = [
            "div.scaffold-layout__list-container",  # Primary selector
            "div.jobs-search-results-list",         # Fallback 1
            "div.jobs-search-results",              # Fallback 2
            "div.scaffold-layout__list",            # Fallback 3
            "main.scaffold-layout__main"            # Fallback 4
        ]

        for selector in container_selectors:
            try:
                container = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                print(f"Found container with selector: {selector}")
                break
            except:
                continue

        if not container:
            print("Could not find job listings container. Current page source:")
            print(self.driver.page_source[:2000])  # Print first 2000 chars for debugging
            return []

        # Find all job cards with multiple selectors
        job_cards = []
        for selector in [
            "li.jobs-search-results__list-item",  # Primary selector
            "li.scaffold-layout__list-item",       # Fallback 1
            "div.job-card-container",              # Fallback 2
            "section.jobs-search-results__list-item" # Fallback 3
        ]:
            try:
                job_cards = container.find_elements(By.CSS_SELECTOR, selector)
                if job_cards:
                    print(f"Found {len(job_cards)} jobs with selector: {selector}")
                    break
            except:
                continue

        job_results = []
        for card in job_cards:
            try:
                job = self.scrape_job_card(card)
                if job:
                    job_results.append(job)
            except Exception as e:
                print(f"Failed to process a job card: {e}")
                continue
                
        return job_results


def JWT(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"error": "Token lipsă"}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user_id = data["user_id"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expirat"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token invalid"}), 401

        return f(*args, **kwargs)
    return decorated
