from flask import Blueprint, request, jsonify
from services import process_cv_service, process_cv_with_esco_service
from mongo import get_documents_collection, save_multiple_job_results, clean_mongo_doc
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
import os, json, time, random, urllib, datetime
from utils import LOCAL_EMB, LOCAL_CHAT, load_llm_hf, build_prompt, llm_json_extract, load_linkedin_cookies, FixedJobSearch
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from linkedin_scraper import JobSearch, actions
from linkedin_scraper.job_search import JobSearch as OriginalJobSearch
from linkedin_scraper.jobs import Job

api_bp = Blueprint('api', __name__)

@api_bp.route('/process_cv', methods=['POST'])
def process_cv():
    file = request.files.get('file')
    emb = request.form.get('emb', 'minilm')
    model = request.form.get('model', 'tinyllama')
    gguf = request.form.get('gguf')  # optional

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    result = process_cv_service(file, emb, model, gguf)
    return jsonify(result)


@api_bp.route('/process_cv_with_esco', methods=['POST'])
def process_cv_with_esco():
    file = request.files.get('file')
    emb = request.form.get('emb', 'minilm')
    model = request.form.get('model', 'tinyllama')
    top = int(request.form.get('top', 3))

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    result = process_cv_with_esco_service(file, emb, model, top)
    return jsonify(result)



@api_bp.route('/documents/<utilizator_id>', methods=['GET'])
def get_documents(utilizator_id):
    collection = get_documents_collection()
    docs = collection.find({"utilizator_id": ObjectId(utilizator_id)})
    results = []
    for doc in docs:
        doc['_id'] = str(doc['_id'])
        doc['utilizator_id'] = str(doc['utilizator_id'])
        results.append(doc)
    return jsonify(results)


@api_bp.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    collection = get_documents_collection()
    result = collection.delete_one({"_id": ObjectId(document_id)})
    if result.deleted_count == 1:
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Document not found"}), 404


@api_bp.route('/documents/<document_id>', methods=['PUT'])
def update_document(document_id):
    collection = get_documents_collection()
    data = request.get_json()
    emb_key = data.get("emb", "minilm")
    model_key = data.get("model", "tinyllama")

    document = collection.find_one({"_id": ObjectId(document_id)})
    if not document:
        return jsonify({"error": "Document not found"}), 404

    raw_text = document["continut_text"]

    # Vector nou și completare LLM
    from services import process_cv_service  # dacă nu e deja importată
    result = process_cv_service(file=raw_text, emb_key=emb_key, model_key=model_key, gguf_path=None)

    # Vector nou
    emb_model = SentenceTransformer(LOCAL_EMB[emb_key], device="cpu", trust_remote_code=True)
    vector = emb_model.encode(raw_text, normalize_embeddings=True)

    update = {
        "continut_vector": vector.tolist(),
        "date_extrase": {
            "competente": result.get("skills", []),
            "job_titles": result.get("job_titles", []),
            "experienta": result.get("experienta", [])  # poate lipsi, dar o includem
        }
    }

    collection.update_one({"_id": ObjectId(document_id)}, {"$set": update})
    return jsonify({"status": "updated", "new_data": update})


@api_bp.route('/linkedin/login', methods=['POST'])
def linkedin_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user_id = data.get('user_id', email.split('@')[0])  # fallback dacă nu se trimite user_id

    if not email or not password:
        return jsonify({"error": "Email și parola sunt necesare."}), 400

    cookie_dir = "linkedin_cookies"
    os.makedirs(cookie_dir, exist_ok=True)
    cookie_file = os.path.join(cookie_dir, f"{user_id}.json")

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--headless=new")
    # options.add_argument("--headless")  # poți activa pentru rulare fără UI

    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://www.linkedin.com/login")
        time.sleep(3)

        driver.find_element(By.ID, "username").send_keys(email)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(5)

        if "feed" not in driver.current_url:
            return jsonify({"error": "Login eșuat. Verifică datele sau CAPTCHA."}), 401

        cookies = driver.get_cookies()
        with open(cookie_file, "w") as f:
            json.dump(cookies, f)

        return jsonify({"message": "Autentificare reușită", "cookie_file": cookie_file})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        driver.quit()


@api_bp.route('/linkedin/search-jobs', methods=['POST'])
def linkedin_search_jobs():
    try:
        data = request.get_json()
        search_term = data.get("search_term")
        user_id = data.get("user_id", "default")

        if not search_term:
            return jsonify({"error": "Câmpul 'search_term' este obligatoriu."}), 400

        cookie_file = f"linkedin_cookies/{user_id}.json"
        if not os.path.exists(cookie_file):
            return jsonify({"error": f"Nu există cookie pentru user_id: {user_id}"}), 404

        # Configurare Chrome
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=chrome_options)

        try:
            # Autentificare cu cookie-uri
            print("Logging in...")
            load_linkedin_cookies(driver, cookie_file=cookie_file)
            time.sleep(random.uniform(3, 5))

            # Inițializare și căutare joburi
            print("Starting search...")
            job_search = FixedJobSearch(driver=driver, close_on_complete=False, scrape=False)
            job_listings = job_search.search(search_term)

            # Formatul de răspuns
            jobs_data = []
            for job in job_listings:
                job_doc = {
                    "title": job.job_title,
                    "company": job.company,
                    "location": job.location,
                    "salary": job.salary,
                    "url": job.linkedin_url,
                    "user_id": user_id,
                    "search_term": search_term,
                }
                jobs_data.append(job_doc)

            if jobs_data:
                save_multiple_job_results(jobs_data)

            return jsonify({
                "count": len(jobs_data),
                "results": [clean_mongo_doc(job) for job in jobs_data]
            })

        except Exception as e:
            driver.save_screenshot("error_screenshot.png")
            return jsonify({"error": str(e), "screenshot": "error_screenshot.png"}), 500

        finally:
            driver.quit()

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500