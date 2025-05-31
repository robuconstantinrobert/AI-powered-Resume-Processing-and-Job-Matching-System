from flask import Blueprint, request, jsonify
from services import process_cv_service, process_cv_with_esco_service
from mongo import get_documents_collection, save_multiple_job_results, clean_mongo_doc, create_user, get_user_by_email, get_user_by_id, get_documents_by_user_id, get_jobs_by_cv_id, get_jobs_collection
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
import os, json, time, random, urllib, datetime
from utils import LOCAL_EMB, LOCAL_CHAT, load_llm_hf, build_prompt, llm_json_extract, load_linkedin_cookies, FixedJobSearch, JWT
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from linkedin_scraper import JobSearch, actions
from linkedin_scraper.job_search import JobSearch as OriginalJobSearch
from linkedin_scraper.jobs import Job
import hashlib, jwt
from datetime import datetime, timedelta

SECRET_KEY = "cheia_mea_secreta"

api_bp = Blueprint('api', __name__)

@api_bp.route('/process_cv', methods=['POST'])
def process_cv():
    file = request.files.get('file')
    emb = request.form.get('emb', 'minilm')
    model = request.form.get('model', 'tinyllama')
    gguf = request.form.get('gguf')
    user_id = request.form.get('user_id')
    file_name = request.form.get('file_name', 'document.pdf')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    result = process_cv_service(file, emb, model, gguf, user_id, file_name)
    return jsonify(result)


@api_bp.route('/process_cv_with_esco', methods=['POST'])
def process_cv_with_esco():
    file = request.files.get('file')
    emb = request.form.get('emb', 'minilm')
    model = request.form.get('model', 'tinyllama')
    top = request.form.get('top', 3)
    user_id = request.form.get('user_id')
    file_name = request.form.get('file_name', 'document.pdf')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    result = process_cv_with_esco_service(file, emb, model, top, user_id, file_name)
    return jsonify(result)


@api_bp.route('/documents/<user_id>', methods=['GET'])
def get_documents(user_id):
    try:
        collection = get_documents_collection()
        documents = collection.find({"utilizator_id": ObjectId(user_id)})

        result = []
        for doc in documents:
            doc['_id'] = str(doc['_id'])
            doc['utilizator_id'] = str(doc['utilizator_id'])
            result.append(doc)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    user_id = data.get("utilizator_id")
    file_name = data.get("file_name")

    document = collection.find_one({"_id": ObjectId(document_id)})
    if not document:
        return jsonify({"error": "Document not found"}), 404

    raw_text = document["continut_text"]

    from services import process_cv_service
    result = process_cv_service(file=raw_text, 
        emb_key=emb_key, 
        model_key=model_key, 
        gguf_path=None,
        user_id=user_id,
        file_name=file_name)

    emb_model = SentenceTransformer(LOCAL_EMB[emb_key], device="cpu", trust_remote_code=True)
    vector = emb_model.encode(raw_text, normalize_embeddings=True)

    update = {
        "continut_vector": vector.tolist(),
        "date_extrase": {
            "competente": result.get("skills", []),
            "job_titles": result.get("job_titles", []),
            "suggested_roles": result.get("suggested_roles", [])
        }
    }

    collection.update_one({"_id": ObjectId(document_id)}, {"$set": update})
    return jsonify({"status": "updated", "new_data": update})


@api_bp.route('/linkedin/login', methods=['POST'])
def linkedin_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user_id = data.get('user_id', email.split('@')[0])

    if not email or not password:
        return jsonify({"error": "Email și parola sunt necesare."}), 400

    cookie_dir = "linkedin_cookies"
    os.makedirs(cookie_dir, exist_ok=True)
    cookie_file = os.path.join(cookie_dir, f"{user_id}.json")

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")
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
        #search_term = data.get("search_term")
        doc_id = data.get("_id")
        user_id = data.get("user_id", "default")
        print(doc_id)
        print(user_id)

        if not doc_id:
            return jsonify({"error": "Câmpul 'doc_id' este obligatoriu."}), 400

        cookie_file = f"linkedin_cookies/{user_id}.json"
        if not os.path.exists(cookie_file):
            print("INTRA AICI")
            return jsonify({"error": f"Nu există cookie pentru user_id: {user_id}"}), 404

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        #chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=chrome_options)


        docs_col = get_documents_collection()
        try:
            query = {
              "_id": ObjectId(doc_id),
              "utilizator_id": ObjectId(user_id)
            }
        except Exception:
            return jsonify({
              "error": "doc_id sau user_id are un format incorect."
            }), 400
        
        doc = docs_col.find_one(query)

        if not doc:
            print("INTRA AICI 2")
            return jsonify({"error": f"Document cu _id={doc_id} nu a fost găsit."}), 404
        
        roles = doc.get("date_extrase", {}).get("job_titles", [])
        print("ERROR 7")
        if not roles:
            print("INTRA AICI 3")
            return jsonify({"error": "Nu există roluri sugerate pentru acest document."}), 400

        jobs_data = []
        try:
            print("Logging in...")
            load_linkedin_cookies(driver, cookie_file=cookie_file)
            time.sleep(random.uniform(3, 5))

            print("Starting search...")
            job_search = FixedJobSearch(driver=driver, close_on_complete=False, scrape=False)
            for role in roles:
                job_listings = job_search.search(role)

                for job in job_listings:
                    job_doc = {
                        "title": job.job_title,
                        "company": job.company,
                        "location": job.location,
                        "salary": job.salary,
                        "url": job.linkedin_url,
                        "user_id": user_id,
                        "source_cv_id":  ObjectId(doc_id),
                        "search_term": role,
                        "applied_status": False
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


@api_bp.route("/users/register", methods=["POST"])
def register_user():
    data = request.get_json()
    nume = data.get("nume")
    email = data.get("email")
    parola = data.get("parola")
    preferinte = data.get("preferinte", {})

    if not all([nume, email, parola]):
        return jsonify({"error": "Toate câmpurile (nume, email, parola) sunt necesare."}), 400

    if get_user_by_email(email):
        return jsonify({"error": "Email deja înregistrat."}), 409

    user_id = create_user(nume, email, parola, preferinte)
    return jsonify({"message": "Utilizator creat cu succes.", "user_id": str(user_id)}), 201

@api_bp.route("/users/<user_id>", methods=["GET"])
def get_user(user_id):
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "Utilizatorul nu a fost găsit."}), 404
    user["_id"] = str(user["_id"])
    user.pop("parola_hash", None) 
    return jsonify(user), 200


@api_bp.route('/users/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email = data.get("email")
    parola = data.get("parola")

    if not all([email, parola]):
        return jsonify({"error": "Email și parolă necesare."}), 400

    user = get_user_by_email(email)
    if not user:
        return jsonify({"error": "Utilizator inexistent."}), 404

    parola_hash = hashlib.sha256(parola.encode()).hexdigest()
    if user.get("parola_hash") != parola_hash:
        return jsonify({"error": "Parolă greșită."}), 401

    token = jwt.encode({
        "user_id": str(user["_id"]),
        "exp": datetime.utcnow() + timedelta(days=1)
    }, SECRET_KEY, algorithm="HS256")

    print("DEBUG TOKEN:", jwt.decode(token, SECRET_KEY, algorithms=["HS256"]))


    return jsonify({
        "message": "Autentificare reușită",
        "token": token,
        "user_id": str(user["_id"])
    }), 200


@api_bp.route("/jobs/recommendations", methods=["GET"])
def get_job_recommendations():
    doc_id  = request.args.get("doc_id")
    user_id = request.args.get("user_id")

    if not doc_id or not user_id:
        return jsonify({ "error": "Missing doc_id or user_id query parameter" }), 400

    try:
        query = {
            "source_cv_id": ObjectId(doc_id),
            "user_id":      user_id
        }
    except Exception:
        return jsonify({ "error": "Invalid doc_id format" }), 400

    all_jobs = get_jobs_by_cv_id(doc_id)
    filtered = [job for job in all_jobs if job.get("user_id") == user_id]

    out = [clean_mongo_doc(j) for j in filtered]
    return jsonify(out), 200

@api_bp.route("/cvs", methods=["GET"])
def get_cvs_for_user():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    cvs = get_documents_by_user_id(user_id)
    docs = []
    for cv in cvs:
        docs.append({
            "id":   str(cv["_id"]),
            "name": cv.get("file_name", "Untitled CV")
        })
    return jsonify(docs), 200


from bson import ObjectId
from flask import request, jsonify, current_app as app

@api_bp.route("/jobs/<job_id>/apply", methods=["PUT"])
def mark_job_as_applied(job_id):
    data     = request.get_json(silent=True) or {}
    user_id  = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        job_oid = ObjectId(job_id)      
    except Exception:
        return jsonify({"error": "Invalid job_id format"}), 400

    coll = get_jobs_collection()                      

    result = coll.find_one_and_update(
        {"_id": job_oid, "user_id": user_id},   
        {"$set": {"applied_status": True}},
        return_document=True               
    )

    if result is None:
        return jsonify({"error": "Job not found for this user"}), 404

    job_json = {
        "_id"           : str(result["_id"]),
        "title"         : result["title"],
        "company"       : result["company"],
        "location"      : result["location"],
        "salary"        : result.get("salary"),
        "url"           : result["url"],
        "search_term"   : result.get("search_term"),
        "applied_status": result.get("applied_status", False)
    }
    return jsonify(job_json), 200


@api_bp.route("/jobs/cleanup", methods=["DELETE"])
def delete_applied_jobs():
    cv_id   = request.args.get("doc_id")
    user_id = request.args.get("user_id")
    if not cv_id or not user_id:
        return jsonify({"error": "Missing doc_id or user_id"}), 400

    try:
        cv_oid = ObjectId(cv_id)
    except Exception:
        return jsonify({"error": "Invalid doc_id format"}), 400

    coll   = get_jobs_collection()
    result = coll.delete_many({
        "source_cv_id": cv_oid,   
        "user_id":      user_id,  
        "applied_status": True   
    })

    return jsonify({"deleted": result.deleted_count}), 200

from datetime import datetime
from bson import ObjectId
from flask import request, jsonify

@api_bp.route("/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """Return aggregate counters for the header dashboard."""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        user_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"error": "Invalid user_id format"}), 400

    docs_col  = get_documents_collection()
    jobs_col  = get_jobs_collection()

    total_resumes   = docs_col.count_documents({"utilizator_id": user_oid})
    total_jobs      = jobs_col.count_documents({"user_id": user_id})
    applied_jobs    = jobs_col.count_documents({
        "user_id": user_id,
        "applied_status": True
    })
    pending_jobs    = total_jobs - applied_jobs

    last_resume_doc = docs_col.find(
        {"utilizator_id": user_oid},
        {"created_at": 1}
    ).sort("created_at", -1).limit(1)
    last_resume_ts  = next(last_resume_doc, {}).get("created_at")

    last_job_doc = jobs_col.find(
        {"user_id": user_id},
        {"_id": 0, "timestamp": 1}
    ).sort("timestamp", -1).limit(1)
    last_job_ts  = next(last_job_doc, {}).get("timestamp")

    return jsonify({
        "total_resumes":   total_resumes,
        "total_jobs":      total_jobs,
        "applied_jobs":    applied_jobs,
        "pending_jobs":    pending_jobs,
        "last_resume_at":  last_resume_ts,
        "last_job_at":     last_job_ts,
    }), 200


