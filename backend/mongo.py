from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import hashlib


def save_result_mongo(data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['cv_database']
    collection = db['cv_results']
    result = collection.insert_one(data)
    return result.inserted_id

def get_mongo_client():
    return MongoClient("mongodb://localhost:27017")

def get_documents_collection():
    client = get_mongo_client()
    db = client["cv_database"]
    return db["documents"]

def get_jobs_collection():
    client = get_mongo_client()
    db = client["cv_database"]
    return db["job_results"]


def save_job_result(data):
    collection = get_jobs_collection()
    result = collection.insert_one(data)
    return result.inserted_id


def save_multiple_job_results(data_list):
    if not data_list:
        return []
    collection = get_jobs_collection()
    result = collection.insert_many(data_list)
    return result.inserted_ids


def get_documents_by_user_id(user_id):
    collection = get_documents_collection()
    return list(collection.find({"utilizator_id": ObjectId(user_id)}))

def get_jobs_by_cv_id(cv_id):
    collection = get_jobs_collection()
    return list(collection.find({"source_cv_id": ObjectId(cv_id)}))


def clean_mongo_doc(doc):
    return {
        key: str(value) if isinstance(value, ObjectId) else value
        for key, value in doc.items()
    }

def get_users_collection():
    client = MongoClient("mongodb://localhost:27017")
    db = client["cv_database"]
    return db["users"]

def hash_password(parola):
    return hashlib.sha256(parola.encode()).hexdigest()

def create_user(nume, email, parola, preferinte=None):
    collection = get_users_collection()
    user = {
        "nume": nume,
        "email": email,
        "parola_hash": hash_password(parola),
        "preferinte": preferinte or {},
        "created_at": datetime.utcnow(),
        "profile_status": "activ"
    }
    return collection.insert_one(user).inserted_id

def get_user_by_email(email):
    collection = get_users_collection()
    return collection.find_one({"email": email})

def get_user_by_id(user_id):
    collection = get_users_collection()
    return collection.find_one({"_id": ObjectId(user_id)})