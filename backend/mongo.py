from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime


def save_result_mongo(data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['cv_database']
    collection = db['cv_results']
    result = collection.insert_one(data)
    return result.inserted_id

def get_mongo_client():
    return MongoClient("mongodb://localhost:27017")  # sau din .env

def get_documents_collection():
    client = get_mongo_client()
    db = client["cv_database"]
    return db["documents"]

def get_jobs_collection():
    client = get_mongo_client()
    db = client["cv_database"]
    return db["job_results"]


def save_job_result(data):
    """
    Salvează un singur job în colecția `job_results`.
    """
    collection = get_jobs_collection()
    result = collection.insert_one(data)
    return result.inserted_id


def save_multiple_job_results(data_list):
    """
    Salvează o listă de joburi în colecția `job_results`.
    """
    if not data_list:
        return []
    collection = get_jobs_collection()
    result = collection.insert_many(data_list)
    return result.inserted_ids


def get_jobs_by_cv_id(cv_id):
    """
    Returnează toate joburile legate de un document CV anume.
    """
    collection = get_jobs_collection()
    return list(collection.find({"source_cv_id": ObjectId(cv_id)}))

def clean_mongo_doc(doc):
    """
    Converteste valorile de tip ObjectId in stringuri pentru JSON serializare.
    """
    return {
        key: str(value) if isinstance(value, ObjectId) else value
        for key, value in doc.items()
    }