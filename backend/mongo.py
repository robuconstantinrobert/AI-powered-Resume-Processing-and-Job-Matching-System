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
