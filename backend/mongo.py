from pymongo import MongoClient

def save_result_mongo(data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['cv_database']
    collection = db['cv_results']
    result = collection.insert_one(data)
    return result.inserted_id
