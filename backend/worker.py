import json
from io import BytesIO
from bson.objectid import ObjectId

from services import process_cv_service  

def process_cv_worker(
    file_bytes: bytes,
    emb_key: str,
    model_key: str,
    gguf_path: str,
    user_id: str,
    file_name: str
) -> dict:
    """
    Runs one CV through your service in isolation.
    Returns the same dict that process_cv_service would.
    """
    file_obj = BytesIO(file_bytes)
    
    result = process_cv_service(
        file_obj, emb_key, model_key, gguf_path, user_id, file_name
    )
    
    return result
