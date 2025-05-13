from flask import Blueprint, request, jsonify
from services import process_cv_service, process_cv_with_esco_service
from mongo import get_documents_collection
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer

from utils import LOCAL_EMB, LOCAL_CHAT, load_llm_hf, build_prompt, llm_json_extract

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
