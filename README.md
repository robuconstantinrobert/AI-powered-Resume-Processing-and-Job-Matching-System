# AI-powered-Resume-Processing-and-Job-Matching-System

First time setup backend:
1) Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
2) python3 -m venv env
3) Then use env\Scripts\activate to activate the virtual environment

First time setup frontend:
1) npm startcl
2) npm install react-router-dom

Vectorization models (NLP - Natural Language Processing) that I can use:
1) Sentence Transformers
    -> Simple, fast, and high-quality sentence-level embeddings (vectorization) for semantic similarity or matching tasks. https://github.com/UKPLab/sentence-transformers 
2) spaCy
    -> General-purpose NLP with support for named entity recognition (NER), part-of-speech tagging, and other preprocessing tasks. You can integrate it with transformer-based models for more advanced vectorization. https://github.com/explosion/spaCy 
3) BERT
    -> Fine-grained NLP tasks requiring high accuracy, such as text classification, token-level prediction, or document similarity. https://github.com/huggingface/transformers 