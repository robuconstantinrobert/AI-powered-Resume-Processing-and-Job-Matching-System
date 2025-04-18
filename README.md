# AI-powered-Resume-Processing-and-Job-Matching-System

First time setup backend:
1) Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
2) python3 -m venv env (using python 3.10)
3) Then use env\Scripts\activate to activate the virtual environment

First time setup frontend:
1) npm startcl
2) npm install react-router-dom

Vectorization models (NLP - Natural Language Processing) that I can use:
1) Sentence Transformers
    -> Simple, fast, and high-quality sentence-level embeddings (vectorization) for semantic similarity or matching tasks. https://github.com/UKPLab/sentence-transformers 
2) spaCy
    -> General-purpose NLP with support for named entity recognition (NER), part-of-speech tagging, and other preprocessing tasks. You can integrate it with transformer-based models for more advanced vectorization. https://github.com/explosion/spaCy 
    -> python -m spacy download en_core_web_sm
    -> python -m spacy download en_core_web_lg
    -> Further implementation using: ro_core_news_lg for ROMANIAN 
3) BERT
    -> Fine-grained NLP tasks requiring high accuracy, such as text classification, token-level prediction, or document similarity. https://github.com/huggingface/transformers 


CPU float precision (no CUDA, simplest)
python app.py --pdf "CV.pdf" --model phi2

Fastest on pure CPU 4-bit
python app.py --pdf "CV.pdf" --gguf models/phi-2.Q4_K_M.gguf

CUDA usage
python app.py --pdf "CV.pdf" --model tinyllama

RAM dependent 
python app.py --pdf "CV.pdf" --model gptneo


Used classifications from:
https://www.onetcenter.org/dictionary/29.2/excel/knowledge.html
https://esco.ec.europa.eu/en/use-esco/download/email

*These are used in order to have a knowledge base of possible job titles, knowledge skills and tools that can be used
