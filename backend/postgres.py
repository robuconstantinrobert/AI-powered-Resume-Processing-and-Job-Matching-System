import psycopg2
import numpy as np

def fetch_embedding_pg(sha1_hash, emb_model):
    conn = psycopg2.connect(dbname='vector_database', user='postgres', password='password', host='localhost', port=5432)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM embeddings_cache WHERE sha1_hash=%s AND emb_model=%s", (sha1_hash, emb_model))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return np.array(row[0])
    return None

def save_embedding_pg(sha1_hash, emb_model, embedding):
    conn = psycopg2.connect(dbname='vector_database', user='postgres', password='password', host='localhost', port=5432)
    cur = conn.cursor()
    cur.execute("INSERT INTO embeddings_cache (sha1_hash, emb_model, embedding) VALUES (%s, %s, %s)", (sha1_hash, emb_model, embedding.tolist()))
    conn.commit()
    cur.close()
    conn.close()
