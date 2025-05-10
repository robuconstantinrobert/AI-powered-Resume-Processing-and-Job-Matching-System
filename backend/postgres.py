import psycopg2
import numpy as np

def save_embedding_pg(vector):
    conn = psycopg2.connect(dbname='vector_database', user='postgres', password='password', host='localhost', port=5432)
    cur = conn.cursor()
    cur.execute("INSERT INTO cv_vectors (vector) VALUES (%s)", (vector.tolist(),))
    conn.commit()
    cur.close()
    conn.close()
