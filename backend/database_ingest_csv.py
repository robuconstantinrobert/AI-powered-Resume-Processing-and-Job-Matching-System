"""
load_esco_to_pg.py
------------------
python database_ingest_csv.py \
       --db vector_database \
       --user postgres \
       --password password \
       --host localhost \
       --port 5432 \
       --csv_occup occupations_en.csv \
       --csv_skills skills_en.csv \
       --csv_rel   occupationSkillRelations_en.csv
"""
import argparse, os, sys, psycopg2, psycopg2.extras, pathlib

# ─────────────────────────────── CLI args ───────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True,  help="database name to create/use")
ap.add_argument("--user", default="postgres")
ap.add_argument("--password", default=os.getenv("PGPASSWORD", ""))
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", default=5432, type=int)
ap.add_argument("--csv_occup", required=True, help="occupations CSV path")
ap.add_argument("--csv_skills", required=True, help="skills CSV path")
ap.add_argument("--csv_rel",   required=True, help="occupation‑skill rel CSV path")
args = ap.parse_args()

# Cast paths to Posix for COPY (backslashes confuse psql on Windows)
csv_occ  = pathlib.Path(args.csv_occup).resolve().as_posix()
csv_skill= pathlib.Path(args.csv_skills).resolve().as_posix()
csv_rel  = pathlib.Path(args.csv_rel).resolve().as_posix()

# ────────────────────────── helpers ────────────────────────────
def connect(dbname):
    return psycopg2.connect(
        dbname=dbname, user=args.user, password=args.password,
        host=args.host, port=args.port)

def db_exists(cur, name):
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,))
    return cur.fetchone() is not None

# ────────────────── 1. create DB if missing ────────────────────
conn = connect("postgres")
conn.autocommit = True
cur = conn.cursor()
if not db_exists(cur, args.db):
    cur.execute(f'CREATE DATABASE "{args.db}"')
    print(f"[info] database {args.db} created.")
cur.close(); conn.close()

# ──────────────── 2. (re)create tables  ────────────────────────
ddl = """
DROP TABLE IF EXISTS occupation CASCADE;
DROP TABLE IF EXISTS skill CASCADE;
DROP TABLE IF EXISTS occupation_skill_rel CASCADE;

CREATE TABLE occupation(
    conceptType              text,
    conceptUri               text PRIMARY KEY,
    iscoGroup                text,
    preferredLabel           text,
    altLabels                text,
    hiddenLabels             text,
    status                   text,
    modifiedDate             date,
    regulatedProfessionNote  text,
    scopeNote                text,
    definition               text,
    inScheme                 text,
    description              text,
    code                     text
);

CREATE TABLE skill(
    conceptType    text,
    conceptUri     text PRIMARY KEY,
    skillType      text,
    reuseLevel     text,
    preferredLabel text,
    altLabels      text,
    hiddenLabels   text,
    status         text,
    modifiedDate   date,
    scopeNote      text,
    definition     text,
    inScheme       text,
    description    text
);

CREATE TABLE occupation_skill_rel(
    occupationUri  text,
    relationType   text,
    skillType      text,
    skillUri       text
);
"""
conn = connect(args.db)
cur  = conn.cursor()
cur.execute(ddl)
conn.commit()
print("[info] tables created.")

# ─────────────── 3. fast bulk‑load with COPY ────────────────
def copy_csv(table, csv_path):
    sql = f"""COPY {table} FROM STDIN
             WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"')"""
    with open(csv_path, "r", encoding="utf8") as f:
        cur.copy_expert(sql, f)
    print(f"[info] loaded {table} from {csv_path}")

copy_csv("occupation", csv_occ)
copy_csv("skill",      csv_skill)
copy_csv("occupation_skill_rel", csv_rel)

conn.commit()
cur.close(); conn.close()
print("[info] ALL DONE ✔")
