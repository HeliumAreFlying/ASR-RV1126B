import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import logging
import sqlite3
import tempfile
import shutil
from tqdm import tqdm
from constant import N
from multiprocessing import Pool, cpu_count
jieba.setLogLevel(logging.ERROR)

def create_schema(cursor):
    cursor.execute('CREATE TABLE IF NOT EXISTS ngrams (order_n INTEGER NOT NULL, gram TEXT NOT NULL, count INTEGER DEFAULT 1, PRIMARY KEY (order_n, gram)) WITHOUT ROWID')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_gram ON ngrams(order_n, gram)')

def process_chunk_to_db(lines_chunk):
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'part.db')
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        create_schema(cur)
        for line in lines_chunk:
            line = line.strip()
            if not line: continue
            tokens = jieba.lcut(line)
            L = len(tokens)
            for i in range(L):
                for o in range(1, N + 1):
                    if i + o <= L:
                        k = "".join(tokens[i:i + o])
                        cur.execute('INSERT OR IGNORE INTO ngrams (order_n, gram, count) VALUES (?, ?, 0)', (o, k))
                        cur.execute('UPDATE ngrams SET count = count + 1 WHERE order_n = ? AND gram = ?', (o, k))
        conn.commit()
        conn.close()
        return db_path
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(db_path):
            os.remove(db_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

def merge_databases(output_db, db_paths):
    if os.path.exists(output_db):
        os.remove(output_db)
    conn_out = sqlite3.connect(output_db)
    cur_out = conn_out.cursor()
    create_schema(cur_out)
    conn_out.close()
    for db_path in tqdm(db_paths, "Merging Databases"):
        if not db_path or not os.path.exists(db_path):
            continue
        try:
            conn_src = sqlite3.connect(db_path)
            cur_src = conn_src.cursor()
            conn_dest = sqlite3.connect(output_db)
            cur_dest = conn_dest.cursor()
            cur_src.execute("SELECT order_n, gram, count FROM ngrams")
            rows = cur_src.fetchall()
            if rows:
                cur_dest.executemany('INSERT OR IGNORE INTO ngrams (order_n, gram, count) VALUES (?, ?, 0)', [(r[0], r[1]) for r in rows])
                cur_dest.executemany('UPDATE ngrams SET count = count + ? WHERE order_n = ? AND gram = ?', [(r[2], r[0], r[1]) for r in rows])
                conn_dest.commit()
            conn_src.close()
            conn_dest.close()
            os.remove(db_path)
            shutil.rmtree(os.path.dirname(db_path), ignore_errors=True)
        except Exception as e:
            print(f"Merge error: {e}")
            try:
                if 'conn_src' in locals(): conn_src.close()
                if 'conn_dest' in locals(): conn_dest.close()
            except: pass
    conn_final = sqlite3.connect(output_db)
    cur_final = conn_final.cursor()
    cur_final.execute('DELETE FROM ngrams WHERE order_n > 1 AND count <= 1')
    conn_final.commit()
    conn_final.close()

def train_entry():
    target_files = ["corpus_cleaned_metadata.txt", "corpus_cleaned_novels.txt", "corpus_cleaned_thu.txt"]
    all_lines = []
    for f in target_files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as rf:
                all_lines.extend(rf.readlines())
    n_chunks = cpu_count() * 2
    chunk_size = max(1, len(all_lines) // n_chunks)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    db_paths = []
    with Pool(processes=min(cpu_count(), len(chunks))) as pool:
        for res in tqdm(pool.imap_unordered(process_chunk_to_db, chunks), total=len(chunks), desc="Processing Chunks"):
            if res:
                db_paths.append(res)
    merge_databases("ngram.db", db_paths)

if __name__ == '__main__':
    train_entry()