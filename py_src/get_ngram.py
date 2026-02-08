import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import sqlite3
import logging
from tqdm import tqdm
from constant import N
from multiprocessing import Pool, cpu_count
jieba.setLogLevel(logging.ERROR)

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS ngrams
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       order_n
                       INTEGER
                       NOT
                       NULL,
                       gram
                       TEXT
                       NOT
                       NULL,
                       count
                       INTEGER
                       DEFAULT
                       1,
                       UNIQUE
                   (
                       order_n,
                       gram
                   ) ON CONFLICT REPLACE
                       )
                   ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_gram ON ngrams(order_n, gram)')
    conn.commit()
    return conn


def process_single_file(args):
    file_path, n, db_path = args
    temp_conn = sqlite3.connect(db_path)
    temp_conn.isolation_level = None
    cursor = temp_conn.cursor()

    if not os.path.exists(file_path):
        temp_conn.close()
        return 0

    processed_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = jieba.lcut(line)
                num_tokens = len(tokens)
                for i in range(num_tokens):
                    for order in range(1, n + 1):
                        if i + order <= num_tokens:
                            gram_key = "".join(tokens[i: i + order])
                            cursor.execute(
                                'INSERT OR IGNORE INTO ngrams (order_n, gram, count) VALUES (?, ?, 1)',
                                (order, gram_key)
                            )
                            cursor.execute(
                                'UPDATE ngrams SET count = count + 1 WHERE order_n = ? AND gram = ?',
                                (order, gram_key)
                            )
                processed_lines += 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    finally:
        temp_conn.close()

    return processed_lines


def train_ngram_parallel(file_list, n=3, db_path="ngram.db"):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = init_db(db_path)
    conn.close()

    tasks = [(f, n, db_path) for f in file_list]

    total_files = len(tasks)

    with Pool(processes=min(cpu_count(), 8)) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, tasks), total=total_files, desc="Training"):
            pass

    final_conn = sqlite3.connect(db_path)
    cursor = final_conn.cursor()

    cursor.execute('DELETE FROM ngrams WHERE count <= 1 AND order_n > 1')

    final_conn.commit()
    final_conn.close()


if __name__ == '__main__':
    target_files = [
        "corpus_cleaned_metadata.txt",
        "corpus_cleaned_novels.txt",
        "corpus_cleaned_thu.txt"
    ]
    train_ngram_parallel(target_files, n=N, db_path="ngram.db")