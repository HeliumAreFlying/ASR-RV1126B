import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import logging
import gc
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
jieba.setLogLevel(logging.ERROR)

def process_single_file(args):
    file_path, n = args
    local_ngram = {i: defaultdict(int) for i in range(1, n + 1)}

    if not os.path.exists(file_path):
        return local_ngram

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                tokens = jieba.lcut(line)
                num_tokens = len(tokens)
                for i in range(num_tokens):
                    for order in range(1, n + 1):
                        if i + order <= num_tokens:
                            gram_key = "".join(tokens[i: i + order])
                            local_ngram[order][gram_key] += 1
    except Exception:
        pass
    return local_ngram

def train_ngram_parallel(file_list, n=3):
    total_ngram = {i: defaultdict(int) for i in range(1, n + 1)}
    tasks = [(f, n) for f in file_list]

    with Pool(processes=min(len(file_list), cpu_count())) as pool:
        for local_result in tqdm(pool.imap_unordered(process_single_file, tasks),
                                 total=len(tasks), desc="Training"):
            for order in local_result:
                for gram, count in local_result[order].items():
                    total_ngram[order][gram] += count

    for order in range(2, n + 1):
        total_ngram[order] = {k: v for k, v in total_ngram[order].items() if v > 1}

    return total_ngram

if __name__ == '__main__':
    target_files = ["corpus_cleaned_metadata.txt", "corpus_cleaned_novels.txt", "corpus_cleaned_thu.txt"]
    N = 3
    model_data = train_ngram_parallel(target_files, n=N)

    gc.collect()

    with open("ngram.pkl", "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)