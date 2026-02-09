import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import logging
import pypinyin
from constant import encodings, pinyin_mode
from multiprocessing import Pool, cpu_count
jieba.setLogLevel(logging.ERROR)

def get_shared_token_key(token):
    pinyins = pypinyin.lazy_pinyin(token, style=pinyin_mode)
    return ','.join(pinyins)

def process_chunk_task(sentences_chunk):
    local_unigram = {}
    local_bigram = {}
    for s in sentences_chunk:
        tokens = [t for t in jieba.lcut(s) if len(t) > 1]
        for i, token in enumerate(tokens):
            local_unigram[token] = local_unigram.get(token, 0) + 1
            if i > 0:
                bi_key = f"{tokens[i - 1]}\t{token}"
                local_bigram[bi_key] = local_bigram.get(bi_key, 0) + 1
    return local_unigram, local_bigram

def get_parallel_data(sentences):
    num_workers = cpu_count()
    chunk_size = max(1, len(sentences) // num_workers)
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    with Pool(num_workers) as pool:
        results = pool.map(process_chunk_task, chunks)

    combined_unigram = {}
    combined_bigram = {}
    for uni, bi in results:
        for k, v in uni.items(): combined_unigram[k] = combined_unigram.get(k, 0) + v
        for k, v in bi.items(): combined_bigram[k] = combined_bigram.get(k, 0) + v
    return combined_unigram, combined_bigram

def get_filepaths(directory, extension="txt"):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root, _file))
    return filepaths

def get_lines_with_auto_encoding_mode(filepath):
    for encoding_mode in encodings:
        try:
            with open(filepath, "r", encoding=encoding_mode) as r:
                return r.readlines()
        except:
            pass
    return []

def wrap_token_dict(unigram_dict):
    token_dict = {}
    for token, freq in unigram_dict.items():
        if freq <= 0:
            continue
        shared_key = get_shared_token_key(token)
        if shared_key not in token_dict:
            token_dict[shared_key] = {}
        token_dict[shared_key][token] = freq
    return token_dict

def entry():
    token_dict_dump_dir = "token_dict.json"

    m_sentences = get_lines_with_auto_encoding_mode("corpus_cleaned_metadata.txt")
    m_unigram, bigram_dict = get_parallel_data(list(set(s.strip() for s in m_sentences if len(s.strip()) >= 4)))

    n_sentences = get_lines_with_auto_encoding_mode("corpus_cleaned_novels.txt")
    n_unigram, _ = get_parallel_data(list(set(s.strip() for s in n_sentences if len(s.strip()) >= 4)))

    t_sentences = get_lines_with_auto_encoding_mode("corpus_cleaned_thu.txt")
    t_unigram, _ = get_parallel_data(list(set(s.strip() for s in t_sentences if len(s.strip()) >= 2)))

    m_total = sum(m_unigram.values()) if m_unigram else 1
    unigram_dict = m_unigram

    for other_uni, weight in [(n_unigram, 0.1), (t_unigram, 0.25)]:
        other_total = sum(other_uni.values())
        if other_total > 0:
            factor = (m_total / other_total) * weight
            for k, v in other_uni.items():
                unigram_dict[k] = unigram_dict.get(k, 0) + int(v * factor)

    clean_bigram = {k: v for k, v in bigram_dict.items() if v > 0}
    final_token_dict = wrap_token_dict(unigram_dict)

    with open(token_dict_dump_dir, "w", encoding="utf-8") as w:
        json.dump({"unigram": final_token_dict, "bigram": clean_bigram}, w, ensure_ascii=False)

def check():
    if os.path.exists("token_dict.json"):
        with open("token_dict.json", "r", encoding="utf-8") as r:
            data = json.load(r)
            target = data["unigram"]
            vocab_size = sum(len(sub) for sub in target.values())
            print(f"vocab size is equal to {vocab_size}")

if __name__ == '__main__':
    entry()
    check()