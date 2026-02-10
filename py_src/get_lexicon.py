import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import jieba
import logging
import pypinyin
from multiprocessing import Pool, cpu_count
from constant import encodings, pinyin_mode, min_token_length_for_bi_key, soft_percent_for_bi_key_when_lower_than_min_token_length
jieba.setLogLevel(logging.ERROR)

def get_shared_token_key(token):
    pinyins = pypinyin.lazy_pinyin(token, style=pinyin_mode)
    return ','.join(pinyins)

def process_chunk_task(sentences_chunk):
    local_unigram = {}
    for s in sentences_chunk:
        tokens = jieba.lcut(s)
        for i, token in enumerate(tokens):
            local_unigram[token] = local_unigram.get(token, 0) + 1
    return local_unigram

def get_parallel_data(sentences):
    num_workers = cpu_count()
    chunk_size = max(1, len(sentences) // num_workers)
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    with Pool(num_workers) as pool:
        results = pool.map(process_chunk_task, chunks)

    combined_unigram = {}
    for uni in results:
        for k, v in uni.items(): combined_unigram[k] = combined_unigram.get(k, 0) + v

    return combined_unigram

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
    m_unigram = get_parallel_data(list(set(s.strip() for s in m_sentences if len(s.strip()) >= 4)))

    n_sentences = get_lines_with_auto_encoding_mode("corpus_cleaned_novels.txt")
    n_unigram = get_parallel_data(list(set(s.strip() for s in n_sentences if len(s.strip()) >= 4)))

    t_sentences = get_lines_with_auto_encoding_mode("corpus_cleaned_thu.txt")
    t_unigram = get_parallel_data(list(set(s.strip() for s in t_sentences if len(s.strip()) >= 2)))

    m_total = sum(m_unigram.values()) if m_unigram else 1
    unigram_dict = m_unigram

    for other_uni, weight in [(n_unigram, 0.1), (t_unigram, 0.25)]:
        other_total = sum(other_uni.values())
        if other_total > 0:
            factor = (m_total / other_total) * weight
            for k, v in other_uni.items():
                unigram_dict[k] = unigram_dict.get(k, 0) + int(v * factor)

    invalid_token_dict = dict()
    for token, freq in unigram_dict.items():
        if len(token) < min_token_length_for_bi_key:
            invalid_token_dict[token] = freq

    invalid_tokens = list(invalid_token_dict.keys())
    invalid_tokens.sort(key=lambda invalid_token : invalid_token_dict[invalid_token], reverse=True)
    soft_invalid_tokens = invalid_tokens[round(soft_percent_for_bi_key_when_lower_than_min_token_length * len(invalid_tokens)):]
    for soft_invalid_token in soft_invalid_tokens:
        del unigram_dict[soft_invalid_token]

    final_token_dict = wrap_token_dict(unigram_dict)

    with open(token_dict_dump_dir, "w", encoding="utf-8") as w:
        json.dump(final_token_dict, w, ensure_ascii=False)

def check():
    if os.path.exists("token_dict.json"):
        with open("token_dict.json", "r", encoding="utf-8") as r:
            target = json.load(r)
            vocab_size = sum(len(sub) for sub in target.values())
            print(f"vocab size is equal to {vocab_size}")

if __name__ == '__main__':
    entry()
    check()