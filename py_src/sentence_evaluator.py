import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import re
import math
import jieba
import pickle
import logging
from constant import N
jieba.setLogLevel(logging.ERROR)

def calculate_ngram_score(sentence, ngram_dicts, n_order=3):
    line = "".join(re.findall(r'[\u4e00-\u9fa5]+', sentence))
    if not line: return -999.0

    tokens = jieba.lcut(line)
    n_tokens = len(tokens)

    unigrams = ngram_dicts.get(1, {})
    total_unigram_count = sum(unigrams.values())
    vocab_size = len(unigrams)

    alpha = 0.0001
    total_log_prob = 0
    match_chars = 0

    for i in range(n_tokens):
        prob = 0
        found_match = False

        for order in range(min(i + 1, n_order), 0, -1):
            gram_str = "".join(tokens[i - order + 1: i + 1])
            current_dict = ngram_dicts.get(order, {})
            gram_count = current_dict.get(gram_str, 0)

            if gram_count > 0:
                if order == 1:
                    prob = (gram_count + alpha) / (total_unigram_count + alpha * vocab_size)
                else:
                    prev_gram_str = "".join(tokens[i - order + 1: i])
                    prev_dict = ngram_dicts.get(order - 1, {})
                    prev_count = prev_dict.get(prev_gram_str, 0)
                    prob = (gram_count + alpha) / (prev_count + alpha * vocab_size)

                if order < n_order:
                    prob *= (0.01 ** (n_order - order))
                found_match = True
                break

        if not found_match:
            prob = alpha / (total_unigram_count + alpha * vocab_size)
            prob *= (0.1 ** n_order)

        total_log_prob += math.log10(prob)

        if unigrams.get(tokens[i], 0) > 0:
            match_chars += len(tokens[i])

    avg_lp = total_log_prob / n_tokens
    match_ratio = match_chars / len(line) if line else 0
    length_bonus = math.log10(len(line)) * 100
    return (avg_lp + 15) * 10 * (match_ratio ** 2) + length_bonus


if __name__ == '__main__':
    with open("ngram.pkl", "rb") as f:
        ngram_data = pickle.load(f)

    test_cases = [
        "这是一个非常正常的句子。", "研究人员正在实验室里进行科学实验。",
        "虽然但是，他的衣服里藏着一个巨大的秘密。", "阿巴阿巴运算速度极其缓慢的香蕉皮。",
        "炮眼打好了，炸药怎么装？", "母亲叮嘱我学习要深钻细研。",
        "母亲叮嘱我学习要深钻戏言。", "稀有词汇如魑魅魍魉和狴犴在代码中出现。",
        "炮也打好了，炸药怎么装？", "的了是在和一有我个中"
    ]

    results = []
    for s in test_cases:
        results.append((s, calculate_ngram_score(s, ngram_data, n_order=N)))

    for s, sc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{s:<30} | {sc:.2f}")