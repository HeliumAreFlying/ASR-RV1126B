import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import re
import math
import jieba
import logging
import sqlite3
from time import time
from constant import N
jieba.setLogLevel(logging.ERROR)

class NgramModel:
    def __init__(self, db_path="ngram.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.isolation_level = None
        self.total_unigram_count = self.get_total_unigram_count()
        self.vocab_size = self.get_vocab_size()

    def get_count(self, order, gram):
        cursor = self.conn.cursor()
        cursor.execute("SELECT count FROM ngrams WHERE order_n=? AND gram=?", (order, gram))
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_total_unigram_count(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT SUM(count) FROM ngrams WHERE order_n=1")
        result = cursor.fetchone()
        return result[0] if result and result[0] else 1

    def get_vocab_size(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ngrams WHERE order_n=1")
        result = cursor.fetchone()
        return result[0] if result else 0

def calculate_ngram_score(sentence, model, n_order=3):
    line = "".join(re.findall(r'[\u4e00-\u9fa5]+', sentence))
    if not line: return -999.0

    tokens = jieba.lcut(line)
    n_tokens = len(tokens)

    alpha = 0.0001
    total_log_prob = 0
    match_chars = 0

    for i in range(n_tokens):
        prob = 0
        found_match = False

        for order in range(min(i + 1, n_order), 0, -1):
            gram_str = "".join(tokens[i - order + 1: i + 1])
            gram_count = model.get_count(order, gram_str)

            if gram_count > 0:
                if order == 1:
                    prob = (gram_count + alpha) / (model.total_unigram_count + alpha * model.vocab_size)
                else:
                    prev_gram_str = "".join(tokens[i - order + 1: i])
                    prev_count = model.get_count(order - 1, prev_gram_str)
                    prob = (gram_count + alpha) / (prev_count + alpha * model.vocab_size)

                if order < n_order:
                    prob *= (0.01 ** (n_order - order))
                found_match = True
                break

        if not found_match:
            prob = alpha / (model.total_unigram_count + alpha * model.vocab_size)
            prob *= (0.1 ** n_order)

        total_log_prob += math.log10(prob)

        if model.get_count(1, tokens[i]) > 0:
            match_chars += len(tokens[i])

    avg_lp = total_log_prob / n_tokens
    match_ratio = match_chars / len(line) if line else 0
    length_bonus = math.log10(len(line)) * 100
    return (avg_lp + 15) * 10 * (match_ratio ** 2) + length_bonus


if __name__ == '__main__':
    ngram_model = NgramModel("ngram.db")

    test_cases = [
        "这是一个非常正常的句子。", "研究人员正在实验室里进行科学实验。",
        "虽然但是，他的衣服里藏着一个巨大的秘密。", "阿巴阿巴运算速度极其缓慢的香蕉皮。",
        "炮眼打好了，炸药怎么装？", "母亲叮嘱我学习要深钻细研。",
        "母亲叮嘱我学习要深钻戏言。", "稀有词汇如魑魅魍魉和狴犴在代码中出现。",
        "炮也打好了，炸药怎么装？", "的了是在和一有我个中"
    ]

    start_time = time()
    results = []
    for s in test_cases:
        results.append((s, calculate_ngram_score(s, ngram_model, n_order=N)))
    end_time = time()

    for s, sc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{s:<30} | {sc:.2f}")

    print(f"\n平均用时 {(end_time - start_time) / len(test_cases) : .4f} 秒")