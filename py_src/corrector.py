import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import json
import re
import pypinyin
from time import time
from sentence_evaluator import NgramModel, calculate_ngram_score
from constant import N, correct_threshold, corrector_window_size, pinyin_mode

class SentenceCorrector:
    def __init__(self, model_path="ngram.db", lexicon_path="token_dict.json"):
        self.ngram_model = NgramModel(model_path)

        with open(lexicon_path, "r", encoding="utf-8") as r:
            lexicon_data = json.load(r)
            self.homophone_dict = lexicon_data.get("unigram", {})

    def get_candidates_by_pinyin(self, text):
        pinyin_results = [r[0] for r in pypinyin.pinyin(text, style=pinyin_mode)]
        py_key = ",".join(pinyin_results)
        candidates = self.homophone_dict.get(py_key, {})
        return list(candidates.keys())

    def get_ngram_count(self, gram, order_n):
        self.cursor.execute('SELECT count FROM ngrams WHERE order_n = ? AND gram = ?', (order_n, gram))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def single_correct(self, sentence, original_score, threshold):
        if original_score >= threshold:
            return sentence, original_score, False

        best_sentence = sentence
        best_score = original_score
        is_corrected = False

        char_list = list(sentence)
        n = len(char_list)

        for i in range(n):
            if not re.match(r'[\u4e00-\u9fa5]', char_list[i]):
                continue

            for window_size in range(1, corrector_window_size + 1):
                if i + window_size > n:
                    continue

                target_text = "".join(char_list[i: i + window_size])
                candidates = self.get_candidates_by_pinyin(target_text)

                for cand in candidates:
                    if cand == target_text:
                        continue

                    temp_chars = list(char_list)
                    temp_chars[i: i + window_size] = list(cand)
                    test_sentence = "".join(temp_chars)

                    new_score = calculate_ngram_score(test_sentence, self.ngram_model, n_order=N)

                    if new_score > best_score:
                        best_score = new_score
                        best_sentence = test_sentence
                        is_corrected = True

        return best_sentence, best_score, is_corrected

    def correct(self, sentence, threshold, n_correct = 4):
        best_sentence, is_corrected = sentence, False
        best_score = calculate_ngram_score(sentence, self.ngram_model, n_order=N)

        for n in range(n_correct):
            better_sentence, better_score, current_sentence_is_corrected = self.single_correct(best_sentence, best_score, threshold)
            best_sentence = better_sentence
            best_score = max(best_score, better_score)
            if not current_sentence_is_corrected:
                break
            else:
                is_corrected = is_corrected or current_sentence_is_corrected

        return best_sentence, best_score, is_corrected

if __name__ == '__main__':
    corrector = SentenceCorrector()

    test_cases = [
        "目亲叮嘱我学习要深钻戏言。",
        "炮也打好了，炸药怎么装？",
        "平果园里有很多的平果。",
        "今天去哪里玩比较好？",
        "泥居然七负我！"
    ]

    print(f"{'原句':<20} | {'纠正后':<20} | {'最终分数':<8} | {'状态'}")
    print("-" * 75)

    test_results = []

    start_time = time()
    for sent in test_cases:
        res, score, status = corrector.correct(sent, threshold=correct_threshold)
        status_str = "已纠正" if status else "无需纠正"
        test_results.append(f"{sent:<20} | {res:<20} | {score:<8.2f} | {status_str}")
    end_time = time()

    for test_result in test_results:
        print(test_result)

    print(f"\n平均用时 {(end_time - start_time) / len(test_cases) : .4f} 秒")