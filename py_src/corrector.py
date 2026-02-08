import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

import json
import pickle
import re
import pypinyin
from sentence_evaluator import calculate_ngram_score

class SentenceCorrector:
    def __init__(self, model_path="ngram.pkl", lexicon_path="token_dict.json"):
        with open(model_path, "rb") as f:
            self.ngram_data = pickle.load(f)

        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                lexicon_data = json.load(f)
                self.homophone_dict = lexicon_data.get("unigram", {})
        except FileNotFoundError:
            self.homophone_dict = {}

    def get_candidates_by_pinyin(self, text):
        py_key = ",".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.TONE))
        candidates = self.homophone_dict.get(py_key, {})
        return list(candidates.keys())

    def correct(self, sentence, threshold=193.0):
        original_score = calculate_ngram_score(sentence, self.ngram_data, n_order=4)

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

            for window_size in range(1, 4):
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

                    new_score = calculate_ngram_score(test_sentence, self.ngram_data, n_order=4)

                    if new_score > best_score:
                        best_score = new_score
                        best_sentence = test_sentence
                        is_corrected = True

        return best_sentence, best_score, is_corrected


if __name__ == '__main__':
    corrector = SentenceCorrector()

    test_cases = [
        "母亲叮嘱我学习要深钻戏言。",
        "炮也打好了，炸药怎么装？",
        "平果园里有很多的苹果。",
        "今天去哪里玩比较好？",
        "你居然七负我！"
    ]

    print(f"{'原句':<20} | {'纠正后':<20} | {'最终分数':<8} | {'状态'}")
    print("-" * 75)

    for sent in test_cases:
        res, score, status = corrector.correct(sent)
        status_str = "已纠正" if status else "无需纠正"
        print(f"{sent:<20} | {res:<20} | {score:<8.2f} | {status_str}")