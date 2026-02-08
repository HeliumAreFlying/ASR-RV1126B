import json
from get_lexicon import get_shared_token_key

def debug_tokens_status(target_words, bigram_pairs, data_path="token_dict.json"):
    with open(data_path, "r", encoding="utf-8") as r:
        data = json.load(r)
        unigram = data.get("unigram", {})
        bigram = data.get("bigram", {})

    print(f"{'词语':<10} | {'拼音键':<20} | {'Unigram 词频':<10}")
    print("-" * 50)

    for word in target_words:
        s_key = get_shared_token_key(word)
        count = unigram.get(s_key, {}).get(word, 0)
        print(f"{word:<10} | {s_key:<20} | {count:<10}")

    print("\n" + f"{'Bigram 搭配':<20} | {'出现频次':<10}")
    print("-" * 40)

    for pair in bigram_pairs:
        bi_key = "\t".join(pair)
        count = bigram.get(bi_key, 0)
        print(f"{bi_key:<20} | {count:<10}")

def debug_sentence_logic(sentence, data):
    import jieba
    tokens = jieba.lcut(sentence)
    bigram = data["bigram"]

    print(f"\n{'Token对':<20} | {'Bigram频次':<10} | {'是否存在搭配'}")
    print("-" * 50)

    for i in range(1, len(tokens)):
        bi_key = f"{tokens[i - 1]}\t{tokens[i]}"
        count = bigram.get(bi_key, 0)
        print(f"{bi_key:<20} | {count:<10} | {'√' if count > 0 else 'X'}")

if __name__ == '__main__':
    words_to_check = ["戏言", "细研", "深钻", "母亲", "叮嘱"]
    pairs_to_check = [("深钻", "细研"), ("深钻", "戏言")]
    debug_tokens_status(words_to_check, pairs_to_check)

    with open("token_dict.json", "r", encoding="utf-8") as r:
        data = json.load(r)
        debug_sentence_logic("阿巴阿巴运算速度极其缓慢的香蕉皮", data)

    sent1 = "炮也打好了"
    sent2 = "炮眼打好了"
    print(f"原句分词: {jieba.lcut(sent1)}")
    print(f"目标句分词: {jieba.lcut(sent2)}")