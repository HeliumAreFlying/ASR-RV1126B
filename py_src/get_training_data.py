import os
import re
from tqdm import tqdm
from constant import encodings

def process_metadata_csv(csv_path, output_file):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    if not os.path.exists(csv_path):
        return
    with open(csv_path, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        lines_from_file = f.readlines()
        for line in tqdm(lines_from_file, desc="Processing metadata"):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            text_content = parts[-1]
            clean_sent = "".join(pattern.findall(text_content))
            if len(clean_sent) >= 5:
                out_f.write(clean_sent + '\n')

def get_valid_file_cursor(filepath):
    for encoding in encodings:
        try:
            f = open(filepath, 'r', encoding=encoding)
            f.readline()
            f.seek(0)
            return f
        except:
            continue
    raise RuntimeError("cannot find valid file cursor")

def process_thu_dict(dict_dir, output_file):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    file_list = [os.path.join(root, f) for root, _, files in os.walk(dict_dir) for f in files if f.endswith('.txt')]

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(file_list, desc="Processing THU Dict"):
            f = get_valid_file_cursor(file_path)
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    word = "".join(pattern.findall(parts[0]))
                    if len(word) >= 2:
                        out_f.write(word + '\n')
            f.close()

def generate_cleaned_corpus(novels_root_dir, output_file):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    file_list = [
        os.path.join(root, f) for root, _, files in os.walk(novels_root_dir)
        for f in files if f.endswith('.txt')
    ]

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(file_list, desc="Cleaning Novels"):
            f = get_valid_file_cursor(file_path)
            content = f.read()

            if content:
                content = re.sub(r'\s+', ' ', content)
                sentences = re.split(r'[。！？，、；：：“”‘’（）《》\s\-\.]', content)

                for sent in sentences:
                    clean_sent = "".join(pattern.findall(sent))
                    if len(clean_sent) >= 5:
                        out_f.write(clean_sent + '\n')

            f.close()

if __name__ == "__main__":
    novels_root_path = r'novels'
    metadata_csv = r"metadata.csv"
    thu_dict_path = r'qinghua'

    novels_output = r'corpus_cleaned_novels.txt'
    metadata_output = r'corpus_cleaned_metadata.txt'
    thu_dict_output = r'corpus_cleaned_thu.txt'

    generate_cleaned_corpus(novels_root_path, novels_output)
    process_metadata_csv(metadata_csv, metadata_output)
    process_thu_dict(thu_dict_path, thu_dict_output)