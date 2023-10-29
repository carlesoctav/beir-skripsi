import json
from datasets import Dataset, concatenate_datasets
from tqdm   import tqdm

indo_corpus = "mmarco/indonesian/corpus.jsonl"
indo_query = "mmarco/indonesian/queries.jsonl"
english_corpus = "msmarco/corpus.jsonl"
english_query = "msmarco/queries.jsonl"

english_corpus_list = []
indo_corpus_list = []

english_query_list = []
indo_query_list = []

allign_corpus = []
allign_query = []


with open(english_corpus, "r") as f:
    english_corpus = f.readlines()
    for line in tqdm(english_corpus, desc="reading english corpus"):
        line = json.loads(line)
        english_corpus_list.append(line)

with open(indo_corpus, "r") as f:
    indo_corpus = f.readlines()
    for line in tqdm(indo_corpus, desc="reading indo corpus"):
        line = json.loads(line)
        indo_corpus_list.append(line)

with open(english_query, "r") as f:
    english_query = f.readlines()
    for line in tqdm(english_query, desc="reading english query"):
        line = json.loads(line)
        english_query_list.append(line)

with open(indo_query, "r") as f:
    indo_query = f.readlines()
    for line in tqdm(indo_query, desc="reading indo query"):
        line = json.loads(line)
        indo_query_list.append(line)


for id, en in tqdm(zip(indo_corpus_list, english_corpus_list), desc="alligning corpus"):
    if id["_id"] == en["_id"]:
        jsonl = { "text_en": en["text"], "text_id": id["text"]}
        allign_corpus.append(jsonl) 

for id, en in tqdm(zip(indo_query_list, english_query_list), desc="alligning query"):
    if id["_id"] == en["_id"]:
        jsonl = { "text_en": en["text"], "text_id": id["text"]}
        allign_query.append(jsonl)


corpus_dataset = Dataset.from_list(allign_corpus)
query_dataset = Dataset.from_list(allign_query)

total_dataset = concatenate_datasets([corpus_dataset, query_dataset])

total_dataset.push_to_hub("carles-undergrad-thesis/msmarco-en-id-parallel-sentences")