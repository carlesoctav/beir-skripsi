from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer
import logging
from pyprojroot import here
import json
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
list_of_path = [
('mmarco train set','datasets/mmarco/indonesian/corpus.jsonl', 'datasets/mmarco/indonesian/queries.jsonl', 'datasets/mmarco/indonesian/qrels/train.tsv'),
('mmarco dev set','datasets/mmarco/indonesian/corpus.jsonl', 'datasets/mmarco/indonesian/queries.jsonl', 'datasets/mmarco/indonesian/qrels/dev.tsv'),
('mrtydi dev set','datasets/mrtydi/indonesian/corpus.jsonl', 'datasets/mrtydi/indonesian/queries.jsonl', 'datasets/mrtydi/indonesian/qrels/dev.tsv'),
('miracl dev set','datasets/miracl/corpus.jsonl', 'datasets/miracl/queries.jsonl', 'datasets/miracl/dev.tsv'),
]

for dataset_name, corpus_path, query_path, qrels_path in list_of_path:
    corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

    whitespace_corpus_length_dist = {}
    whitespace_query_length_dist = {}
    bert_corpus_length_dist = {}
    bert_query_length_dist = {}

    # for corpus in tqdm(corpus.values(), desc="Processing corpus"):
    #     len_corpus_words = len(corpus.get("text").split(" "))
    #     len_corpus_tokens = len(tokenizer(corpus.get("text"))["input_ids"])

    #     if len_corpus_words in whitespace_corpus_length_dist:
    #         whitespace_corpus_length_dist[len_corpus_words] += 1
    #     else:
    #         whitespace_corpus_length_dist[len_corpus_words] = 1

    #     if len_corpus_tokens in bert_corpus_length_dist:
    #         bert_corpus_length_dist[len_corpus_tokens] += 1
    #     else:
    #         bert_corpus_length_dist[len_corpus_tokens] = 1
        
    # with open(f"whitespace_corpus_length_dist_{dataset_name}.json", "w") as f:
    #     json.dump(whitespace_corpus_length_dist, f)

    # with open(f"bert_corpus_length_dist_{dataset_name}.json", "w") as f:
    #     json.dump(bert_corpus_length_dist, f)

    for query in tqdm(queries.values(), desc="Processing queries"):
        
        len_query_words = len(query.split(" "))
        len_query_tokens = len(tokenizer(query)["input_ids"])

        if len_query_words in whitespace_query_length_dist:
            whitespace_query_length_dist[len_query_words] += 1
        else:
            whitespace_query_length_dist[len_query_words] = 1

        if len_query_tokens in bert_query_length_dist:
            bert_query_length_dist[len_query_tokens] += 1
        else:
            bert_query_length_dist[len_query_tokens] = 1

    with open(f"whitespace_query_length_dist_{dataset_name}.json", "w") as f:
        json.dump(whitespace_query_length_dist, f)
    
    with open(f"bert_query_length_dist_{dataset_name}.json", "w") as f:
        json.dump(bert_query_length_dist, f)

    print(f"Done processing {dataset_name}")

    
        