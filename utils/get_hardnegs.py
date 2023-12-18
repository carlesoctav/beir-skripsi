from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.train import TrainRetriever
import pathlib, os, tqdm
import logging
from pyprojroot import here
from datasets import Dataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "mmarco"

corpus_path = str(here('datasets/mmarco/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mmarco/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mmarco/indonesian/qrels/train.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

hostname = "localhost" 
index_name = "mmarco-indo" 
language = "indonesian" 
initialize = False 
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards, language=language)

bm25 = EvaluateRetrieval(model)

# bm25.retriever.index(corpus)

triplets = []
# print(qrels)
qids = list(qrels)
# print(qids)
hard_negatives_max = 10

for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25"):
    query_id, query_text = qids[idx], queries[qids[idx]]
    pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
    pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
    hits = bm25.retriever.es.lexical_multisearch(texts=pos_doc_texts, top_hits=hard_negatives_max+1)
    for (pos_text, hit) in zip(pos_doc_texts, hits):
        jsonl = {}
        jsonl["qid"] = query_id
        jsonl["pos"] = pos_docs
        neg_list_ids = []
        for (neg_id, _) in hit.get("hits"):
            if neg_id not in pos_docs:
                neg_list_ids.append(neg_id)
                jsonl["neg"] = neg_list_ids
        triplets.append(jsonl)

hardnegs = Dataset.from_list(triplets)
hardnegs.push_to_hub("carles-undergrad-thesis/mmarco-hardnegs-bm25")