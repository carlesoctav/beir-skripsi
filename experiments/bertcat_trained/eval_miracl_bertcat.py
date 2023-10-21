from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import pathlib, os
import logging
import random
from pyprojroot import here

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


corpus_path = str(here('datasets/miracl/corpus.jsonl'))
query_path = str(here('datasets/miracl/queries.jsonl'))
qrels_path = str(here('datasets/miracl/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


#########################################
#### (1) RETRIEVE Top-100 docs using BM25
#########################################

#### Provide parameters for Elasticsearch
hostname = "localhost" #localhost
index_name = "miracl-indo" # trec-covid
initialize = True # False
language = "indonesian" 

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize,language=language)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

################################################
#### (2) RERANK Top-100 docs using Cross-Encoder
################################################

#### Reranking using Cross-Encoder models #####
#### https://www.sbert.net/docs/pretrained_cross-encoders.html
cross_encoder_model = CrossEncoder('carles-undergrad-thesis/indobert-crossencoder-mmarco', max_length = 512)

#### Or use MiniLM, TinyBERT etc. CE models (https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

reranker = Rerank(cross_encoder_model, batch_size=256)

# Rerank top-100 results using the reranker provided
rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)


mrr = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="hole")


#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))