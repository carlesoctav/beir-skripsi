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


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])



corpus_path = str(here('datasets/miracl/corpus.jsonl'))
query_path = str(here('datasets/miracl/queries.jsonl'))
qrels_path = str(here('datasets/miracl/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


hostname = "localhost" #localhost
index_name = "miracl-indo" # trec-covid
initialize = True # False
language = "indonesian" 

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize,language=language)
retriever = EvaluateRetrieval(model)


results = retriever.retrieve(corpus, queries)
cross_encoder_model = CrossEncoder('carles-undergrad-thesis/Indobertcat', max_length = 256)
reranker = Rerank(cross_encoder_model, batch_size=256)

rerank_results = reranker.rerank(corpus, queries, results, top_k=1000)

ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="hole")

top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))