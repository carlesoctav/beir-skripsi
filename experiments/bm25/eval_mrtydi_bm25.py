from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random
import logging
from pyprojroot import here

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


corpus_path = str(here('datasets/mrtydi/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mrtydi/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mrtydi/indonesian/qrels/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

hostname = "localhost" 
index_name = "mrtydi-indo" 

initialize = True

language = "indonesian" 

number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)
retriever = EvaluateRetrieval(model)

results = retriever.retrieve(corpus, queries)

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")


query_id, scores_dict = random.choice(list(results.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))