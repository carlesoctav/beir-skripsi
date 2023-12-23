from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os
import datetime
import logging
import random
from pyprojroot import here

random.seed(42)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])




corpus_path = str(here('datasets/mmarco/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mmarco/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mmarco/indonesian/qrels/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

corpus_ids, query_ids = list(corpus), list(queries)

corpus_set = set()
for query_id in qrels:
    corpus_set.update(list(qrels[query_id].keys()))
corpus_new = {corpus_id: corpus[corpus_id] for corpus_id in corpus_set}

remaining_corpus = list(set(corpus_ids) - corpus_set)
sample = 1000000 - len(corpus_set)

for corpus_id in random.sample(remaining_corpus, sample):
    corpus_new[corpus_id] = corpus[corpus_id]

hostname = "localhost"
index_name = "mmarco"
initialize = True
language = "indonesian" 
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, language=language)
bm25 = EvaluateRetrieval(model)
 

bm25.retriever.index(corpus_new)


time_taken_all = {}

for query_id in query_ids:
    query = queries[query_id]
    
    
    start = datetime.datetime.now()
    results = bm25.retriever.es.lexical_search(text=query, top_hits=10) 
    end = datetime.datetime.now()
    
    
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    time_taken_all[query_id] = time_taken
    logging.info("{}: {} {:.2f}ms".format(query_id, query, time_taken))

time_taken = list(time_taken_all.values())
logging.info("Average time taken: {:.2f}ms".format(sum(time_taken)/len(time_taken_all)))