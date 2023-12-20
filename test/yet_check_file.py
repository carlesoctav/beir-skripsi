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

corpus_path = str(here('datasets/miracl/corpus.jsonl'))
query_path = str(here('datasets/miracl/queries.jsonl'))
qrels_path = str(here('datasets/miracl/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


print(qrels)