from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import util as utils

import pathlib, os, sys
import numpy as np
import torch
import logging
import datetime
from pyprojroot import here
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "mmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = here("datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path+"/indonesian").load(split="dev")
corpus_ids, query_ids = list(corpus), list(queries)

number_docs = 1000000
reduced_corpus = [corpus[corpus_id] for corpus_id in corpus_ids[:number_docs]]



model_path = "carles-undergrad-thesis/indobert-mmarco-hardnegs-bm25"
model = models.SentenceBERT(model_path=model_path)
normalize = False


logging.info("Computing Document Embeddings...")
if normalize:
    corpus_embs = model.encode_corpus(reduced_corpus, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
else:
    corpus_embs = model.encode_corpus(reduced_corpus, batch_size=128, convert_to_tensor=True)

#### Saving benchmark times
time_taken_all = {}

for query_id in query_ids:
    query = queries[query_id]
    
    #### Compute query embedding and retrieve similar scores using dot-product
    start = datetime.datetime.now()
    if normalize:
        query_emb = model.encode_queries([query], batch_size=1, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    else:
        query_emb = model.encode_queries([query], batch_size=1, convert_to_tensor=True, show_progress_bar=False)
    
    #### Dot product for normalized embeddings is equal to cosine similarity
    sim_scores = utils.dot_score(query_emb, corpus_embs)
    sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(sim_scores, 10, dim=1, largest=True, sorted=True)
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    time_taken_all[query_id] = time_taken
    logging.info("{}: {} {:.2f}ms".format(query_id, query, time_taken))

time_taken = list(time_taken_all.values())
logging.info("Average time taken: {:.2f}ms".format(sum(time_taken)/len(time_taken_all)))

#### Measuring Index size consumed by document embeddings
corpus_embs = corpus_embs.cpu()
cpu_memory = sys.getsizeof(np.asarray([emb.numpy() for emb in corpus_embs]))

logging.info("Number of documents: {}, Dim: {}".format(len(corpus_embs), len(corpus_embs[0])))
logging.info("Index size (in MB): {:.2f}MB".format(cpu_memory*0.000001))