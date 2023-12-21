from sentence_transformers import SentenceTransformer, models, losses, InputExample
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import pathlib, os, gzip, json
import logging
import random
from pyprojroot import here

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


dataset = "mmarco"
corpus_path = str(here('datasets/mmarco/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mmarco/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mmarco/indonesian/qrels/train.tsv'))

corpus, queries, _ = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


train_batch_size = 32           
max_seq_length = 250           
num_negs = 5

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join("datasets", "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
missing_bm25_counter = 0
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn):
        data = json.loads(line)
        pos_pids = [item['pid'] for item in data['pos']]
        qid = data['qid']
        neg_pids = set()
        try:
            for item in data['neg']["bm25"]:
                pid = item['pid']
                neg_pids.add(pid)

                if len(neg_pids) >= num_negs:
                    break
        except:
            missing_bm25_counter += 1
            continue
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[qid] = {'query': queries[qid], 'pos': pos_pids, 'hard_neg': list(neg_pids)}

print("Missing BM25 counter: {}".format(missing_bm25_counter))        
logging.info("Train queries: {}".format(len(train_queries)))

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.

class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

model_name = "indolem/indobert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


retriever = TrainRetriever(model=model, batch_size=train_batch_size)
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score, scale=1)
ir_evaluator = retriever.load_dummy_evaluator()

model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-hardnegs-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)


num_epochs = 5
evaluation_steps = 10000
warmup_steps = int(0.1 * num_epochs * len(train_dataset) / train_batch_size)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=num_epochs,
                output_path=model_save_path,
                evaluation_steps=evaluation_steps,
                use_amp=True)


model.save_to_hub("indobert-mmarco-hardnegs-bm25", "carles-undergrad-thesis")