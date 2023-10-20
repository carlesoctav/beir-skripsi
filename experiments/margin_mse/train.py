from sentence_transformers import SentenceTransformer, models, InputExample
from beir import util, LoggingHandler, losses
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
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = here("datasets")
data_path = util.download_and_unzip(url, out_dir)


corpus, queries, _ = GenericDataLoader(data_path+"/indonesian").load(split="train")

#################################
#### Parameters for Training ####
#################################

train_batch_size = 10       # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^2))


triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

logging.info("Loading MMMARCO hard-negatives...")

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_scores = dict(zip(pos_pids, [item['ce-score'] for item in data['pos']]))
        
        #Get all the hard negatives
        neg_pids = set()
        neg_scores = {}
        for system_negs in data['neg'].values():
            for item in system_negs:
                pid = item['pid']
                score = item['ce-score']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    neg_scores[pid] = score
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'pos_scores': pos_scores, 
                                          'hard_neg': neg_pids, 'hard_neg_scores': neg_scores}
        
logging.info("Train queries: {}".format(len(train_queries)))

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

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)
        pos_score = float(query['pos_scores'][pos_id])

        neg_id = query['hard_neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)
        neg_score = float(query['hard_neg_scores'][neg_id])

        return InputExample(texts=[query_text, pos_text, neg_text], label=(pos_score - neg_score))

    def __len__(self):
        return len(self.queries)

# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
model_name = "indolem/indobert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)

#### Training SBERT with dot-product (default) using Margin MSE Loss
train_loss = losses.MarginMSELoss(model=retriever.model)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v3-margin-MSE-loss-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 5
evaluation_steps = 10000
warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1)    #10% 

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)

model.save_to_hub("indobert-mmarco-margin-mse", "carles-undergrad-thesis")