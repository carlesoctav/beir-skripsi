'''
This example shows how to train a SOTA Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained using hard negatives which were specially mined with different dense and lexical search methods for MSMARCO. 

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

Running this script:
python train_msmarco_v3.py
'''

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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download msmarco.zip dataset and unzip the dataset
dataset = "mmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = here("datasets")
data_path = util.download_and_unzip(url, out_dir)

### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, _ = GenericDataLoader(data_path+"/indonesian").load(split="train")

#################################
#### Parameters for Training ####
#################################

train_batch_size = 32           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n))
max_seq_length = 250            # Max length for passages. Increasing it, requires more GPU memory (O(n^2))
ce_score_margin = 3             # Margin for the CrossEncoder score between negative and positive passagesu
num_negs_per_system = 5         # We used different systems to mine hard negatives. Number of hard negatives to add from each system

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
#### We chose a margin between positive and negative passage scores => above which consider negative as hard negative. 
#### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        #Get the hard negatives
        neg_pids = set()
        for system_negs in data['neg'].values():
            negs_added = 0
            for item in system_negs:
                pid = item['pid']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'hard_neg': list(neg_pids)}

        
logging.info("Train queries: {}".format(len(train_queries)))

for key, value in train_queries.items():
    print(key, value)
    break

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

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
model_name = "indolem/indobert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)

print(len(train_dataset))

train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)

#### training SBERT with dot-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score, scale=1)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v3-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 5
evaluation_steps = 10000
warmup_steps = int(0.1 * num_epochs * len(train_dataset) / train_batch_size )    #10% of train data for warm-up
print(f"==>> warmup_steps: {warmup_steps}")

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=num_epochs,
                output_path=model_save_path,
                evaluation_steps=evaluation_steps,
                use_amp=True)


model.save_to_hub("indobert-mmarco-hardnegs-bm25", "carles-undergrad-thesis")