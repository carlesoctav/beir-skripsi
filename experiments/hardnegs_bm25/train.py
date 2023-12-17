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


corpus_path = str(here('datasets/mmarco/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mmarco/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mmarco/indonesian/qrels/dev.tsv'))

corpus, queries, _ = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

train_batch_size = 32           
max_seq_length = 250            
ce_score_margin = 3             
num_negs_per_system = 5         

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
        pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin
        
        #Get the hard negatives
        neg_pids = set()
        for system_negs in data['neg'].values():
            negs_added = 0
            for item in system_negs:
                if item['ce-score'] > ce_score_threshold:
                    continue

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


model_name = "indolem/indobert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


retriever = TrainRetriever(model=model, batch_size=train_batch_size)


train_dataset = MSMARCODataset(train_queries, corpus=corpus)

print(len(train_dataset))

train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)

#### training SBERT with dot-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score, scale=1)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-hardnegs-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)


num_epochs = 5
evaluation_steps = 10000
warmup_steps = int(0.1 * num_epochs * len(train_dataset) / train_batch_size )    #10% of train data for warm-up
print(f"==>> warmup_steps: {warmup_steps}")

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=num_epochs,
                output_path=model_save_path,
                evaluation_steps=evaluation_steps,
                use_amp=True)


# model.save_to_hub("indobert-mmarco-hardnegs-bm25", "carles-undergrad-thesis")