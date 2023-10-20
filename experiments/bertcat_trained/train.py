from beir.datasets.data_loader import GenericDataLoader
import gzip
import pathlib, os
import logging
import random
from pyprojroot import here
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import tarfile
import tqdm
import time



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = 'indolem/indobert-base-uncased'
train_batch_size = 32
num_epochs = 5
model_save_path = 'output/training_mm-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pos_neg_ration = 3
max_train_samples = 500_000
model = CrossEncoder(model_name, num_labels=1, max_length=512)

dataset = "mmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = here("datasets")
data_path = str(here("datasets/mmarco"))

corpus, queries, qrels = GenericDataLoader(data_path+"/indonesian").load(split="train")


train_filepath = os.path.join(out_dir, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download "+os.path.basename(train_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)


train_samples = []

cnt = 0
with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split()

        query = queries[qid]
        if (cnt % (pos_neg_ration+1)) == 0:
            passage = corpus[pos_id]["text"]
            label = 1
        else:
            passage = corpus[neg_id]["text"]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

for x in train_samples[:5]:
    print(x)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = SequentialEvaluator([], main_score_function=lambda x: time.time())


warmup_steps = 0.1 * num_epochs * len(train_samples) / train_batch_size
print(f"==>> warmup_steps: {warmup_steps}")
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          evaluation_steps=1_000_000_000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)

#Save latest model
model.save(model_save_path+'-latest')

model.model.push_to_hub('carles-undergrad-thesis/indobert-crossencoder-mmarco')
model.tokenizer.push_to_hub('carles-undergrad-thesis/indobert-crossencoder-mmarco')