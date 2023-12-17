from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
from pyprojroot import here
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "mmarco"
corpus_path = str(here('datasets/mmarco/indonesian/corpus.jsonl'))
query_path = str(here('datasets/mmarco/indonesian/queries.jsonl'))
qrels_path = str(here('datasets/mmarco/indonesian/qrels/dev.tsv'))

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

model_name = "indolem/indobert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


retriever = TrainRetriever(model=model, batch_size=32)

train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, scale=1.0, similarity_fct=util.dot_score)

ir_evaluator = retriever.load_dummy_evaluator()

model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v1-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)


num_epochs = 5
evaluation_steps = 1_000_000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)


# model.save_to_hub("st-indobert-mmarco-v1", "carles-undergrad-thesis")