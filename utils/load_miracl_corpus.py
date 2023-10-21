import datasets
import json


lang='id'  # or any of the 16 languages
miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)['train']
miracl_corpus = miracl_corpus.rename_column('docid', '_id')
miracl_corpus.to_json('miracl_corpus.jsonl', orient='records', lines=True)

