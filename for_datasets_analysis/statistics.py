import matplotlib.pyplot as plt
import json
import statistics
import pandas as pd

def describe(dict):

    with open(dict, "r") as f:
        data = json.load(f)

    list_of_data = []

    for key, value in data.items():
        for i in range(value):
            list_of_data.append(int(key))

    df = pd.DataFrame(list_of_data)
    print("Dataset: ", dict)
    
    percentile = [0, 0.5, 0.95, 1]
    print("Percentile: ")
    print(df.quantile(percentile))
    print("=========")





if __name__ == "__main__":
    list_of_dist_whitespace = [
    "for_datasets_analysis/whitespace_corpus_length_dist_mmarco train set.json",
    "for_datasets_analysis/whitespace_query_length_dist_mmarco train set.json",
    "for_datasets_analysis/whitespace_query_length_dist_mmarco dev set.json",

    "for_datasets_analysis/whitespace_corpus_length_dist_mrtydi dev set.json",
    "for_datasets_analysis/whitespace_query_length_dist_mrtydi dev set.json",

    "for_datasets_analysis/whitespace_corpus_length_dist_miracl dev set.json",
    "for_datasets_analysis/whitespace_query_length_dist_miracl dev set.json"
    ]

    list_of_dist_bert = [
    "for_datasets_analysis/bert_corpus_length_dist_mmarco train set.json",
    "for_datasets_analysis/bert_query_length_dist_mmarco train set.json",
    "for_datasets_analysis/bert_query_length_dist_mmarco dev set.json",

    "for_datasets_analysis/bert_corpus_length_dist_mrtydi dev set.json",
    "for_datasets_analysis/bert_query_length_dist_mrtydi dev set.json",

    "for_datasets_analysis/bert_corpus_length_dist_miracl dev set.json",
    "for_datasets_analysis/bert_query_length_dist_miracl dev set.json"
    ]

    
    for i, dist in enumerate(list_of_dist_bert):
        describe(dist)