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
    percentile = [0.1, 0.25, 0.5, 0.75, 0.95]
    print(df.describe())
    print("Percentile: ")
    print(df.quantile(percentile))





if __name__ == "__main__":
    list_of_dist = [
    "for_datasets_analysis/bert_corpus_length_dist_miracl dev set.json", 
    "for_datasets_analysis/whitespace_corpus_length_dist_mmarco train set.json",
    "for_datasets_analysis/bert_corpus_length_dist_mmarco train set.json", 
    "for_datasets_analysis/whitespace_corpus_length_dist_miracl dev set.json", 
    "for_datasets_analysis/bert_corpus_length_dist_mrtydi dev set.json",
    "for_datasets_analysis/whitespace_corpus_length_dist_mrtydi dev set.json"

    ]

    
    for i, dist in enumerate(list_of_dist):
        describe(dist)