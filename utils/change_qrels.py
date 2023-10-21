import csv
# read in qrels.tsv as a list of dictionaries
with open('utils/qrels.tsv', mode='r') as file:
    reader = csv.reader(file, delimiter='\t')
    rows = [["query-id", "corpus-id", "score"]]
    for row in reader:
        if row[3] == '1':
            rows.append([row[0], row[2], row[3]])


# write out qrels.tsv
with open('qrels.tsv', mode='w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(rows)


