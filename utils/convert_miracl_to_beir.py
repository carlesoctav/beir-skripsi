import csv
import json

# Open the TSV file and read its contents
with open('utils/query.tsv', 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')

    rows = []
    for row in reader:
        jsonl = {
            "_id": row[0],
            "text": row[1],
        }
        rows.append(jsonl)
    print(rows[:10])


with open('query.jsonl', 'w') as jsonlfile:
    for row in rows:
        json.dump(row, jsonlfile)
        jsonlfile.write('\n')
