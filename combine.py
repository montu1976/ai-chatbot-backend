files = ["dataset1.jsonl", "dataset2.jsonl", "dataset3.jsonl"]

with open("dataset.jsonl", "w", encoding="utf-8") as outfile:
    for fname in files:
        with open(fname, "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)

print("All files combined into dataset.jsonl!")