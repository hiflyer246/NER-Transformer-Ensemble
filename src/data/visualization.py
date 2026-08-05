from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_from_disk

from src.config.labels import ID2LABEL

# Load dataset
dataset = load_from_disk("data/raw/conll2003")

counter = Counter()

for sample in dataset["train"]:
    counter.update(sample["ner_tags"])

labels = [ID2LABEL[i] for i in sorted(counter.keys())]
counts = [counter[i] for i in sorted(counter.keys())]

plt.figure(figsize=(10, 5))
plt.bar(labels, counts)

plt.title("CoNLL-2003 Entity Label Distribution")
plt.xlabel("NER Labels")
plt.ylabel("Number of Tokens")

plt.tight_layout()

plt.show()