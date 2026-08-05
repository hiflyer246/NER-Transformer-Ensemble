from datasets import load_from_disk
from src.config.labels import ID2LABEL
from collections import Counter

# Load dataset
dataset = load_from_disk("data/raw/conll2003")

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"Training Samples   : {len(train)}")
print(f"Validation Samples : {len(validation)}")
print(f"Test Samples       : {len(test)}")

print("\n")

# Sentence lengths
lengths = [len(sample["tokens"]) for sample in train]

print(f"Average Sentence Length : {sum(lengths)/len(lengths):.2f}")
print(f"Maximum Sentence Length : {max(lengths)}")
print(f"Minimum Sentence Length : {min(lengths)}")

print("\n")

# Count labels
counter = Counter()

for sample in train:
    counter.update(sample["ner_tags"])

print("=" * 70)
print("ENTITY DISTRIBUTION")
print("=" * 70)

for idx in sorted(counter.keys()):
    print(f"{ID2LABEL[idx]:<10} : {counter[idx]}")