from datasets import load_from_disk
from src.config.labels import ID2LABEL

# Load dataset
dataset = load_from_disk("data/raw/conll2003")

# Get first training sample
sample = dataset["train"][0]

print("=" * 70)
print("FIRST TRAINING SAMPLE")
print("=" * 70)

print("\nSentence:\n")
print(" ".join(sample["tokens"]))

print("\nToken-wise Labels:\n")

print(f"{'TOKEN':<20} {'LABEL'}")
print("-" * 35)

for token, tag in zip(sample["tokens"], sample["ner_tags"]):
    print(f"{token:<20} {ID2LABEL[tag]}")