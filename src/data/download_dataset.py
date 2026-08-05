import os
from datasets import load_dataset

print("Downloading CoNLL-2003...")

dataset = load_dataset("lhoestq/conll2003")

os.makedirs("data/raw", exist_ok=True)

dataset.save_to_disk("data/raw/conll2003")

print("===================================")
print("Dataset downloaded successfully!")
print(dataset)
print("===================================")