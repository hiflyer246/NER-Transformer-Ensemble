from datasets import load_from_disk
from transformers import AutoTokenizer
import os

MODEL_NAME = "bert-large-cased"

print("Loading dataset...")
dataset = load_from_disk("data/raw/conll2003")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            label_ids.append(example["ner_tags"][word_idx])

        else:
            label_ids.append(-100)

        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids

    return tokenized_inputs


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=False,
)

print("Removing unused columns...")

tokenized_dataset = tokenized_dataset.remove_columns(
    ["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"]
)

print(tokenized_dataset)

os.makedirs("data/processed/bert-large", exist_ok=True)

tokenized_dataset.save_to_disk("data/processed/bert-large")

print("\nDataset successfully saved!")