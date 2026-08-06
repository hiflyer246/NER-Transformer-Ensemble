import argparse
import os

from datasets import load_from_disk

from src.utils.config_loader import get_config
from src.model.tokenizer_factory import TokenizerFactory


# =====================================================
# Command Line Arguments
# =====================================================

parser = argparse.ArgumentParser(
    description="Tokenize CoNLL-2003 dataset for Transformer models"
)

parser.add_argument(
    "--model",
    required=True,
    choices=["bert", "roberta", "deberta", "xlmr"],
    help="Model name to preprocess dataset for",
)

args = parser.parse_args()

# =====================================================
# Load Configuration
# =====================================================

config = get_config(args.model)

print("=" * 80)
print("Configuration")
print("=" * 80)
print(config)

# =====================================================
# Load Dataset
# =====================================================

print("\nLoading CoNLL-2003 Dataset...")

dataset = load_from_disk("data/raw/conll2003")

# =====================================================
# Load Tokenizer
# =====================================================

print("\nLoading Tokenizer...")

tokenizer = TokenizerFactory.build(config)

# =====================================================
# Tokenization Function
# =====================================================

def tokenize_and_align_labels(example):

    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        max_length=config.max_length,
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


# =====================================================
# Tokenize Dataset
# =====================================================

print("\nTokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=False,
)

# =====================================================
# Remove Unused Columns
# =====================================================

print("\nRemoving Unused Columns...")

tokenized_dataset = tokenized_dataset.remove_columns(
    [
        "id",
        "tokens",
        "pos_tags",
        "chunk_tags",
        "ner_tags",
    ]
)

print(tokenized_dataset)

# =====================================================
# Save Dataset
# =====================================================

os.makedirs(config.dataset_path, exist_ok=True)

tokenized_dataset.save_to_disk(config.dataset_path)

print("\nDataset Saved Successfully!")
print("Location:", config.dataset_path)

print("\nDone.")