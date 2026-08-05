from datasets import load_from_disk
from transformers import AutoModelForTokenClassification

from src.config.labels import ID2LABEL, LABEL2ID
from src.config.training_config import MODEL_NAME

from src.training.trainer_builder import build_trainer

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

dataset = load_from_disk("data/processed/bert-large")

print("=" * 70)
print("Loading Model...")
print("=" * 70)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

print("=" * 70)
print("Building Trainer...")
print("=" * 70)

trainer = build_trainer(model, dataset)

print("\nTrainer Created Successfully!")

