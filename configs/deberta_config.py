from configs.base_config import BaseConfig

config = BaseConfig(
    model_name="microsoft/deberta-v3-large",
    output_dir="models/deberta-large",
    dataset_path="data/processed/deberta-large",

    # Memory
    train_batch_size=2,
    eval_batch_size=2,
    gradient_accumulation_steps=4,

    # Mixed Precision
    fp16=False,
    bf16=False,
)