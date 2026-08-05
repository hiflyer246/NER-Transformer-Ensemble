from configs.base_config import BaseConfig

config = BaseConfig(
    model_name="microsoft/deberta-v3-large",
    output_dir="models/deberta-large",
    dataset_path="data/processed/deberta-large",
)