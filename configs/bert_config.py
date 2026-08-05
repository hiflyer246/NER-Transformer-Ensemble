from configs.base_config import BaseConfig

config = BaseConfig(
    model_name="bert-large-cased",
    output_dir="models/bert-large",
    dataset_path="data/processed/bert-large",
)