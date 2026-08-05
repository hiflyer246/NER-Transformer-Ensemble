from src.utils.config_loader import get_config
from src.data.dataset_factory import DatasetFactory

config = get_config("bert")

dataset = DatasetFactory.load(config)

print()
print(dataset)

print()

print("Train Samples      :", len(dataset["train"]))
print("Validation Samples :", len(dataset["validation"]))
print("Test Samples       :", len(dataset["test"]))