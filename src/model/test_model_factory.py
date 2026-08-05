from src.utils.config_loader import get_config
from src.model.model_factory import ModelFactory

config = get_config("bert")

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = ModelFactory.build(config)

print()

print("Model Loaded Successfully!")

print(model.__class__.__name__)