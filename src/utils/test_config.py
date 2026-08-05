from src.utils.config_loader import get_config

config = get_config("bert")

print("=" * 50)

print(config)

print("=" * 50)

print("Model :", config.model_name)

print("Output:", config.output_dir)