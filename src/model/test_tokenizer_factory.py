from src.utils.config_loader import get_config
from src.model.tokenizer_factory import TokenizerFactory

config = get_config("bert")

print("=" * 60)
print("Loading Tokenizer...")
print("=" * 60)

tokenizer = TokenizerFactory.build(config)

print()

print(type(tokenizer))

print()

print(tokenizer.tokenize("Microsoft hired John in Seattle"))