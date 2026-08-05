from transformers import DataCollatorForTokenClassification

from src.model.tokenizer_factory import TokenizerFactory


def build_data_collator(config):

    tokenizer = TokenizerFactory.build(config)

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    return collator