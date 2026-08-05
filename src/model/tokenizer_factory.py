from transformers import AutoTokenizer


class TokenizerFactory:

    @staticmethod
    def build(config):

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
        )

        return tokenizer