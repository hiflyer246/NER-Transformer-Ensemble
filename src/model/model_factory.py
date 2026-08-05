from transformers import AutoModelForTokenClassification

from src.config.labels import ID2LABEL, LABEL2ID


class ModelFactory:

    @staticmethod
    def build(config):

        model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        return model