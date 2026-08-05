from configs.bert_config import config as bert_config
from configs.roberta_config import config as roberta_config
from configs.deberta_config import config as deberta_config
from configs.xlmr_config import config as xlmr_config


def get_config(model_name: str):

    configs = {
        "bert": bert_config,
        "roberta": roberta_config,
        "deberta": deberta_config,
        "xlmr": xlmr_config,
    }

    if model_name not in configs:
        raise ValueError(f"Unsupported model: {model_name}")

    return configs[model_name]