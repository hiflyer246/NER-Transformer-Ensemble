import argparse

from src.utils.config_loader import get_config
from src.data.dataset_factory import DatasetFactory
from src.model.model_factory import ModelFactory
from src.evaluation.evaluator import ModelEvaluator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["bert", "roberta", "deberta", "xlmr"],
    )

    args = parser.parse_args()

    config = get_config(args.model)

    dataset = DatasetFactory.load(config)

    model = ModelFactory.build(config)

    model = model.from_pretrained(config.output_dir)

    evaluator = ModelEvaluator(
        config,
        model,
        dataset,
    )

    evaluator.evaluate()

    evaluator.predict()


if __name__ == "__main__":
    main()