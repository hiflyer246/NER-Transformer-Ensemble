import argparse
import json
from pathlib import Path

from transformers import Trainer

from src.utils.config_loader import get_config
from src.data.dataset_factory import DatasetFactory
from src.model.model_factory import ModelFactory
from src.training.data_collator import build_data_collator
from src.evaluation.metrics import compute_metrics


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained NER model"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["bert", "roberta", "deberta", "xlmr"],
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Loading Configuration")
    print("=" * 80)

    config = get_config(args.model)

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    dataset = DatasetFactory.load(config)

    print("=" * 80)
    print("Loading Model")
    print("=" * 80)

    model = ModelFactory.build(config)

    print("=" * 80)
    print("Loading Trained Weights")
    print("=" * 80)

    model = model.from_pretrained(config.output_dir)

    trainer = Trainer(
        model=model,
        data_collator=build_data_collator(config),
        compute_metrics=compute_metrics,
    )

    print("=" * 80)
    print("Evaluating")
    print("=" * 80)

    metrics = trainer.evaluate(dataset["test"])

    print(metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print()
    print("Metrics saved to")
    print(metrics_file)


if __name__ == "__main__":
    main()