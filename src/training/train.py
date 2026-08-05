import argparse
from pathlib import Path

from src.utils.config_loader import get_config
from src.data.dataset_factory import DatasetFactory
from src.model.model_factory import ModelFactory
from src.training.trainer_builder import build_trainer


def main():

    parser = argparse.ArgumentParser(
        description="Train a Transformer model for Named Entity Recognition"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["bert", "roberta", "deberta", "xlmr"],
        help="Model to train",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Loading Configuration")
    print("=" * 80)

    config = get_config(args.model)

    print(config)
    print()

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    dataset = DatasetFactory.load(config)

    print()

    print("=" * 80)
    print("Loading Model")
    print("=" * 80)

    model = ModelFactory.build(config)

    print()

    print("=" * 80)
    print("Building Trainer")
    print("=" * 80)

    trainer = build_trainer(config, model, dataset)

    print()

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)

    if config.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print()

    print("=" * 80)
    print("Saving Model")
    print("=" * 80)

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(config.output_dir)

    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(config.output_dir)

    print()

    print("=" * 80)
    print("Training Completed Successfully")
    print("=" * 80)

    print(f"Model saved to : {config.output_dir}")


if __name__ == "__main__":
    main()