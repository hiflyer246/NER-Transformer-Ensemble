import argparse

from src.utils.config_loader import get_config
from src.data.dataset_factory import DatasetFactory
from src.model.model_factory import ModelFactory
from src.training.trainer_builder import build_trainer


def main():

    parser = argparse.ArgumentParser()

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

    trainer = build_trainer(config,model,dataset)

    print()

    print("=" * 80)
    print("Pipeline Ready")
    print("=" * 80)

    print()

    print("Model :", config.model_name)
    print("Output:", config.output_dir)

    # DO NOT TRAIN YET
    # trainer.train()


if __name__ == "__main__":
    main()