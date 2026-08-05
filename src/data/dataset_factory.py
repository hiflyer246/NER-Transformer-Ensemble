from datasets import load_from_disk


class DatasetFactory:

    @staticmethod
    def load(config):
        print(f"Loading dataset from: {config.dataset_path}")

        dataset = load_from_disk(config.dataset_path)

        return dataset