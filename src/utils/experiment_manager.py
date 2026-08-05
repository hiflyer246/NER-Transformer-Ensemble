from pathlib import Path
from datetime import datetime


class ExperimentManager:

    @staticmethod
    def create(config):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        experiment_name = f"{config.model_name.replace('/','_')}_{timestamp}"

        model_dir = Path("models") / experiment_name

        output_dir = Path("outputs") / experiment_name

        model_dir.mkdir(parents=True, exist_ok=True)

        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "model_dir": model_dir,
            "output_dir": output_dir,
            "experiment_name": experiment_name,
        }