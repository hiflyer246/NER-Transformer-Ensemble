from src.utils.config_loader import get_config
from src.utils.experiment_manager import ExperimentManager

config = get_config("bert")

exp = ExperimentManager.create(config)

print(exp)