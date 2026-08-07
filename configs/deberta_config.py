from dataclasses import dataclass


@dataclass
class BaseConfig:

    # Model
    model_name: str

    # Paths
    output_dir: str
    dataset_path: str

    # Training
    learning_rate: float = 1e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 3
    weight_decay: float = 0.01

    # Tokenization
    max_length: int = 512

    # Reproducibility
    seed: int = 42

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Logging
    logging_steps: int = 100
    logging_dir: str = "outputs/logs"
    report_to: str = "none"

    # Evaluation
    eval_strategy: str = "epoch"
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True

    # ===============================
    # Checkpoint Saving
    # ===============================
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    save_safetensors: bool = True

    # Resume
    resume_from_checkpoint: bool = False

    # Mixed Precision
    fp16: bool = False
    bf16: bool = False