from dataclasses import dataclass


@dataclass
class BaseConfig:
    """
    Base configuration shared by all transformer models.
    """

    # =====================================================
    # Model Information
    # =====================================================

    model_name: str
    output_dir: str
    dataset_path: str

    # =====================================================
    # Training Hyperparameters
    # =====================================================

    learning_rate: float = 2e-5

    # Optimized for Tesla T4 (16 GB)
    train_batch_size: int = 4
    eval_batch_size: int = 4

    epochs: int = 3

    weight_decay: float = 0.01

    max_length: int = 512

    seed: int = 42

    # Effective batch size = 4 × 2 = 8
    gradient_accumulation_steps: int = 2

    # =====================================================
    # Logging
    # =====================================================

    logging_steps: int = 100

    logging_dir: str = "outputs/logs"

    report_to: str = "none"

    # =====================================================
    # Evaluation
    # =====================================================

    eval_strategy: str = "epoch"

    metric_for_best_model: str = "f1"

    greater_is_better: bool = True

    # =====================================================
    # Saving
    # =====================================================

    save_strategy: str = "epoch"

    save_total_limit: int = 2

    save_steps: int = 500

    load_best_model_at_end: bool = True

    save_safetensors: bool = True

    # =====================================================
    # Resume Training
    # =====================================================

    resume_from_checkpoint: bool = False

    # =====================================================
    # Hardware
    # =====================================================

    # Mixed precision for Tesla T4
    fp16: bool = True

    # T4 does not support BF16 efficiently
    bf16: bool = False