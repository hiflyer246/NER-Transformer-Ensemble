from configs.base_config import BaseConfig

config = BaseConfig(

    model_name="microsoft/deberta-v3-large",

    output_dir="models/deberta-large",

    dataset_path="data/processed/deberta-large",

    # =====================================================
    # Training
    # =====================================================

    learning_rate=1e-5,

    train_batch_size=2,

    eval_batch_size=2,

    # TEMPORARY FOR DEBUGGING
    gradient_accumulation_steps=1,

    epochs=3,

    weight_decay=0.01,

    # =====================================================
    # Mixed Precision
    # =====================================================

    fp16=False,

    bf16=False,
)