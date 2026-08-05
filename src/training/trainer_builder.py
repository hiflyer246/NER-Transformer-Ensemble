from transformers import Trainer, TrainingArguments

from src.evaluation.metrics import compute_metrics
from src.training.data_collator import build_data_collator
from src.training.callbacks import early_stopping


def build_trainer(config, model, dataset):

    training_args = TrainingArguments(

        # =====================================================
        # Output
        # =====================================================

        output_dir=config.output_dir,

        # =====================================================
        # Training
        # =====================================================

        do_train=True,
        do_eval=True,

        learning_rate=config.learning_rate,

        per_device_train_batch_size=config.train_batch_size,

        per_device_eval_batch_size=config.eval_batch_size,

        gradient_accumulation_steps=config.gradient_accumulation_steps,

        num_train_epochs=config.epochs,

        weight_decay=config.weight_decay,

        seed=config.seed,

        # =====================================================
        # Evaluation
        # =====================================================

        eval_strategy=config.eval_strategy,

        metric_for_best_model=config.metric_for_best_model,

        greater_is_better=config.greater_is_better,

        # =====================================================
        # Saving
        # =====================================================

        save_strategy=config.save_strategy,

        save_steps=config.save_steps,

        save_total_limit=config.save_total_limit,

        load_best_model_at_end=config.load_best_model_at_end,

        # =====================================================
        # Logging
        # =====================================================

        logging_strategy="steps",

        logging_steps=config.logging_steps,

        logging_dir=config.logging_dir,

        report_to=config.report_to,

        logging_first_step=True,

        # =====================================================
        # Mixed Precision
        # =====================================================

        fp16=config.fp16,

        bf16=config.bf16,

        # =====================================================
        # Data Loader
        # =====================================================

        dataloader_num_workers=2,

        dataloader_pin_memory=True,

        remove_unused_columns=False,
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        data_collator=build_data_collator(config),

        compute_metrics=compute_metrics,

        callbacks=[early_stopping],
    )

    return trainer