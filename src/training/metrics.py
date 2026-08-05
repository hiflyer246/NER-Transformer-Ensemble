import evaluate
import numpy as np

from src.config.labels import ID2LABEL

seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [
            ID2LABEL[p]
            for p, l in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [
            ID2LABEL[l]
            for p, l in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }