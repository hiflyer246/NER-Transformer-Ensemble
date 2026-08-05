import numpy as np
import evaluate

# Load SeqEval metric
seqeval = evaluate.load("seqeval")

LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def compute_metrics(eval_prediction):
    """
    Computes Precision, Recall, F1 and Accuracy
    using SeqEval.
    """

    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        pred_tags = []
        label_tags = []

        for p, l in zip(prediction, label):

            if l != -100:

                pred_tags.append(LABELS[p])
                label_tags.append(LABELS[l])

        true_predictions.append(pred_tags)
        true_labels.append(label_tags)

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