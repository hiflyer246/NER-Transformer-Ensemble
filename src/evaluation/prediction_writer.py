import json
from pathlib import Path


def save_predictions(predictions, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "predictions.json"

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print("Predictions saved:", output_file)