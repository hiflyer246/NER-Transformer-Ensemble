from pathlib import Path


def save_report(report, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "classification_report.txt"

    with open(report_file, "w") as f:
        f.write(report)

    print("Classification report saved:", report_file)