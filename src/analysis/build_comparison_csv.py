"""Consolidate the four per-checkpoint evaluation_mbert.py reports
(analysis_results/{text_only,scratch,pretrained,finetune}.json) into two
CSVs for the final vision-ablation comparison: overall macro metrics, and
a per-class breakdown for every metadata head."""
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
MODELS = ["text_only", "scratch", "pretrained", "finetune"]
TASKS = ["period", "genre", "language", "provenience"]

MACRO_METRICS = [
    "loss", "mlm_acc", "mlm_top3_acc", "mlm_top5_acc", "mlm_mrr",
    "period_acc", "period_macro_f1", "genre_acc", "genre_macro_f1",
    "language_acc", "language_macro_f1", "provenience_acc", "provenience_macro_f1",
]


def main():
    reports = {}
    for m in MODELS:
        with open(os.path.join(RESULTS_DIR, f"{m}.json"), encoding="utf-8") as f:
            reports[m] = json.load(f)

    # 1. Macro metrics
    macro_path = os.path.join(RESULTS_DIR, "macro_comparison.csv")
    with open(macro_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + MODELS)
        for metric in MACRO_METRICS:
            row = [metric] + [round(reports[m]["metrics"].get(metric, float("nan")), 4) for m in MODELS]
            writer.writerow(row)
    print(f"Wrote {macro_path}")

    # 2. Per-class F1/support
    per_class_path = os.path.join(RESULTS_DIR, "per_class_comparison.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task", "class", "support"] + [f"{m}_f1" for m in MODELS] + [f"{m}_precision" for m in MODELS] + [f"{m}_recall" for m in MODELS]
        writer.writerow(header)
        for task in TASKS:
            classes = [c for c in reports["text_only"]["per_class"].get(task, {})
                       if c not in ("accuracy", "macro avg", "weighted avg")]
            for cls in classes:
                row = [task, cls]
                support = reports["text_only"]["per_class"][task].get(cls, {}).get("support", "")
                row.append(int(support) if support != "" else "")
                for m in MODELS:
                    stats = reports[m]["per_class"].get(task, {}).get(cls)
                    row.append(round(stats["f1-score"], 4) if stats else "")
                for m in MODELS:
                    stats = reports[m]["per_class"].get(task, {}).get(cls)
                    row.append(round(stats["precision"], 4) if stats else "")
                for m in MODELS:
                    stats = reports[m]["per_class"].get(task, {}).get(cls)
                    row.append(round(stats["recall"], 4) if stats else "")
                writer.writerow(row)
    print(f"Wrote {per_class_path}")


if __name__ == "__main__":
    main()
