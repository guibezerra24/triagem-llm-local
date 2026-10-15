from .common import LABELS


def classification_metrics(records):
    if not records:
        raise ValueError("Conjunto vazio")
    matrix = {label: {pred: 0 for pred in LABELS + ["INVALIDO"]} for label in LABELS}
    for row in records:
        matrix[row["label"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "support": sum(matrix[label].values())}
    return {"n": len(records), "accuracy": sum(r["label"] == r["prediction"] for r in records) / len(records),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
            "valid_rate": sum(r["prediction"] in LABELS for r in records) / len(records),
            "mean_seconds": sum(r["seconds"] for r in records) / len(records),
            "per_class": per_class, "confusion_matrix": matrix}
