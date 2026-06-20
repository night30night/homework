"""
Evaluate the trained model on val.csv and generate prediction examples.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import RumorDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_on_val():
    print("Loading model...")
    detector = RumorDetector()

    # Load val data
    val_path = os.path.join(BASE_DIR, "val.csv")
    val_df = pd.read_csv(val_path)
    print(f"Evaluating on {len(val_df)} validation samples...")

    correct = 0
    results = []
    for i, row in val_df.iterrows():
        result = detector.predict(row["text"])
        pred = result["prediction"]
        true_label = row["label"]
        if pred == true_label:
            correct += 1
        results.append(
            {
                "id": row["id"],
                "text": row["text"][:100],
                "true_label": true_label,
                "pred_label": pred,
                "confidence": result["confidence"],
                "explanation": result["explanation"],
                "correct": pred == true_label,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_df)}...")

    accuracy = correct / len(val_df)
    print(f"\n{'='*60}")
    print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{len(val_df)})")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Rumor", "Rumor"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Show some examples
    print(f"\n{'='*60}")
    print("Example Predictions:")
    print(f"{'='*60}")

    # Show 3 correct and 2 incorrect examples
    correct_results = [r for r in results if r["correct"]]
    incorrect_results = [r for r in results if not r["correct"]]

    for label, subset in [("Correct", correct_results[:3]), ("Incorrect", incorrect_results[:2])]:
        for r in subset:
            true_str = "Rumor" if r["true_label"] == 1 else "Non-Rumor"
            pred_str = "Rumor" if r["pred_label"] == 1 else "Non-Rumor"
            print(f"\n[{label}] Text: {r['text']}...")
            print(f"  True: {true_str} | Pred: {pred_str} | Conf: {r['confidence']:.2%}")
            print(f"  Explanation: {r['explanation'][:200]}...")

    return accuracy


if __name__ == "__main__":
    evaluate_on_val()
