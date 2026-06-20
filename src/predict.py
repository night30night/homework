"""
Unified Prediction Interface for Rumor Detection
Loads the trained model and provides a simple API for inference.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explainer import RumorExplainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


class RumorDetector:
    """
    Rumor detection model that outputs both classification and explanation.

    Usage:
        detector = RumorDetector()
        pred, confidence, explanation = detector.predict("Your text here")
        print(f"Prediction: {'Rumor' if pred == 1 else 'Non-Rumor'}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Explanation: {explanation}")
    """

    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR
        self.explainer = RumorExplainer(checkpoint_dir)

    def predict(self, text: str):
        """
        Predict whether a tweet is a rumor and provide explanation.

        Args:
            text: Input tweet text (English)

        Returns:
            dict with keys:
                - prediction: int (0=non-rumor, 1=rumor)
                - confidence: float (0-1)
                - explanation: str
        """
        pred_class, confidence, explanation = self.explainer.predict_and_explain(text)
        return {
            "prediction": pred_class,
            "confidence": confidence,
            "explanation": explanation,
            "label": "Rumor" if pred_class == 1 else "Non-Rumor",
        }

    def batch_predict(self, texts: list):
        """Predict for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


# ── Demo ─────────────────────────────────────────────────────────────────
def demo():
    detector = RumorDetector()

    test_texts = [
        # Likely rumor
        "BREAKING: The government is hiding the truth about the incident! Wake up people!",
        # Likely non-rumor
        "According to the official statement released by the White House, the bill was signed today.",
        # Ambiguous
        "OMG I can't believe what just happened at the airport! This is insane!",
    ]

    for text in test_texts:
        result = detector.predict(text)
        print("=" * 60)
        print(f"Text: {text[:100]}...")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
        print(f"Explanation: {result['explanation']}")
        print()


if __name__ == "__main__":
    demo()
