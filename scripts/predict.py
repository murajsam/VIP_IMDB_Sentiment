"""
scripts/predict.py
------------------
Jednostavan CLI za proveru istreniranog modela nad JEDNOM recenzijom.
Koristan za brzu demonstraciju na odbrani (bez pokretanja cele aplikacije) i
kao osnova za Streamlit aplikaciju u seminarskom radu.

Pokretanje (iz korena projekta):
    python scripts/predict.py "This movie was fantastic, I really enjoyed it!"
    python scripts/predict.py            # zatim ukucajte tekst i pritisnite Enter
"""

import os
import sys
import json

import numpy as np
import torch

# Omogućavamo uvoz paketa src iz korena projekta.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.preprocessing import SentimentPreprocessor           # noqa: E402
from src.model import ModelConfig, build_model                # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")


def load_model():
    """Učitava sačuvani model, njegovu konfiguraciju i preprocessor."""
    with open(os.path.join(MODEL_DIR, "best_model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = ModelConfig(**meta["model"])
    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                                     map_location="cpu"))
    model.eval()
    pre = SentimentPreprocessor.load(os.path.join(MODEL_DIR, "preprocessor.json"))
    return model, pre


@torch.no_grad()
def predict(text: str, model, pre) -> dict:
    x = torch.from_numpy(pre.transform_text(text)).unsqueeze(0)  # (1, max_len)
    probs = torch.softmax(model(x), dim=1).squeeze(0).numpy()
    label = int(np.argmax(probs))
    return {
        "sentiment": "POZITIVAN" if label == 1 else "NEGATIVAN",
        "prob_pozitivan": float(probs[1]),
        "prob_negativan": float(probs[0]),
    }


def main():
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pt")):
        print("Model nije pronađen. Prvo pokrenite trening: python -m src.train --quick")
        sys.exit(1)

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("Unesite recenziju (na engleskom): ").strip()

    model, pre = load_model()
    result = predict(text, model, pre)
    print("\nRecenzija:", text)
    print("Predikcija:", result["sentiment"])
    print(f"Verovatnoća (pozitivan): {result['prob_pozitivan']:.1%}")
    print(f"Verovatnoća (negativan): {result['prob_negativan']:.1%}")


if __name__ == "__main__":
    main()
