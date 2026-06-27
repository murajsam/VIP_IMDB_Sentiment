"""
utils.py
--------
Pomoćne funkcije koje se koriste u celom projektu:
  * fiksiranje random seed-ova (reproduktivnost),
  * evidentiranje verzija biblioteka,
  * merenje vremena,
  * brojanje parametara i veličine modela,
  * pomoćne funkcije za crtanje grafikona (learning curve, konfuziona matrica,
    ROC i PR krive).

Sve funkcije su namerno kratke i komentarisane kako bi se lako objasnile na odbrani.
"""

from __future__ import annotations

import os
import time
import json
import random
import platform
from contextlib import contextmanager

import numpy as np


# ----------------------------------------------------------------------------
# 1. Reproduktivnost
# ----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fiksira sve izvore slučajnosti kako bi eksperimenti bili ponovljivi.

    Fiksiranje seed-a je obavezan zahtev domaćeg zadatka (reproduktivnost).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinističko ponašanje (na CPU-u ovo je dovoljno za ponovljivost).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def library_versions() -> dict:
    """Vraća verzije ključnih biblioteka radi evidencije (upisuje se u MLflow)."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ["numpy", "pandas", "sklearn", "torch", "mlflow"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_installed"
    return versions


# ----------------------------------------------------------------------------
# 2. Merenje vremena
# ----------------------------------------------------------------------------
@contextmanager
def timer():
    """Kontekst menadžer za merenje proteklog vremena u sekundama.

    Upotreba:
        with timer() as t:
            ...  # posao
        print(t())   # proteklo vreme
    """
    start = time.perf_counter()
    elapsed = {"value": None}
    yield lambda: (time.perf_counter() - start) if elapsed["value"] is None else elapsed["value"]
    elapsed["value"] = time.perf_counter() - start


# ----------------------------------------------------------------------------
# 3. Veličina modela
# ----------------------------------------------------------------------------
def count_parameters(model) -> int:
    """Broj trenirajućih parametara modela."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model) -> float:
    """Veličina modela u megabajtima (na osnovu broja i tipa parametara)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


# ----------------------------------------------------------------------------
# 4. Čuvanje JSON-a
# ----------------------------------------------------------------------------
def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# 5. Grafikoni (koriste se u train.py i u EDA notebook-u)
# ----------------------------------------------------------------------------
def plot_learning_curve(history: dict, title: str, path: str) -> None:
    """Crta krivu učenja: gubitak (loss) po epohi za trening i validaciju."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Trening gubitak")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validacioni gubitak")
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], marker="^", label="Validaciona tačnost")
    plt.xlabel("Epoha")
    plt.ylabel("Vrednost")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(path)


def plot_confusion_matrix(cm, class_names, title: str, path: str) -> None:
    """Crta konfuzionu matricu (2x2 za binarni problem)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predviđena klasa")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title(title)
    # Upisujemo brojeve u ćelije.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save_fig(path)


def plot_roc_pr(y_true, y_score, title: str, roc_path: str, pr_path: str) -> dict:
    """Crta ROC i PR krivu i vraća površine ispod krivih (AUC vrednosti)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # ROC kriva
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Slučajno pogađanje")
    plt.xlabel("Stopa lažno pozitivnih (FPR)")
    plt.ylabel("Stopa stvarno pozitivnih (TPR)")
    plt.title(f"ROC kriva – {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(roc_path)

    # PR kriva
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR (AP = {pr_auc:.3f})")
    plt.xlabel("Odziv (Recall)")
    plt.ylabel("Preciznost (Precision)")
    plt.title(f"PR kriva – {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(pr_path)

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def _save_fig(path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
