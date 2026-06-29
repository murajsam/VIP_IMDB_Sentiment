"""
train.py
--------
Glavna skripta za treniranje i evaluaciju modela.

Ispunjava zahteve domaćeg zadatka:
  * kros-validacija (StratifiedKFold),
  * optimizacija/pretraga hiperparametara (6 konfiguracija iz experiments.yaml),
  * automatsko logovanje eksperimenata u MLflow (hiperparametri, gubitak po
    epohi, metrike po fold-u),
  * fiksiranje seed-ova i evidentiranje verzija biblioteka,
  * poređenje verzija modela kroz tabele i grafikone (learning curve,
    konfuziona matrica, ROC/PR krive, vreme treniranja i inferencije, veličina
    modela).

Pokretanje:
    python -m src.train                 # koristi configs/experiments.yaml
    python -m src.train --quick         # brzi režim (manji podskup, manje epoha)
    python -m src.train --config putanja.yaml
"""

from __future__ import annotations

import os
import sys
import argparse
import platform

# MLflow 3.x podrazumevano blokira lokalni "file store" (mlruns/). Pošto koristimo
# klasičnu mlruns/ evidenciju (koju otvara `mlflow ui`), eksplicitno je dozvoljavamo.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from . import data as D
from . import utils
from .preprocessing import build_preprocessor
from .model import ModelConfig, build_model

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
MODEL_DIR = os.path.join(ROOT, "models")
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------------
# Trening i evaluacija jednog modela
# ----------------------------------------------------------------------------
def train_one(model, train_loader, val_loader, epochs, lr, weight_decay):
    """Trenira model i vraća istoriju (gubitak i tačnost po epohi)."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for _ in range(epochs):
        # --- trening ---
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        history["train_loss"].append(running / n)

        # --- validacija ---
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Vraća (val_loss, accuracy, y_true, y_pred, y_prob_pozitivan)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_prob = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]  # verovatnoća klase "pozitivan"
        preds = logits.argmax(dim=1)
        all_true.append(yb.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    acc = accuracy_score(y_true, y_pred)
    return total_loss / n, acc, y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Standardni skup metrika za binarnu klasifikaciju."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# ----------------------------------------------------------------------------
# Kros-validacija za jednu konfiguraciju
# ----------------------------------------------------------------------------
def run_experiment(exp_cfg, X, y, data_cfg, vocab_size, class_names):
    """Pokreće k-fold kros-validaciju za jednu konfiguraciju modela."""
    name = exp_cfg["name"]
    print(f"\n=== Eksperiment: {name} ===")
    lr = exp_cfg.get("lr", data_cfg["lr"])
    weight_decay = exp_cfg.get("weight_decay", data_cfg["weight_decay"])

    skf = StratifiedKFold(n_splits=data_cfg["n_folds"], shuffle=True,
                          random_state=data_cfg["seed"])

    fold_metrics = []
    train_times, infer_times = [], []
    first_history = None
    oof_true, oof_prob, oof_pred = [], [], []   # "out-of-fold" predikcije
    n_params, size_mb = None, None

    parent_ctx = mlflow.start_run(run_name=name) if _HAS_MLFLOW else _Dummy()
    with parent_ctx:
        if _HAS_MLFLOW:
            mlflow.log_params({k: v for k, v in exp_cfg.items() if k != "name"})
            mlflow.log_params({"lr": lr, "weight_decay": weight_decay,
                               "n_folds": data_cfg["n_folds"], "epochs": data_cfg["epochs"],
                               "batch_size": data_cfg["batch_size"],
                               "num_words": data_cfg["num_words"], "max_len": data_cfg["max_len"]})

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            utils.set_seed(data_cfg["seed"] + fold)
            tr_ds = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
            va_ds = TensorDataset(torch.from_numpy(X[va_idx]), torch.from_numpy(y[va_idx]))
            tr_loader = DataLoader(tr_ds, batch_size=data_cfg["batch_size"], shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=data_cfg["batch_size"], shuffle=False)

            mcfg = ModelConfig(
                vocab_size=vocab_size, max_len=data_cfg["max_len"],
                d_model=exp_cfg["d_model"], n_heads=exp_cfg["n_heads"],
                n_layers=exp_cfg["n_layers"], dim_ff=exp_cfg["dim_ff"],
                dropout=exp_cfg["dropout"], pos_encoding=exp_cfg["pos_encoding"],
                activation=exp_cfg["activation"],
            )
            model = build_model(mcfg)
            if n_params is None:
                n_params = utils.count_parameters(model)
                size_mb = utils.model_size_mb(model)

            # --- trening (mereno vreme) ---
            with utils.timer() as t_train:
                history = train_one(model, tr_loader, va_loader,
                                    data_cfg["epochs"], lr, weight_decay)
            train_time = t_train()

            # --- evaluacija + merenje vremena inferencije ---
            criterion = nn.CrossEntropyLoss()
            with utils.timer() as t_inf:
                _, _, yt, yp, ypr = evaluate(model, va_loader, criterion)
            infer_ms_per_sample = 1000.0 * t_inf() / len(yt)

            m = compute_metrics(yt, yp, ypr)
            fold_metrics.append(m)
            train_times.append(train_time)
            infer_times.append(infer_ms_per_sample)
            oof_true.append(yt); oof_prob.append(ypr); oof_pred.append(yp)
            if first_history is None:
                first_history = history

            print(f"  fold {fold}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
                  f"roc_auc={m['roc_auc']:.3f} (train {train_time:.1f}s)")

            # MLflow: ugnježdeni run po fold-u (gubitak po epohi + metrike)
            if _HAS_MLFLOW:
                with mlflow.start_run(run_name=f"{name}_fold{fold}", nested=True):
                    mlflow.log_params({"fold": fold})
                    for ep in range(len(history["train_loss"])):
                        mlflow.log_metric("train_loss", history["train_loss"][ep], step=ep)
                        mlflow.log_metric("val_loss", history["val_loss"][ep], step=ep)
                        mlflow.log_metric("val_acc", history["val_acc"][ep], step=ep)
                    for k, v in m.items():
                        mlflow.log_metric(k, v)
                    mlflow.log_metric("train_time_s", train_time)
                    mlflow.log_metric("infer_ms_per_sample", infer_ms_per_sample)

        # --- agregacija preko foldova ---
        agg = {k: (float(np.mean([fm[k] for fm in fold_metrics])),
                   float(np.std([fm[k] for fm in fold_metrics]))) for k in fold_metrics[0]}

        oof_true = np.concatenate(oof_true)
        oof_prob = np.concatenate(oof_prob)
        oof_pred = np.concatenate(oof_pred)

        # Grafikoni
        lc_path = os.path.join(FIG_DIR, f"learning_curve_{name}.png")
        utils.plot_learning_curve(first_history, f"Kriva učenja – {name}", lc_path)

        cm = confusion_matrix(oof_true, oof_pred)
        cm_path = os.path.join(FIG_DIR, f"confusion_matrix_{name}.png")
        utils.plot_confusion_matrix(cm, class_names, f"Konfuziona matrica – {name}", cm_path)

        roc_path = os.path.join(FIG_DIR, f"roc_{name}.png")
        pr_path = os.path.join(FIG_DIR, f"pr_{name}.png")
        auc_info = utils.plot_roc_pr(oof_true, oof_prob, name, roc_path, pr_path)

        summary = {
            "name": name,
            "accuracy_mean": agg["accuracy"][0], "accuracy_std": agg["accuracy"][1],
            "precision_mean": agg["precision"][0], "recall_mean": agg["recall"][0],
            "f1_mean": agg["f1"][0], "f1_std": agg["f1"][1],
            "roc_auc_mean": agg["roc_auc"][0], "pr_auc_mean": agg["pr_auc"][0],
            "train_time_s_mean": float(np.mean(train_times)),
            "infer_ms_per_sample_mean": float(np.mean(infer_times)),
            "n_params": int(n_params), "size_mb": float(size_mb),
        }

        if _HAS_MLFLOW:
            mlflow.log_metrics({
                "accuracy_mean": summary["accuracy_mean"], "f1_mean": summary["f1_mean"],
                "roc_auc_mean": summary["roc_auc_mean"], "pr_auc_mean": summary["pr_auc_mean"],
                "train_time_s_mean": summary["train_time_s_mean"],
                "infer_ms_per_sample_mean": summary["infer_ms_per_sample_mean"],
                "n_params": summary["n_params"], "size_mb": summary["size_mb"],
            })
            for p in [lc_path, cm_path, roc_path, pr_path]:
                mlflow.log_artifact(p, artifact_path="figures")

    return summary


class _Dummy:
    """Zamena za MLflow kontekst ako MLflow nije instaliran."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------------
# Poređenje konfiguracija + finalni model
# ----------------------------------------------------------------------------
def save_comparison(summaries):
    import pandas as pd
    df = pd.DataFrame(summaries).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    csv_path = os.path.join(TAB_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False)

    import matplotlib.pyplot as plt
    # Poređenje F1 i ROC-AUC po konfiguraciji
    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = np.arange(len(df))
    ax.bar(xpos - 0.2, df["f1_mean"], width=0.4, label="F1 (prosek)")
    ax.bar(xpos + 0.2, df["roc_auc_mean"], width=0.4, label="ROC-AUC (prosek)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(df["name"], rotation=30, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Vrednost metrike")
    ax.set_title("Poređenje konfiguracija modela")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    utils._save_fig(os.path.join(FIG_DIR, "comparison_metrics.png"))

    # Poređenje vreme treninga vs veličina modela
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xpos, df["train_time_s_mean"], color="#c47f17")
    ax.set_xticks(xpos)
    ax.set_xticklabels(df["name"], rotation=30, ha="right")
    ax.set_ylabel("Prosečno vreme treninga po fold-u [s]")
    ax.set_title("Vreme treniranja po konfiguraciji")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    utils._save_fig(os.path.join(FIG_DIR, "comparison_time.png"))

    print("\n[train] Tabela poređenja sačuvana u:", csv_path)
    print(df.to_string(index=False))
    return df


def train_final_model(best_cfg, X, y, X_test, y_test, data_cfg, vocab_size, class_names):
    """Trenira finalni model (najbolja konfiguracija) na celom podskupu i
    evaluira ga na zvaničnom IMDB test skupu; čuva model za Streamlit."""
    print(f"\n=== Finalni model: {best_cfg['name']} (evaluacija na test skupu) ===")
    utils.set_seed(data_cfg["seed"])
    lr = best_cfg.get("lr", data_cfg["lr"])
    weight_decay = best_cfg.get("weight_decay", data_cfg["weight_decay"])

    tr_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    tr_loader = DataLoader(tr_ds, batch_size=data_cfg["batch_size"], shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=data_cfg["batch_size"], shuffle=False)

    mcfg = ModelConfig(
        vocab_size=vocab_size, max_len=data_cfg["max_len"],
        d_model=best_cfg["d_model"], n_heads=best_cfg["n_heads"],
        n_layers=best_cfg["n_layers"], dim_ff=best_cfg["dim_ff"],
        dropout=best_cfg["dropout"], pos_encoding=best_cfg["pos_encoding"],
        activation=best_cfg["activation"],
    )
    model = build_model(mcfg)
    train_one(model, tr_loader, te_loader, data_cfg["epochs"], lr, weight_decay)

    criterion = nn.CrossEntropyLoss()
    _, _, yt, yp, ypr = evaluate(model, te_loader, criterion)
    test_metrics = compute_metrics(yt, yp, ypr)
    print("  Test metrike:", {k: round(v, 3) for k, v in test_metrics.items()})

    cm = confusion_matrix(yt, yp)
    utils.plot_confusion_matrix(cm, class_names, "Konfuziona matrica – TEST skup",
                                os.path.join(FIG_DIR, "confusion_matrix_TEST.png"))
    utils.plot_roc_pr(yt, ypr, "TEST skup",
                      os.path.join(FIG_DIR, "roc_TEST.png"),
                      os.path.join(FIG_DIR, "pr_TEST.png"))

    # Čuvanje modela + konfiguracije (za seminarski / Streamlit)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
    utils.save_json({"model": mcfg.__dict__, "test_metrics": test_metrics,
                     "num_words": data_cfg["num_words"], "max_len": data_cfg["max_len"]},
                    os.path.join(MODEL_DIR, "best_model_meta.json"))
    print("  Model sačuvan u:", os.path.join(MODEL_DIR, "best_model.pt"))
    return test_metrics


# ----------------------------------------------------------------------------
# Glavna funkcija
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "experiments.yaml"))
    ap.add_argument("--quick", action="store_true", help="Brzi režim (manji podskup/epohe)")
    ap.add_argument("--experiment_name", default="IMDB_Sentiment_Transformer")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    experiments = cfg["experiments"]

    if args.quick:
        data_cfg.update({"subset_size": 3000, "n_folds": 3, "epochs": 3})
        print("[train] BRZI REŽIM: subset=3000, folds=3, epochs=3")

    utils.set_seed(data_cfg["seed"])
    versions = utils.library_versions()
    utils.save_json(versions, os.path.join(ROOT, "results", "versions.json"))
    print("[train] Verzije biblioteka:", versions)

    # Priprema podataka (reproduktivan pipeline)
    pre = build_preprocessor(data_cfg["num_words"], data_cfg["max_len"])
    pre.save(os.path.join(MODEL_DIR, "preprocessor.json"))
    dcfg = D.DataConfig(num_words=data_cfg["num_words"], max_len=data_cfg["max_len"],
                        subset_size=data_cfg["subset_size"], seed=data_cfg["seed"])
    x_tr, y_tr, x_te, y_te = D.get_train_test(dcfg)
    X = pre.transform_ranks(x_tr)
    y = np.asarray(y_tr, dtype=np.int64)
    X_test = pre.transform_ranks(x_te)
    y_test = np.asarray(y_te, dtype=np.int64)
    print(f"[train] Trening primeri: {len(X)} | Test primeri: {len(X_test)} | "
          f"vokabular: {pre.vocab_size} | max_len: {data_cfg['max_len']}")

    if _HAS_MLFLOW:
        import pathlib
        mlruns_dir = pathlib.Path(ROOT) / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())  # ispravan file:/// URI na Windows-u
        mlflow.set_experiment(args.experiment_name)

    summaries = []
    for exp in experiments:
        summaries.append(run_experiment(exp, X, y, data_cfg, pre.vocab_size, D.CLASS_NAMES))

    df = save_comparison(summaries)

    # Finalni model = konfiguracija sa najboljim prosečnim F1
    best_name = df.iloc[0]["name"]
    best_cfg = next(e for e in experiments if e["name"] == best_name)
    train_final_model(best_cfg, X, y, X_test, y_test, data_cfg, pre.vocab_size, D.CLASS_NAMES)

    print("\n[train] GOTOVO. Rezultati: results/  |  MLflow: mlruns/  |  Model: models/")


if __name__ == "__main__":
    main()
