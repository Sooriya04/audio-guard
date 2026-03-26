"""
evaluate.py — Evaluation utilities and standalone evaluation script.

Usage
-----
  python evaluate.py --model checkpoints/best_model --csv data/processed/metadata.csv
  python evaluate.py --model checkpoints/best_model --test-dir data/processed/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

import config as cfg
from dataset import DeepfakeAudioDataset, load_records_from_csv, load_records_from_dirs
from model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Core metrics
# ──────────────────────────────────────────────────────────

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Given raw logits (N, 2) and integer labels (N,), return a dict of metrics.
    """
    probs   = _softmax(logits)
    preds   = probs.argmax(axis=1)
    fake_prob = probs[:, 1]          # probability of class "fake"

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels, fake_prob)),
    }
    return metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────
#  Full evaluation pipeline
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    tag: str = "eval",
) -> Dict:
    model.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Evaluating [{tag}]", unit="batch"):
        input_values   = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        outputs = model(input_values=input_values, attention_mask=attention_mask)
        all_logits.append(outputs.logits.float().cpu().numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs  = _softmax(logits)
    preds  = probs.argmax(axis=1)

    metrics = compute_metrics(logits, labels)

    # Detailed classification report
    report = classification_report(
        labels, preds,
        target_names=list(cfg.LABEL_MAP.values()),
        output_dict=True,
        zero_division=0,
    )

    # False-positive / false-negative analysis
    fp = int(((preds == 1) & (labels == 0)).sum())   # real → predicted fake
    fn = int(((preds == 0) & (labels == 1)).sum())   # fake → predicted real
    tn = int(((preds == 0) & (labels == 0)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())

    return {
        "metrics":    metrics,
        "report":     report,
        "confusion":  {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "logits":     logits,
        "probs":      probs,
        "preds":      preds,
        "labels":     labels,
    }


# ──────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, out_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {out_path}")


def plot_roc_curve(labels, probs, out_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    auc          = roc_auc_score(labels, probs[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved → {out_path}")


def plot_precision_recall(labels, probs, out_path: Path):
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="darkorange")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"P-R curve saved → {out_path}")


def plot_confidence_hist(probs, labels, out_path: Path):
    """Histogram of fake-class probability, separated by true label."""
    fake_prob = probs[:, 1]
    fig, ax   = plt.subplots(figsize=(7, 4))
    ax.hist(fake_prob[labels == 0], bins=40, alpha=0.6, label="Real",  color="steelblue")
    ax.hist(fake_prob[labels == 1], bins=40, alpha=0.6, label="Fake",  color="salmon")
    ax.axvline(cfg.CONFIDENCE_THRESHOLD, color="black", linestyle="--", label=f"Threshold={cfg.CONFIDENCE_THRESHOLD}")
    ax.set_xlabel("P(fake)", fontsize=12)
    ax.set_ylabel("Count",   fontsize=12)
    ax.set_title("Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Confidence histogram saved → {out_path}")


# ──────────────────────────────────────────────────────────
#  Full report
# ──────────────────────────────────────────────────────────

def generate_report(
    model,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path = cfg.RESULTS_DIR,
    tag: str = "test",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    result  = evaluate_model(model, loader, device, tag=tag)
    metrics = result["metrics"]
    labels  = result["labels"]
    probs   = result["probs"]
    preds   = result["preds"]

    print("\n── Evaluation Report ────────────────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={result['confusion']['tp']}  TN={result['confusion']['tn']}")
    print(f"    FP={result['confusion']['fp']}  FN={result['confusion']['fn']}")
    print("─────────────────────────────────────────────────────\n")

    # Save JSON
    report_path = out_dir / f"{tag}_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "metrics":   metrics,
                "confusion": result["confusion"],
                "report":    result["report"],
            },
            f, indent=2,
        )
    log.info(f"JSON report → {report_path}")

    # Plots
    plot_confusion_matrix(labels, preds,  out_dir / f"{tag}_confusion_matrix.png")
    plot_roc_curve(labels, probs,          out_dir / f"{tag}_roc_curve.png")
    plot_precision_recall(labels, probs,   out_dir / f"{tag}_precision_recall.png")
    plot_confidence_hist(probs, labels,    out_dir / f"{tag}_confidence_hist.png")

    return metrics


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved deepfake audio detector.")
    parser.add_argument("--model",    type=Path, required=True, help="Path to saved model directory")
    parser.add_argument("--csv",      type=Path, default=None,  help="Metadata CSV path")
    parser.add_argument("--test-dir", type=Path, default=None,  help="Processed audio root dir (has real/ and fake/)")
    parser.add_argument("--out-dir",  type=Path, default=cfg.RESULTS_DIR, help="Results output directory")
    parser.add_argument("--tag",      type=str,  default="test", help="Tag for output files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(str(args.model), device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model))

    if args.csv and args.csv.exists():
        records = load_records_from_csv(args.csv)
    elif args.test_dir:
        records = load_records_from_dirs(
            args.test_dir / "real",
            args.test_dir / "fake",
        )
    else:
        records = load_records_from_dirs(cfg.REAL_PROC, cfg.FAKE_PROC)

    dataset = DeepfakeAudioDataset(records, feature_extractor, augment=False)
    loader  = DataLoader(dataset, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False, num_workers=4)

    generate_report(model, loader, device, out_dir=args.out_dir, tag=args.tag)
