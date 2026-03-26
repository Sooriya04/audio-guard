"""
train.py — Training script for the Deepfake Audio Detection System.

Usage
-----
  # Quickstart (reads from config.py defaults)
  python train.py

  # Custom paths
  python train.py --csv data/processed/metadata.csv --output checkpoints/run1

  # Resume from checkpoint
  python train.py --resume checkpoints/run1/checkpoint-500
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup
from tqdm import tqdm

import config as cfg
from dataset import build_dataloaders
from evaluate import compute_metrics
from model import build_model, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────
#  One epoch of training
# ──────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, unit="batch")
    for batch_idx, batch in enumerate(pbar):
        input_values  = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels        = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.FP16 and device.type == "cuda"):
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % cfg.LOGGING_STEPS == 0:
            lr_now = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr",   lr_now,      global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

    avg_loss = total_loss / n_batches
    log.info(f"Epoch {epoch} — avg train loss: {avg_loss:.4f}")
    return global_step


# ──────────────────────────────────────────────────────────
#  Validation pass
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    loader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False, unit="batch"):
        input_values   = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast(enabled=cfg.FP16 and device.type == "cuda"):
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss  += outputs.loss.item()
        all_logits.append(outputs.logits.float().cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits.numpy(), labels.numpy())
    metrics["loss"] = total_loss / len(loader)

    log.info(
        f"Epoch {epoch} [VAL] — "
        f"loss:{metrics['loss']:.4f}  acc:{metrics['accuracy']:.4f}  "
        f"f1:{metrics['f1']:.4f}  roc_auc:{metrics['roc_auc']:.4f}"
    )

    for k, v in metrics.items():
        writer.add_scalar(f"val/{k}", v, global_step)

    return metrics


# ──────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────

def train(
    output_dir: Path = cfg.CHECKPOINTS,
    csv_path: Optional[Path] = None,
    resume_from: Optional[Path] = None,
):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature extractor ─────────────────────────────────
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        cfg.MODEL_NAME,
        sampling_rate=cfg.SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # ── DataLoaders ───────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        feature_extractor=feature_extractor,
        csv_path=csv_path,
    )

    # ── Model ─────────────────────────────────────────────
    if resume_from:
        log.info(f"Resuming from {resume_from}")
        model = load_model(str(resume_from), device)
    else:
        model = build_model().to(device)

    # ── Optimiser + Scheduler ─────────────────────────────
    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": cfg.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer  = AdamW(params, lr=cfg.LEARNING_RATE)
    total_steps = len(train_loader) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler     = GradScaler(enabled=cfg.FP16 and device.type == "cuda")

    writer      = SummaryWriter(log_dir=str(cfg.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")))
    best_metric = -float("inf")
    best_ckpt   = None
    global_step = 0
    history     = []

    log.info(f"Training for {cfg.NUM_EPOCHS} epoch(s) — {total_steps} total steps")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, writer, global_step,
        )

        val_metrics = validate(model, val_loader, device, epoch, writer, global_step)
        history.append({"epoch": epoch, **val_metrics})

        # ── Save checkpoint ─────────────────────────────
        ckpt_dir = output_dir / f"checkpoint-epoch{epoch}"
        model.save_pretrained(str(ckpt_dir))
        feature_extractor.save_pretrained(str(ckpt_dir))
        log.info(f"  Checkpoint saved → {ckpt_dir}")

        # ── Track best ──────────────────────────────────
        score = val_metrics[cfg.METRIC_FOR_BEST]
        if score > best_metric:
            best_metric = score
            best_ckpt   = ckpt_dir
            best_dir    = output_dir / "best_model"
            model.save_pretrained(str(best_dir))
            feature_extractor.save_pretrained(str(best_dir))
            log.info(f"  ★ New best ({cfg.METRIC_FOR_BEST}={best_metric:.4f}) → {best_dir}")

    writer.close()

    # ── Final evaluation on test set ────────────────────
    log.info("\n── Final Test Evaluation (best model) ──────────────")
    best_model = load_model(str(output_dir / "best_model"), device)
    test_metrics = validate(best_model, test_loader, device, epoch="test", writer=SummaryWriter(), global_step=0)

    results = {
        "best_val_metric": best_metric,
        "best_checkpoint": str(best_ckpt),
        "test_metrics":    test_metrics,
        "training_history": history,
    }
    results_path = cfg.RESULTS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    return results


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the deepfake audio detector.")
    parser.add_argument("--csv",     type=Path, default=None,         help="Path to metadata CSV")
    parser.add_argument("--output",  type=Path, default=cfg.CHECKPOINTS, help="Output directory for checkpoints")
    parser.add_argument("--resume",  type=Path, default=None,         help="Resume from checkpoint directory")
    args = parser.parse_args()

    results = train(
        output_dir=args.output,
        csv_path=args.csv,
        resume_from=args.resume,
    )
    print(f"\nTest F1    : {results['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
