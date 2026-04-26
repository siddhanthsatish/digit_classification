"""
Training script for both CustomCNN and VGG16 models on SVHN.
All hyperparameters are loaded from config.json.

Usage:
    python scripts/train.py --model custom
    python scripts/train.py --model vgg16
    python scripts/train.py --model custom --config my_config.json
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models.custom_cnn import CustomCNN, custom_cnn_kwargs_from_cfg
from src.models.vgg16_finetune import build_vgg16


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        if collect:
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    metrics = (total_loss / total, correct / total)
    if collect:
        return metrics, (np.array(all_labels), np.array(all_preds))
    return metrics


# ── Visualisations ────────────────────────────────────────────────────────────

def save_loss_accuracy_curves(history, model_name, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val",   marker="o", markersize=3)
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss"); ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", marker="o", markersize=3)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val",   marker="o", markersize=3)
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved loss/accuracy curves → {path}")


def save_confusion_matrix(y_true, y_pred, model_name, out_dir, num_classes):
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [str(c) if c < 10 else "bg" for c in present]

    cm = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def save_per_class_accuracy(y_true, y_pred, model_name, out_dir, num_classes):
    classes = list(range(num_classes))
    labels  = [str(c) if c < 10 else "bg" for c in classes]
    accs = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((y_pred[mask] == c).mean() * 100)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, accs, color="steelblue", edgecolor="white")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Class"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} — Per-Class Accuracy (Test Set)")
    ax.axhline(np.mean(accs), color="red", linestyle="--", label=f"Mean {np.mean(accs):.1f}%")
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved per-class accuracy → {path}")


def save_lr_schedule(history, model_name, out_dir):
    if "lr" not in history or not history["lr"]:
        return
    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, history["lr"], marker="o", markersize=3, color="darkorange")
    ax.set_title(f"{model_name} — Learning Rate Schedule")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved LR schedule → {path}")


def save_train_val_gap(history, model_name, out_dir):
    gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    epochs = range(1, len(gap) + 1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, [g * 100 for g in gap], color="purple", marker="o", markersize=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"{model_name} — Generalisation Gap (Train Acc − Val Acc)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (pp)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_generalisation_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved generalisation gap → {path}")


# ── Main training function ────────────────────────────────────────────────────

def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = cfg[args.model]

    # Override batch size based on device
    if device.type == "cuda":
        model_cfg = dict(model_cfg)
        model_cfg["batch_size"] = 1024 if args.model == "custom" else 256
    print(f"Batch size: {model_cfg['batch_size']}")
    num_classes = cfg["num_classes"]
    out_dir = cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    log_path = os.path.join(out_dir, f"{args.model}_train.log")
    log_file = open(log_path, "w")
    old_stdout = sys.stdout

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data)
        def flush(self):
            for s in self.streams: s.flush()

    try:
        sys.stdout = _Tee(old_stdout, log_file)
        print(f"Logging to {log_path}")

        use_extra = device.type == "cuda"
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=cfg["data_dir"],
            batch_size=model_cfg["batch_size"],
            cfg=cfg,
            use_extra=use_extra,
        )

        if args.model == "custom":
            cc_kw = custom_cnn_kwargs_from_cfg(model_cfg)
            model = CustomCNN(num_classes=num_classes, **cc_kw).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=model_cfg["lr"],
                weight_decay=model_cfg["weight_decay"],
            )
        else:
            model = build_vgg16(
                num_classes=num_classes,
                freeze_features=model_cfg["freeze_features"],
            ).to(device)
            optimizer = optim.Adam([
                {"params": model.features.parameters(),
                 "lr": model_cfg["lr"] * model_cfg["backbone_lr_scale"]},
                {"params": model.classifier.parameters(),
                 "lr": model_cfg["lr"]},
            ], weight_decay=model_cfg["weight_decay"])

        # AMP scaler for mixed-precision training on CUDA
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        # torch.compile for fused kernels (PyTorch 2.x)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("  torch.compile enabled")
            except Exception:
                pass

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=model_cfg["scheduler_patience"],
            factor=model_cfg["scheduler_factor"],
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

        for epoch in range(1, model_cfg["epochs"] + 1):
            print(f"\nEpoch {epoch}/{model_cfg['epochs']}")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, collect=False)
            scheduler.step(vl_loss)

            current_lr = optimizer.param_groups[-1]["lr"]
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)
            history["lr"].append(current_lr)

            print(f"  train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | lr={current_lr:.2e}")

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                patience_counter = 0
                ckpt_path = os.path.join(cfg["ckpt_dir"], f"{args.model}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Saved best checkpoint → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= model_cfg["patience"]:
                    print(f"  Early stopping triggered after {epoch} epochs.")
                    break

        config_snapshot_path = os.path.join(out_dir, f"{args.model}_config_used.json")
        with open(config_snapshot_path, "w") as f:
            json.dump({"model": args.model, "num_classes": cfg["num_classes"],
                       "model_cfg": model_cfg, "dataset_cfg": cfg["dataset"]}, f, indent=2)
        print(f"  Saved config snapshot → {config_snapshot_path}")

        ckpt_path = os.path.join(cfg["ckpt_dir"], f"{args.model}_best.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        (test_loss, test_acc), (y_true, y_pred) = evaluate(
            model, test_loader, criterion, device, collect=True
        )
        print(f"\nTest loss={test_loss:.4f}  acc={test_acc:.4f}")

        history["test_loss"] = test_loss
        history["test_acc"]  = test_acc

        with open(os.path.join(out_dir, f"{args.model}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print("\nGenerating report visualisations...")
        save_loss_accuracy_curves(history, args.model, out_dir)
        save_lr_schedule(history, args.model, out_dir)
        save_train_val_gap(history, args.model, out_dir)

        save_confusion_matrix(y_true, y_pred, args.model, out_dir, num_classes)
        save_per_class_accuracy(y_true, y_pred, args.model, out_dir, num_classes)

        digit_names = [str(i) for i in range(10)] + ["bg"]
        present_classes = sorted(set(y_true.tolist()))
        present_names   = [digit_names[c] for c in present_classes]
        report = classification_report(y_true, y_pred, labels=present_classes,
                                       target_names=present_names)
        report_path = os.path.join(out_dir, f"{args.model}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Saved classification report → {report_path}")
        print(report)

    finally:
        sys.stdout = old_stdout
        log_file.close()

    print(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["custom", "vgg16"], default="custom")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    train(args, cfg)
