r"""
Train graph model B for stitch-point edge prediction.

Usage:
    python -m sew_point_conntect.train --image_dir D:\project\changrui\CAB-F\sew_point_connect\images --annotation_dir D:\project\changrui\CAB-F\sew_point_connect\annotations
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import GraphDataset, build_graph_samples, collate_graphs, split_samples, summarize_samples
from .model import EdgeGraphNet


ROOT = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    tp = float(((pred == 1) & (labels == 1)).sum().item())
    fp = float(((pred == 1) & (labels == 0)).sum().item())
    fn = float(((pred == 0) & (labels == 1)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = float((pred == labels).float().mean().item())
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_epoch(model, dataloader, device, criterion, optimizer=None, threshold: float = 0.5, phase_name: str = ""):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_edges = 0
    all_logits = []
    all_labels = []

    for batch in dataloader:
        batch_logits = []
        batch_labels = []
        for sample in batch:
            node_x = sample.node_x.to(device)
            edge_index = sample.edge_index.to(device)
            edge_attr = sample.edge_attr.to(device)
            edge_patch = sample.edge_patch.to(device)
            edge_y = sample.edge_y.to(device)
            logits = model(node_x, edge_index, edge_attr, edge_patch)
            batch_logits.append(logits)
            batch_labels.append(edge_y)

        logits = torch.cat(batch_logits, dim=0)
        labels = torch.cat(batch_labels, dim=0)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_edges = int(labels.numel())
        total_loss += float(loss.item()) * batch_edges
        total_edges += batch_edges
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_edges == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, threshold=threshold)
    metrics["loss"] = total_loss / total_edges
    return metrics


def train(args):
    set_seed(args.seed)

    samples = build_graph_samples(
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
        k_neighbors=args.k_neighbors,
        radius_multiplier=args.radius_multiplier,
        default_spacing=args.default_spacing,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
    )
    if not samples:
        raise RuntimeError(f"在 {args.annotation_dir} 下没有找到可训练的图样本。")

    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    train_stats = summarize_samples(train_samples)
    val_stats = summarize_samples(val_samples)

    print(f"Image dir: {args.image_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(
        f"Train samples={train_stats['num_samples']} avg_nodes={train_stats['avg_nodes']:.2f} "
        f"avg_edges={train_stats['avg_edges']:.2f} pos_ratio={train_stats['pos_ratio']:.4f}"
    )
    print(
        f"Val   samples={val_stats['num_samples']} avg_nodes={val_stats['avg_nodes']:.2f} "
        f"avg_edges={val_stats['avg_edges']:.2f} pos_ratio={val_stats['pos_ratio']:.4f}"
    )

    train_ds = GraphDataset(train_samples)
    val_ds = GraphDataset(val_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_samples[0].node_dim
    edge_dim = train_samples[0].edge_dim
    model = EdgeGraphNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count / 1e6:.3f}M")

    pos_edges = max(train_stats["pos_edges"], 1)
    neg_edges = max(train_stats["neg_edges"], 1)
    pos_weight = torch.tensor([neg_edges / pos_edges], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx: int):
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(args.warmup_epochs, 1)
        progress = (epoch_idx - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_dl,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            threshold=args.threshold,
            phase_name=f"Epoch {epoch:03d}/{args.epochs} [Train]",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_dl,
            device=device,
            criterion=criterion,
            optimizer=None,
            threshold=args.threshold,
            phase_name=f"Epoch {epoch:03d}/{args.epochs} [Val]",
        )
        scheduler.step()

        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "args": vars(args),
                    "best_f1": best_f1,
                    "threshold": args.threshold,
                },
                os.path.join(args.save_dir, "best.pth"),
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.5f} val_f1={val_metrics['f1']:.4f} "
            f"val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
            f"{' *best*' if is_best else ''}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "args": vars(args),
            "best_f1": best_f1,
            "threshold": args.threshold,
        },
        os.path.join(args.save_dir, "last.pth"),
    )
    print(f"\nDone. Best val F1: {best_f1:.4f}")
    print(f"Checkpoints saved to {args.save_dir}")


def build_argparser():

    parser = argparse.ArgumentParser(description="Train stitch-point edge GNN.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of source images.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory of edge-labeled JSON files.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--radius_multiplier", type=float, default=2.5)
    parser.add_argument("--default_spacing", type=float, default=28.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch_width", type=int, default=96)
    parser.add_argument("--patch_height", type=int, default=24)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
