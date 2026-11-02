#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: average pairwise token cosine similarity across GEN stages.

Expected NPZ contains token tensors. Supported token shapes:
  [T, C]
  [B, T, C]
  [N, B, T, C]  -> first dimensions are merged as samples

Example:
python plot_figure4_contranorm_similarity.py \
  --npz vis_dump/sample_tokens.npz \
  --series "Baseline:baseline_before_cross,baseline_after_cross,baseline_after_norm,baseline_after_ffn" \
  --series "Ours:ours_before_cross,ours_after_cross,ours_after_contranorm,ours_after_ffn" \
  --stage-labels "Before cross" "After cross" "After norm" "After FFN" \
  --out figures/figure4_token_cosine_similarity
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=str, help="NPZ file containing token tensors.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Format: 'Label:key1,key2,key3'. Each key is one stage tensor in the NPZ file.",
    )
    parser.add_argument(
        "--stage-labels",
        nargs="+",
        default=None,
        help="Stage names for x axis. Length should equal number of keys in each --series.",
    )
    parser.add_argument("--out", default="figures/figure4_token_cosine_similarity", type=str)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=7.2)
    parser.add_argument("--height", type=float, default=4.4)
    parser.add_argument("--title", default="Average pairwise token cosine similarity", type=str)
    parser.add_argument("--ylabel", default="Average pairwise cosine similarity", type=str)
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save numerical values to <out>.csv.",
    )
    return parser.parse_args()


def parse_series(series_arg: str) -> Tuple[str, List[str]]:
    if ":" not in series_arg:
        raise ValueError(f"Invalid --series '{series_arg}'. Expected 'Label:key1,key2'.")
    label, key_str = series_arg.split(":", 1)
    keys = [k.strip() for k in key_str.split(",") if k.strip()]
    if not keys:
        raise ValueError(f"No keys found in --series '{series_arg}'.")
    return label.strip(), keys


def to_samples_tokens_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:  # [T, C]
        x = x[None, :, :]
    elif x.ndim == 3:  # [B, T, C]
        pass
    elif x.ndim >= 4:
        # Merge all leading dimensions except T and C.
        t, c = x.shape[-2], x.shape[-1]
        x = x.reshape(-1, t, c)
    else:
        raise ValueError(f"Unsupported token tensor shape: {x.shape}")
    return x


def average_pairwise_cosine(x: np.ndarray, eps: float = 1e-8) -> float:
    """Mean off-diagonal cosine similarity over samples."""
    x = to_samples_tokens_channels(x)
    values = []
    for sample in x:
        norm = np.linalg.norm(sample, axis=-1, keepdims=True)
        sample = sample / np.maximum(norm, eps)
        sim = sample @ sample.T
        t = sim.shape[0]
        if t <= 1:
            continue
        off_diag_sum = sim.sum() - np.trace(sim)
        values.append(off_diag_sum / (t * (t - 1)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def split_out_path(out: str) -> Tuple[str, List[str]]:
    root, ext = os.path.splitext(out)
    if ext.lower() in [".png", ".pdf", ".svg"]:
        return root, [ext.lower()[1:]]
    return out, ["png", "pdf"]


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)

    parsed = [parse_series(s) for s in args.series]
    lengths = {len(keys) for _, keys in parsed}
    if len(lengths) != 1:
        raise ValueError("All --series entries must have the same number of keys.")
    n_stages = next(iter(lengths))

    if args.stage_labels is None:
        stage_labels = [f"Stage {i}" for i in range(n_stages)]
    else:
        if len(args.stage_labels) != n_stages:
            raise ValueError("--stage-labels length must match number of keys in each --series.")
        stage_labels = args.stage_labels

    records = []
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    x_pos = np.arange(n_stages)

    for label, keys in parsed:
        scores = []
        for stage, key in zip(stage_labels, keys):
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in {args.npz}. Available keys: {data.files}")
            score = average_pairwise_cosine(data[key])
            scores.append(score)
            records.append({"series": label, "stage": stage, "key": key, "avg_pairwise_cosine": score})
        ax.plot(x_pos, scores, marker="o", markersize=4.8, linewidth=2.0, label=label)

    ax.set_title(args.title, fontsize=13, pad=8)
    ax.set_ylabel(args.ylabel, fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_labels, fontsize=9, rotation=0)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.30)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()

    root, exts = split_out_path(args.out)
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    for ext in exts:
        fig.savefig(f"{root}.{ext}", dpi=args.dpi, bbox_inches="tight")

    if args.save_csv:
        pd.DataFrame(records).to_csv(f"{root}.csv", index=False)

    print("Saved:", ", ".join(f"{root}.{ext}" for ext in exts))
    if args.save_csv:
        print("Saved:", f"{root}.csv")


if __name__ == "__main__":
    main()
