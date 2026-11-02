#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: refined MSA loss curve over temporal positions.

Example:
python plot_figure2_loss_curve.py \
  --csv baseline.csv "Baseline" \
  --csv gen_assa.csv "GEN-ASSA" \
  --csv ours.csv "GEN-ASSA + ContraNorm" \
  --out figures/figure2_refined_msa_loss
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {"seconds_to_latest", "refined_msa_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs=2,
        action="append",
        metavar=("PATH", "LABEL"),
        required=True,
        help="CSV path and legend label. Use multiple --csv entries for multiple methods.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/figure2_refined_msa_loss",
        help="Output path without extension, or with .png/.pdf extension.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Refined MSA loss across temporal positions",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Seconds to latest frame",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Refined MSA loss",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=7.2)
    parser.add_argument("--height", type=float, default=4.4)
    parser.add_argument(
        "--show_mean_in_legend",
        action="store_true",
        help="Append mean refined_msa_loss to each legend label.",
    )
    return parser.parse_args()


def load_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    df = df.sort_values("seconds_to_latest").reset_index(drop=True)
    return df


def split_out_path(out: str) -> Tuple[str, List[str]]:
    root, ext = os.path.splitext(out)
    if ext.lower() in [".png", ".pdf", ".svg"]:
        return root, [ext.lower()[1:]]
    return out, ["png", "pdf"]


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    for csv_path, label in args.csv:
        df = load_curve(csv_path)
        mean_loss = df["refined_msa_loss"].mean()
        final_label = f"{label} (mean={mean_loss:.4f})" if args.show_mean_in_legend else label
        ax.plot(
            df["seconds_to_latest"],
            df["refined_msa_loss"],
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            label=final_label,
        )

    ax.set_title(args.title, fontsize=13, pad=8)
    ax.set_xlabel(args.xlabel, fontsize=11)
    ax.set_ylabel(args.ylabel, fontsize=11)
    ax.grid(True, linewidth=0.6, alpha=0.30)
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()

    root, exts = split_out_path(args.out)
    for ext in exts:
        fig.savefig(f"{root}.{ext}", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", ", ".join(f"{root}.{ext}" for ext in exts))


if __name__ == "__main__":
    main()
