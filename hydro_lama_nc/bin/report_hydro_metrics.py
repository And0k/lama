#!/usr/bin/env python3
"""Report hydro-metrics from TensorBoard training logs.

Reads TensorBoard event files from a training run directory and produces:
    1. Loss component curves (data, stability, bbl, gradient, smooth, bounds)
    2. Metric progression (SSIM, RMSE, correlation, stability_violation, bbl_gradient)
    3. CSV table of all metrics at each logged step
    4. PNG plots saved to the output directory

Usage:
    python bin/report_hydro_metrics.py <tb_logdir> [--outdir <dir>]

Example:
    python bin/report_hydro_metrics.py /tmp/lama_nc_output/tb_logs/nc_training/run0
"""

import argparse
import glob
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

LOSS_TAGS = [
    "train_gen_l1", "train_gen_l2", "train_gen_smooth", "train_gen_bounds",
    "train_gen_grad", "train_gen_stability", "train_gen_bbl", "train_gen_obs_mse",
    "train_gen_adv", "train_gen_fm",
    "val_gen_l1", "val_gen_l2", "val_gen_smooth", "val_gen_bounds",
    "val_gen_grad", "val_gen_stability", "val_gen_bbl", "val_gen_obs_mse",
]

METRIC_TAGS = [
    "val_ssim_mean", "val_rmse_mean", "val_correlation_mean",
    "val_stability_violation_mean", "val_bbl_gradient_mean",
    "val_domain_rmse_mean",
]

PHYSICS_LOSS_TAGS = [
    "train_gen_stability", "train_gen_bbl", "train_gen_grad",
    "train_gen_smooth", "train_gen_bounds", "train_gen_obs_mse",
]


def read_tb_events(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """Read all scalar events from TensorBoard event files.

    Returns:
        Dict mapping tag → list of (step, value) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.error("tensorboard not installed. Install with: pip install tensorboard")
        sys.exit(1)

    tag_data = {}
    event_files = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))

    if not event_files:
        logger.warning("No event files found in %s", logdir)
        return tag_data

    ea = EventAccumulator(logdir)
    ea.Reload()
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        tag_data[tag] = [(e.step, e.value) for e in events]

    return tag_data


def plot_losses(tag_data: dict, outdir: str):
    """Plot loss component curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all train losses
    for tag in LOSS_TAGS:
        if tag.startswith("train_") and tag in tag_data:
            steps, vals = zip(*tag_data[tag])
            label = tag.replace("train_gen_", "")
            axes[0].plot(steps, vals, label=label, alpha=0.8)
    axes[0].set(title="Train losses", xlabel="step", ylabel="loss")
    axes[0].legend(fontsize=7)
    axes[0].set_yscale("log")

    # Right: physics losses only
    for tag in PHYSICS_LOSS_TAGS:
        if tag in tag_data:
            steps, vals = zip(*tag_data[tag])
            label = tag.replace("train_gen_", "")
            axes[1].plot(steps, vals, label=label, alpha=0.8)
    axes[1].set(title="Physics losses (train)", xlabel="step", ylabel="loss")
    axes[1].legend(fontsize=7)
    axes[1].set_yscale("log")

    fig.tight_layout()
    path = os.path.join(outdir, "hydro_loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_metrics(tag_data: dict, outdir: str):
    """Plot validation metric progression."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metric_groups = {
        "SSIM": ("val_ssim_mean", axes[0, 0]),
        "RMSE": ("val_rmse_mean", axes[0, 1]),
        "Stability violation": ("val_stability_violation_mean", axes[1, 0]),
        "BBL gradient": ("val_bbl_gradient_mean", axes[1, 1]),
    }

    for title, (tag, ax) in metric_groups.items():
        if tag in tag_data:
            steps, vals = zip(*tag_data[tag])
            ax.plot(steps, vals, "o-", markersize=3)
            ax.set(title=title, xlabel="step")
        else:
            ax.set(title=f"{title} (not logged)")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="grey")

    fig.suptitle("Validation metrics", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, "hydro_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def write_csv(tag_data, outdir):
    """Write all logged values to a CSV table."""
    import csv

    all_tags = sorted(set(LOSS_TAGS + METRIC_TAGS) & set(tag_data.keys()))
    if not all_tags:
        logger.info("No matching tags found — skipping CSV")
        return

    # Collect all steps
    all_steps = set()
    for tag in all_tags:
        for step, _ in tag_data[tag]:
            all_steps.add(step)
    all_steps = sorted(all_steps)

    path = os.path.join(outdir, "hydro_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + all_tags)
        for step in all_steps:
            row = [step]
            tag_to_val = dict(tag_data.get(tag, []))
            for tag in all_tags:
                row.append(tag_to_val.get(step, ""))
            writer.writerow(row)
    logger.info("Saved: %s", path)


def print_summary(tag_data: dict):
    """Print a terminal summary of latest metric values."""
    logger.info("=" * 70)
    logger.info("%s", " " * 13 + "Metric" + " " * 21 + "Latest" + " " * 4 + "Best")
    logger.info("-" * 70)

    for tag in METRIC_TAGS:
        if tag not in tag_data:
            continue
        vals = [v for _, v in tag_data[tag]]
        name = tag.replace("val_", "").replace("_mean", "")
        latest = vals[-1]
        if "violation" in tag or "gradient" in tag or "rmse" in tag:
            best = min(vals)
        else:
            best = max(vals)
        logger.info("  %s: %10.6f %10.6f", name, latest, best)

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tb_logdir", help="Path to TensorBoard log directory")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: <tb_logdir>/report)")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.tb_logdir, "report")
    os.makedirs(outdir, exist_ok=True)

    logger.info("Reading TensorBoard events from: %s", args.tb_logdir)
    tag_data = read_tb_events(args.tb_logdir)
    logger.info("Found %d tags", len(tag_data))

    if not tag_data:
        logger.error("No data found. Check the log directory path.")
        sys.exit(1)

    plot_losses(tag_data, outdir)
    plot_metrics(tag_data, outdir)
    write_csv(tag_data, outdir)
    print_summary(tag_data)

    logger.info("All reports saved to: %s", outdir)


if __name__ == "__main__":
    main()