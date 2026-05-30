#!/usr/bin/env python3
"""Error analysis for NetCDF inpainting predictions.

Computes RMSE, Correlation, SSIM, and physics-aware metrics (stability, BBL gradient)
to analyze prediction quality. Saves visualizations of best/worst samples.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(indir: str, variables: list[str]) -> pd.DataFrame:
    """Load prediction files and compute metrics."""
    from netcdf.evaluation import RMSEScore, CorrelationScore, SSIMScore, StabilityViolationScore, BBLGradientScore

    indir = Path(indir)
    records = []

    npy_files = sorted(indir.glob("*_orig.npy"))
    if not npy_files:
        logger.error("No *_orig.npy files found in %s", indir)
        return pd.DataFrame()

    rmse_scorer = RMSEScore()
    corr_scorer = CorrelationScore()
    ssim_scorer = SSIMScore()
    stab_scorer = StabilityViolationScore()
    bbl_scorer = BBLGradientScore()

    for orig_path in npy_files:
        base = orig_path.stem.replace("_orig", "")
        inp_path = indir / f"{base}_inpainted.npy"
        mask_path = indir / f"{base}_mask.npy"

        if not inp_path.exists() or not mask_path.exists():
            continue

        orig = np.load(orig_path)
        inp = np.load(inp_path)
        mask = np.load(mask_path)

        orig_t = torch.from_numpy(orig).unsqueeze(0)
        inp_t = torch.from_numpy(inp).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        rmse_scorer.reset()
        corr_scorer.reset()
        ssim_scorer.reset()
        stab_scorer.reset()
        bbl_scorer.reset()

        rmse_val = rmse_scorer(inp_t, orig_t, mask_t)
        corr_val = corr_scorer(inp_t, orig_t, mask_t)
        ssim_val = ssim_scorer(inp_t, orig_t, mask_t)
        stab_val = stab_scorer(inp_t, orig_t, mask_t)
        bbl_val = bbl_scorer(inp_t, orig_t, mask_t)

        records.append({
            "sample_id": base,
            "rmse": rmse_val.item(),
            "correlation": corr_val.item(),
            "ssim": ssim_val.item(),
            "stability_violation": stab_val.item(),
            "bbl_gradient": bbl_val.item(),
        })

    return pd.DataFrame(records)


def save_visualizations(df: pd.DataFrame, indir: str, top_n: int = 10):
    """Save comparison plots for best/worst samples."""
    import matplotlib.pyplot as plt
    from netcdf.visualization import plot_comparison

    indir = Path(indir)

    worst = df.nsmallest(top_n, "rmse")
    best = df.nlargest(top_n, "rmse")

    for direction, subset in [("worst", worst), ("best", best)]:
        out_dir = indir / f"viz_{direction}"
        out_dir.mkdir(exist_ok=True)

        for _, row in subset.iterrows():
            base = row["sample_id"]
            orig = np.load(indir / f"{base}_orig.npy")
            inp = np.load(indir / f"{base}_inpainted.npy")
            mask = np.load(indir / f"{base}_mask.npy")

            masked = orig.copy()
            m = mask[0] if mask.shape[0] == 1 else mask[0, 0]
            masked[0, m > 0] = 0

            try:
                fig = plot_comparison(
                    original=orig,
                    masked=masked,
                    result=inp,
                    titles=["Original", "Masked", "Inpainted"],
                )
                fig.savefig(out_dir / f"{base}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("Failed to save viz for %s: %s", base, exc)


def main(args):
    df = load_predictions(args.indir, args.variables)

    if df.empty:
        logger.error("No predictions found to analyze.")
        return

    df.to_csv(Path(args.indir) / "analysis_results.csv", index=False)
    logger.info("Saved analysis to %s/analysis_results.csv", args.indir)

    summary = df[["rmse", "correlation", "ssim", "stability_violation", "bbl_gradient"]].describe()
    logger.info("Metrics summary:\n%s", summary)

    save_visualizations(df, args.indir, top_n=args.top_n)
    logger.info("Saved visualizations to %s/viz_best and vizard_worst", args.indir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NetCDF inpainting errors")
    parser.add_argument("indir", type=str, help="Directory containing .npy prediction files")
    parser.add_argument("--variables", nargs="+", default=["thetao", "so", "uo"],
                        help="Variable names used in prediction")
    parser.add_argument("--top-n", type=int, default=10, help="Number of best/worst samples to visualize")

    main(parser.parse_args())