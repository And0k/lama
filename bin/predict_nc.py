#!/usr/bin/env python3
"""Hydra-powered inference script for NetCDF oceanographic data using LaMa inpainting."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.factory import build_model
from netcdf.data.dataset import NetCDFDataset
from netcdf.visualization import plot_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_inference(cfg: DictConfig):
    """Run inference on NetCDF data using LaMa inpainting."""
    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    dev = torch.device(device)

    logger.info("Building NetCDF dataset ...")
    
    # Get dataset config
    ds_cfg = cfg.dataset
    dataset = NetCDFDataset(
        filepaths=ds_cfg.filepaths,
        variables=ds_cfg.variables,
        scaling=ds_cfg.get("scaling"),
        time_indices=ds_cfg.get("time_indices"),
        depth_indices=ds_cfg.get("depth_indices"),
        lat_indices=ds_cfg.get("lat_indices"),
        lon_indices=ds_cfg.get("lon_indices"),
        slice_mode=ds_cfg.get("slice_mode"),
        mask_generator=ds_cfg.get("mask_generator"),
    )
    
    if len(dataset) == 0:
        logger.error("Dataset is empty – check file paths and indices.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    C, H, W = dataset.get_shape()
    logger.info("Dataset shape per sample: (C=%d, H=%d, W=%d)", C, H, W)

    in_channels = max(C, 1)
    
    # For big-lama checkpoints, we need 4 input channels (image + mask concatenated)
    # The model was trained with input_nc=4 and expects 3 output channels
    is_big_lama = cfg.checkpoint and "big-lama" in cfg.checkpoint
    model_in_channels = 4 if is_big_lama else in_channels
    model_out_channels = 3 if is_big_lama else in_channels
    
    logger.info("Loading LaMa model from %s (in_channels=%d)", cfg.checkpoint, model_in_channels)
    
    # For big-lama checkpoint: we need a model that takes 4 channels (image+mask) and outputs 3
    # For regular checkpoint: model takes C channels and outputs C channels
    if cfg.checkpoint and "big-lama" in cfg.checkpoint:
        # Build model expecting 4 input channels, 3 output channels for big-lama
        model = build_model(in_channels=model_in_channels, out_channels=model_out_channels)
        # Load checkpoint weights
        checkpoint = torch.load(cfg.checkpoint, map_location=dev)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    else:
        model = build_model(in_channels=model_in_channels, out_channels=model_out_channels)
        if cfg.checkpoint:
            checkpoint = torch.load(cfg.checkpoint, map_location=dev)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    model.to(dev)
    model.eval()

    processed = 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(dev)
        masks = batch["mask"].to(dev)
        meta = batch["meta"]

        with torch.no_grad():
            # For big-lama checkpoint: concatenate image with mask (4 channels expected)
            if in_channels == 1 and cfg.checkpoint and "big-lama" in cfg.checkpoint:
                images_in = torch.cat([images, masks, masks, masks], dim=1)
                inpainted = model(images_in)
            elif in_channels == 3 and cfg.checkpoint and "big-lama" in cfg.checkpoint:
                images_in = torch.cat([images, masks], dim=1)
                inpainted = model(images_in)
            else:
                # Try with masks first, fall back to without masks only if the model explicitly doesn't accept masks
                try:
                    inpainted = model(images, masks)
                except TypeError as e:
                    # Only fall back if the error is about unexpected keyword argument 'masks'
                    if "unexpected keyword argument" in str(e) and "masks" in str(e):
                        inpainted = model(images)
                    else:
                        # Re-raise other TypeErrors (e.g., shape mismatch, wrong dtype)
                        raise

        for i in range(images.shape[0]):
            fp = meta["filepath"][i] if isinstance(meta["filepath"], (list, tuple)) else meta["filepath"]
            
            # Convert tensor indices to scalar values for naming
            t_idx = meta["time_index"]
            t_idx = t_idx[i].item() if hasattr(t_idx, 'item') else (t_idx[i] if isinstance(t_idx, (list, tuple)) else t_idx)
            
            d_idx = meta.get("depth_index", -1)
            if d_idx != -1:
                d_idx = d_idx[i].item() if hasattr(d_idx, 'item') else (d_idx[i] if isinstance(d_idx, (list, tuple)) else d_idx)
            
            lat_idx = meta.get("lat_index", -1)
            if lat_idx != -1:
                lat_idx = lat_idx[i].item() if hasattr(lat_idx, 'item') else (lat_idx[i] if isinstance(lat_idx, (list, tuple)) else lat_idx)
            
            lon_idx = meta.get("lon_index", -1)
            if lon_idx != -1:
                lon_idx = lon_idx[i].item() if hasattr(lon_idx, 'item') else (lon_idx[i] if isinstance(lon_idx, (list, tuple)) else lon_idx)
            
            slice_mode = meta.get("slice_mode", "unknown")
            base_name = f"{Path(fp).stem}_t{t_idx}_d{d_idx}_lat{lat_idx}_lon{lon_idx}_{slice_mode}_b{batch_idx}_i{i}"

            orig_np = images[i].detach().cpu().numpy()
            # For big-lama, output is 3 channels; take first channel
            inp_np = inpainted[i].detach().cpu().numpy()
            if is_big_lama and inp_np.shape[0] > 1:
                inp_np = inp_np[:1]  # Take only first channel for single-variable input
            mask_np = masks[i].cpu().numpy()

            # Clamp to valid range
            orig_np = np.clip(orig_np, 0.0, 1.0)
            inp_np = np.clip(inp_np, 0.0, 1.0)

            np.save(out_path / f"{base_name}_orig.npy", orig_np)
            np.save(out_path / f"{base_name}_inpainted.npy", inp_np)
            np.save(out_path / f"{base_name}_mask.npy", mask_np)

            # Create masked version (zero out masked regions)
            masked_np = orig_np.copy()
            mask_2d = mask_np[0] if mask_np.shape[0] == 1 else mask_np[0, 0]
            masked_np[0, mask_2d > 0] = 0.0

            try:
                fig = plot_comparison(
                    original=orig_np,
                    masked=masked_np,
                    result=inp_np,
                    titles=["Original", "Masked", "Inpainted"],
                )
                fig.savefig(out_path / f"{base_name}.png", dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as exc:
                logger.warning("Visualization failed for %s: %s", base_name, exc)

            processed += 1
            if cfg.get("max_samples") and processed >= cfg.max_samples:
                logger.info("Reached max_samples limit (%d). Stopping.", cfg.max_samples)
                return

    logger.info("Inference complete. %d slices processed. Results in %s", processed, cfg.out_dir)


@hydra.main(config_path="../configs/nc/prediction", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    # Initialize empty filepaths if not set
    if not cfg.dataset.filepaths:
        logger.error("No NetCDF filepaths provided. Set nc.data.filepaths in config or override.")
        return
    
    # Resolve checkpoint path - Hydra uses 'no' as placeholder
    if cfg.checkpoint == "no":
        logger.error("No checkpoint provided. Set checkpoint in config or override.")
        return
        
    run_inference(cfg)


if __name__ == "__main__":
    main()