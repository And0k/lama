from typing import Optional

import torch
from omegaconf import DictConfig

from training.datamodule import NetCDFDataModule
from models.factory import build_model


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except Exception:
        pass
    return torch.tensor(x)


def run(cfg: DictConfig, smoke: bool = True) -> dict:
    """Run a minimal smoke pipeline using provided Hydra `cfg`.

    - instantiates DataModule, sets it up to compute `in_channels`
    - builds a model via `models.factory.build_model`
    - pulls one batch from dataloader and runs forward (and a tiny backward step if `smoke`)

    Returns a dict with a few diagnostics (shapes, device).
    """
    # Create DataModule and setup to get `in_channels`
    dm = NetCDFDataModule(dataset_cfg=cfg.data.dataset,
                          batch_size=int(cfg.training.get("batch_size", 1)),
                          num_workers=int(cfg.training.get("num_workers", 0)))
    dm.setup()

    in_ch = int(dm.in_channels)

    # determine num_classes (fallback to in_ch)
    num_classes = None
    try:
        # DictConfig supports mapping access
        num_classes = int(cfg.model.get("num_classes", in_ch))
    except Exception:
        try:
            num_classes = int(getattr(cfg.model, "num_classes", in_ch))
        except Exception:
            num_classes = in_ch

    # generator config if provided
    generator_cfg = None
    try:
        generator_cfg = cfg.model.get("generator", None)
    except Exception:
        generator_cfg = getattr(cfg.model, "generator", None)

    model = build_model(in_channels=in_ch, num_classes=num_classes, generator_cfg=generator_cfg)
    model.to("cpu")
    model.eval()

    dl = dm.train_dataloader()
    it = iter(dl)
    try:
        batch = next(it)
    except Exception as e:
        return {"error": f"Failed to get batch from DataLoader: {e}"}

    # expect batch to be a mapping with 'image'
    img = None
    if isinstance(batch, dict):
        img = batch.get("image", None)
    elif isinstance(batch, (list, tuple)):
        # try first element
        first = batch[0]
        if isinstance(first, dict):
            img = first.get("image", None)
        else:
            img = first

    if img is None:
        return {"error": "Could not find 'image' in dataset sample"}

    img = _to_tensor(img).float()
    # ensure batch dimension
    if img.ndim == 3:
        img = img.unsqueeze(0)

    img = img.to("cpu")

    # forward pass
    try:
        with torch.no_grad():
            out = model(img)
    except Exception as e:
        return {"error": f"Forward failed: {e}"}

    diagnostics = {
        "input_shape": tuple(img.shape),
        "output_shape": tuple(out.shape) if hasattr(out, "shape") else None,
        "in_channels": in_ch,
        "num_classes": num_classes,
    }

    # tiny training step if requested
    if smoke:
        try:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            opt.zero_grad()
            loss = out.sum()
            loss.backward()
            opt.step()
            diagnostics["train_step"] = "ok"
        except Exception as e:
            diagnostics["train_step_error"] = str(e)

    return diagnostics
