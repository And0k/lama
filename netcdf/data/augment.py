"""Oceanography-safe data augmentations for NetCDF datasets."""

import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional, Callable
import albumentations as A
from omegaconf import OmegaConf


class NetCDSBaseTransform:
    """Base class for NetCDF-specific transforms that work on sample dicts."""
    
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Apply transform to sample.
        
        Args:
            sample: Dict with keys 'image' (Tensor), 'mask' (Tensor), 'meta' (Dict)
            
        Returns:
            Transformed sample dict
        """
        raise NotImplementedError


class RandomFlipLatLon(NetCDSBaseTransform):
    """Randomly flip latitude and/or longitude dimensions.
    
    Note: For scalar fields (temperature, salinity), flipping is safe.
    For vector fields (uo, vo), latitude flip requires sign change of vo,
    and longitude flip requires sign change of uo.
    """
    
    def __init__(self, flip_lat: bool = True, flip_lon: bool = True, p: float = 0.5):
        self.flip_lat = flip_lat
        self.flip_lon = flip_lon
        self.p = p
        
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        if np.random.rand() > self.p:
            return sample
            
        image = sample["image"]
        mask = sample["mask"]
        meta = sample["meta"].copy()
        
        # Get variable names for sign adjustment
        variables = meta.get("variables", [])
        
        # Flip latitude (dimension -2 for image tensor CxHxW)
        if self.flip_lat and image.ndim >= 3:
            image = torch.flip(image, [-2])  # Flip H dimension
            mask = torch.flip(mask, [-2])
            
            # Adjust sign for meridional velocity (vo) if present
            if "vo" in variables:
                vo_idx = variables.index("vo")
                image[vo_idx] = -image[vo_idx]
                
        # Flip longitude (dimension -1 for image tensor CxHxW)
        if self.flip_lon and image.ndim >= 3:
            image = torch.flip(image, [-1])  # Flip W dimension
            mask = torch.flip(mask, [-1])
            
            # Adjust sign for zonal velocity (uo) if present
            if "uo" in variables:
                uo_idx = variables.index("uo")
                image[uo_idx] = -image[uo_idx]
                
        sample["image"] = image
        sample["mask"] = mask
        sample["meta"] = meta
        
        return sample


class RandomCropSpatial(NetCDSBaseTransform):
    """Randomly crop spatial dimensions (latitude/longitude)."""
    
    def __init__(self, crop_size: Union[int, Tuple[int, int]], p: float = 0.5):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.p = p
        
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        if np.random.rand() > self.p:
            return sample
            
        image = sample["image"]
        mask = sample["mask"]
        meta = sample["meta"].copy()
        
        _, H, W = image.shape
        crop_h, crop_w = self.crop_size
        
        if crop_h >= H or crop_w >= W:
            return sample  # Can't crop larger than image
            
        # Random crop bounds
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        
        # Crop image and mask
        image = image[:, top:top + crop_h, left:left + crop_w]
        mask = mask[:, top:top + crop_h, left:left + crop_w]
        
        # Update metadata if needed (store crop info)
        meta["crop_info"] = {
            "top": top,
            "left": left,
            "height": crop_h,
            "width": crop_w,
            "original_height": H,
            "original_width": W
        }
        
        sample["image"] = image
        sample["mask"] = mask
        sample["meta"] = meta
        
        return sample


class DepthDropout(NetCDSBaseTransform):
    """Randomly zero out depth levels (for profile data or when depth is a spatial dim)."""
    
    def __init__(self, dropout_frac: float = 0.1, p: float = 0.5):
        self.dropout_frac = dropout_frac
        self.p = p
        
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        if np.random.rand() > self.p:
            return sample
            
        image = sample["image"]
        mask = sample["mask"]
        
        # For 2D field data (CxHxW), apply dropout to one spatial dimension
        # For profile data (CxD), apply dropout to depth dimension
        if image.ndim == 3:  # CxHxW - treat H as potential depth dimension
            _, H, W = image.shape
            # Randomly choose whether to apply to H or W dimension
            if np.random.rand() > 0.5 and H > 1:  # Apply to first spatial dim (H)
                dropout_size = max(1, int(H * self.dropout_frac))
                if dropout_size < H:
                    start = np.random.randint(0, H - dropout_size + 1)
                    image[:, start:start + dropout_size, :] = 0
                    mask[:, start:start + dropout_size, :] = 0
            elif W > 1:  # Apply to second spatial dim (W)
                dropout_size = max(1, int(W * self.dropout_frac))
                if dropout_size < W:
                    start = np.random.randint(0, W - dropout_size + 1)
                    image[:, :, start:start + dropout_size] = 0
                    mask[:, :, start:start + dropout_size] = 0
        elif image.ndim == 2:  # CxD - profile data
            C, D = image.shape
            if D > 1:
                dropout_size = max(1, int(D * self.dropout_frac))
                if dropout_size < D:
                    start = np.random.randint(0, D - dropout_size + 1)
                    image[:, start:start + dropout_size] = 0
                    mask[:, start:start + dropout_size] = 0
                    
        sample["image"] = image
        sample["mask"] = mask
        
        return sample


class GaussianNoise(NetCDSBaseTransform):
    """Add Gaussian noise to simulate measurement uncertainty."""
    
    def __init__(self, sigma: float = 0.01, p: float = 0.5):
        self.sigma = sigma
        self.p = p
        
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        if np.random.rand() > self.p:
            return sample
            
        image = sample["image"]
        noise = torch.randn_like(image) * self.sigma
        sample["image"] = image + noise
        
        return sample


class ToFixedSize(NetCDSBaseTransform):
    """Pad or resize to fixed size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 mode: str = "pad",  # "pad" or "interpolate"
                 p: float = 1.0):  # Always apply by default
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.mode = mode
        self.p = p
        
    def __call__(self, sample: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]:
        if np.random.rand() > self.p:
            return sample
            
        image = sample["image"]
        mask = sample["mask"]
        _, H, W = image.shape
        target_h, target_w = self.size
        
        if self.mode == "pad":
            # Compute padding
            pad_h = max(0, target_h - H)
            pad_w = max(0, target_w - W)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Apply padding
            if pad_h > 0 or pad_w > 0:
                image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom))
                mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom))
                
        elif self.mode == "interpolate":
            # Interpolate/resize
            if image.shape[-2:] != (target_h, target_w):
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(target_h, target_w), 
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), size=(target_h, target_w), 
                    mode="nearest"
                ).squeeze(0)
                
        sample["image"] = image
        sample["mask"] = mask
        
        return sample


def get_netcdf_transforms(
    transform_variant: str = "none",
    out_size: Optional[Union[int, Tuple[int, int]]] = None,
    flip_lat: bool = True,
    flip_lon: bool = False,
    noise_sigma: float = 0.01,
    depth_dropout: float = 0.1,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    crop_p: float = 0.5,
) -> Callable[[Dict], Dict]:
    """Factory function for NetCDF-specific transforms.
    
    Args:
        transform_variant: "none", "light", or "full"
        out_size: Target size for fixed output (None for no resizing)
        flip_lat: Whether to apply random latitude flip
        flip_lon: Whether to apply random longitude flip
        noise_sigma: Standard deviation for Gaussian noise
        depth_dropout: Fraction of depth to randomly zero out
        crop_size: Size for random spatial cropping
        crop_p: Probability of applying crop
        
    Returns:
        Compose callable that transforms sample dicts
    """
    transforms = []
    
    if transform_variant == "none":
        pass  # No transforms
        
    elif transform_variant == "light":
        if flip_lat or flip_lon:
            transforms.append(RandomFlipLatLon(flip_lat=flip_lat, flip_lon=flip_lon, p=0.5))
        if noise_sigma > 0:
            transforms.append(GaussianNoise(sigma=noise_sigma, p=0.5))
        if depth_dropout > 0:
            transforms.append(DepthDropout(dropout_frac=depth_dropout, p=0.5))
        if crop_size is not None:
            transforms.append(RandomCropSpatial(crop_size=crop_size, p=crop_p))
            
    elif transform_variant == "full":
        if flip_lat or flip_lon:
            transforms.append(RandomFlipLatLon(flip_lat=flip_lat, flip_lon=flip_lon, p=0.5))
        if noise_sigma > 0:
            transforms.append(GaussianNoise(sigma=noise_sigma, p=0.5))
        if depth_dropout > 0:
            transforms.append(DepthDropout(dropout_frac=depth_dropout, p=0.5))
        if crop_size is not None:
            transforms.append(RandomCropSpatial(crop_size=crop_size, p=crop_p))
        # Add more aggressive augmentations for full variant
        transforms.append(GaussianNoise(sigma=noise_sigma*2, p=0.3))  # Stronger noise
        
    else:
        raise ValueError(f"Unknown transform_variant: {transform_variant}")
        
    # Add fixed size transformation if requested
    if out_size is not None:
        transforms.append(ToFixedSize(size=out_size, mode="pad", p=1.0))
        
    if len(transforms) == 0:
        # Return identity transform
        def identity(sample):
            return sample
        return identity
        
    def compose_transforms(sample):
        for transform in transforms:
            sample = transform(sample)
        return sample
        
    return compose_transforms