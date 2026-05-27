"""Visualization utilities for NetCDF oceanographic data."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import logging
import torch

logger = logging.getLogger(__name__)


def plot_slice(
    data: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a 2D slice of data.

    Args:
        data: 2D numpy array to plot.
        title: Title for the plot.
        cmap: Colormap to use.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        ax: Matplotlib axes to plot on. If None, create new figure.

    Returns:
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def plot_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    result: np.ndarray,
    titles: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot original, masked, and inpainted results side by side.

    Args:
        original: Original image of shape (H, W) or (C, H, W).
        masked: Image with mask applied (missing values) of same shape as original.
        result: Inpainted result of same shape as original.
        titles: List of three titles for the subplots.
        cmap: Colormap to use.
        figsize: Figure size.

    Returns:
        The matplotlib figure object.
    """
    if titles is None:
        titles = ["Original", "Masked", "Result"]

    # Handle multi-channel images by taking first channel or averaging
    def prepare_for_display(img):
        if img.ndim == 3:
            if img.shape[0] == 1:
                return img[0]
            elif img.shape[0] == 3:
                # RGB-like, transpose for display
                return np.transpose(img, (1, 2, 0))
            else:
                # Take first channel
                return img[0]
        return img

    orig_disp = prepare_for_display(original)
    masked_disp = prepare_for_display(masked)
    result_disp = prepare_for_display(result)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    plot_slice(orig_disp, title=titles[0], cmap=cmap, ax=axes[0])
    plot_slice(masked_disp, title=titles[1], cmap=cmap, ax=axes[1])
    plot_slice(result_disp, title=titles[2], cmap=cmap, ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_mask(
    mask: np.ndarray,
    title: str = "Mask",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a binary mask.

    Args:
        mask: Binary mask of shape (H, W) or (1, H, W).
        title: Title for the plot.
        ax: Matplotlib axes to plot on.

    Returns:
        The matplotlib axes object.
    """
    if mask.ndim == 3:
        mask = mask[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a colormap that shows 0 as black and 1 as white
    im = ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, ticks=[0, 1])
    return ax


def create_gif(
    frames: List[np.ndarray],
    title: str = "",
    cmap: str = "viridis",
    duration: int = 200,
    loop: int = 0,
) -> List[np.ndarray]:
    """
    Create a list of frames suitable for making a GIF (returns list of RGB arrays).

    Args:
        frames: List of 2D arrays to include in the GIF.
        title: Title to add to each frame.
        cmap: Colormap to use.
        duration: Duration of each frame in milliseconds.
        loop: Number of times to loop (0 means infinite).

    Returns:
        List of RGB arrays suitable for saving as GIF with imageio.
    """
    # This function prepares frames; actual GIF saving would need imageio
    # For now, we'll just return the frames normalized for display
    processed_frames = []
    for frame in frames:
        # Normalize to [0, 1] for display
        if frame.ndim == 3:
            if frame.shape[0] == 1:
                disp_frame = frame[0]
            else:
                disp_frame = np.transpose(frame, (1, 2, 0))
        else:
            disp_frame = frame
        
        # Normalize
        if np.nanmax(disp_frame) > np.nanmin(disp_frame):
            disp_frame = (disp_frame - np.nanmin(disp_frame)) / (np.nanmax(disp_frame) - np.nanmin(disp_frame))
        else:
            disp_frame = np.zeros_like(disp_frame)
        
        # Convert to RGB using colormap
        # Simple approach: duplicate grayscale to RGB
        if disp_frame.ndim == 2:
            rgb_frame = np.stack([disp_frame, disp_frame, disp_frame], axis=-1)
        else:
            rgb_frame = disp_frame
        
        processed_frames.append(rgb_frame)
    
    return processed_frames


def visualize_sample(
    sample: Dict[str, Union[np.ndarray, torch.Tensor, Dict]],
    idx: int = 0,
    channel: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a sample from the dataset.

    Args:
        sample: Dictionary from NetCDFDataset with keys 'image', 'mask', 'meta'.
        idx: Batch index if sample is a batch.
        channel: Channel index to visualize for multi-channel images.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib figure object.
    """
    # Extract data
    image = sample['image']
    mask = sample['mask']
    meta = sample['meta']
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # Handle batch dimension
    if image.ndim == 4:
        image = image[idx]
    if mask.ndim == 4:
        mask = mask[idx]
    
    # Select channel if needed
    if image.ndim == 3:
        image_chan = image[channel]
    else:
        image_chan = image
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original
    plot_slice(image_chan, title=f"Original (Channel {channel})", ax=axes[0])
    
    # Plot mask
    if mask.ndim == 3:
        mask_chan = mask[0]  # Assume single channel mask
    else:
        mask_chan = mask
    plot_mask(mask_chan, title="Mask", ax=axes[1])
    
    # Plot masked image (original where mask==0, gray where mask==1)
    masked_image = image_chan.copy()
    masked_image[mask_chan > 0] = np.nanmean(image_chan)  # Fill masked area with mean
    plot_slice(masked_image, title="Masked Image", ax=axes[2])
    
    # Add metadata to figure title
    time_idx = meta.get('time_index', 'N/A')
    file_path = meta.get('filepath', 'Unknown')
    fig.suptitle(f"Time index: {time_idx}\nFile: {file_path}", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    return fig