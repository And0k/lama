import numpy as np


def vertical_random_lines_mask(width: int, height: int, n_lines: int = 20, seed: int | None = None) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.uint8)
    if seed is not None:
        np.random.seed(seed)
    xs = np.random.choice(width, min(n_lines, width), replace=False)
    for x in xs:
        mask[:, x] = 0
    return mask * 255


def add_model_points(mask: np.ndarray, x_num: int = 20, y_num: int = 30, y_law: str = "log") -> np.ndarray:
    h, w = mask.shape
    xs = np.linspace(0, w - 1, x_num, dtype=int)
    if y_law == "log":
        ys_raw = np.logspace(0, np.log10(h), y_num, endpoint=False) - 1
        ys = np.clip(ys_raw.astype(int), 0, h - 1)
    else:
        ys = np.linspace(0, h - 1, y_num, dtype=int)

    for x in xs:
        for y in ys:
            mask[y, x] = 0
    return mask


def generate_vertical_mask(
    shape: tuple[int, int],
    n_lines: int = 20,
    x_num: int = 20,
    y_num: int = 30,
    y_law: str = "log",
    seed: int | None = None,
) -> np.ndarray:
    height, width = shape
    mask = vertical_random_lines_mask(width, height, n_lines=n_lines, seed=seed)
    mask = add_model_points(mask, x_num=x_num, y_num=y_num, y_law=y_law)
    return mask[None, ...]  # Add channel dimension