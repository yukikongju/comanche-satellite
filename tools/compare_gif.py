"""Compare reference and generated GIF maps for Comanche Voxel Space compatibility."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

REFERENCE_COLOR_PATH: Path = Path("../engine/maps/gif/map0.color.gif")
REFERENCE_HEIGHT_PATH: Path = Path("../engine/maps/gif/map0.height.gif")
GENERATED_COLOR_PATH: Path = Path("../assets/map0_color.gif")
GENERATED_HEIGHT_PATH: Path = Path("../assets/map0_height.gif")


def load_gif_arrays(gif_path: Path) -> tuple[str, NDArray[np.uint8], NDArray[np.uint8]]:
    """Load raw indexed/grayscale array and expanded RGB array from a GIF file.

    Parameters
    ----------
    gif_path : Path
        Path to the target ``.gif`` file.

    Returns
    -------
    tuple[str, NDArray[np.uint8], NDArray[np.uint8]]
        Tuple of ``(pil_mode, raw_array, rgb_array)`` where ``raw_array`` preserves
        palette indices (in ``"P"`` mode) and ``rgb_array`` has shape ``(H, W, 3)``.

    Examples
    --------
    >>> mode, raw, rgb = load_gif_arrays(Path("elev_raw.gif"))
    >>> rgb.ndim
    3

    Caveats
    -------
    Matplotlib's ``plt.imsave`` writes palette-indexed (``"P"``) GIFs. Inspecting
    ``raw_array`` reveals the actual 0-255 byte indices read by C engines that
    ignore the GIF color table.
    """
    with Image.open(gif_path) as img:
        mode = img.mode
        raw_array = np.asarray(img, dtype=np.uint8)
        rgb_array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return mode, raw_array, rgb_array


def summarize_gif(label: str, gif_path: Path) -> dict[str, Any]:
    """Compute structural, palette, and pixel statistics for a GIF map.

    Parameters
    ----------
    label : str
        Identifier label for the map in the summary table.
    gif_path : Path
        Path to the ``.gif`` file to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary of summary metrics including shape, palette mode, unique colors,
        raw index bounds, RGB channel statistics, and luminance percentiles.

    Examples
    --------
    >>> stats = summarize_gif("gen_height", Path("elev_raw.gif"))
    >>> "raw_min" in stats
    True

    Caveats
    -------
    If a C voxel engine reads raw GIF raster bytes directly as height values without
    resolving the GIF palette, ``raw_min``/``raw_max`` and ``index_to_luma_corr``
    will expose scrambled elevation ordering.
    """
    mode, raw, rgb = load_gif_arrays(gif_path)
    luma = rgb.mean(axis=-1)
    flat_rgb = rgb.reshape(-1, 3)
    unique_colors = int(np.unique(flat_rgb, axis=0).shape[0])

    # Correlation between raw palette index (0-255) and visual brightness
    if raw.ndim == 2 and np.std(raw) > 0 and np.std(luma) > 0:
        index_luma_corr = float(np.corrcoef(raw.ravel(), luma.ravel())[0, 1])
    else:
        index_luma_corr = float("nan")

    # Local spatial roughness (mean absolute pixel-to-pixel step)
    dx = np.abs(np.diff(luma, axis=1)).mean()
    dy = np.abs(np.diff(luma, axis=0)).mean()
    spatial_roughness = float(0.5 * (dx + dy))

    return {
        "label": label,
        "file": gif_path.name,
        "mode": mode,
        "shape": f"{rgb.shape[0]}x{rgb.shape[1]}",
        "is_square_pow2": (
            rgb.shape[0] == rgb.shape[1] and (rgb.shape[0] & (rgb.shape[0] - 1) == 0)
        ),
        "unique_rgb_colors": unique_colors,
        "raw_min": int(raw.min()),
        "raw_max": int(raw.max()),
        "raw_mean": round(float(raw.mean()), 2),
        "raw_std": round(float(raw.std()), 2),
        "luma_min": round(float(luma.min()), 2),
        "luma_p05": round(float(np.percentile(luma, 5)), 2),
        "luma_median": round(float(np.median(luma)), 2),
        "luma_mean": round(float(luma.mean()), 2),
        "luma_p95": round(float(np.percentile(luma, 95)), 2),
        "luma_max": round(float(luma.max()), 2),
        "luma_std": round(float(luma.std()), 2),
        "pct_zero_luma": round(float((luma == 0).mean() * 100.0), 2),
        "pct_saturated_luma": round(float((luma >= 254).mean() * 100.0), 2),
        "index_to_luma_corr": round(index_luma_corr, 3),
        "spatial_roughness": round(spatial_roughness, 2),
    }


def compare_gif_maps(map_specs: list[tuple[str, Path]]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame across multiple GIF files.

    Parameters
    ----------
    map_specs : list[tuple[str, Path]]
        Sequence of ``(label, gif_path)`` pairs to evaluate.

    Returns
    -------
    pd.DataFrame
        Transposed comparison table indexed by metric name.

    Examples
    --------
    >>> df = compare_gif_maps([("gen", Path("elev_raw.gif"))])
    >>> "mode" in df.index
    True
    """
    records = [summarize_gif(label, path) for label, path in map_specs]
    return pd.DataFrame(records).set_index("label").T


# --- Execution ---
MAP_FILES: list[tuple[str, Path]] = [
    ("ref_color", REFERENCE_COLOR_PATH),
    ("gen_color", GENERATED_COLOR_PATH),
    ("ref_height", REFERENCE_HEIGHT_PATH),
    ("gen_height", GENERATED_HEIGHT_PATH),
]

comparison_df = compare_gif_maps(MAP_FILES)
print(comparison_df.to_string())
