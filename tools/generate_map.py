"""
Generating color and height map using Microsoft Planetary Computer STAC API
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio

from PIL import Image
from pyproj import Transformer
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/search"
STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
USER_AGENT: str = "comanche-voxel-map-builder/1.0"
LOCATION_QUERY: str = "Mount St. Helens, Washington"
MAP_SIZE_METERS: float = 10240.0
GRID_SIZE: int = 1024

# Example: Grand Canyon, AZ [min_lon, min_lat, max_lon, max_lat]
#  TARGET_BBOX: list[float] = [-112.18, 36.05, -112.06, 36.15]
DATE_RANGE: str = "2024-06-01/2024-09-30"
MAX_CLOUD_COVER: float = 5.0

DEM_SMOOTH_SIGMA: float = 1.2
SUN_AZIMUTH_DEG: float = 315.0
SUN_ALTITUDE_DEG: float = 35.0
SHADE_STRENGTH: float = 0.55
RGB_GAMMA: float = 1.25
SKY_RGB: tuple[int, int, int] = (110, 165, 225)

RGB_GIF_PATH: Path = Path("../assets/map1_color.gif")
ELEV_GIF_PATH: Path = Path("../assets/map1_height.gif")

def open_catalog(url: str) -> pystac_client.Client:
    """Open the Planetary Computer STAC catalog with automatic SAS signing.

    Parameters
    ----------
    url : str
        Base STAC API endpoint URL.

    Returns
    -------
    pystac_client.Client
        Authenticated STAC client that signs asset URLs in place.
    """
    return pystac_client.Client.open(
        url,
        modifier=planetary_computer.sign_inplace,
    )


def search_best_sentinel_item(
    catalog: pystac_client.Client,
    bbox: list[float],
    date_range: str,
    max_cloud: float,
) -> Any:
    """Find the lowest-cloud Sentinel-2 L2A item intersecting the bounding box.

    Parameters
    ----------
    catalog : pystac_client.Client
        Open STAC client.
    bbox : list[float]
        WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].
    date_range : str
        ISO8601 interval string (e.g., "2024-06-01/2024-09-30").
    max_cloud : float
        Maximum allowed cloud cover percentage.

    Returns
    -------
    Any
        Signed PySTAC Item with the lowest cloud cover.
    """
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    return min(items, key=lambda item: item.properties["eo:cloud_cover"])


def search_dem_item(
    catalog: pystac_client.Client,
    bbox: list[float],
) -> Any:
    """Find the Copernicus GLO-30 DEM tile intersecting the bounding box.

    Parameters
    ----------
    catalog : pystac_client.Client
        Open STAC client.
    bbox : list[float]
        WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].

    Returns
    -------
    Any
        Signed PySTAC Item for the elevation tile.
    """
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    )
    return next(search.items())


def read_cropped_grid(
    cog_href: str,
    wgs84_bbox: list[float],
    grid_size: int,
) -> NDArray[Any]:
    """Stream and resample a bounding box from a remote COG to (bands, grid_size, grid_size).

    Caveats
    -------
    Sentinel-2 uses UTM (meters) while Copernicus DEM uses EPSG:4326 (degrees),
    so `transform_bounds` converts the input bbox into each dataset's native CRS.
    """
    with rasterio.open(cog_href) as src:
        native_bounds = transform_bounds("EPSG:4326", src.crs, *wgs84_bbox)
        window = from_bounds(*native_bounds, transform=src.transform)
        return src.read(
            window=window,
            out_shape=(src.count, grid_size, grid_size),
            resampling=rasterio.enums.Resampling.bilinear,
        )


def plot_satellite_grids(
    rgb_hwc: NDArray[np.uint8],
    elev_hwc: NDArray[np.float32],
) -> Figure:
    """Render side-by-side plots of the Sentinel-2 RGB and Copernicus DEM grids.

    Parameters
    ----------
    rgb_hwc : NDArray[np.uint8]
        Array of shape (height, width, 3) containing RGB channels.
    elev_hwc : NDArray[np.float32]
        Array of shape (height, width, 1) containing elevation in meters.

    Returns
    -------
    Figure
        Matplotlib figure displaying the colormap and heightmap side by side.
    """
    #  rgb_image = np.moveaxis(rgb_bands, 0, -1)
    #  elevation_grid = elev_bands[0]

    fig, (ax_rgb, ax_elev) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    ax_rgb.imshow(rgb_hwc)
    ax_rgb.set_title("Sentinel-2 True Color (rgb_raw)")
    ax_rgb.axis("off")

    elev_plot = ax_elev.imshow(elev_hwc, cmap="terrain")
    ax_elev.set_title("Copernicus DEM 30m (elev_raw)")
    ax_elev.axis("off")
    fig.colorbar(elev_plot, ax=ax_elev, fraction=0.046, pad=0.04, label="Elevation (m)")

    return fig

def geocode_location(place_name: str, user_agent: str) -> tuple[float, float]:
    """Resolve a human-readable location name to (latitude, longitude) via OSM Nominatim.

    Parameters
    ----------
    place_name : str
        Location query string (e.g., "Mont-Tremblant, Quebec").
    user_agent : str
        Descriptive User-Agent header required by Nominatim's usage policy.

    Returns
    -------
    tuple[float, float]
        Center coordinates as (center_lat, center_lon) in WGS84 decimal degrees.

    Caveats
    -------
    Nominatim enforces a maximum rate of 1 request per second and requires a
    non-generic User-Agent header. Raises ValueError if no match is found.
    """
    params = urllib.parse.urlencode({"q": place_name, "format": "json", "limit": 1})
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}",
        headers={"User-Agent": user_agent},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        results: list[dict[str, str]] = json.loads(response.read().decode("utf-8"))

    if not results:
        raise ValueError(f"No coordinates found for location: {place_name!r}")

    return float(results[0]["lat"]), float(results[0]["lon"])

def bbox_from_center(
    center_lat: float,
    center_lon: float,
    size_meters: float,
) -> list[float]:
    """Compute a physically square WGS84 bounding box from a center point and width.

    Parameters
    ----------
    center_lat : float
        Center latitude in decimal degrees (WGS84).
    center_lon : float
        Center longitude in decimal degrees (WGS84).
    size_meters : float
        Width and height of the square terrain patch in meters.

    Returns
    -------
    list[float]
        WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].

    Examples
    --------
    >>> bbox_from_center(36.10, -112.12, 10240.0)
    [-112.1768..., 36.0538..., -112.0631..., 36.1461...]
    """
    local_crs = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    half_size = size_meters / 2.0
    min_lon, min_lat = to_wgs84.transform(-half_size, -half_size)
    max_lon, max_lat = to_wgs84.transform(half_size, half_size)

    return [min_lon, min_lat, max_lon, max_lat]

def save_grid_to_gif(
    grid: NDArray[Any],
    output_path: Path,
    cmap: str | None = None,
) -> None:
    """Save a single (height, width, bands) or (height, width) raster grid as a GIF image.

    Parameters
    ----------
    grid : NDArray[Any]
        Raster array of shape (3, height, width) for RGB or (1, height, width)
        for single-band elevation data.
    output_path : Path
        Destination `.gif` file path.
    cmap : str | None
        Matplotlib colormap name for single-band grids (ignored for 3-band RGB).

    Examples
    --------
    >>> save_grid_to_gif(rgb_raw, Path("rgb_raw.gif"))
    >>> save_grid_to_gif(elev_raw, Path("elev_raw.gif"), cmap="terrain")

    Caveats
    -------
    GIF quantizes colors to 256 entries, so use these files for visual preview
    rather than as input to the C Voxel Space engine.
    """
    #  image_data = np.moveaxis(grid, 0, -1) if grid.shape[0] == 3 else grid[0]
    #  plt.imsave(output_path, image_data, format="gif", cmap=cmap)
    plt.imsave(output_path, grid, format="gif", cmap=cmap)

def read_aligned_grids(
    rgb_href: str,
    dem_href: str,
    wgs84_bbox: list[float],
    grid_size: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], float]:
    """Read RGB in UTM and reproject DEM onto the exact same pixel grid.

    Returns
    -------
    tuple[NDArray[np.uint8], NDArray[np.float32], float]
        (rgb_hwc of shape (H, W, 3), elev_hw of shape (H, W), pixel_size_meters).
    """
    with rasterio.open(rgb_href) as rgb_src:
        utm_bounds = transform_bounds("EPSG:4326", rgb_src.crs, *wgs84_bbox)
        window = from_bounds(*utm_bounds, transform=rgb_src.transform)
        rgb_chw = rgb_src.read(
            window=window,
            out_shape=(3, grid_size, grid_size),
            resampling=rasterio.enums.Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        dst_transform = rasterio.transform.from_bounds(*utm_bounds, grid_size, grid_size)
        dst_crs = rgb_src.crs
        pixel_size_m = (utm_bounds[2] - utm_bounds[0]) / grid_size

    elev_hw = np.empty((grid_size, grid_size), dtype=np.float32)
    with rasterio.open(dem_href) as dem_src:
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=elev_hw,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rasterio.enums.Resampling.cubic,
        )

    rgb_hwc = np.moveaxis(rgb_chw, 0, -1).astype(np.uint8)
    return rgb_hwc, elev_hw, pixel_size_m


def smooth_and_normalize_height(
    elev_hw: NDArray[np.float32],
    sigma: float,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Smooth 30m DEM stair-steps and quantize elevation to full-range uint8 (0..255)."""
    smoothed = gaussian_filter(elev_hw, sigma=sigma)
    min_h = float(smoothed.min())
    max_h = float(smoothed.max())
    norm = (smoothed - min_h) / max(max_h - min_h, 1e-6)
    height_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return smoothed, height_u8


def bake_hillshade_into_rgb(
    rgb_hwc: NDArray[np.uint8],
    elev_meters: NDArray[np.float32],
    pixel_size_m: float,
    azimuth_deg: float,
    altitude_deg: float,
    shade_strength: float,
    gamma: float,
) -> NDArray[np.uint8]:
    """Apply gamma correction and directional terrain hillshading to the colormap."""
    dy, dx = np.gradient(elev_meters, pixel_size_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)

    az_rad = np.deg2rad(azimuth_deg)
    alt_rad = np.deg2rad(altitude_deg)
    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(
        az_rad - aspect
    )
    hillshade = np.clip(shaded, 0.0, 1.0)

    rgb_float = (rgb_hwc.astype(np.float32) / 255.0) ** gamma
    lighting = (1.0 - shade_strength) + shade_strength * hillshade
    lit_rgb = np.clip(rgb_float * lighting[..., np.newaxis] * 255.0, 0, 255)
    return lit_rgb.astype(np.uint8)


def export_comanche_bins(
    color_hwc: NDArray[np.uint8],
    height_hw: NDArray[np.uint8],
    color_path: Path,
    height_path: Path,
) -> None:
    """Write raw headerless binary arrays for C (H*W*3 RGB bytes and H*W height bytes)."""
    color_path.write_bytes(np.ascontiguousarray(color_hwc).tobytes())
    height_path.write_bytes(np.ascontiguousarray(height_hw).tobytes())

def prepare_comanche_height_gif(height_u8: NDArray[np.uint8]) -> Image.Image:
    img = Image.fromarray(height_u8, mode="P")
    identity_palette = [c for i in range(256) for c in (i, i, i)]
    img.putpalette(identity_palette)
    return img

def prepare_comanche_color_gif(
    color_hwc: NDArray[np.uint8], sky_rgb: tuple[int, int, int]
) -> Image.Image:
    rgb_img = Image.fromarray(color_hwc, mode="RGB")
    quantized = rgb_img.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    
    shifted_indices = np.asarray(quantized, dtype=np.uint8) + 1
    full_palette = list(sky_rgb) + quantized.getpalette()[: 255 * 3]
    
    img = Image.fromarray(shifted_indices, mode="P")
    img.putpalette(full_palette)
    return img

def save_image_to_gif(img: Image.Image, output_path: Path) -> None:
    """Save a PIL Image to disk as a GIF file.

    Parameters
    ----------
    img : Image.Image
        Source PIL Image (e.g., paletted mode "P" or "RGB").
    output_path : Path
        Destination `.gif` file path.

    Examples
    --------
    >>> save_image_to_gif(rgb_clean, Path("../assets/map0_color.gif"))
    >>> save_image_to_gif(elev_clean, Path("../assets/map0_height.gif"))

    Caveats
    -------
    Creates parent directories if they do not exist and writes the image directly
    without re-quantizing or altering an existing mode "P" color table.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="GIF")

# --- Execution ---
center_lat, center_lon = geocode_location(LOCATION_QUERY, USER_AGENT)
target_bbox = bbox_from_center(center_lat, center_lon, MAP_SIZE_METERS)
print(f"{LOCATION_QUERY} -> lat={center_lat:.5f}, lon={center_lon:.5f}")

catalog = open_catalog(STAC_URL)

s2_item = search_best_sentinel_item(catalog, target_bbox, DATE_RANGE, MAX_CLOUD_COVER)
dem_item = search_dem_item(catalog, target_bbox)

print(f"Sentinel-2 ID: {s2_item.id} | Clouds: {s2_item.properties['eo:cloud_cover']}%")
print(f"DEM ID:        {dem_item.id}")

#
#  rgb_raw = read_cropped_grid(s2_item.assets["visual"].href, target_bbox, GRID_SIZE)
#  elev_raw = read_cropped_grid(dem_item.assets["data"].href, target_bbox, GRID_SIZE)


# Reproject DEM onto Sentinel-2 UTM grid; stretch heigh contrast; add hillshading based on gamma curve
rgb_raw, elev_raw, pixel_size_m = read_aligned_grids(
    s2_item.assets["visual"].href,
    dem_item.assets["data"].href,
    target_bbox,
    GRID_SIZE,
)
elev_smooth, height_u8 = smooth_and_normalize_height(elev_raw, DEM_SMOOTH_SIGMA)
color_shaded = bake_hillshade_into_rgb(
    rgb_raw,
    elev_smooth,
    pixel_size_m,
    SUN_AZIMUTH_DEG,
    SUN_ALTITUDE_DEG,
    SHADE_STRENGTH,
    RGB_GAMMA,
)

print(f"RGB shape: {rgb_raw.shape}, dtype: {rgb_raw.dtype}")
print(f"DEM shape: {elev_raw.shape}, min/max m: {elev_raw.min():.1f} / {elev_raw.max():.1f}")

fig = plot_satellite_grids(rgb_raw, elev_raw)
plt.show()

#  rgb_clean = prepare_comanche_color_gif(color_shaded, SKY_RGB)
#  elev_clean = prepare_comanche_height_gif(height_u8)

save_grid_to_gif(rgb_raw, RGB_GIF_PATH)
save_grid_to_gif(elev_raw, ELEV_GIF_PATH, cmap="terrain")
#  save_image_to_gif(rgb_clean, RGB_GIF_PATH)
#  save_image_to_gif(elev_clean, ELEV_GIF_PATH)


#  export_comanche_bins(color_shaded, height_u8, OUTPUT_COLOR_BIN, OUTPUT_HEIGHT_BIN)
