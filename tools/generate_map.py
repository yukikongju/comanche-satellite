"""
Generating color and height map using Microsoft Planetary Computer STAC API
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
GRID_SIZE: int = 1024

# Example: Grand Canyon, AZ [min_lon, min_lat, max_lon, max_lat]
TARGET_BBOX: list[float] = [-112.18, 36.05, -112.06, 36.15]
DATE_RANGE: str = "2024-06-01/2024-09-30"
MAX_CLOUD_COVER: float = 5.0


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
    rgb_bands: NDArray[np.uint8],
    elev_bands: NDArray[np.float32],
) -> Figure:
    """Render side-by-side plots of the Sentinel-2 RGB and Copernicus DEM grids.

    Parameters
    ----------
    rgb_bands : NDArray[np.uint8]
        Array of shape (3, height, width) containing RGB channels.
    elev_bands : NDArray[np.float32]
        Array of shape (1, height, width) containing elevation in meters.

    Returns
    -------
    Figure
        Matplotlib figure displaying the colormap and heightmap side by side.
    """
    rgb_image = np.moveaxis(rgb_bands, 0, -1)
    elevation_grid = elev_bands[0]

    fig, (ax_rgb, ax_elev) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    ax_rgb.imshow(rgb_image)
    ax_rgb.set_title("Sentinel-2 True Color (rgb_raw)")
    ax_rgb.axis("off")

    elev_plot = ax_elev.imshow(elevation_grid, cmap="terrain")
    ax_elev.set_title("Copernicus DEM 30m (elev_raw)")
    ax_elev.axis("off")
    fig.colorbar(elev_plot, ax=ax_elev, fraction=0.046, pad=0.04, label="Elevation (m)")

    return fig


# --- Execution ---
catalog = open_catalog(STAC_URL)

s2_item = search_best_sentinel_item(catalog, TARGET_BBOX, DATE_RANGE, MAX_CLOUD_COVER)
dem_item = search_dem_item(catalog, TARGET_BBOX)

print(f"Sentinel-2 ID: {s2_item.id} | Clouds: {s2_item.properties['eo:cloud_cover']}%")
print(f"DEM ID:        {dem_item.id}")

rgb_raw = read_cropped_grid(s2_item.assets["visual"].href, TARGET_BBOX, GRID_SIZE)
elev_raw = read_cropped_grid(dem_item.assets["data"].href, TARGET_BBOX, GRID_SIZE)

print(f"RGB shape: {rgb_raw.shape}, dtype: {rgb_raw.dtype}")
print(f"DEM shape: {elev_raw.shape}, min/max m: {elev_raw.min():.1f} / {elev_raw.max():.1f}")

fig = plot_satellite_grids(rgb_raw, elev_raw)
plt.show()
