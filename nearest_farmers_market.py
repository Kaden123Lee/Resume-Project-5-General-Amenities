
"""
nearest_farmers_market.py
-------------------------
Find the closest farmers market to a given latitude/longitude from the provided
"Farmers_Markets" CSV.

Supported data shapes
---------------------
- Explicit WGS84 columns: "Latitude", "Longitude" (preferred if present).
- Projected coordinates: "x", "y" (assumed EPSG:2229 StatePlane CA Zone 5, US feet unless you pass data_epsg).

Return value
------------
A dict with keys:
  - name
  - address
  - city
  - zip_code
  - latitude
  - longitude
  - accepts_snap_ebt (if available)
  - distance_km
  - raw_index

Usage
-----
from nearest_farmers_market import find_closest_market

result = find_closest_market(34.05, -118.25)  # downtown LA
print(result)

Or, if your CSV is in a different path:
find_closest_market(34.05, -118.25, data_path="path/to/Farmers_Markets.csv")
"""

from typing import Optional, Dict, Tuple, List
import os
import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None

# Default to the uploaded dataset in this environment
DEFAULT_DATA_PATH = "/mnt/data/Farmers_Markets_chp_-5926993295703360347.csv"

def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0088  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def _compose_address(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            if s:
                parts.append(s)
    return ", ".join(parts)

def find_closest_market(
    lat: float,
    lon: float,
    *,
    data_path: Optional[str] = None,
    data_epsg: Optional[str] = None,
) -> Dict:
    """
    Return the closest farmers market to (lat, lon). If the CSV does not provide
    Latitude/Longitude but has projected (x,y), EPSG:2229 is assumed unless you
    provide data_epsg.
    """
    path = data_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Prefer explicit Latitude/Longitude if present
    has_latlon = "Latitude" in df.columns and "Longitude" in df.columns
    has_xy = "x" in df.columns and "y" in df.columns

    if has_latlon:
        df = df.dropna(subset=["Latitude","Longitude"]).copy()
        df["__lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["__lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["__lat","__lon"])
        dists = _haversine_km(lat, lon, df["__lat"].to_numpy(), df["__lon"].to_numpy())
        idx = int(np.argmin(dists))
        row = df.iloc[idx]
        return {
            "name": str(row.get("Name","")),
            "address": _compose_address(row, ["Address","City","Zip Code"]),
            "city": str(row.get("City","")) if "City" in df.columns else None,
            "zip_code": str(row.get("Zip Code","")) if "Zip Code" in df.columns else None,
            "latitude": float(row["__lat"]),
            "longitude": float(row["__lon"]),
            "accepts_snap_ebt": str(row.get("Site Accepts SNAP EBT","")) if "Site Accepts SNAP EBT" in df.columns else None,
            "distance_km": float(dists[idx]),
            "raw_index": int(row.name),
        }

    if has_xy:
        if Transformer is None:
            raise RuntimeError(
                "This CSV uses projected coordinates (x,y). Install pyproj first:\n"
                "  pip install pyproj\n"
                "Then re-run. If your data is not EPSG:2229, provide data_epsg='EPSG:XXXX'."
            )
        target_epsg = (data_epsg or "EPSG:2229").upper()
        try:
            to_proj = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
        except Exception as e:
            raise RuntimeError(f"Could not create transformer to {target_epsg}: {e}")

        X0, Y0 = to_proj.transform(lon, lat)  # always_xy => (lon, lat)
        df = df.dropna(subset=["x","y"]).copy()
        df["__x"] = pd.to_numeric(df["x"], errors="coerce")
        df["__y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["__x","__y"])

        dx = df["__x"].to_numpy() - X0
        dy = df["__y"].to_numpy() - Y0
        planar = np.sqrt(dx*dx + dy*dy)
        # If StatePlane feet, convert to km; otherwise assume meters
        if target_epsg in {"EPSG:2229","EPSG:2230","EPSG:102645","EPSG:102646"}:
            dist_km = (planar * 0.3048) / 1000.0
        else:
            dist_km = planar / 1000.0

        idx = int(np.argmin(dist_km))
        row = df.iloc[idx]

        # Try to provide lat/lon if possible
        try:
            to_wgs84 = Transformer.from_crs(target_epsg, "EPSG:4326", always_xy=True)
            lon_w, lat_w = to_wgs84.transform(float(row["__x"]), float(row["__y"]))
        except Exception:
            lon_w, lat_w = (None, None)

        return {
            "name": str(row.get("Name","")),
            "address": _compose_address(row, ["Address","City","Zip Code"]),
            "city": str(row.get("City","")) if "City" in df.columns else None,
            "zip_code": str(row.get("Zip Code","")) if "Zip Code" in df.columns else None,
            "latitude": float(lat_w) if lat_w is not None else None,
            "longitude": float(lon_w) if lon_w is not None else None,
            "accepts_snap_ebt": str(row.get("Site Accepts SNAP EBT","")) if "Site Accepts SNAP EBT" in df.columns else None,
            "distance_km": float(dist_km[idx]),
            "raw_index": int(row.name),
        }

    raise ValueError(
        "CSV does not contain recognizable coordinates. Expected ('Latitude','Longitude') or ('x','y')."
    )

if __name__ == "__main__":
    # Simple smoke test from Downtown LA
    try:
        print(find_closest_market(34.0505, -118.2551))
    except Exception as e:
        print("Self-test note:", e)
