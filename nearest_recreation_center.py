"""
nearest_recreation_center.py
----------------------------
Utility to find the closest recreation center to a given latitude/longitude
from the provided recreation center CSV data.

Data assumptions
- The CSV has separate "latitude" and "longitude" columns
- Other useful columns include "Name", "addrln1", "city", "zip"

Usage
-----
from nearest_recreation_center import find_closest_recreation_center

result = find_closest_recreation_center(34.05, -118.25)
print(result["name"], result["distance_km"], "km away")

You can also specify a custom CSV path:
find_closest_recreation_center(34.05, -118.25, data_path="/path/to/recreation_centers.csv")
"""

import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Default path points to the uploaded file location in this workspace.
DEFAULT_DATA_PATH = "/mnt/data/recreation_centers.csv"

@dataclass
class RecreationCenterRecord:
    name: str
    address: str
    city: Optional[str]
    zip_code: Optional[str]
    latitude: float
    longitude: float
    distance_km: float
    org_name: Optional[str]
    phone: Optional[str]
    hours: Optional[str]

# Module-level cache so repeated calls are fast
_df_cache: Optional[pd.DataFrame] = None

def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized Haversine distance in kilometers between a single point and arrays of points.
    """
    R = 6371.0088  # Mean Earth radius in kilometers
    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def _load_recreation_centers(data_path: Optional[str]) -> pd.DataFrame:
    """
    Load and normalize the recreation centers dataframe.
    Caches the dataframe for subsequent calls.
    """
    global _df_cache
    path = data_path or DEFAULT_DATA_PATH
    if _df_cache is not None and getattr(_df_cache, "_source_path", None) == path:
        return _df_cache

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Ensure required columns exist
    for col in ["Name", "latitude", "longitude"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}")

    # Extract coordinates from dedicated columns
    df["__lat"] = pd.to_numeric(df["latitude"], errors='coerce')
    df["__lon"] = pd.to_numeric(df["longitude"], errors='coerce')

    # Drop rows without coordinates
    df = df.dropna(subset=["__lat", "__lon"]).copy()

    # Cache with the source path
    df._source_path = path
    _df_cache = df
    return df

def find_closest_recreation_center(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Return the closest recreation center to the provided coordinate as a dict with keys:
      - name
      - address
      - city
      - zip_code
      - latitude
      - longitude
      - distance_km
      - org_name
      - phone
      - hours

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    data_path : Optional[str]
        Optional path to the CSV if different from the default.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found.
    ValueError
        If required columns are missing or no rows contain coordinates.
    """
    df = _load_recreation_centers(data_path)

    if df.empty:
        raise ValueError("No recreation center records with valid coordinates were found in the CSV.")

    distances = _haversine_km(lat, lon, df["__lat"].to_numpy(), df["__lon"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = RecreationCenterRecord(
        name         = str(row.get("Name", "")),
        address      = str(row.get("addrln1", "")),
        city         = str(row.get("city", "")) if "city" in df.columns else None,
        zip_code     = str(row.get("zip", "")) if "zip" in df.columns else None,
        latitude     = float(row["__lat"]),
        longitude    = float(row["__lon"]),
        distance_km  = float(distances[idx]),
        org_name     = str(row.get("org_name", "")) if "org_name" in df.columns else None,
        phone        = str(row.get("phones", "")) if "phones" in df.columns else None,
        hours        = str(row.get("hours", "")) if "hours" in df.columns else None,
    )
    return asdict(rec)

if __name__ == "__main__":
    # Simple smoke test near Downtown LA
    result = find_closest_recreation_center(34.0505, -118.2551)
    print(json.dumps(result, indent=2))