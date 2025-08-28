"""
nearest_business.py
-------------------
Utility to find the closest business to a given latitude/longitude
from the provided business CSV data with X/Y coordinates.

Data assumptions
- The CSV has "X" and "Y" columns (projected coordinates, likely Web Mercator)
- Other useful columns include "Business_Name", "Address", "City", "Zip", "Business_Category"

Usage
-----
from nearest_business import find_closest_business

result = find_closest_business(34.05, -118.25)
print(result["business_name"], result["distance_km"], "km away")

You can also specify a custom CSV path:
find_closest_business(34.05, -118.25, data_path="/path/to/businesses.csv")
"""

import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Default path points to the uploaded file location in this workspace.
DEFAULT_DATA_PATH = "/mnt/data/businesses.csv"

@dataclass
class BusinessRecord:
    business_name: str
    address: str
    city: str
    zip_code: str
    business_category: str
    latitude: float
    longitude: float
    distance_km: float
    object_id: Optional[str]

# Module-level cache so repeated calls are fast
_df_cache: Optional[pd.DataFrame] = None

def _web_mercator_to_latlon(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Web Mercator coordinates (EPSG:3857) to latitude/longitude (EPSG:4326).
    """
    lon = x * 180 / 20037508.34
    lat = np.arctan(np.exp(y * np.pi / 20037508.34)) * 360 / np.pi - 90
    return lat, lon

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

def _load_businesses(data_path: Optional[str]) -> pd.DataFrame:
    """
    Load and normalize the businesses dataframe.
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
    for col in ["Business_Name", "X", "Y"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}")

    # Convert Web Mercator coordinates to lat/lon
    df["__x"] = pd.to_numeric(df["X"], errors='coerce')
    df["__y"] = pd.to_numeric(df["Y"], errors='coerce')
    
    # Drop rows without coordinates
    df = df.dropna(subset=["__x", "__y"]).copy()
    
    # Convert to lat/lon
    lat, lon = _web_mercator_to_latlon(df["__x"].to_numpy(), df["__y"].to_numpy())
    df["__lat"] = lat
    df["__lon"] = lon

    # Cache with the source path
    df._source_path = path
    _df_cache = df
    return df

def find_closest_business(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Return the closest business to the provided coordinate as a dict with keys:
      - business_name
      - address
      - city
      - zip_code
      - business_category
      - latitude
      - longitude
      - distance_km
      - object_id

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
    df = _load_businesses(data_path)

    if df.empty:
        raise ValueError("No business records with valid coordinates were found in the CSV.")

    distances = _haversine_km(lat, lon, df["__lat"].to_numpy(), df["__lon"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = BusinessRecord(
        business_name      = str(row.get("Business_Name", "")),
        address           = str(row.get("Address", "")),
        city              = str(row.get("City", "")),
        zip_code          = str(row.get("Zip", "")),
        business_category = str(row.get("Business_Category", "")),
        latitude          = float(row["__lat"]),
        longitude         = float(row["__lon"]),
        distance_km       = float(distances[idx]),
        object_id         = str(row.get("OBJECTID", "")) if "OBJECTID" in df.columns else None,
    )
    return asdict(rec)

if __name__ == "__main__":
    # Simple smoke test near Downtown LA
    result = find_closest_business(34.05, -118.25)
    print(json.dumps(result, indent=2))