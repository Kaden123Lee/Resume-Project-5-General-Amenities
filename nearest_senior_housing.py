"""
nearest_senior_housing.py
-------------------------
Utility to find the closest senior housing facility to a given latitude/longitude
from the provided senior housing CSV data.

Data assumptions
- The CSV has separate "Lat" and "Lon" columns
- Other useful columns include "Name", "Address", "City", "Zipcode"

Usage
-----
from nearest_senior_housing import find_closest_senior_housing

result = find_closest_senior_housing(34.05, -118.25)
print(result["name"], result["distance_km"], "km away")

You can also specify a custom CSV path:
find_closest_senior_housing(34.05, -118.25, data_path="/path/to/senior_housing.csv")
"""

import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Default path points to the uploaded file location in this workspace.
DEFAULT_DATA_PATH = "/mnt/data/senior_housing.csv"

@dataclass
class SeniorHousingRecord:
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    latitude: float
    longitude: float
    distance_km: float
    object_id: Optional[str]
    global_id: Optional[str]

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

def _load_senior_housing(data_path: Optional[str]) -> pd.DataFrame:
    """
    Load and normalize the senior housing dataframe.
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
    for col in ["Name", "Lat", "Lon"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}")

    # Extract coordinates from dedicated columns
    df["__lat"] = pd.to_numeric(df["Lat"], errors='coerce')
    df["__lon"] = pd.to_numeric(df["Lon"], errors='coerce')

    # Drop rows without coordinates
    df = df.dropna(subset=["__lat", "__lon"]).copy()

    # Cache with the source path
    df._source_path = path
    _df_cache = df
    return df

def find_closest_senior_housing(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Return the closest senior housing facility to the provided coordinate as a dict with keys:
      - name
      - address
      - city
      - state
      - zip_code
      - latitude
      - longitude
      - distance_km
      - object_id
      - global_id

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
    df = _load_senior_housing(data_path)

    if df.empty:
        raise ValueError("No senior housing records with valid coordinates were found in the CSV.")

    distances = _haversine_km(lat, lon, df["__lat"].to_numpy(), df["__lon"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = SeniorHousingRecord(
        name         = str(row.get("Name", "")),
        address      = str(row.get("Address", "")),
        city         = str(row.get("City", "")),
        state        = str(row.get("State", "")),
        zip_code     = str(row.get("Zipcode", "")),
        latitude     = float(row["__lat"]),
        longitude    = float(row["__lon"]),
        distance_km  = float(distances[idx]),
        object_id    = str(row.get("OBJECTID", "")) if "OBJECTID" in df.columns else None,
        global_id    = str(row.get("GlobalID", "")) if "GlobalID" in df.columns else None,
    )
    return asdict(rec)

if __name__ == "__main__":
    # Simple smoke test near Downtown LA
    result = find_closest_senior_housing(34.05, -118.25)
    print(json.dumps(result, indent=2))