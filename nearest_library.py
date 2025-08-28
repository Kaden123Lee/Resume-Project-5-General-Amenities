
"""
nearest_library.py
------------------
Utility to find the closest public library to a given latitude/longitude
from the provided "Public_Library_Facilities" CSV.

Data assumptions
- The CSV has a column "Address and Location" that ends with coordinates in the
  form "(<lat>, <lon>)". Example: "...\nLos Angeles\n(34.0408, -118.18)"
- Other useful columns include "Library Name", "City", "Zip Code".

Usage
-----
from nearest_library import find_closest_library

result = find_closest_library(34.05, -118.25)
print(result["library_name"], result["distance_km"], "km away")

You can also specify a custom CSV path:
find_closest_library(34.05, -118.25, data_path="/path/to/Public_Library_Facilities.csv")
"""

import math
import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Default path points to the uploaded file location in this workspace.
DEFAULT_DATA_PATH = "/mnt/data/Public_Library_Facilities_-1446258977375260289 (1).csv"

_COORD_REGEX = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*$")

@dataclass
class LibraryRecord:
    library_name: str
    address: str
    city: Optional[str]
    zip_code: Optional[str]
    latitude: float
    longitude: float
    distance_km: float

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

def _parse_lat_lon_from_address(addr: str) -> Optional[Tuple[float, float]]:
    """
    Extract (lat, lon) from the end of the 'Address and Location' field.
    Returns None if not parseable.
    """
    if not isinstance(addr, str):
        return None
    m = _COORD_REGEX.search(addr)
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    return lat, lon

def _load_libraries(data_path: Optional[str]) -> pd.DataFrame:
    """
    Load and normalize the libraries dataframe, extracting latitude/longitude.
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
    for col in ["Library Name", "Address and Location"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}")

    # Extract coordinates
    coords = df["Address and Location"].apply(_parse_lat_lon_from_address)
    df["__lat"] = coords.apply(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    df["__lon"] = coords.apply(lambda t: t[1] if isinstance(t, tuple) else np.nan)

    # Drop rows without coordinates
    df = df.dropna(subset=["__lat", "__lon"]).copy()

    # Cache with the source path
    df._source_path = path  # type: ignore[attr-defined]
    _df_cache = df
    return df

def find_closest_library(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Return the closest library to the provided coordinate as a dict with keys:
      - library_name
      - address
      - city
      - zip_code
      - latitude
      - longitude
      - distance_km

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
    df = _load_libraries(data_path)

    if df.empty:
        raise ValueError("No library records with valid coordinates were found in the CSV.")

    distances = _haversine_km(lat, lon, df["__lat"].to_numpy(), df["__lon"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = LibraryRecord(
        library_name = str(row.get("Library Name", "")),
        address      = str(row.get("Address and Location", "")),
        city         = str(row.get("City", "")) if "City" in df.columns else None,
        zip_code     = str(row.get("Zip Code", "")) if "Zip Code" in df.columns else None,
        latitude     = float(row["__lat"]),
        longitude    = float(row["__lon"]),
        distance_km  = float(distances[idx]),
    )
    return asdict(rec)

if __name__ == "__main__":
    # Simple smoke test near Downtown LA
    result = find_closest_library(34.0505, -118.2551)
    print(json.dumps(result, indent=2))
