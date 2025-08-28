"""
nearest_campground.py
---------------------
Utility to find the closest campground to a given latitude/longitude
from the provided Campgrounds CSV.

Data assumptions
- The CSV has columns: OBJECTID, Name, Address Line 1, City, State, ZIP Code, x, y
- x, y are projected coordinates (same units across dataset).
- Other useful columns include Organization and Category.

Usage
-----
from nearest_campground import find_closest_campground

result = find_closest_campground(34.05, -118.25)
print(result["name"], result["distance_km"], "km away")

You can also specify a custom CSV path:
find_closest_campground(34.05, -118.25, data_path="/path/to/campgrounds.csv")
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Default path points to your uploaded dataset
DEFAULT_DATA_PATH = "Arts and Recreation.csv"

@dataclass
class CampgroundRecord:
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    x: float
    y: float
    distance_km: float

_df_cache: Optional[pd.DataFrame] = None

def _euclidean_distance(x1: float, y1: float, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance in kilometers (assuming units are meters)."""
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2) / 1000.0

def _load_art(data_path: Optional[str]) -> pd.DataFrame:
    """Load the campgrounds dataset, caching for speed."""
    global _df_cache
    path = data_path or DEFAULT_DATA_PATH
    if _df_cache is not None and getattr(_df_cache, "_source_path", None) == path:
        return _df_cache

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    for col in ["Name", "x", "y"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}")

    df = df.dropna(subset=["x", "y"]).copy()
    df._source_path = path  # type: ignore[attr-defined]
    _df_cache = df
    return df

def find_closest_art(x: float, y: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Return the closest campground to the provided coordinate.
    Parameters
    ----------
    x : float
        X coordinate (same projection as CSV).
    y : float
        Y coordinate.
    """
    df = _load_art(data_path)
    if df.empty:
        raise ValueError("No campground records with valid coordinates found in CSV.")

    distances = _euclidean_distance(x, y, df["x"].to_numpy(), df["y"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = CampgroundRecord(
        name      = str(row.get("Name", "")),
        address   = str(row.get("Address Line 1", "")),
        city      = str(row.get("City", "")),
        state     = str(row.get("State", "")),
        zip_code  = str(row.get("ZIP Code", "")),
        x         = float(row["x"]),
        y         = float(row["y"]),
        distance_km = float(distances[idx]),
    )
    return asdict(rec)

if __name__ == "__main__":
    # Example test using approximate coordinates near LA
    result = find_closest_art(6429574.5, 1797450.2)
    print(json.dumps(result, indent=2))
