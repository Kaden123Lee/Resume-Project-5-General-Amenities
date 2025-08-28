"""
nearest_trail_poi.py
---------------------
Utility to find the closest trail Point of Interest (POI)
to a given coordinate from the provided Trails/POI dataset.

Data assumptions
- The CSV has columns: X, Y, OBJECTID, Name, POI_type, trail_name, etc.
- X, Y are projected coordinates (same units across dataset).
- Other columns give useful metadata.

Usage
-----
from nearest_trail_poi import find_closest_poi

# Example: find closest POI to a point near Baldwin Hills
result = find_closest_poi(6446000.0, 1829000.0)
print(result["Name"], result["POI_type"], "on", result["trail_name"])
"""

import os
import json
from typing import Dict, Optional
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = "/mnt/data/Trails_POI.csv"

_df_cache = None

def _euclidean_distance(x1: float, y1: float, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance in kilometers (assuming X,Y are meters)."""
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2) / 1000.0

def _load_trail(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the POI dataset, caching for speed."""
    global _df_cache
    path = data_path or DEFAULT_DATA_PATH
    if _df_cache is not None and getattr(_df_cache, "_source_path", None) == path:
        return _df_cache

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    for col in ["X", "Y", "Name", "POI_type"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found: {list(df.columns)}")

    df = df.dropna(subset=["X", "Y"]).copy()
    df._source_path = path  # mark where it came from
    _df_cache = df
    return df

def find_closest_trail(x: float, y: float, *, data_path: Optional[str] = None) -> Dict:
    """
    Find the closest POI to the given X, Y coordinate.
    
    Parameters
    ----------
    x : float
        X coordinate (same projection as dataset).
    y : float
        Y coordinate.
    data_path : str, optional
        Path to a CSV if not using the default.
    """
    df = _load_trail(data_path)
    if df.empty:
        raise ValueError("No POIs with valid coordinates found.")

    distances = _euclidean_distance(x, y, df["X"].to_numpy(), df["Y"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    # include all fields + distance
    result = row.to_dict()
    result["distance_km"] = float(distances[idx])
    return result

if __name__ == "__main__":
    # quick test
    poi = find_closest_trail(6446000.0, 1829000.0)
    print(json.dumps(poi, indent=2))