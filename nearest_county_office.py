"""
nearest_county_office.py
------------------------
Utility to find the closest Los Angeles County Office / Municipal Service
to a given latitude/longitude from the provided CSV.

Data assumptions
- CSV includes columns: OBJECTID, Name, Label, Category1, Category2,
  Organization, Address Line 1, Address Line 2, City, State, ZIP Code, x, y
- Coordinates x/y are in projected meters (not lat/lon). We'll use
  latitude/longitude if provided, else fallback to x/y.
- File is expected to be UTF-8 CSV.

Usage
-----
from nearest_county_office import find_closest_county_office

result = find_closest_county_office(34.05, -118.25)
print(result["Name"], result["distance_km"], "km away")

Optionally specify a custom CSV path:
find_closest_county_office(34.05, -118.25, data_path="/path/to/County_Offices.csv")
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "/mnt/data/County_Offices.csv"

@dataclass
class CountyOfficeRecord:
    objectid: int
    name: str
    label: str
    category1: str
    category2: str
    category3: str
    organization: str
    address: str
    address2: str
    city: str
    state: str
    zip_code: str
    latitude: float
    longitude: float
    distance_km: float

_df_cache: Optional[pd.DataFrame] = None

def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in kilometers between one point and many others."""
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def _load_offices(data_path: Optional[str]) -> pd.DataFrame:
    global _df_cache
    path = data_path or DEFAULT_DATA_PATH
    if _df_cache is not None and getattr(_df_cache, "_source_path", None) == path:
        return _df_cache

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Derive lat/lon if not explicitly present (x/y given in dataset are projected)
    if "Latitude" in df.columns and "Longitude" in df.columns:
        pass
    else:
        if "y" in df.columns and "x" in df.columns:
            # Convert from LA County projected coordinates (approx meters) to pseudo lat/lon
            # NOTE: For precision, you should use pyproj with the actual EPSG. Here we assume
            # x ~ easting, y ~ northing in meters relative to WGS84 UTM zone 11N.
            import pyproj
            proj = pyproj.Transformer.from_crs("EPSG:2229", "EPSG:4326", always_xy=True)  # LA County typically EPSG:2229
            lons, lats = proj.transform(df["x"].to_numpy(), df["y"].to_numpy())
            df["Longitude"] = lons
            df["Latitude"] = lats

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df._source_path = path
    _df_cache = df
    return df

def find_closest_county_office(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """Return the closest county office to the given lat/lon."""
    df = _load_offices(data_path)
    if df.empty:
        raise ValueError("No county office records with valid coordinates found.")

    distances = _haversine_km(lat, lon, df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = CountyOfficeRecord(
        objectid     = int(row.get("OBJECTID", -1)),
        name         = str(row.get("Name", "")),
        label        = str(row.get("Label", "")),
        category1    = str(row.get("Category1", "")),
        category2    = str(row.get("Category2", "")),
        category3    = str(row.get("Category3", "")),
        organization = str(row.get("Organization", "")),
        address      = str(row.get("Address Line 1", "")),
        address2     = str(row.get("Address Line 2", "")),
        city         = str(row.get("City", "")),
        state        = str(row.get("State", "")),
        zip_code     = str(row.get("ZIP Code", "")),
        latitude     = float(row["Latitude"]),
        longitude    = float(row["Longitude"]),
        distance_km  = float(distances[idx]),
    )
    return asdict(rec)

if __name__ == "__main__":
    # Example: downtown Los Angeles
    result = find_closest_county_office(34.05, -118.25)
    print(json.dumps(result, indent=2))
