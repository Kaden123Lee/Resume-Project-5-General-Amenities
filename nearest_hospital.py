"""
nearest_hospital.py
-------------------
Utility to find the closest hospital to a given latitude/longitude
from the provided Hospitals CSV.

Data assumptions
- CSV includes columns: FACNAME, FAC_TYPE_CODE, CAPACITY, ADDRESS, City, ZIP Code,
  CONTACT_PHONE_NUMBER, CONTACT_EMAIL, COUNTY_NAME, LICENSE_STATUS_DESCRIPTION,
  LATITUDE, LONGITUDE.
- Lat/Lon are WGS84 coordinates (decimal degrees).

Usage
-----
from nearest_hospital import find_closest_hospital

result = find_closest_hospital(34.05, -118.25)
print(result["FACNAME"], result["distance_km"], "km away")

Optionally specify a custom CSV path:
find_closest_hospital(34.05, -118.25, data_path="/path/to/hospitals.csv")
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "/mnt/data/Hospitals.csv"

@dataclass
class HospitalRecord:
    facname: str
    fac_type: str
    capacity: Optional[int]
    address: str
    city: str
    zip_code: str
    county: str
    district: str
    license_status: str
    contact_phone: str
    contact_email: str
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

def _load_hospitals(data_path: Optional[str]) -> pd.DataFrame:
    global _df_cache
    path = data_path or DEFAULT_DATA_PATH
    if _df_cache is not None and getattr(_df_cache, "_source_path", None) == path:
        return _df_cache

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    for col in ["FACNAME", "LATITUDE", "LONGITUDE"]:
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in CSV. Found: {list(df.columns)}")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    df._source_path = path
    _df_cache = df
    return df

def find_closest_hospital(lat: float, lon: float, *, data_path: Optional[str] = None) -> Dict:
    """Return the closest hospital to the given lat/lon."""
    df = _load_hospitals(data_path)
    if df.empty:
        raise ValueError("No hospital records with valid coordinates found.")

    distances = _haversine_km(lat, lon, df["LATITUDE"].to_numpy(), df["LONGITUDE"].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    rec = HospitalRecord(
        facname        = str(row.get("FACNAME", "")),
        fac_type       = str(row.get("FAC_TYPE_CODE", "")),
        capacity       = int(row["CAPACITY"]) if not pd.isna(row.get("CAPACITY")) else None,
        address        = str(row.get("ADDRESS", "")),
        city           = str(row.get("City", "")),
        zip_code       = str(row.get("ZIP Code", "")),
        county         = str(row.get("COUNTY_NAME", "")),
        district       = str(row.get("DISTRICT_NAME", "")),
        license_status = str(row.get("LICENSE_STATUS_DESCRIPTION", "")),
        contact_phone  = str(row.get("CONTACT_PHONE_NUMBER", "")),
        contact_email  = str(row.get("CONTACT_EMAIL", "")),
        latitude       = float(row["LATITUDE"]),
        longitude      = float(row["LONGITUDE"]),
        distance_km    = float(distances[idx]),
    )
    return asdict(rec)

if __name__ == "__main__":
    # Example: downtown Los Angeles
    result = find_closest_hospital(34.05, -118.25)
    print(json.dumps(result, indent=2))
