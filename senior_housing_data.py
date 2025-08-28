import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os

# Path to the Senior Housing CSV - UPDATED PATH
DATA_PATH = r"datasets\Senior_Housing.csv"

# Cache DataFrame
_df_cache = None


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute distance (km) between two (lat, lon) points using Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_senior_housing(data_path=DATA_PATH):
    """
    Load and cache the senior housing dataset.
    """
    global _df_cache
    if _df_cache is None:
        # Check if file exists before trying to read it
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        
        df = pd.read_csv(data_path, low_memory=False)

        # Ensure Lat / Lon columns exist
        if not {"Lat", "Lon"}.issubset(df.columns):
            raise ValueError("CSV missing required 'Lat' and 'Lon' columns")

        _df_cache = df
    return _df_cache


def find_closest_senior_housing(lat, lon, data_path=DATA_PATH):
    """
    Find the closest senior housing facility to a given (lat, lon).
    """
    df = load_senior_housing(data_path).copy()

    # Compute distance for each row
    df["distance_km"] = df.apply(
        lambda row: haversine(lat, lon, row["Lat"], row["Lon"]), axis=1
    )

    nearest = df.loc[df["distance_km"].idxmin()]

    return {
        "name": nearest.get("Name"),
        "address": nearest.get("Address"),
        "city": nearest.get("City"),
        "state": nearest.get("State"),
        "zip": nearest.get("Zipcode"),
        "latitude": nearest.get("Lat"),
        "longitude": nearest.get("Lon"),
        "distance_km": nearest.get("distance_km"),
    }


def get_all_senior_housing():
    """
    Get all senior housing facilities with their details.
    """
    df = load_senior_housing()
    return df.to_dict('records')