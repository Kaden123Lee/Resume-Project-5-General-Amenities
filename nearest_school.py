"""
nearest_school.py
------------------
Find the closest school (row) in a Los Angeles County Education dataset
given a latitude/longitude.

Expected columns (case-insensitive):
  - Name (or Label)
  - Category1, Category2, Category3
  - Address Line 1, Address Line 2, City, State, ZIP Code
  - Organization
  - Enrollment (optional)
  - Latitude, Longitude  (preferred; uses these directly)
  - Source, Source ID, Source Date (optional)

Notes:
  * We intentionally DO NOT use the "x"/"y" columns in these datasets,
    because they are projected coordinates (not lon/lat).
  * If some rows are missing lat/lon and you provide a geocode_fn(address)->{lat, lon},
    those rows can be geocoded on the fly and cached in "<data_path>-geocoded-cache.csv".
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import pandas as pd

__all__ = ["find_closest_school"]


# -------------------------
# Helpers
# -------------------------

def _read_csv_robust(path: Path | str) -> pd.DataFrame:
    """Read CSV with fallbacks to avoid pandas engine/low_memory issues."""
    path = str(path)
    # Try fast C engine first with low_memory disabled for consistent dtypes
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False, engine="c")
    except Exception:
        pass
    # Retry C engine without low_memory in case of older pandas
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", engine="c")
    except Exception:
        pass
    # Fallback to Python engine (cannot use low_memory here)
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", engine="python")
    except Exception:
        # Last resort: let pandas decide
        return pd.read_csv(path, dtype=str)


def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first existing column (case-insensitive match) from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    R_km = 6371.0088
    R_mi = R_km * 0.62137119
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_mi * c


def _build_address(row: pd.Series) -> str:
    parts = []
    for col in ("Address Line 1", "Address Line 2", "City", "State", "ZIP Code"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    # If everything is missing, fall back to Name
    if not parts and "Name" in row and pd.notna(row["Name"]):
        parts.append(str(row["Name"]).strip())
    return ", ".join(parts)


def _geocode_missing(df: pd.DataFrame, lat_col: str, lon_col: str, geocode_fn: Callable[[str], Any], cache_path: Path) -> pd.DataFrame:
    """Geocode rows with missing lat/lon using provided geocode function and cache results."""
    cache = None
    if cache_path.is_file():
        try:
            cache = _read_csv_robust(cache_path)
        except Exception:
            cache = None
    if cache is None:
        cache = pd.DataFrame(columns=["address", "lat", "lon"])
    else:
        cache["address"] = cache["address"].astype(str)
    
    addr_to_cached = {
        a: (float(la), float(lo))
        for a, la, lo in zip(cache.get("address", []), cache.get("lat", []), cache.get("lon", []))
        if pd.notna(a) and pd.notna(la) and pd.notna(lo)
    }
    
    updated_rows = []
    for idx, row in df.loc[df[lat_col].isna() | df[lon_col].isna()].iterrows():
        address = _build_address(row)
        if not address:
            continue
        if address in addr_to_cached:
            la, lo = addr_to_cached[address]
        else:
            try:
                result = geocode_fn(address)
            except TypeError:
                # Support geocode_fn(address: str) -> {lat, lon}
                result = geocode_fn(address=address)
            if not result:
                continue
            la = float(result.get("lat"))
            lo = float(result.get("lon"))
            # Update cache
            addr_to_cached[address] = (la, lo)
            updated_rows.append({"address": address, "lat": la, "lon": lo})
        df.at[idx, lat_col] = la
        df.at[idx, lon_col] = lo
    
    if updated_rows:
        new_cache = pd.concat([cache, pd.DataFrame(updated_rows)], ignore_index=True)
        # Drop duplicates keeping first
        new_cache = new_cache.drop_duplicates(subset=["address"], keep="first")
        new_cache.to_csv(cache_path, index=False)
    
    return df


# -------------------------
# Public API
# -------------------------

def find_closest_school(
    lat: float,
    lon: float,
    data_path: str | Path,
    geocode_fn: Optional[Callable[[str], Any]] = None,
    max_rows: Optional[int] = None,
    return_all: bool = False
) -> Dict[str, Any] | pd.DataFrame:
    """Return the closest school to (lat, lon) from the dataset at data_path.
    
    Parameters
    ----------
    lat, lon : float
        Location to search from.
    data_path : str | Path
        Path to the Education CSV file.
    geocode_fn : callable(address) -> {'lat': ..., 'lon': ...}, optional
        If provided, rows missing coordinates will be geocoded and cached.
    max_rows : int, optional
        If set, limit the number of rows processed (useful for very large files).
    return_all : bool
        If True, return a DataFrame of all rows with a 'distance_miles' column.
        Otherwise return a dict describing the single nearest school.
    """
    data_path = Path(data_path)
    df = _read_csv_robust(data_path)
    if max_rows is not None:
        df = df.head(max_rows)
    
    # Identify columns
    name_col = _pick_column(df, ("Name", "Label"))
    org_col = _pick_column(df, ("Organization",))
    cat1_col = _pick_column(df, ("Category1",))
    cat2_col = _pick_column(df, ("Category2",))
    cat3_col = _pick_column(df, ("Category3",))
    enroll_col = _pick_column(df, ("Enrollment",))
    src_col = _pick_column(df, ("Source",))
    src_id_col = _pick_column(df, ("Source ID",))
    src_date_col = _pick_column(df, ("Source Date",))
    
    # Latitude/Longitude columns (avoid x/y projected coords)
    lat_col = _pick_column(df, ("Latitude", "Lat"))
    lon_col = _pick_column(df, ("Longitude", "Lon", "Long"))
    
    # If no lat/lon columns, we cannot compute distances (unless we geocode every row)
    if lat_col is None or lon_col is None:
        if geocode_fn is None:
            raise ValueError("This dataset has no latitude/longitude columns. Provide 'geocode_fn' to geocode rows.")
        # Create empty lat/lon columns for geocoding
        lat_col = lat_col or "Latitude"
        lon_col = lon_col or "Longitude"
        if lat_col not in df.columns: df[lat_col] = None
        if lon_col not in df.columns: df[lon_col] = None
    
    # Convert numeric
    df[lat_col] = _to_float(df[lat_col])
    df[lon_col] = _to_float(df[lon_col])
    
    # Geocode missing if a callback is provided
    if geocode_fn is not None and (df[lat_col].isna().any() or df[lon_col].isna().any()):
        cache_path = data_path.with_suffix(data_path.suffix + "-geocoded-cache.csv")
        df = _geocode_missing(df, lat_col, lon_col, geocode_fn, cache_path)
        # Ensure numeric after geocode
        df[lat_col] = _to_float(df[lat_col])
        df[lon_col] = _to_float(df[lon_col])
    
    # Drop rows with missing coordinates
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        raise ValueError("No rows in the dataset have usable latitude/longitude.")
    
    # Compute distance
    df["distance_miles"] = [
        _haversine_miles(float(lat), float(lon), float(row_lat), float(row_lon))
        for row_lat, row_lon in zip(df[lat_col], df[lon_col])
    ]
    df["distance_km"] = df["distance_miles"] * 1.609344
    
    # Sort and pick
    df_sorted = df.sort_values("distance_miles", ascending=True).reset_index(drop=True)
    if return_all:
        return df_sorted
    
    top = df_sorted.iloc[0]
    def _get(c):
        return None if c is None else top.get(c)
    
    address = _build_address(top)
    
    # enrollment may be missing / non-numeric
    enrollment_val = _get(enroll_col)
    try:
        enrollment_val = int(float(enrollment_val)) if enrollment_val not in (None, "", "nan") else None
    except Exception:
        enrollment_val = None
    
    result = {
        "name": _get(name_col),
        "address": address,
        "organization": _get(org_col),
        "category1": _get(cat1_col),
        "category2": _get(cat2_col),
        "category3": _get(cat3_col),
        "enrollment": enrollment_val,
        "latitude": float(top[lat_col]),
        "longitude": float(top[lon_col]),
        "distance_miles": float(top["distance_miles"]),
        "distance_km": float(top["distance_km"]),
        "source": _get(src_col),
        "source_id": _get(src_id_col),
        "source_date": _get(src_date_col),
    }
    return result


if __name__ == "__main__":
    # Tiny smoke test (replace with your own path & coordinates)
    import json
    example_path = Path("raw_data/Education.csv")
    if example_path.exists():
        try:
            nearest = find_closest_school(34.0353, -118.2624, example_path)
            print(json.dumps(nearest, indent=2))
        except Exception as e:
            print("Error:", e)
    else:
        print("Tip: Place your Education CSV at", example_path)
