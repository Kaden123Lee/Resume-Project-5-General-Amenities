# nearest_fire_station.py
# Find the closest Public Safety record (Fire Stations, etc.) to a given lat/lon.

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import pandas as pd

__all__ = ["find_closest_fire_station"]


# -------------------------
# CSV + misc helpers
# -------------------------

def _read_csv_robust(path: Path | str) -> pd.DataFrame:
    """Read CSV with fallbacks (avoids low_memory+python engine issue)."""
    path = str(path)
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False, engine="c")
    except Exception:
        pass
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", engine="c")
    except Exception:
        pass
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(path, dtype=str)

def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_km = 6371.0088
    R_mi = R_km * 0.62137119
    import math as _m
    phi1, phi2 = _m.radians(lat1), _m.radians(lat2)
    dphi = phi2 - phi1
    dlambda = _m.radians(lon2 - lon1)
    a = _m.sin(dphi/2.0) ** 2 + _m.cos(phi1) * _m.cos(phi2) * _m.sin(dlambda/2.0) ** 2
    c = 2 * _m.atan2(_m.sqrt(a), _m.sqrt(1 - a))
    return R_mi * c

def _build_address(row: pd.Series) -> str:
    parts = []
    for col in ("Address Line 1", "Address Line 2", "City", "State", "ZIP Code"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    if not parts and "Name" in row and pd.notna(row["Name"]):
        parts.append(str(row["Name"]).strip())
    return ", ".join(parts)


# -------------------------
# Optional: convert projected X/Y → WGS84 lat/lon
# -------------------------

def _xy_to_latlon_columns(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Tuple[str, str]]:
    """
    Try converting projected State Plane coords (feet) to WGS84.
    This dataset's X/Y look like Los Angeles County State Plane CA Zone V (US ft).
    We'll attempt EPSG:2229 (and ESRI:102645) if pyproj is available.

    Returns new (lat_col, lon_col) names if successful; otherwise None.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None

    try:
        from pyproj import Transformer
    except Exception:
        return None  # pyproj not available

    # Candidates: (EPSG authority, code)
    candidates = ["EPSG:2229", "ESRI:102645"]  # CA Zone V (US ft) variants
    x = _to_float(df[x_col])
    y = _to_float(df[y_col])
    mask = x.notna() & y.notna()
    if not mask.any():
        return None

    sample = df.loc[mask, [x_col, y_col]].head(1000)
    for source_crs in candidates:
        try:
            tr = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
            lon_s, lat_s = tr.transform(sample[x_col].astype(float).values,
                                        sample[y_col].astype(float).values)
            # Plausibility check for LA area
            import numpy as np
            ok = (lat_s > 32.0) & (lat_s < 35.5) & (lon_s < -117.0) & (lon_s > -119.9)
            if np.mean(ok) >= 0.75:  # 75%+ plausible
                # Apply to full column
                lon_all, lat_all = tr.transform(x.astype(float).values, y.astype(float).values)
                df["_lat_from_xy"] = pd.Series(lat_all, index=df.index)
                df["_lon_from_xy"] = pd.Series(lon_all, index=df.index)
                return ("_lat_from_xy", "_lon_from_xy")
        except Exception:
            continue

    return None


# -------------------------
# Public API
# -------------------------

def find_closest_fire_station(
    lat: float,
    lon: float,
    data_path: str | Path,
    geocode_fn: Optional[Callable[[str], Any]] = None,
    prefer_xy: bool = True,
    max_rows: Optional[int] = None,
    return_all: bool = False,
) -> Dict[str, Any] | pd.DataFrame:
    """
    Return the closest Public Safety record (e.g., Fire Station) to (lat, lon).

    Parameters
    ----------
    lat, lon : float
        Query location.
    data_path : str | Path
        Path to the Public Safety CSV (Fire Stations etc.).
    geocode_fn : callable(address) -> {'lat': ..., 'lon': ...}, optional
        Used to fill in rows missing coordinates. Results are cached to
        "<data_path>-geocoded-cache.csv".
    prefer_xy : bool
        If True and no Latitude/Longitude columns exist, attempt to convert
        projected X/Y (State Plane CA Zone V ft) to WGS84 using pyproj.
        If that fails, fall back to geocoding (if provided).
    max_rows : int, optional
        Limit rows processed (useful for very large files).
    return_all : bool
        If True, return all rows with distance columns; else return nearest dict.
    """
    data_path = Path(data_path)
    df = _read_csv_robust(data_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # Identify columns
    name_col   = _pick_column(df, ("Name", "Label"))
    org_col    = _pick_column(df, ("Organization",))
    cat1_col   = _pick_column(df, ("Category1",))
    cat2_col   = _pick_column(df, ("Category2",))
    cat3_col   = _pick_column(df, ("Category3",))
    disp_col   = _pick_column(df, ("Display",))
    upd_col    = _pick_column(df, ("Last Update", "LastUpdate", "Updated"))
    src_col    = _pick_column(df, ("Source",))
    src_id_col = _pick_column(df, ("Source ID", "SourceID"))
    src_dt_col = _pick_column(df, ("Source Date", "SourceDate"))

    # Coordinates: prefer explicit lat/lon if present
    lat_col = _pick_column(df, ("Latitude", "Lat"))
    lon_col = _pick_column(df, ("Longitude", "Lon", "Long"))

    # If missing, optionally try X/Y → lat/lon
    if (lat_col is None or lon_col is None) and prefer_xy:
        x_col = _pick_column(df, ("x", "X"))
        y_col = _pick_column(df, ("y", "Y"))
        if x_col and y_col:
            xy_cols = _xy_to_latlon_columns(df, x_col, y_col)
            if xy_cols:
                lat_col, lon_col = xy_cols

    # Still missing? fall back to geocoding
    if lat_col is None or lon_col is None:
        if geocode_fn is None:
            raise ValueError(
                "No Latitude/Longitude columns found. "
                "Pass a geocode_fn (e.g., your geocode_address) or provide data with lat/lon."
            )
        # create missing columns
        lat_col = lat_col or "Latitude"
        lon_col = lon_col or "Longitude"
        if lat_col not in df.columns: df[lat_col] = None
        if lon_col not in df.columns: df[lon_col] = None

        # Geocode rows missing coords
        cache_path = data_path.with_suffix(data_path.suffix + "-geocoded-cache.csv")
        df = _geocode_missing(df, lat_col, lon_col, geocode_fn, cache_path=None)

    # Convert numeric + drop NAs
    df[lat_col] = _to_float(df[lat_col])
    df[lon_col] = _to_float(df[lon_col])
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        raise ValueError("No rows in the dataset have usable coordinates.")

    # Distances
    df["distance_miles"] = [
        _haversine_miles(float(lat), float(lon), float(row_lat), float(row_lon))
        for row_lat, row_lon in zip(df[lat_col], df[lon_col])
    ]
    df["distance_km"] = df["distance_miles"] * 1.609344

    df_sorted = df.sort_values("distance_miles", ascending=True).reset_index(drop=True)
    if return_all:
        return df_sorted

    top = df_sorted.iloc[0]
    def _get(c): return None if c is None else top.get(c)

    address = _build_address(top)
    result = {
        "name": _get(name_col),
        "address": address,
        "organization": _get(org_col),
        "category1": _get(cat1_col),
        "category2": _get(cat2_col),
        "category3": _get(cat3_col),
        "latitude": float(top[lat_col]),
        "longitude": float(top[lon_col]),
        "distance_miles": float(top["distance_miles"]),
        "distance_km": float(top["distance_km"]),
        "display_flag": _get(disp_col),
        "last_update": _get(upd_col),
        "source": _get(src_col),
        "source_id": _get(src_id_col),
        "source_date": _get(src_dt_col),
    }
    return result


# ---------- geocoding support (cached) ----------

def _geocode_missing(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    geocode_fn: Callable[[str], Any],
    cache_path: Optional[Path],
) -> pd.DataFrame:
    # Load cache (if any)
    cache = None
    if cache_path and Path(cache_path).is_file():
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
    missing_mask = df[lat_col].isna() | df[lon_col].isna()
    for idx, row in df.loc[missing_mask].iterrows():
        address = _build_address(row)
        if not address:
            continue
        if address in addr_to_cached:
            la, lo = addr_to_cached[address]
        else:
            try:
                result = geocode_fn(address)
            except TypeError:
                result = geocode_fn(address=address)
            if not result:
                continue
            la = float(result.get("lat"))
            lo = float(result.get("lon"))
            addr_to_cached[address] = (la, lo)
            updated_rows.append({"address": address, "lat": la, "lon": lo})
        df.at[idx, lat_col] = la
        df.at[idx, lon_col] = lo

    if cache_path and updated_rows:
        new_cache = pd.concat([cache, pd.DataFrame(updated_rows)], ignore_index=True)
        new_cache = new_cache.drop_duplicates(subset=["address"], keep="first")
        new_cache.to_csv(cache_path, index=False)

    return df
