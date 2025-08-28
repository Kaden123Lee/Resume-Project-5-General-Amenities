# nearest_transportation.py
# Find the closest Transportation record (Amtrak stations, etc.) to a given lat/lon.

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import pandas as pd

__all__ = ["find_closest_transport"]


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
    LA County 'x,y' are typically State Plane CA Zone V (US ft).
    Attempt EPSG:2229 (and ESRI:102645) if pyproj is available.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None
    try:
        from pyproj import Transformer
    except Exception:
        return None  # pyproj not available

    candidates = ["EPSG:2229", "ESRI:102645"]
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
            import numpy as np
            ok = (lat_s > 32.0) & (lat_s < 35.5) & (lon_s < -117.0) & (lon_s > -119.9)
            if np.mean(ok) >= 0.75:
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

def find_closest_transport(
    lat: float,
    lon: float,
    data_path: str | Path,
    geocode_fn: Optional[Callable[[str], Any]] = None,
    prefer_xy: bool = True,
    max_rows: Optional[int] = None,
    return_all: bool = False,
) -> Dict[str, Any] | pd.DataFrame:
    """
    Return the closest Transportation record (e.g., Amtrak station) to (lat, lon).

    Parameters
    ----------
    lat, lon : float
        Query location.
    data_path : str | Path
        Path to the Transportation CSV.
    geocode_fn : callable(address) -> {'lat': ..., 'lon': ...}, optional
        Used to fill in rows missing coordinates. You can pass your geocode_address().
    prefer_xy : bool
        If True and no Latitude/Longitude columns exist, attempt to convert
        projected X/Y (State Plane CA Zone V ft) to WGS84 using pyproj.
        If that fails, fall back to geocoding (if provided).
    max_rows : int, optional
        Limit rows processed.
    return_all : bool
        If True, return the full DataFrame (sorted) with distance columns.
    """
    data_path = Path(data_path)
    df = _read_csv_robust(data_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # Column discovery
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

    # Coordinates first choice
    lat_col = _pick_column(df, ("Latitude", "Lat"))
    lon_col = _pick_column(df, ("Longitude", "Lon", "Long"))

    # If missing, try X/Y
    if (lat_col is None or lon_col is None) and prefer_xy:
        x_col = _pick_column(df, ("x", "X"))
        y_col = _pick_column(df, ("y", "Y"))
        if x_col and y_col:
            xy_cols = _xy_to_latlon_columns(df, x_col, y_col)
            if xy_cols:
                lat_col, lon_col = xy_cols

    # If still missing, optionally geocode each row
    if lat_col is None or lon_col is None:
        if geocode_fn is None:
            raise ValueError(
                "No Latitude/Longitude columns found. "
                "Pass a geocode_fn (e.g., your geocode_address) or provide data with lat/lon."
            )
        lat_col = lat_col or "Latitude"
        lon_col = lon_col or "Longitude"
        if lat_col not in df.columns: df[lat_col] = None
        if lon_col not in df.columns: df[lon_col] = None

        # Geocode rows lacking coords (no cache here; stations set is small)
        missing_mask = df[lat_col].isna() | df[lon_col].isna()
        for idx, row in df.loc[missing_mask].iterrows():
            address = _build_address(row)
            if not address:
                continue
            try:
                res = geocode_fn(address)
            except TypeError:
                res = geocode_fn(address=address)
            if not res:
                continue
            df.at[idx, lat_col] = float(res.get("lat"))
            df.at[idx, lon_col] = float(res.get("lon"))

    # Clean
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

    result = {
        "name": _get(name_col),
        "address": _build_address(top),
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
