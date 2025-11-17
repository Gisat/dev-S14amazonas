import xarray as xr
import pandas as pd
from osgeo import osr, ogr
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Tuple, Optional
import requests
from shapely.geometry import shape

import time
import numpy as np
from scipy.stats import ttest_ind_from_stats
from copy import copy
import logging

DEBUG = False
APPLY_SIEVE_FILTER = True

logger = logging.getLogger(__name__)

# -------------------------
# Config / Constants
# -------------------------
S1_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel1/search.json"
DATE_RE = re.compile(r'_(\d{8})T\d{6}_')  # e.g., ..._20211201T091308_...

# -------------------------
# Datacube utils
# -------------------------
def get_spatial_extent(spatial_extent) -> dict:

    """Get spatial bounds in WGS84."""
    # x_coord = arr.coords['x'].values
    # y_coord = arr.coords['y'].values
    #
    # west, east = float(x_coord.min()), float(x_coord.max())
    # south, north = float(y_coord.min()), float(y_coord.max())
    west, east, south, north = spatial_extent["west"], spatial_extent["east"], spatial_extent["south"], spatial_extent["north"]
    source_epsg = spatial_extent.get("crs", "EPSG:4326").split(":")[-1]
    # ------------------------------------
    # Build polygon from bbox (in source CRS)
    # ------------------------------------
    if int(source_epsg) != 4326:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(west, south)
        ring.AddPoint(east, south)
        ring.AddPoint(east, north)
        ring.AddPoint(west, north)
        ring.AddPoint(west, south)  # close ring

        geom = ogr.Geometry(ogr.wkbPolygon)
        geom.AddGeometry(ring)
        geom = geom.Clone()  # 2D is fine here

        geom_wkt = geom.ExportToWkt()
        print(geom_wkt)

        # ------------------------------------
        # Define CRS and transformation
        # ------------------------------------
        source_epsg = int(source_epsg)  # e.g. 3857 or 32633; must be int
        CATALOG_EPSG = 4326

        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(source_epsg)

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(CATALOG_EPSG)

        # Assign SRS to geometry (good practice)
        geom.AssignSpatialReference(src_srs)

        trans_to_catalog = osr.CoordinateTransformation(src_srs, dst_srs)

        # Sanity check (optional)
        if trans_to_catalog is None:
            raise RuntimeError("Failed to create CoordinateTransformation")

        # ------------------------------------
        # Transform and get envelope
        # ------------------------------------
        catalog_aoi_geom = geom.Clone()
        catalog_aoi_geom.Transform(trans_to_catalog)

        west, east, south, north = catalog_aoi_geom.GetEnvelope()

    return {'west': west, 'east': east, 'south': south, 'north': north}, [south, west, north, east]

def get_temporal_extent(arr: xr.DataArray) -> dict:
    """Get temporal extent from time dimension."""
    time_dim = 't'
    if 't' in arr.dims:
        times = arr.coords[time_dim].values
        times = pd.to_datetime(times).to_pydatetime()
        start = pd.to_datetime(times.min())
        end = pd.to_datetime(times.max())
        log_string = '\n'.join([f'{i} {v}' for i, v in enumerate(times)])
        return {'start': start, 'end': end, 'times': times}



# -------------------------
# Utilities
# -------------------------
def parse_date_from_title(title: str) -> Optional[datetime]:
    """Extract YYYYMMDD as datetime from a Sentinel-1 title. Return None if not found."""
    m = DATE_RE.search(title)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")


def intersection_ratio_bbox2_in_bbox1(b1: List[float], b2: List[float]) -> float:
    """
    Fraction of bbox2's area inside bbox1. bboxes are [minx, miny, maxx, maxy] in lon/lat.
    """
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
        return 0.0

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    return 0.0 if area2 <= 0 else inter_area / area2


def get_intervals(start_dt: str, end_dt: str, acq_frequency: int = 6) -> List[Tuple[datetime, datetime]]:
    """
    Build phase-based intervals between start_date and end_date (inclusive of the interval containing end_date).
    Phases:
      - Phase 1: 2015-04-28 .. 2021-12-16 step=acq_frequency
      - Phase 2: 2021-12-16 .. 2025-03-30 step=acq_frequency*2
      - Phase 3: 2025-03-30 .. end step=acq_frequency
    """
    # start_dt = start_date.to_pydatetime()
    # end_dt = end_date.to_pydatetime()
    MASTER_START = datetime(2015, 4, 28)
    PHASE1_END = datetime(2021, 12, 16)
    PHASE2_END = datetime(2025, 3, 30)
    GEN_END = end_dt + timedelta(days=25)

    points = []
    cur = MASTER_START
    while cur < PHASE1_END:
        points.append(cur)
        cur += timedelta(days=acq_frequency)

    cur = PHASE1_END
    while cur < PHASE2_END:
        points.append(cur)
        cur += timedelta(days=acq_frequency * 2)

    cur = PHASE2_END
    while cur < GEN_END:
        points.append(cur)
        cur += timedelta(days=acq_frequency)

    intervals = list(zip(points[:-1], points[1:]))
    return [(s, e)
            for (s, e) in intervals
            if (s >= start_dt and e <= end_dt)]


def create_timewindow_groups(intervals, window_size=10):
    out = {}
    for i in range(len(intervals) - window_size + 1):
        win = intervals[i:i + window_size]
        key = win[5][0]  # start time of 6th interval
        out[key] = win
    return out


def fetch_s1_features(bbox: List[float], start_iso: str, end_iso: str) -> List[dict]:
    """
    Query the Copernicus Dataspace Resto API for Sentinel-1 features within bbox and datetime range.
    Only returns features in JSON 'features' list.
    """
    params = {
        "box": ",".join(map(str, bbox)),
        "page": 1,
        "maxRecords": 1000,
        "status": "ONLINE",
        "dataset": "ESA-DATASET",
        "processingLevel": "LEVEL1",
        "productType": "IW_GRDH_1S-COG",
        "startDate": f"{start_iso.strftime('%Y-%m-%dT00:00:00Z')}",
        "completionDate": f"{end_iso.strftime('%Y-%m-%dT23:59:59.999999999Z')}",
    }
    r = requests.get(S1_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    logger.info(f"s1query {r.url}")
    return r.json().get("features", [])


def build_index_by_date_orbit(features: List[dict]) -> Dict[Tuple[datetime, str], List[dict]]:
    """
    Keep only GRDH (non-CARD_BS, non-COG) scenes and index them by (date, orbitDirection),
    ordered chronologically by date.
    """
    temp = defaultdict(list)
    for ft in features:
        props = ft.get("properties", {})
        title = props.get("title", "")
        if "_GRDH_" in title and "_CARD_BS" not in title: # and "_COG." not in title:
            dt = parse_date_from_title(title)
            if dt is None:
                continue
            orbit = props.get("orbitDirection", "UNKNOWN")
            temp[(dt, orbit)].append(ft)

    # Sort keys by datetime
    ordered = OrderedDict(sorted(temp.items(), key=lambda kv: kv[0][0]))
    return ordered

def filter_index_to_dates(index_do: Dict[Tuple[datetime, str], List[dict]],
                          dt_list: List[datetime]) -> Dict[Tuple[datetime, str], List[dict]]:
    """Keep only entries whose date is in dt_list (exact date match)."""
    dates = set(dt_list)
    return {k: v for k, v in index_do.items() if k[0] in dates}


def filter_index_by_orbit(index_do, selected_orbit):
    """
    Filter (date, orbit) → [features] dictionary to keep only the chosen orbit direction.
    Returns a new dict preserving chronological order.
    """
    if not selected_orbit:
        return index_do  # no orbit chosen, keep all

    return dict(
        (k, v) for k, v in index_do.items() if k[1] == selected_orbit
    )


def pick_orbit_direction(index_do: Dict[Tuple[datetime, str], List[dict]],
                         aoi_bbox: List[float]) -> Optional[str]:
    """
    Choose orbit direction deterministically:
      1) If only one orbit exists → return it.
      2) Else compare counts per orbit → choose higher count.
      3) If tie, compute each orbit's minimum (worst-case) fraction of scene-bbox inside AOI → prefer higher.
      4) If still tied → return None.
    """
    if not index_do:
        return None

    orbits = [k[1] for k in index_do.keys()]
    counts = Counter(orbits)
    if len(counts) == 1:
        return next(iter(counts))

    # Step 2: counts
    top_count = max(counts.values())
    leaders = [o for o, c in counts.items() if c == top_count]
    if len(leaders) == 1:
        return leaders[0]

    # Step 3: tie-break by worst-case overlap
    # For each orbit, compute the MIN overlap ratio across its scenes; pick the orbit with higher MIN
    min_overlap_by_orbit = {}
    for orbit in leaders:
        min_ratio = float("inf")
        for (dt, ob), fts in index_do.items():
            if ob != orbit:
                continue
            for ft in fts:
                try:
                    # Use feature geometry bounds as scene bbox
                    scene_bbox = list(shape(ft["geometry"]).bounds)
                except Exception:
                    continue
                ratio = intersection_ratio_bbox2_in_bbox1(scene_bbox, aoi_bbox)
                if ratio < min_ratio:
                    min_ratio = ratio
        if min_ratio == float("inf"):
            min_ratio = 0.0
        min_overlap_by_orbit[orbit] = min_ratio

    # Compare worst-case overlap; if tie, return None
    max_min_overlap = max(min_overlap_by_orbit.values())
    overlap_leaders = [o for o, r in min_overlap_by_orbit.items() if r == max_min_overlap]
    return overlap_leaders[0] if len(overlap_leaders) == 1 else None


def get_scene_indices(index_do: Dict[Tuple[datetime, str], List[dict]],
                      feature_names: List[str]) -> List[Tuple[int, str]]:
    """
    Returns indices & filenames from feature_names that match ANY date present in the index.
    """
    dates_yymmdd = {k[0].strftime("%Y%m%d") for k in index_do.keys()}
    matched = []
    for i, name in enumerate(feature_names):
        if any(d in name for d in dates_yymmdd):
            matched.append((i, name))
    return matched


#################################################################################
##################### BACKSCATTER CHANGE DETECTION ##############################
#################################################################################

def _as_dtype(a, dtype):
    return a if (a.dtype == dtype and a.flags['C_CONTIGUOUS']) else np.ascontiguousarray(a, dtype=dtype)


def _nanmean_along(a, axis, mask=None):
    """Numerically stable nanmean with float64 accumulation."""
    if mask is None:
        mask = np.isfinite(a)
    sums = np.where(mask, a, 0).sum(axis=axis, dtype=np.float64)
    counts = mask.sum(axis=axis, dtype=np.int64).astype(np.float64)
    out = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)
    return out, counts


def create_nan_mask(numpy_array, vv_vh_bandcount):
    """Return combined invalid mask (VV or VH invalid). Expects (T,H,W)."""
    mask_vv = np.isnan(numpy_array[:vv_vh_bandcount]) | (numpy_array[:vv_vh_bandcount] < -99)
    mask_vh = np.isnan(numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount]) | (
            numpy_array[vv_vh_bandcount:2 * vv_vh_bandcount] < -99
    )
    return mask_vv | mask_vh


def ttest_from_stats(
    past,
    future,
    min_valid_per_window=3,
    valid_mask=None,
    compute_dtype=np.float32,
    alternative="greater"
):
    """
    Column-wise Welch t-test (one-sided: past > future) using summary stats.
    NaNs are ignored. Outputs:
      t_stat_full  : (H, W) compute_dtype
      p_one_full   : (H, W) compute_dtype
      insufficient_full : (H, W) bool  (True where either group has < min_valid samples)

    Parameters
    ----------
    past, future : np.ndarray, shape (T, H, W)
        Time-series stacks for the two windows being compared.
    min_valid_per_window : int
        Minimum non-NaN samples required in each window per pixel.
    valid_mask : np.ndarray | None, shape (H, W)
        Optional boolean mask of pixels to evaluate.
    compute_dtype : np.dtype
        Dtype for returned t and p arrays (e.g., np.float32).
    alternative : {"greater", "less"}
        "greater" : H1 is mean(past) > mean(future)  (decrease in future)
        "less"    : H1 is mean(past) < mean(future)  (increase in future)
    """
    assert past.shape == future.shape, "past and future must have same shape (T,H,W)"
    if alternative not in ("greater", "less"):
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")

    T, H, W = past.shape
    start_time = time.time()

    # Prepare outputs
    t_full  = np.zeros((H, W), dtype=compute_dtype)
    p_full  = np.ones((H, W),  dtype=compute_dtype)
    ins_full = np.ones((H, W), dtype=bool)

    # Valid pixel mask
    vm = np.ones((H, W), dtype=bool) if valid_mask is None else valid_mask
    if not np.any(vm):
        return t_full, p_full, ins_full

    # Views as (T, N) and select only valid pixels (to reduce work/memory)
    N = H * W
    idx = vm.ravel()
    past_v   = past.reshape(T, N)[:, idx]
    future_v = future.reshape(T, N)[:, idx]

    # Counts (ignore NaNs)
    n1 = np.sum(np.isfinite(past_v),   axis=0).astype(np.int32, copy=False)
    n2 = np.sum(np.isfinite(future_v), axis=0).astype(np.int32, copy=False)

    # Pixels that have enough samples in BOTH groups
    test_idx = (n1 >= min_valid_per_window) & (n2 >= min_valid_per_window)

    out_t = np.zeros(idx.sum(), dtype=np.float64)
    out_p = np.ones(idx.sum(),  dtype=np.float64)
    insufficient_v = ~test_idx

    if np.any(test_idx):
        # Summary stats in float64 for stability
        # Sample mean and sample variance (ddof=1)
        m1 = np.nanmean(past_v[:,  test_idx], axis=0).astype(np.float64, copy=False)
        m2 = np.nanmean(future_v[:, test_idx], axis=0).astype(np.float64, copy=False)
        v1 = np.nanvar(past_v[:,   test_idx], axis=0, ddof=1).astype(np.float64, copy=False)
        v2 = np.nanvar(future_v[:,  test_idx], axis=0, ddof=1).astype(np.float64, copy=False)

        # Convert variance -> std (ttest_ind_from_stats expects std)
        s1 = np.sqrt(np.maximum(v1, 0.0))
        s2 = np.sqrt(np.maximum(v2, 0.0))

        nn1 = n1[test_idx].astype(np.float64, copy=False)
        nn2 = n2[test_idx].astype(np.float64, copy=False)

        try:
            # SciPy >= 1.9: supports one-sided alternative
            t_v, p_v = ttest_ind_from_stats(
                m1, s1, nn1,
                m2, s2, nn2,
                equal_var=False,
                alternative=alternative
            )
        except TypeError:
            # Older SciPy fallback: compute two-sided, convert to one-sided "greater"
            t_v, p2_v = ttest_ind_from_stats(
                m1, s1, nn1,
                m2, s2, nn2,
                equal_var=False
            )
            # Ensure array types
            t_v  = np.asarray(t_v,  dtype=np.float64)
            p2_v = np.asarray(p2_v, dtype=np.float64)

            p_v = np.ones_like(p2_v, dtype=np.float64)
            good = np.isfinite(t_v) & np.isfinite(p2_v)

            if alternative == "greater":
                pos = good & (t_v > 0)
                # For t > 0, one-sided p = two-sided p / 2
                p_v[pos] = p2_v[pos] / 2.0
                # For t <= 0, one-sided p = 1 - (two-sided p / 2)
                p_v[good & ~pos] = 1.0 - (p2_v[good & ~pos] / 2.0)
            else:
                # H1: mean(past) < mean(future)
                neg = good & (t_v < 0)
                p_v[neg] = p2_v[neg] / 2.0
                p_v[good & ~neg] = 1.0 - (p2_v[good & ~neg] / 2.0)

        # Sanitize NaNs/Infs in t
        t_v = np.asarray(t_v, dtype=np.float64)
        p_v = np.asarray(p_v, dtype=np.float64)
        bad = ~np.isfinite(t_v)
        if np.any(bad):
            t_v[bad] = 0.0

        out_t[test_idx] = t_v
        out_p[test_idx] = p_v

    # Scatter back to (H, W)
    t_full.ravel()[idx]  = out_t.astype(compute_dtype, copy=False)
    p_full.ravel()[idx]  = out_p.astype(compute_dtype, copy=False)
    ins_full.ravel()[idx] = insufficient_v
    return t_full, p_full, ins_full

def apply_threshold(stat_array, pol_item, DEC_array_threshold,
                    stat_item_name=None, previous_stat_array_bool=None):
    """
    Convert a stat array into a binary mask and accumulate into DEC. Dead pixels should be masked upstream.
    """

    stat_array_copy = copy(stat_array)

    if stat_item_name == 'std':
        pol_thr = 2.0 if pol_item == "VH" else 1.5
        stat_array = np.where(np.isnan(stat_array) | (stat_array < pol_thr), 0, 1)

    elif stat_item_name == 'mean_change':
        # Looking for decreases (future - past <= threshold)
        pol_thr = -1.75
        stat_array = np.where(np.isnan(stat_array) | (stat_array > pol_thr), 0, 1)

    elif stat_item_name == 'pval':
        t_stat = stat_array[0]
        p_val = stat_array[1]
        insufficient = stat_array[2].astype(bool)
        pvalue_thr = 0.05

        is_significant = (p_val < pvalue_thr) & (~insufficient)
        stat_array = is_significant.astype(np.uint8)

        if previous_stat_array_bool is not None:
            stat_array[~previous_stat_array_bool.astype(bool)] = 0

    elif stat_item_name == 'ratio_slope':
        slope = stat_array[0]
        r2 = stat_array[1]
        insufficient = stat_array[2].astype(bool)

        slope_mask = np.isfinite(slope) & (slope >= 0.025) & ~insufficient
        r2_mask    = np.isfinite(r2)    & (r2 >= 0.60)     & ~insufficient
        stat_array = (slope_mask & r2_mask).astype(np.uint8)

    elif stat_item_name == 'ratio_mean_change':
        stat_thr = 2.0
        stat_array = np.where(np.isnan(stat_array) | (stat_array < stat_thr), 0, 1)

    DEC_array_threshold += stat_array.astype(int)
    if DEBUG:
        return DEC_array_threshold, stat_array_copy, stat_array.astype(int)
    else:
        return DEC_array_threshold, None, stat_array.astype(int)


def calculate_lsfit_r(vv_vh_r, min_valid=3, center_time=True, valid_mask=None, compute_dtype=np.float32):
    T, H, W = vv_vh_r.shape
    x = np.arange(T, dtype=np.float64)  # keep time in float64
    if center_time:
        x = x - x.mean()

    slope_full = np.full((H, W), np.nan, dtype=compute_dtype)
    r2_full = np.full((H, W), np.nan, dtype=compute_dtype)
    insufficient_full = np.ones((H, W), dtype=bool)

    vm = np.ones((H, W), dtype=bool) if valid_mask is None else valid_mask
    if not np.any(vm):
        return slope_full, r2_full, insufficient_full

    vv_vh_r = _as_dtype(vv_vh_r, compute_dtype)

    idx = vm.ravel()
    y = vv_vh_r.reshape(T, -1)[:, idx]  # (T,N) compute_dtype
    mask = np.isfinite(y)
    n = mask.sum(axis=0)
    insufficient = n < min_valid

    # means with float64 accumulation
    y_mean, _ = _nanmean_along(y.astype(np.float64, copy=False), axis=0, mask=mask)

    x2 = x[:, None]  # (T,1), float64
    y_center = np.where(mask, y.astype(np.float64, copy=False) - y_mean[None, :], 0.0)

    Sxx = ((x2 ** 2) * mask).sum(axis=0, dtype=np.float64)
    Sxy = (x2 * y_center).sum(axis=0, dtype=np.float64)

    with np.errstate(invalid='ignore', divide='ignore'):
        slope64 = np.divide(Sxy, Sxx, out=np.zeros_like(Sxy), where=Sxx > 0)

    yhat = slope64[None, :] * x2 + y_mean[None, :]
    resid = np.where(mask, y.astype(np.float64, copy=False) - yhat, 0.0)

    SSE = (resid ** 2).sum(axis=0, dtype=np.float64)
    SST = (np.where(mask, (y.astype(np.float64, copy=False) - y_mean[None, :]) ** 2, 0.0)
           ).sum(axis=0, dtype=np.float64)

    r2_64 = np.zeros_like(SSE)
    good_sst = SST > 0
    r2_64[good_sst] = 1.0 - (SSE[good_sst] / SST[good_sst])

    slope64[insufficient] = np.nan
    r2_64[insufficient] = np.nan

    slope_full.reshape(-1)[idx] = slope64.astype(compute_dtype, copy=False)
    r2_full.reshape(-1)[idx] = r2_64.astype(compute_dtype, copy=False)
    insufficient_full.reshape(-1)[idx] = insufficient
    return slope_full, r2_full, insufficient_full


def _dead_mask_for_window(stack, past_len, future_len):
    """Pixels with all-NaN in past or in future (axis=0). stack: (T,H,W)."""
    past = stack[0:past_len, :, :]
    future = stack[past_len:past_len + future_len, :, :]
    past_dead = ~np.any(np.isfinite(past), axis=0)
    future_dead = ~np.any(np.isfinite(future), axis=0)
    return past_dead | future_dead


def apply_stat_datacube(numpy_stack_pol_dict, window_size=10, compute_dtype=np.float32):
    """
    Multi-criteria change detection with early skipping of dead pixels
    (all-NaN in past or future window). Dead pixels => DEC outputs = 0.
    Heavy ops (ttest, lsfit) are computed only on valid pixels.
    """
    start_time = time.time()
    VV = _as_dtype(numpy_stack_pol_dict["VV"], compute_dtype)
    VH = _as_dtype(numpy_stack_pol_dict["VH"], compute_dtype)

    bands, dim1, dim2 = numpy_stack_pol_dict["VV"].shape
    assert window_size % 2 == 0, "window_size must be even (split equally into past/future)"
    assert bands >= window_size, f"Need at least {window_size} time steps; got {bands}"
    past_len = future_len = window_size // 2

    R = VV - VH

    # Build dead masks per signal, then combine
    dead_vv = _dead_mask_for_window(VV, past_len, future_len)
    dead_vh = _dead_mask_for_window(VH, past_len, future_len)
    dead_r = _dead_mask_for_window(R, past_len, future_len)
    dead_any = dead_vv | dead_vh | dead_r  # (H,W)
    valid_mask = ~dead_any
    if not DEBUG: del dead_vv, dead_vh, dead_r

    # If everything is dead, return zeros fast
    if not np.any(valid_mask):
        DEC_array_threshold = np.zeros((dim1, dim2), dtype=np.int32)
        DEC_array_mask = np.zeros((dim1, dim2), dtype=np.uint8)
        return DEC_array_threshold, DEC_array_mask

    DEC_array_threshold = np.zeros((dim1, dim2), dtype=np.int32)

    # --- Per-pol statistics (VV and VH) ---
    for pol_item, stack in (("VV", VV), ("VH", VH)):

        # Restrict to window once (cheap) – nanmean/nanstd will be fast on masked arrays anyway
        past = stack[0:past_len, :, :]
        future = stack[past_len:past_len + future_len, :, :]
        used = stack[0:past_len + future_len, :, :]

        # means/std with float64 reductions
        Stack_p_MIN, _ = _nanmean_along(past.astype(np.float64, copy=False), axis=0)
        Stack_f_MIN, _ = _nanmean_along(future.astype(np.float64, copy=False), axis=0)
        POL_std = np.nanstd(used.astype(np.float64, copy=False), axis=0).astype(compute_dtype, copy=False)

        # Mask out dead pixels before thresholding so they never vote
        POL_std[dead_any] = np.nan
        DEC_array_threshold, _, _ = apply_threshold(
            POL_std, pol_item, DEC_array_threshold, stat_item_name="std"
        )
        if not DEBUG:
            del POL_std

        # Mean change (future - past)
        POL_mean_change = (Stack_f_MIN - Stack_p_MIN).astype(compute_dtype, copy=False)
        POL_mean_change[dead_any] = np.nan
        DEC_array_threshold, _, POL_mean_change_bool = apply_threshold(
            POL_mean_change, pol_item, DEC_array_threshold, stat_item_name="mean_change"
        )

        # Paired t-test (compute only on valid pixels)
        t_t, t_p, mask_sufficient = ttest_from_stats(
            past, future, min_valid_per_window=3, valid_mask=valid_mask, compute_dtype=compute_dtype,
            alternative="greater"
        )
        DEC_array_threshold, _, _ = apply_threshold(
            [t_t, t_p, mask_sufficient], pol_item, DEC_array_threshold,
            stat_item_name="pval", previous_stat_array_bool=POL_mean_change_bool)

    # --- Ratio-based stats (VV - VH) ---
    ratio = R
    # Linear trend only on valid pixels
    ratio_slope, ratio_r2, mask_sufficient_ratio = calculate_lsfit_r(
        ratio, valid_mask=valid_mask, compute_dtype=compute_dtype
    )
    DEC_array_threshold, _, _ = apply_threshold(
        [ratio_slope, ratio_r2, mask_sufficient_ratio],
        pol_item="RATIO",
        DEC_array_threshold=DEC_array_threshold,
        stat_item_name="ratio_slope"
    )
    if not DEBUG:
        del ratio_slope, ratio_r2

    # Mean change of ratio (future - past) with correct axis
    ratio_mean_change = (
            _nanmean_along(R[past_len:past_len + future_len].astype(np.float64, copy=False), axis=0)[0]
            - _nanmean_along(R[0:past_len].astype(np.float64, copy=False), axis=0)[0]
    ).astype(compute_dtype, copy=False)
    ratio_mean_change[dead_any] = np.nan
    DEC_array_threshold, _, ratio_mean_change_bool = apply_threshold(
        ratio_mean_change, pol_item="RATIO", DEC_array_threshold=DEC_array_threshold,
        stat_item_name="ratio_mean_change"
    )
    if not DEBUG: del ratio_mean_change

    # T-test on ratio (valid pixels only)
    t_t, t_p, mask_sufficient = ttest_from_stats(
        ratio[0:past_len, :, :], ratio[past_len:past_len + future_len, :, :],
        min_valid_per_window=3, valid_mask=valid_mask, compute_dtype=compute_dtype, alternative="less")
    DEC_array_threshold, _, _ = apply_threshold(
        [t_t, t_p, mask_sufficient], pol_item="RATIO", DEC_array_threshold=DEC_array_threshold,
        stat_item_name="pval", previous_stat_array_bool=ratio_mean_change_bool)

    # --- Final mask post-processing ---
    DEC_array_mask = np.zeros_like(DEC_array_threshold, dtype=np.uint8)
    DEC_array_mask[DEC_array_threshold > 4] = 1

    # Force dead pixels to zero in both outputs (per your requirement)
    DEC_array_threshold[dead_any] = 0
    DEC_array_mask[dead_any] = 0

    # Morphological hole-filling on the binary mask (only on valid area)
    # if np.any(DEC_array_mask) and APPLY_SIEVE_FILTER:
    #     logger.info(f"sieve filtering")
    #     inverted = np.logical_not(DEC_array_mask.astype(bool))
    #     processed = remove_small_holes(inverted, area_threshold=10)
    #     DEC_array_mask = np.logical_not(processed).astype(np.uint8)
    #     # Keep dead pixels at 0
    #     DEC_array_mask[dead_any] = 0

    return DEC_array_threshold, DEC_array_mask


def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Simple UDF: Check S1 observation frequency via STAC and aggregate temporally.
    """
    acq_frequency = 12
    arr = cube


    # Get temporal extent
    start_str_time = context["start_time"]
    end_str_time = context["end_time"]
    epsg_code = context["epsg"]
    spatial_extent = context["spatial_extent"]

    # Get spatial extent
    spatial_extent_4326, bbox_4326 = get_spatial_extent(spatial_extent)
    logger.info(f"Spatial extent in EPSG:{epsg_code}: {spatial_extent_4326} {bbox_4326}")

    start_time = datetime.strptime(start_str_time, "%Y-%m-%d")
    end_time = datetime.strptime(end_str_time, "%Y-%m-%d")

    temporal_extent = get_temporal_extent(arr)

    # 1) Fetch & build index
    feats = fetch_s1_features(bbox_4326, temporal_extent["start"], temporal_extent["end"])
    index_do = build_index_by_date_orbit(feats)

    template_array = np.zeros_like(arr[0, 0, :, :])

    # 2) Filter to your dates of interest
    logger.info(f"featuresdateorb: {index_do.keys()}")
    logger.info(f"filteringscenesusing: {temporal_extent['times']}")
    index_do = filter_index_to_dates(index_do, temporal_extent["times"])
    logger.info(f"AfterTimeFilter: {index_do.keys()}")
    # 3) Decide the orbit direction (or None if tie after tie-break)
    # selected_orbit = pick_orbit_direction(index_do, bbox)

    days_interval = get_intervals(start_time, end_time, acq_frequency)
    group_days_interval = create_timewindow_groups(days_interval)

    arr = 10 * np.log10(arr)

    # 4).
    DEC_array_combined = None
    entered_wininterval_loop = False
    DEC_array_list = []
    win_list = []
    logger.info(f"Processingtimewindows {len(group_days_interval)}")
    for win, win_days_interval in group_days_interval.items():
        entered_wininterval_loop = True
        DEC_array_stack = []
        for orbit_dir in ["ASCENDING", "DESCENDING"]:
            index_orb_do = filter_index_by_orbit(index_do, orbit_dir)

            if len(index_orb_do) == 0:
                DEC_array_stack.append(template_array)
                continue

            vv_list = []
            vh_list = []

            for interval_start, interval_end in win_days_interval:
                vh_window_stack = []
                vv_window_stack = []
                orbit_dir_period = None
                time_points_averaged_str = ""
                for (dt, ob), fts in index_orb_do.items():
                    if interval_start <= dt < interval_end:
                        idx = next((i for i, d in enumerate(temporal_extent["times"]) if d.date() == dt.date()), None)
                        scene_array = arr[idx, :, :, :]
                        vh_band = scene_array[0, :, :]
                        vv_band = scene_array[1, :, :]

                        vh_window_stack.append(vh_band)
                        vv_window_stack.append(vv_band)

                        time_points_averaged_str += f"{dt.date()}, {idx} --"
                # Average over the scenes in the interval
                if len(vh_window_stack) == 0 or len(vv_window_stack) == 0:
                    vh_avg = np.full_like(template_array, np.nan)
                    vv_avg = np.full_like(template_array, np.nan)
                else:
                    vh_avg = np.nanmean(vh_window_stack, axis=0)
                    vv_avg = np.nanmean(vv_window_stack, axis=0)
                logger.info(f"AvgInfo: shapes {vh_avg.shape} {vv_avg.shape}  {interval_start.date()} to {interval_end.date()}, win: {win}, {time_points_averaged_str} -- Orbit: {orbit_dir}, -- avg {len(vh_window_stack)} scenes.")

                vh_list.append(vh_avg)
                vv_list.append(vv_avg)

            vh_array_stack = np.stack(vh_list, axis=0)
            vv_array_stack = np.stack(vv_list, axis=0)
            DEC_array_threshold, DEC_array_mask = apply_stat_datacube({"VV": vv_array_stack, "VH": vh_array_stack}, window_size=10)
            DEC_array_stack.append(DEC_array_mask)

        DEC_array_combined = np.nanmax(np.stack(DEC_array_stack, axis=0), axis=0)
        DEC_array_list.append(DEC_array_combined)
        win_list.append(win)

    if not entered_wininterval_loop:
        DEC_array_list = [template_array]
        win_list = [datetime(1, 1, 1, 0, 0, 0, 0)]

    if len(DEC_array_list) == 0:
        DEC_array_list = [template_array]
        win_list = [datetime(1, 1, 1, 0, 0, 0, 0)]

    DEC_array_combined_arraystack = np.stack(DEC_array_list, axis=0)
    # create xarray with single timestamp
    output_xarraycube = xr.DataArray(
        DEC_array_combined_arraystack[:, np.newaxis, :, :],   #DEC_array_combined[np.newaxis, np.newaxis, :, :],   # add a time dimension
        dims=["t", "bands", "y", "x"],
        coords={
            "t": win_list,
            "bands": ["DEC"],# win is your datetime.datetime object
            "y": arr.coords["y"],
            "x": arr.coords["x"],
        }
    )

    return output_xarraycube














