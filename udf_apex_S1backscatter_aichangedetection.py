import functools
import sys
import xarray as xr
from osgeo import osr, ogr
import re
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import requests
import datetime
import numpy as np
import pandas as pd
from shapely.geometry import shape
from scipy import ndimage as ndi
import logging
from collections import defaultdict, Counter, OrderedDict

# The onnx_deps folder contains the extracted contents of the dependencies archive provided in the job options
sys.path.insert(0, "onnx_deps")
import onnxruntime as ort

DEBUG = False

LOWER_CUTOFF = -30
logger = logging.getLogger(__name__)




# -------------------------
# Time extents
# -------------------------
# Phase boundaries as DATES

MASTER_START = datetime.datetime(2015, 4, 28)
PHASE1_END = datetime.datetime(2021, 12, 16)
PHASE2_END = datetime.datetime(2025, 3, 30)

# ───────────────────────── Step logic (your rule) ─────────────────────────

def step_forward(start_d: datetime.datetime, acq_frequency: int = 6) -> int:
    """
    Decide interval length (N or 2N) from a given START date.

    Rule:
      - If start + N does NOT cross PHASE1_END => use N
      - Else, as long as start < PHASE2_END    => use 2N
      - Once start >= PHASE2_END               => use N
    """
    N = acq_frequency

    if start_d + timedelta(days=N) <= PHASE1_END:
        # Still in Phase 1
        return N
    elif start_d < PHASE2_END:
        # Phase 2 (including the interval that crosses PHASE2_END)
        return 2 * N
    else:
        # Phase 3
        return N


# ───────────────────────── Backwards (no truncation) ─────────────────────────

def prev_interval(cur_start: datetime.datetime, acq_frequency: int = 6) -> Tuple[datetime.date, datetime.date]:
    """
    Given the START of the current interval, find the previous full interval [prev_start, cur_start],
    such that its length is either N or 2N and is consistent with step_forward(prev_start).

    No truncation: length is exactly N or 2N.
    """
    N = acq_frequency

    # Candidate 1: previous interval length N
    cand1_start = cur_start - timedelta(days=N)
    cand1_len   = N
    cand1_ok = (
        step_forward(cand1_start, N) == cand1_len and
        cand1_start + timedelta(days=cand1_len) == cur_start
    )

    # Candidate 2: previous interval length 2N
    cand2_start = cur_start - timedelta(days=2 * N)
    cand2_len   = 2 * N
    cand2_ok = (
        step_forward(cand2_start, N) == cand2_len and
        cand2_start + timedelta(days=cand2_len) == cur_start
    )

    if not cand1_ok and not cand2_ok:
        raise RuntimeError(f"No valid previous interval for start={cur_start}")

    if cand1_ok and not cand2_ok:
        return cand1_start, cur_start
    if cand2_ok and not cand1_ok:
        return cand2_start, cur_start

    # Both valid (rare near boundaries) – prefer 2N by convention
    return cand2_start, cur_start


def back_chain(anchor_start: datetime.datetime, n_back: int, acq_frequency: int = 6) -> List[Tuple[datetime.date, datetime.date]]:
    """
    Build n_back intervals BEFORE anchor_start, going backwards, with no truncation.
    """
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    cur_start = anchor_start

    for _ in range(n_back):
        prev_start, prev_end = prev_interval(cur_start, acq_frequency)
        intervals.append((prev_start, prev_end))
        cur_start = prev_start

    # Reverse to chronological order
    return list(reversed(intervals))


# ───────────────────────── Forwards (no truncation) ─────────────────────────

def forward_chain(anchor_start: datetime.datetime, n_forw: int, acq_frequency: int = 6) -> List[Tuple[datetime.date, datetime.date]]:
    """
    Build n_forw intervals starting from anchor_start (first interval starts at anchor_start).
    """
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    cur_start = anchor_start

    for _ in range(n_forw):
        length = step_forward(cur_start, acq_frequency)
        end = cur_start + timedelta(days=length)
        intervals.append((cur_start, end))
        cur_start = end

    return intervals

# ───────────────────────── Main helper: 5 back + 4 forward ─────────────────────────

def get_context_intervals(
    start_str: str,
    back: int = 5,
    forward: int = 4,
    acq_frequency: int = 6
) -> List[Tuple[datetime.date, datetime.date]]:
    """
    Returns:
      - `back` intervals before start_date
      - the interval starting at start_date
      - `forward` intervals after that

    Total = back + 1 + forward intervals.
    The 6th interval's START is exactly start_date if back=5.
    """
    start_d = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    before = back_chain(start_d, back, acq_frequency)          # 5 intervals before
    after  = forward_chain(start_d, forward + 1, acq_frequency)  # includes anchor as first
    return before + after

def get_overall_start_end(intervals: List[Tuple[datetime.date, datetime.date]]):
    """
    Given a list of intervals [(start, end), ...],
    return (overall_start, overall_end).
    """
    overall_start = min(s for s, e in intervals)
    overall_end   = max(e for s, e in intervals)
    return overall_start, overall_end



# -------------------------
# Config / Constants
# -------------------------
S1_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel1/search.json"
DATE_RE = re.compile(r'_(\d{8})T\d{6}_')  # e.g., ..._20211201T091308_...

# -------------------------
# Datacube utils
# -------------------------
def get_spatial_extent(spatial_extent) -> Tuple[dict, List[float]]:

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
def remove_small_objects(mask, min_size=2, connectivity=1):
    """
    Removes connected components of 1s smaller than min_size.
    Fastest possible implementation (SciPy C backend).
    """
    mask = np.asarray(mask).astype(bool)

    if min_size <= 1:
        return mask

    # Label connected components in the foreground (1-pixels)
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, num = ndi.label(mask, structure=structure)

    if num == 0:
        return mask

    # Compute size of each component
    sizes = ndi.sum(mask, labels, index=np.arange(1, num + 1))

    # Select components to keep
    keep_labels = np.where(sizes >= min_size)[0] + 1

    # Build output
    out = np.isin(labels, keep_labels)

    return out


def parse_date_from_title(title: str) -> Optional[datetime.datetime]:
    """Extract YYYYMMDD as datetime from a Sentinel-1 title. Return None if not found."""
    m = DATE_RE.search(title)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y%m%d")


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
    # logger.info(f"s1query {r.url}")
    return r.json().get("features", [])


def build_index_by_date_orbit(features: List[dict]) -> Dict[Tuple[datetime.datetime, str], List[dict]]:
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

def filter_index_to_dates(index_do: Dict[Tuple[datetime.datetime, str], List[dict]],
                          dt_list: List[datetime.datetime]) -> Dict[Tuple[datetime.datetime, str], List[dict]]:
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


def pick_orbit_direction(index_do: Dict[Tuple[datetime.datetime, str], List[dict]],
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


def get_scene_indices(index_do: Dict[Tuple[datetime.datetime, str], List[dict]],
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
def has_enough_valid(vh_win, vv_win, min_valid_frac=0.01):
    # vh_win, vv_win: (T, Y, X)
    valid = np.isfinite(vh_win) & np.isfinite(vv_win)
    # valid anywhere across time for a pixel counts as valid pixel
    valid_pix = np.any(valid, axis=0)  # (Y, X)
    frac = valid_pix.mean()
    return frac >= min_valid_frac

@functools.lru_cache(maxsize=5)
def load_onnx_model(model_name: str) -> ort.InferenceSession:
    """
    Loads an ONNX model from the onnx_models folder and returns an ONNX runtime session.

    Extracting the model loading code into a separate function allows us to cache the loaded model.
    This prevents the model from being loaded for every chunk of data that is processed, but only once per executor,
    which can save a lot of time, memory and ultimately processing costs.

    Should you have to download the model from a remote location, you can add the download code here, and cache the model.

    Make sure that the arguments of the method you add the @functools.lru_cache decorator to are hashable.
    Be careful with using this decorator for class methods, as the self argument is not hashable.
    In that case you can use a static method or make sure your class is hashable (more difficult): https://docs.python.org/3/faq/programming.html#faq-cache-method-calls.

    More information on this functool can be found here:
    https://docs.python.org/3/library/functools.html#functools.lru_cache
    """
    # The onnx_models folder contains the content of the model archive provided in the job options
    if DEBUG:
        return ort.InferenceSession(f"{model_name}")
    else:
        return ort.InferenceSession(f"onnx_models/{model_name}")

def run_inference(input_np: np.ndarray, ort_session: ort.InferenceSession) -> tuple:
    """
    Run inference using the ONNX runtime session and return predicted labels and probabilities.
    """
    # Get the input name expected by the ONNX model
    input_name = ort_session._inputs_meta[0].name  # Extract input name from metadata

    # Ensure input_np is a NumPy array and reshape to match model input shape
    input_np = input_np.astype(np.float32)  # Ensure correct data type

    # Run inference
    outputs = ort_session.run(None, {input_name: input_np})

    predicted_labels = outputs[0]

    return predicted_labels


def postprocess_output(predicted_labels: np.ndarray, input_shape: tuple) -> tuple:
    """
    Postprocess the output by reshaping the predicted labels and probabilities into the original spatial structure.
    """

    # Reshape to match the (y, x) spatial structure
    predicted_labels = np.squeeze(predicted_labels, axis=-1)  # Remove the last axis
    predicted_labels = np.squeeze(predicted_labels, axis=0)  # Remove the last axis
    return predicted_labels


def create_output_xarray(predicted_labels: np.ndarray,
                         input_xr: xr.DataArray) -> xr.DataArray:
    """
    Create an xarray DataArray with predicted labels and probabilities stacked along the bands dimension.
    """

    return xr.DataArray(
        predicted_labels,
        dims=["bands", "y", "x"],
        coords={
            'y': input_xr.coords['y'],
            'x': input_xr.coords['x']
        }
    )

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Simple UDF: Check S1 observation frequency via STAC and aggregate temporally.
    """

    ## Step 1: Load the ONNX model
    ort_session = load_onnx_model("ml_model.onnx")

    arr = cube

    # Get temporal extent
    spatial_extent = context["spatial_extent"]
    datection_start_time = context["detection_extent"][0]
    detection_end_time = context["detection_extent"][1]
    datacube_window = context["datacube_ai_time_window"]

    start_d = datetime.datetime.strptime(datection_start_time, "%Y-%m-%d")
    end_d = datetime.datetime.strptime(detection_end_time, "%Y-%m-%d")
    delta_days = (end_d - start_d).days
    acq_frequency = abs(delta_days)

    # temporal extent
    days_interval = get_context_intervals(datection_start_time, acq_frequency=acq_frequency)
    start_time, end_time = get_overall_start_end(days_interval)
    # logger.info(f"Processingfromto: {start_time} to {end_time}")

    group_days_interval = create_timewindow_groups(days_interval)

    # Get spatial extent
    spatial_extent_4326, bbox_4326 = get_spatial_extent(spatial_extent)
    # logger.info(f"Spatial extent in EPSG:{epsg_code}: {spatial_extent_4326} {bbox_4326}")

    temporal_extent = get_temporal_extent(arr)

    # Fetch & build index
    feats = fetch_s1_features(bbox_4326, temporal_extent["start"], temporal_extent["end"])
    index_do = build_index_by_date_orbit(feats)

    template_array = np.zeros_like(arr[0, 0, :, :])

    # 2) Filter to your dates of interest
    # logger.info(f"featuresdateorb: {index_do.keys()}")
    # logger.info(f"filteringscenesusing: {temporal_extent['times']}")
    index_do = filter_index_to_dates(index_do, temporal_extent["times"])
    # logger.info(f"AfterTimeFilter: {index_do.keys()}")
    # 3) Decide the orbit direction (or None if tie after tie-break)
    # selected_orbit = pick_orbit_direction(index_do, bbox)

    arr_cube = arr.astype(np.float32)
    arr = np.where(arr_cube >0, 10 * np.log10(arr_cube), np.nan)  # 10 * np.log10(arr)

    # 4).
    DEC_array_combined = None
    entered_wininterval_loop = False

    DEC_temporal_list = []
    win_list = []
    # logger.info(f"Processingtimewindows {len(group_days_interval)}")
    for win, win_days_interval in group_days_interval.items():
        DEC_array_list = []
        entered_wininterval_loop = True
        DEC_array_stack = []
        DEC_array_threshold_stack = []
        for orbit_dir in ["ASCENDING", "DESCENDING"]:
            index_orb_do = filter_index_by_orbit(index_do, orbit_dir)

            if len(index_orb_do) == 0:
                DEC_array_stack.append(template_array)
                DEC_array_threshold_stack.append(template_array)
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
                        idx = next((i for i, d in enumerate(temporal_extent["times"]) if d == dt), None)
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
                # logger.info(f"AvgInfo: shapes {vh_avg.shape} {vv_avg.shape}  {interval_start.date()} to {interval_end.date()}, win: {win}, {time_points_averaged_str} -- Orbit: {orbit_dir}, -- avg {len(vh_window_stack)} scenes.")

                vh_list.append(vh_avg)
                vv_list.append(vv_avg)

            vh_array_stack = np.stack(vh_list, axis=0)
            vv_array_stack = np.stack(vv_list, axis=0)

            vh_array_window = vh_array_stack[:datacube_window, :, :]
            vv_array_window = vv_array_stack[:datacube_window, :, :]

            if not has_enough_valid(vh_array_window, vv_array_window, min_valid_frac= 0.01):
                DEC_array_stack.append(template_array)
                DEC_array_threshold_stack.append(template_array)
                continue

            if np.nanstd(vh_array_window) < 1e-6  and np.nanstd(vv_array_window) < 1e-6:
                DEC_array_stack.append(template_array.astype(np.int16))
                DEC_array_threshold_stack.append(template_array)
                continue

            vh_vv_ratio = vh_array_window - vv_array_window

            vh = np.where(np.isfinite(vh_array_window), vh_array_window, LOWER_CUTOFF)
            vv = np.where(np.isfinite(vv_array_window), vv_array_window, LOWER_CUTOFF)
            vh_vv_ratio = np.where(np.isfinite(vh_vv_ratio), vh_vv_ratio, LOWER_CUTOFF)

            # Stack VH, VV, and VH/VV ratio
            result = np.stack((vh, vv, vh_vv_ratio), axis=-1)
            # logger.info(f"InputNPShape {result.shape} for orbit {orbit_dir} in window starting {win}")

            result = result[:, :256, :256, :]

            input_np = np.transpose(result, (1, 2, 0, 3))
            input_np = input_np.reshape(256, 256, datacube_window * 3)
            input_np = input_np[np.newaxis, ...]

            # Step 3: Perform inference
            predicted_labels = run_inference(input_np, ort_session)

            # Step 4: Postprocess the output
            predicted_labels = postprocess_output(predicted_labels, input_np.shape)

            DEC_array_mask =  (predicted_labels < 0.05).astype(np.int16)
            DEC_array_stack.append(DEC_array_mask)


        DEC_array_combined = np.nanmax(np.stack(DEC_array_stack, axis=0), axis=0)
        DEC_array_list.append(DEC_array_combined)

        win_list.append(win)
        # logger.info(f"DECArraylistlen {len(DEC_array_list)}")
        DEC_temporal_list.append(np.stack(DEC_array_list, axis=0))
        # logger.info(f"DECArrayliststackshape {np.stack(DEC_array_list, axis=0).shape}")


    DEC_temporal_array = np.stack(DEC_temporal_list, axis=0)
    # logger.info(f"DECtemporalarray {DEC_temporal_array.shape}")

    # create xarray with single timestamp
    output_xarraycube = xr.DataArray(
        DEC_temporal_array,   #DEC_array_combined[np.newaxis, np.newaxis, :, :],   # add a time dimension
        dims=["t", "bands", "y", "x"],
        coords={
            "t": win_list,
            "bands": ["DEC"],# win is your datetime.datetime object
            "y": arr_cube.coords["y"],
            "x": arr_cube.coords["x"],
        }
    )

    return output_xarraycube
