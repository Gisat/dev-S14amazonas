from openeo.udf import inspect, UdfData, StructuredData
import datetime
from typing import Dict, List, Tuple, Optional

# Phase boundaries as DATES
PHASE1_END = datetime.date(2021, 12, 16)
PHASE2_END = datetime.date(2025, 3, 30)

# ───────────────────────── Step logic (your rule) ─────────────────────────

def step_forward(start_d: datetime.date, acq_frequency: int = 6) -> int:
    """
    Decide interval length (N or 2N) from a given START date.

    Rule:
      - If start + N does NOT cross PHASE1_END => use N
      - Else, as long as start < PHASE2_END    => use 2N
      - Once start >= PHASE2_END               => use N
    """
    N = acq_frequency

    if start_d + datetime.timedelta(days=N) <= PHASE1_END:
        # Still in Phase 1
        return N
    elif start_d < PHASE2_END:
        # Phase 2 (including the interval that crosses PHASE2_END)
        return 2 * N
    else:
        # Phase 3
        return N


# ───────────────────────── Backwards (no truncation) ─────────────────────────

def prev_interval(cur_start: datetime.date, acq_frequency: int = 6) -> Tuple[datetime.date, datetime.date]:
    """
    Given the START of the current interval, find the previous full interval [prev_start, cur_start],
    such that its length is either N or 2N and is consistent with step_forward(prev_start).

    No truncation: length is exactly N or 2N.
    """
    N = acq_frequency

    # Candidate 1: previous interval length N
    cand1_start = cur_start - datetime.timedelta(days=N)
    cand1_len   = N
    cand1_ok = (
        step_forward(cand1_start, N) == cand1_len and
        cand1_start + datetime.timedelta(days=cand1_len) == cur_start
    )

    # Candidate 2: previous interval length 2N
    cand2_start = cur_start - datetime.timedelta(days=2 * N)
    cand2_len   = 2 * N
    cand2_ok = (
        step_forward(cand2_start, N) == cand2_len and
        cand2_start + datetime.timedelta(days=cand2_len) == cur_start
    )

    if not cand1_ok and not cand2_ok:
        raise RuntimeError(f"No valid previous interval for start={cur_start}")

    if cand1_ok and not cand2_ok:
        return cand1_start, cur_start
    if cand2_ok and not cand1_ok:
        return cand2_start, cur_start

    # Both valid (rare near boundaries) – prefer 2N by convention
    return cand2_start, cur_start


def back_chain(anchor_start: datetime.date, n_back: int, acq_frequency: int = 6) -> List[Tuple[datetime.date, datetime.date]]:
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

def forward_chain(anchor_start: datetime.date, n_forw: int, acq_frequency: int = 6) -> List[Tuple[datetime.date, datetime.date]]:
    """
    Build n_forw intervals starting from anchor_start (first interval starts at anchor_start).
    """
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    cur_start = anchor_start

    for _ in range(n_forw):
        length = step_forward(cur_start, acq_frequency)
        end = cur_start + datetime.timedelta(days=length)
        intervals.append((cur_start, end))
        cur_start = end

    return intervals

# ───────────────────────── Main helper: 5 back + 4 forward ─────────────────────────

def get_context_intervals(
    start_d: datetime.date,
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

def get_temporal_extends(udf_data: UdfData) -> UdfData:
    """
    UDF that takes a temporal extent as input and returns temporal products needed for further processing.
    """
    temporal_extent = udf_data.get_structured_data_list()[0].data
    start_date = temporal_extent[0]
    end_date = temporal_extent[1]
    acq_freq = int(end_date)

    start_datetime = datetime.date.fromisoformat(start_date)
    # end_datetime = datetime.date.fromisoformat(end_date)

    intervals = get_context_intervals(start_datetime, acq_frequency=acq_freq)
    overall_start, overall_end = get_overall_start_end(intervals)

    udf_data.set_structured_data_list([])
    udf_data.set_structured_data_list([
        StructuredData(
            description="Extended temporal interval",
            data=[
                overall_start.isoformat(),
                overall_end.isoformat()
            ],
            type="list"
        )
    ])
    inspect(message="Resulting udf_data", data=str(udf_data.to_dict()))
    return udf_data