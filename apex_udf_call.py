from udf_apex_S1backscatter_changedetection import apply_datacube, fetch_s1_features
from dataclasses import dataclass
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np

temporal_extent = ["2020-03-11", "2020-03-23"]
context = {
    "detection_extent": temporal_extent,
    "acq_frequency": 12,
    "epsg": 32721,
    "spatial_extent": {
        "south": -11.753685105,
        "east": -54.158521354,
        "north": -12.73714945,
        "west": -55.165335994,
         "crs": f"EPSG:4326"
    }
}
# apply_datacube(None, context)


# Original dictionary
temporal_dict = {
    "start": pd.Timestamp("2020-01-17 00:00:00"),
    "end": pd.Timestamp("2020-05-04 00:00:00"),
}


bbox_4326 = [-11.753685105, -55.165335994, -12.73714945, -54.158521354]
feats = fetch_s1_features(bbox_4326, temporal_dict["start"], temporal_dict["end"])
