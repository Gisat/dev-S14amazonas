from udf_apex_S1backscatter_changedetection import apply_datacube



context = {
    "detection_time": "2020-03-30",
    "acq_frequency": 12,
    "epsg": 32721,
    "spatial_extent": {
        "south": -11.753685105,
        "east": -54.158521354,
        "north": -12.73714945,
        "west": -55.165335994,
        "crs": 4326
    }
}
apply_datacube(None, context)