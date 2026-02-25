import openeo
import openeo.processes as eop
from openeo import UDF
from pathlib import Path

connection  = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

#
# crs = "32721"
# west, south, east, north =  770900,8660000, 787040, 8697260

# small road
# crs = "32618"
# west, south, east, north = 788200,199180, 803800,214100

crs = "32721"
west, south, east, north = 733677.7864486989565194,8635458.9404676575213671,763907.4923464423045516,8672119.4999492373317480
spatial_extent = {"south": south, "east": east, "north": north, "west": west, "crs": f"EPSG:{crs}"}
temporal_extent = ["2020-03-11", "2020-03-23"]

####################
# PART 1: Extend temporal extent using UDF
####################
current_dir = Path(__file__).parent
udf_path = current_dir / "udf_createcustomintervals.py"

udf = openeo.UDF.from_file(str(udf_path))
extended_temporal_extent = eop.run_udf(
    data=temporal_extent,
    udf=udf.code,
    runtime="python"
)


####################
# PART 2: Load data with extended temporal extent
####################

s1 = connection.load_collection(
    collection_id="SENTINEL1_GRD",
    bands=["VH", "VV"], spatial_extent=spatial_extent,
).filter_temporal(start_date=extended_temporal_extent[0], end_date=extended_temporal_extent[1])

s1 = s1.filter_bbox(spatial_extent).resample_spatial(resolution=20, align="upper-left") # projection=f"EPSG:{crs}",
s1_backcatter = s1.sar_backscatter(
    elevation_model="COPERNICUS_30",
    coefficient="sigma0-ellipsoid",
    local_incidence_angle=False)

####################
# PART 3: Apply statcube processing
####################
# context_udf = {"start_time": extended_temporal_extent[0], "end_time": extended_temporal_extent[1], "epsg": int(crs), "spatial_extent": spatial_extent}
context_udf = {"spatial_extent": spatial_extent, "detection_start_time": temporal_extent[0], "detection_end_time": temporal_extent[1]}
udf_path = current_dir / "udf_apex_S1backscatter_changedetection.py"
udf = UDF.from_file(str(udf_path), context=context_udf)
output_statmcd = s1_backcatter.apply_dimension(process=udf, dimension="t")
output_statmcd = output_statmcd.rename_labels(dimension="bands", target=["DEC", "DEC_asc", "DEC_asc_threshold", "DEC_des", "DEC_des_threshold"])

context_udf = {"spatial_extent": spatial_extent,
               "detection_start_time": temporal_extent[0], "detection_end_time": temporal_extent[1],
                "datacube_ai_time_window": 5}

udf_path = current_dir / "udf_apex_S1backscatter_aichangedetection.py"
udf_ai = UDF.from_file(str(udf_path), context=context_udf)
output_aimcd = s1_backcatter.apply_neighborhood(
        process=udf_ai,
        size=[
            {"dimension": "x", "value": 192, "unit": "px"},
            {"dimension": "y", "value": 192, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 32, "unit": "px"},
            {"dimension": "y", "value": 32, "unit": "px"},
        ])
output_aimcd = output_aimcd.rename_labels(dimension="bands", target=["MCD_AI"])
output = output_statmcd.merge_cubes(output_aimcd)

job_options = {"executor-memory": "4G",
    "executor-memoryOverhead": "500m",
    "python-memory": "2500m",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "max-executors": 5,
    "soft-errors": True,
   "udf-dependency-archives": [
       f"https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip#onnx_deps",
       f"https://s3.waw3-1.cloudferro.com/swift/v1/amazonas/ml_models/amazonas_ai_cnn.zip#onnx_models"]
               }

# job = connection.create_job(title='test_catalogue_check', process_graph= output, job_options=job_options)
job = output.create_job(job_options=job_options)
job.start_and_wait()