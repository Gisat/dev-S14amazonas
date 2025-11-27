import openeo
import openeo.processes as eop
from openeo import UDF, DataCube

connection  = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

crs = "32721"
west, south, east, north =  770900,8660000, 787040, 8697260
spatial_extent = {"south": south, "east": east, "north": north, "west": west, "crs": f"EPSG:{crs}"}
acq_freq = 12
####################
# PART 1: Extend temporal extent using UDF
####################
temporal_extent = ["2025-03-30", str(acq_freq)]
udf = openeo.UDF.from_file("udf_createcustomintervals.py")
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
    bands=["VH", "VV"]
).filter_temporal(start_date=extended_temporal_extent[0], end_date=extended_temporal_extent[1])

s1 = s1.filter_bbox(spatial_extent).resample_spatial(projection=f"EPSG:{crs}", resolution=20, align="upper-left")
s1_backcatter = s1.sar_backscatter(
    elevation_model="COPERNICUS_30",
    coefficient="sigma0-ellipsoid",
    local_incidence_angle=False)

####################
# PART 3: Apply statcube processing
####################
# context_udf = {"start_time": extended_temporal_extent[0], "end_time": extended_temporal_extent[1], "epsg": int(crs), "spatial_extent": spatial_extent}
context_udf = {"epsg": int(crs), "spatial_extent": spatial_extent, "detection_time": temporal_extent[0], "acq_frequency": acq_freq}
udf = UDF.from_file("/home/eouser/userdoc/src/pythonProject/hans_udf_S1backscatter_updated_oncube.py", context=context_udf)

output = s1_backcatter.apply_dimension(process=udf, dimension="t")
job_options = {"executor-memory": "4G",
    "executor-memoryOverhead": "500m",
    "python-memory": "2500m",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "max-executors": 5,
    "soft-errors": True}

# job = connection.create_job(title='test_catalogue_check', process_graph= output, job_options=job_options)
job = output.create_job(job_options=job_options)
job.start_and_wait()