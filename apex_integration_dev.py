#%%
import openeo
from openeo import UDF, DataCube

connection = openeo.connect(url="openeo.dataspace.copernicus.eu").authenticate_oidc()

temporal_extent = ["2023-01-01", "2023-12-31"]
source_epsg = 32621
west, south, east, north = 732635.6526184550020844,-1360687.7596141430549324, 762820.0796379272360355,-1331475.8172322250902653

# 21LYG small patch
spatial_extent = {"south": south, "east": east, "north": north, "west": west, "crs": crs}
crs = f"EPSG:{source_epsg}"

s1 = connection.load_collection(
    "SENTINEL1_GRD",
    temporal_extent=temporal_extent,
    spatial_extent=spatial_extent,
    bands=["VH", "VV"])
datacube = s1.resample_spatial(resolution=20, method="near", projection=f"EPSG:{source_epsg}")

context_udf = {"start_time": temporal_extent[0], "end_time": temporal_extent[1], "epsg": source_epsg, "spatial_extent": spatial_extent}
udf = UDF.from_file("udf_apex_S1backscatter_changedetection.py", context=context_udf)

output = datacube.apply_dimension(process=udf)
job_options = {"executor-memory": "4G",
    "executor-memoryOverhead": "500m",
    "python-memory": "2500m",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "max-executors": 5,
    "soft-errors": True}

job = output.create_job(job_options=job_options)
job.start_and_wait()

