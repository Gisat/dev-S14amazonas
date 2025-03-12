import openeo
import json
from pathlib import Path
import openeo.processes as eop
from openeo.processes import array_element, subtract, array_create, rename_labels
from openeo.metadata import CollectionMetadata, BandDimension, TemporalDimension, SpatialDimension
import pandas as pd

def create_timeintervals_list(temporal_extent):
    # Convert to datetime format
    start_date = pd.to_datetime(temporal_extent[0])
    end_date = pd.to_datetime(temporal_extent[1])

    # Generate the 12-day intervals
    intervals = pd.date_range(start=start_date, end=end_date, freq='12D')

    # Format the intervals into the desired structure
    year_intervals = [[str(interval.date())] for interval in intervals]

    return year_intervals

# Connect to the openEO backend (e.g., VITO backend)
# You can replace this URL with the specific backend you're using
# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
# connection.authenticate_oidc().authenticate_basic("grasslandwatch@gmail.com", "Productionenvironment1234!")
# connection.authenticate_oidc_device()
# Authenticate (if needed, this could be skipped if not required)
connection.authenticate_oidc()

# Define the area of interest (AOI) using a bounding box (or GeoJSON if available)
# Example bounding box for a region (replace with your desired area)

# full 21LYG extent
# north = -11.7536266487057866
# south = -12.7372447835033018
# east = -54.1584601180090601
# west = -55.1654233290762974

# small buffer
north = -11.8536
south = -12.6372
east = -54.258460
west = -55.06542



aoi = {
    "west": west,
    "east": east,
    "north": north,
    "south": south,
    "crs": "EPSG:4326"
}
crs_epsg = 4326

# aoi = {
#     "west": -55.0801,
#     "east": -54.2801,
#     "north": -11.958,
#     "south": -12.502,
#     "crs": "EPSG:4326"
# }
# crs_epsg = 4326


# -55.059959007,-54.881706667,-12.654733931,-12.447807500
# aoi = {
#     "west": -55.059959007,
#     "east": -54.881706667,
#     "north": -12.447807500,
#     "south": -12.654733931,
#     "crs": "EPSG:4326"
# }

# -55.16530653244335,
# -12.737140243568525,
# -54.15849189244335,
# -11.753681403568525

# Load the DEM collection from the STAC catalog (e.g., "DEM_aspec_30m")
# You can browse collections at https://radiantearth.github.io/stac-browser/#/external/stac.openeo.vito.be/
# collection_id = "DEM_slope_30m"  # Change this to the appropriate collection if needed
# datacube = connection.load_stac(
#     url="https://stac.openeo.vito.be/collections/DEM_slope_10m",
#     spatial_extent=aoi,
#     temporal_extent=None,  # Set this if you want data for a specific time range
#     bands=["SLP10"]  # Specify the band(s) if necessary
# )

url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/SAR/21LYG/21LYG_catalog.json"
bands = ["VV", "VH"]
resample_method = "cubicspline"

temporal_extent = [f"2021-01-04", f"2021-04-24"]
# temporal_extent = [f"2021-05-04", f"2021-08-22"]
# temporal_extent = [f"2021-01-04", f"2021-01-06"]
year_intervals = create_timeintervals_list(temporal_extent)

# model_relative_path = Path("output").joinpath("land_cover", "ml_models", f"{bioregion}_bioregion", str(year), onnx_name)
model_relative_path = f"amazonas/ml_models/amazonas_ai_cnn.zip"
MODEL_URL = f"https://s3.waw3-1.cloudferro.com/swift/v1/{model_relative_path}"

# Add the onnx dependencies to the job options. You can reuse this existing dependencies archive
DEPENDENCY_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


# Define the spatial dimensions based on the bounding box and CRS
x_extent = [aoi["west"], aoi["east"]]
y_extent = [aoi["south"], aoi["north"]]


stac_item = connection.load_stac(
    url=url,
    spatial_extent=aoi,
    temporal_extent=temporal_extent,
    bands=bands
)

stac_item.metadata = CollectionMetadata(
    metadata={},
    dimensions=[
        BandDimension(
            name="bands",
            bands=[openeo.metadata.Band(name=band) for band in bands]
        ),
        TemporalDimension(
            name="t",
            extent=year_intervals
        ),
        SpatialDimension(
            name="x",
            extent=x_extent,
            crs=f"EPSG:{crs_epsg}"
        ),
        SpatialDimension(
            name="y",
            extent=y_extent,
            crs=f"EPSG:{crs_epsg}"
        )
    ]
)

datacube = stac_item.resample_spatial(resolution=20, method=resample_method)
datacube = datacube.resample_spatial(projection=4326)

# Reduce to a single array by collapsing the time dimension into a single stack
datacube_time_as_bands = datacube.apply_dimension(
    dimension='t',
    target_dimension='bands',
    process=lambda d: eop.array_create(data=d)
)
band_names = [band + "_t" + str(i+1).zfill(2) for band in ["VV", "VH"] for i in range(10)]
print(f"band names {band_names}")
datacube_time_as_bands = datacube_time_as_bands.rename_labels('bands', band_names)



# job = datacube_time_as_bands.create_job(out_format="netCDF") #, job_options=job_options)
# job.start_and_wait()
# results = job.get_results()
# results.download_file("result_udf.nc")

# Load the UDF from a file.
udf = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O5_udf_deforestation_detection.py")
# Apply the UDF to the data cube.
datacube_udf = datacube_time_as_bands.apply_neighborhood(
    process=udf,
    size=[
        {"dimension": "x", "value": 400, "unit": "px"},
        {"dimension": "y", "value": 400, "unit": "px"},
    ],
    overlap=[
        {"dimension": "x", "value": 0, "unit": "px"},
        {"dimension": "y", "value": 0, "unit": "px"},
    ])
target = ["MCD", "MCD_threshold"]
datacube_udf = datacube_udf.rename_labels(dimension="bands", target=target)



udf_ai = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O6_udf_amazonas_ai.py")
# Apply the UDF to the data cube.
datacube_ai_udf = datacube_time_as_bands.apply_neighborhood(
    process=udf_ai,
    size=[
        {"dimension": "x", "value": 192, "unit": "px"},
        {"dimension": "y", "value": 192, "unit": "px"},
    ],
    overlap=[
        {"dimension": "x", "value": 32, "unit": "px"},
        {"dimension": "y", "value": 32, "unit": "px"},
    ])
target = ["AIMCD"]
datacube_ai_udf = datacube_ai_udf.rename_labels(dimension="bands", target=target)

change_detection = datacube_udf.merge_cubes(datacube_ai_udf)

# Optionally, define additional operations on the datacube, like rescaling or masking
# For now, we just download the DEM data as is


job_options = {
    "executor-memory": "2500m",
    "executor-memoryOverhead": "2G",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "soft-errors": True,
    "max_executors": 15,
    "udf-dependency-archives": [
    f"{DEPENDENCY_URL}#onnx_deps",
    f"{MODEL_URL}#onnx_models"]
}

# job_options = {
#     "executor-memory": "4G",
#     "executor-memoryOverhead": "3G",
#     "driver-memory": "3G",
#     "driver-memoryOverhead": "3G",
#     "soft-errors": True,
#     # "max_executors": 15,
#     "udf-dependency-archives": [
#     f"{DEPENDENCY_URL}#onnx_deps",
#     f"{MODEL_URL}#onnx_models"]
# }



# Save the result as a GeoTIFF file
job = datacube_ai_udf.create_job(out_format="GTiff", job_options=job_options)
job.start_and_wait()
results = job.get_results()
results.download_file(f"openeo_detection_21LYG_20210306_sieved12_count3_{resample_method}_withmeanchangeremoved.tiff")
