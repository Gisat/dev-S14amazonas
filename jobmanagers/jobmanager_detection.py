import os
from pathlib import Path
from datetime import datetime, timedelta

import openeo
from openeo.extra.job_management import MultiBackendJobManager
from openeo.extra.job_management import create_job_db

import pandas as pd
import geopandas as gpd

from Oa_openeo_utils import get_temporalextents_mastertemporalextent, get_extended_temporalextents_with_padding, get_monthyear_periods_joblist
from O4_openeo_deforestation_detection import changedetection_jm_wrapper
## INPUTS
NUM_PROCESS = 4
past_runs = 0
RUN_NAME =  "poc" #"poc20230503-20240825"
priority = None
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
work_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/detection_jobmanagement_poc")
os.makedirs(work_dir, exist_ok=True)

####
assert input_df_path.exists(), f"Input file {input_df_path} does not exist"
input_df = gpd.read_file(input_df_path)



if priority is not None:
    input_df = input_df[input_df["Priority"] == 3]
input_df = input_df[input_df["Name"] == "20LNR"]
# temporal_extents = get_monthyear_periods_joblist("20230503", "20240825", 1)
temporal_extents = get_monthyear_periods_joblist("20210101", "20241223", 10)

job_df = pd.concat([input_df.assign(startdate=months_year_item[0], enddate = months_year_item[1]) for months_year_item in temporal_extents], ignore_index=True)

job_df.rename(mapper={"xmin": "west", "xmax": "east", "ymin": "south", "ymax": "north"}, axis=1, inplace=True)
job_df["epsg"] = 4326

column_list= ["west", "east", "south", "north", "epsg", "startdate", "enddate",  "Name"]
job_df = job_df[column_list]

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

if RUN_NAME is not None:
    runstr = RUN_NAME
else:
    runstr = datetime.now().strftime("%Y%m%d-%Hh%M")
print(f"runstr: {runstr}")

job_database = create_job_db(
    path= work_dir / f"job_database-{runstr}.csv",
    df=job_df,
    on_exists="skip"
)

job_manager = MultiBackendJobManager(
    poll_sleep=10,
    root_dir=work_dir
)

job_manager.add_backend(
    "cdse",
    connection=connection,
    parallel_jobs=NUM_PROCESS
)

job_manager.run_jobs(
    df=job_df,
    start_job=changedetection_jm_wrapper,
    job_db=job_database
)


