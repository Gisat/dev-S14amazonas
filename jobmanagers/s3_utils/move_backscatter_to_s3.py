import os
from pathlib import Path
import subprocess

######################################################



def copy_to_s3(bucket_name, s3_path, local_path, config_path):
    cmd = ['rclone',
           '--config', config_path,
        'copy',
        '--ignore-existing',
        '-v',
        f'{local_path}',
        f's14amazonas:{bucket_name}/{s3_path}']

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print("Download completed!")


# Example usage:
bucket_name = "deforestation"
s3_path = f"sarbackscatter"

local_path = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"

copy_to_s3(bucket_name, s3_path, local_path, config_path)