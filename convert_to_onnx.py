import onnx
import tensorflow as tf
import tf2onnx
import zipfile
from pathlib import Path
import subprocess

from cnn_architectures import build_vgg16_segmentation_bn

MIN_VALUE = 0.0
MAX_VALUE = 65535.0
LOWER_CUTOFF = -30.0
UPPER_CUTOFF = 0.0

def copy_to_s3(project_name, local_path, config_path, s3_path=""):
    cmd = [
        "rclone", "copy",
        "--config", config_path,
        "--log-level=INFO",
        "--no-gzip-encoding",
        str(local_path),
        f"{project_name}:{s3_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print(f"Upload {local_path} completed!")

# --- Load weights into architecture ---
h5_file_path = "/mnt/hddarchive.nfs/amazonas_dir/model/model_best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score.h5"
rclone_config_path = "/home/eouser/userdoc/rclone_conf/rclone.conf"

base_model = build_vgg16_segmentation_bn((256, 256, 15))
base_model.load_weights(h5_file_path)

# --- Build a wrapped model that includes scaling ---
inp = tf.keras.Input(shape=(256, 256, 15), dtype=tf.float32, name="input_raw")

# Clip to [-30, 0]
x = tf.clip_by_value(inp, LOWER_CUTOFF, UPPER_CUTOFF)

# Min-max scale from [-30, 0] -> [0, 65535]
scale = (MAX_VALUE - MIN_VALUE) / (UPPER_CUTOFF - LOWER_CUTOFF)  # 65535/30
x = (x - LOWER_CUTOFF) * scale + MIN_VALUE

# (Optional) if your network expects a certain range, you can normalize further here.
# e.g., x = x / 65535.0  (ONLY if you trained it that way)

out = base_model(x)

wrapped_model = tf.keras.Model(inputs=inp, outputs=out, name="amazonas_ai_with_scaling")

# --- Convert wrapped model to ONNX ---
spec = (tf.TensorSpec((None, 256, 256, 15), tf.float32, name="input_raw"),)
onnx_model, _ = tf2onnx.convert.from_keras(wrapped_model, input_signature=spec, opset=13)

onnx_model_path = "/mnt/hddarchive.nfs/amazonas_dir/onnxmodel/amazonas_ai_cnn.onnx"
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

onnx_zip_local_path = "/mnt/hddarchive.nfs/amazonas_dir/onnxmodel/amazonas_ai_cnn.zip"
with zipfile.ZipFile(onnx_zip_local_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(onnx_model_path, "ml_model.onnx")

s3_path = Path("amazonas").joinpath("ml_models")
copy_to_s3("gisat", onnx_zip_local_path, rclone_config_path, s3_path=s3_path)

print(f"Model with scaling exported to: {onnx_model_path}")
