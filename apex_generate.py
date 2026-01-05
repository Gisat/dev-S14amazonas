import json
import sys
from pathlib import Path
import openeo.processes as eop
from openeo import UDF

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

def generate() -> dict:
    connection = openeo.connect("openeo.dataspace.copernicus.eu")

    spatial_extent = Parameter.spatial_extent(name="spatial_extent")
    temporal_extent = Parameter.temporal_interval(name="temporal_extent")

    ####################
    # PART 1: Extend temporal extent using UDF
    ####################
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

    s1 = s1.filter_bbox(spatial_extent).resample_spatial(resolution=20, align="upper-left")  # projection=f"EPSG:{crs}",
    s1_backcatter = s1.sar_backscatter(
        elevation_model="COPERNICUS_30",
        coefficient="sigma0-ellipsoid",
        local_incidence_angle=False)

    returns = {
        "description": "A data cube with the newly computed values.\n\nAll dimensions stay the same, except for the dimensions specified in corresponding parameters. There are three cases how the dimensions can change:\n\n1. The source dimension is the target dimension:\n   - The (number of) dimensions remain unchanged as the source dimension is the target dimension.\n   - The source dimension properties name and type remain unchanged.\n   - The dimension labels, the reference system and the resolution are preserved only if the number of values in the source dimension is equal to the number of values computed by the process. Otherwise, all other dimension properties change as defined in the list below.\n2. The source dimension is not the target dimension. The target dimension exists with a single label only:\n   - The number of dimensions decreases by one as the source dimension is 'dropped' and the target dimension is filled with the processed data that originates from the source dimension.\n   - The target dimension properties name and type remain unchanged. All other dimension properties change as defined in the list below.\n3. The source dimension is not the target dimension and the latter does not exist:\n   - The number of dimensions remain unchanged, but the source dimension is replaced with the target dimension.\n   - The target dimension has the specified name and the type other. All other dimension properties are set as defined in the list below.\n\nUnless otherwise stated above, for the given (target) dimension the following applies:\n\n- the number of dimension labels is equal to the number of values computed by the process,\n- the dimension labels are incrementing integers starting from zero,\n- the resolution changes, and\n- the reference system is undefined.",
        "schema": {
            "type": "object",
            "subtype": "datacube"
        }
    }
    ####################
    # PART 3: Apply statcube processing
    ####################:
    context_udf = {"spatial_extent": {"from_parameter": "spatial_extent"}, "detection_extent": {"from_parameter": "temporal_extent"}}
    udf = UDF.from_file("udf_apex_S1backscatter_changedetection.py", context={"from_parameter": "context"}) 
    output_statmcd = s1_backcatter.apply_dimension(process=udf, dimension="t", context=context_udf)
    output_statmcd = output_statmcd.rename_labels(dimension="bands",
                                                  target=["DEC", "DEC_asc", "DEC_asc_threshold", "DEC_des",
                                                          "DEC_des_threshold"])

    return build_process_dict(
        process_graph=output_statmcd,
        process_id="sentinel1_mcd",
        summary="Sentinel-1 change detection 20m resolution.",
        description=(
            Path(__file__).parent / "sentinel1_changedetection.md"
        ).read_text(),
        parameters=[
            spatial_extent,
            temporal_extent,
        ],
        returns=returns,
        categories=["sentinel-1", "change-detection", "forestcover"],
    )


if __name__ == "__main__":
    c = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    # TODO: how to enforce a useful order of top-level keys?
    process_json = json.dump(generate(), sys.stdout, indent=2)
    print(process_json)
    with open("sentinel1_mcd.json", "w") as f:
        json.dump(generate(), f, indent=2)

    with open("sentinel1_mcd.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    west, south, east, north = 733677.7864486989565194, 8635458.9404676575213671, 763907.4923464423045516, 8672119.4999492373317480
    crs = "32721"

    cube = c.datacube_from_json(
        json.dumps(raw),
        parameters={
                "spatial_extent": {"south": south, "east": east, "north": north, "west": west, "crs": f"EPSG:{crs}"},
                "temporal_extent": [
                    "2021-01-01",
                    "2021-01-13"
                ]
        }
    )

    cube.create_job().start_and_wait()


