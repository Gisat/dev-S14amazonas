import os

import pystac
from pystac.extensions.eo import Band, EOExtension
from shapely.geometry import Polygon, mapping
from datetime import datetime
from pathlib import Path
import rasterio
from shapely.geometry import shape

def file_datetime_list(folder_path):
    rasterfile_list = os.listdir(folder_path)
    datetime_list = []
    for rasterfile_list_item in rasterfile_list:
        if not rasterfile_list_item.endswith("tif"):
            continue
        datetime_str = rasterfile_list_item.split("_")[3]
        datetime_list.append(datetime_str)
    return datetime_list

# Function to Extract BBox and Footprint from S3 Asset
def get_bbox_and_footprint(s3_url_item):
    with rasterio.open(s3_url_item) as r:
        bounds = r.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        footprint = Polygon([
            [bounds.left, bounds.bottom],
            [bounds.left, bounds.top],
            [bounds.right, bounds.top],
            [bounds.right, bounds.bottom]
        ])
        return bbox, mapping(footprint)

base_url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/jaxa_forestcover"
root_path = Path("/mnt/hddarchive.nfs/amazonas_dir/openEO/jaxa_forestcover")
datetime_item = "20181213"
area = "amazonas"
bbox, footprint = None, None  # Initialize bbox and footprint


stac_items = []

collection_item = pystac.Item(
    id=f'jaxa-brazil',
    geometry=footprint,
    bbox=bbox,
    datetime=datetime.strptime(datetime_item, "%Y%m%d"),
    properties={},
    href=f"{base_url}/jaxa_forestcover_{area}.json"  # S3 path reference
)

collection_item.common_metadata.gsd = 0.3
collection_item.common_metadata.platform = 'Gisat'
collection_item.common_metadata.instruments = ['ForestView']

tif_url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/jaxa_forestcover/jaxa_forestcover_amazonas.tif"
bbox, footprint = get_bbox_and_footprint(tif_url)

asset = pystac.Asset(href=tif_url, media_type=pystac.MediaType.GEOTIFF, roles=['data'])

forest_mask_bands = [Band.create(name="forest_mask", description=f'Jaxa forest cover', common_name='forest_mask')]
eo_asset = EOExtension.ext(asset, add_if_missing=False)
eo_asset.apply(forest_mask_bands)
collection_item.add_asset("forest_mask", asset)

# Update item geometry and bbox after processing assets
collection_item.bbox = bbox
collection_item.geometry = footprint

stac_items.append(collection_item)


print(f"done create stac items")
# Generate Collection BBox and Footprint
unioned_footprint = shape(stac_items[0].geometry)
for item in stac_items[1:]:
    unioned_footprint = unioned_footprint.union(shape(item.geometry))

collection_bbox = list(unioned_footprint.bounds)
spatial_extent = pystac.SpatialExtent(bboxes=[collection_bbox])

collection_interval = sorted([item.datetime for item in stac_items])
temporal_extent = pystac.TemporalExtent(intervals=[[collection_interval[0], collection_interval[-1]]])

collection_extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)


# Create STAC Collection
collection = pystac.Collection(
    id='jaxa_forest_cover',
    description='Raster images from public S3 bucket',
    extent=collection_extent,
    license='CC-BY-SA-4.0',
    href=f'{base_url}/collection.json'
)

collection.add_items(stac_items)

# Create Catalog
catalog = pystac.Catalog(
    id='catalog-with-collection',
    description='STAC Catalog with Collection for S3-hosted Jaxa world forest data',
    href=f'{base_url}/jaxa_forest_cover_{area}_catalog.json'  # S3 path reference
)
catalog.add_child(collection)

print(f"creating catalog")
catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED, dest_href='/mnt/hddarchive.nfs/amazonas_dir/openEO/jaxa_forestcover')

