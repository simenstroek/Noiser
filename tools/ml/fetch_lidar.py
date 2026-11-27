"""Downloads USGS 3DEP 1 m bare-earth DEM tiles (public domain, no key; via The National Map) for the eight terrain types.

    python tools/ml/fetch_lidar.py [OUT_DIR]       default E:/_NoiserML/lidar

One tile per site; only its central 5000 x 5000 cells are read (the files are cloud-optimised GeoTIFFs), several at once,
since the USGS bucket can drop to about 100 KB/s per connection. Files are named <type>__<site>[_HOLD].tif; the _HOLD ones are kept out of training.
"""

import os
import sys
import urllib.parse
import urllib.request
import json

# type: [(site, lon, lat, held_out)]
SITES = {
    "alpine": [("yosemite", -119.58, 37.74, 0), ("rockymtn", -105.65, 40.30, 0), ("tetons", -110.80, 43.74, 1)],
    "desert": [("moab", -109.55, 38.57, 0), ("joshuatree", -116.00, 34.00, 0), ("deathvalley", -116.85, 36.45, 1)],
    "badlands": [("badlands_sd", -102.30, 43.75, 0), ("troosevelt", -103.45, 46.95, 0), ("paintedesert", -109.78, 35.07, 1)],
    "hills": [("wv", -80.50, 38.50, 0), ("pa", -77.80, 40.80, 0), ("vermont", -72.80, 43.90, 1)],
    "volcanic": [("sthelens", -122.19, 46.20, 0), ("lassen", -121.50, 40.50, 0), ("shasta", -122.19, 41.41, 1)],
    "karst": [("mammoth", -86.10, 37.20, 0), ("ozarks", -91.40, 37.10, 0), ("indiana", -86.50, 38.80, 1)],
    "coastal": [("bigsur", -121.80, 36.30, 0), ("mendocino", -123.78, 39.35, 0), ("olympic", -124.40, 47.90, 1)],
    "plains": [("iowa", -93.60, 42.00, 0), ("kansas", -98.00, 38.50, 0), ("nebraska", -98.50, 41.00, 1)],
}
API = "https://tnmaccess.nationalmap.gov/api/v1/products"


def find(lon, lat):
    q = {"datasets": "Digital Elevation Model (DEM) 1 meter", "bbox": f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}", "max": 20, "outputFormat": "JSON"}
    items = json.load(urllib.request.urlopen(API + "?" + urllib.parse.urlencode(q), timeout=60))["items"]
    items = [i for i in items if (i.get("sizeInBytes") or 0) < 450e6 and i.get("downloadURL", "").endswith(".tif")]
    for i in items:
        i["sizeInBytes"] = i.get("sizeInBytes") or 1
    return max(items, key=lambda i: i["sizeInBytes"]) if items else None


def crop(url, path, size=5000):
    import rasterio
    from rasterio.windows import Window

    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=2, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open("/vsicurl/" + url) as src:
            y, x = max(src.height - size, 0) // 2, max(src.width - size, 0) // 2
            win = Window(x, y, min(size, src.width), min(size, src.height))
            dem = src.read(1, window=win)
            prof = src.profile
            prof.update(width=win.width, height=win.height, transform=src.window_transform(win), driver="GTiff", compress="deflate", tiled=True, blockxsize=512, blockysize=512, predictor=3)
    with rasterio.open(path + ".part", "w", **prof) as dst:
        dst.write(dem, 1)
    os.replace(path + ".part", path)


def job(kind, site, lon, lat, hold, out):
    path = os.path.join(out, f"{kind}__{site}{'_HOLD' if hold else ''}.tif")
    if os.path.exists(path):
        return f"{kind}/{site}: have it"
    try:
        item = find(lon, lat)
        if not item:
            return f"{kind}/{site}: no 1 m tile here"
        crop(item["downloadURL"], path)
        return f"{kind}/{site}: {item['title']}"
    except Exception as e:
        return f"{kind}/{site}: failed ({e})"


def main():
    from concurrent.futures import ThreadPoolExecutor

    out = sys.argv[1] if len(sys.argv) > 1 else "E:/_NoiserML/lidar"
    os.makedirs(out, exist_ok=True)
    jobs = [(k, *site, out) for k, sites in SITES.items() for site in sites]
    with ThreadPoolExecutor(8) as pool:
        for line in pool.map(lambda j: job(*j), jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
