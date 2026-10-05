"""Headless replacement for mm3d SaisieMasq (Pompei patch, HistCat).

Writes a binary mask for the scanned-film case: white = photo content to use,
black = film border / scanner surround to ignore. Largest bright connected
component (downsampled), holes filled, eroded for safety. Kept georeferenced
(copy the source grid) so downstream GDAL steps can use it directly.
"""
import sys
import glob
import numpy as np
import rasterio
from scipy import ndimage

out = sys.argv[1] if len(sys.argv) > 1 else "filtre0.tif"
src = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("OIS-Reech_*.tif"))[0]
with rasterio.open(src) as d:
    a = d.read(1)
    prof = d.profile
    tf = d.transform
if a.ndim == 3:
    a = a.mean(axis=0)
small = a[::8, ::8]
bright = small > 25
lab, n = ndimage.label(bright)
if n:
    keep = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
else:
    keep = bright
keep = ndimage.binary_fill_holes(keep)
keep = ndimage.binary_erosion(keep, ndimage.generate_binary_structure(2, 1), iterations=3)
mask = np.kron(keep.astype(np.uint8), np.full((8, 8), 255, np.uint8))
mask = mask[:a.shape[0], :a.shape[1]]
prof.update(driver="GTiff", dtype="uint8", count=1, compress="none",
            nodata=None, tiled=False)   # Elise Tiff_Im rejects compressed masks
with rasterio.open(out, "w", **prof) as d:
    d.write(mask.astype(np.uint8), indexes=1)
print(f"auto mask {out}: {100 * (mask > 0).mean():.1f}% kept")
