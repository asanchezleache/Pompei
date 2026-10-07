"""Headless replacement for mm3d SaisieMasq (Pompei patch, HistCat).

Writes a binary mask for the scanned-film case: white = photo content to use,
black = film border / scanner surround to ignore. The photo is found by
photo_frame.detect (largest bright region at 1/8, holes filled, data strip cut
by the square trim) and eroded by a safety margin (photo_frame.MARGIN). Kept
georeferenced (copy the source grid) so downstream GDAL steps can use it directly.

  auto_mask_pompei.py [out] [src]           one mask (default filtre0.tif) from src
                                            (default: the first OIS-Reech_*.tif)
  auto_mask_pompei.py --per-image [out]     also one mask per frame, filtre_<frame>.tif,
                                            for every OIS-Reech_*.tif (MyKeyCalculMasq
                                            maps each frame to its own mask; Malt's
                                            per-image masks are copied from them)

Per frame matters in zero-fiducial mode: the photo sits 130-150 px differently in
each scan (madrid_0559_00K/L), so one shared mask is off on the other frames.
"""
import sys
import glob
import os
import rasterio

import photo_frame


def write_mask(src, out):
    with rasterio.open(src) as d:
        a = d.read(1)
        prof = d.profile
    det = photo_frame.detect(a[::photo_frame.STEP, ::photo_frame.STEP])
    m = photo_frame.mask(det, a.shape)
    prof.update(driver="GTiff", dtype="uint8", count=1, compress="none",
                nodata=None, tiled=False)   # Elise Tiff_Im rejects compressed masks
    with rasterio.open(out, "w", **prof) as d:
        d.write(m, indexes=1)
    trim = f", data strip cut on the {det['trimmed']}" if det["trimmed"] else ""
    print(f"auto mask {out} ({os.path.basename(src)}): {100 * (m > 0).mean():.1f}% kept{trim}")


def frames():
    return sorted(f for f in glob.glob("OIS-Reech_*.tif") if not f.endswith("_Masq.tif"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--per-image"]
    out = args[0] if len(args) > 0 else "filtre0.tif"
    src = args[1] if len(args) > 1 else frames()[0]
    write_mask(src, out)
    if "--per-image" in sys.argv:
        for f in frames():
            write_mask(f, f"filtre_{f}")
