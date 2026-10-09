"""Refine find_GCP's anchor pairs before aero.sh uses them (HistCat, session "Pompei cities", 2026-10-09; problem 14 of
01_data/03_imagery/pompei/run_ledger.md).

find_GCP (IGN binaries) does not resolve displacements of a couple of metres between the trial ortho and the reference:
on a synthetic case with a known shift of (+1.85, -1.50) m it reports (0, 0), and larger shifts come out 0.3-0.8 m
off. The anchors then keep the residual of the coarse (1/10) alignment, the bundle fits them exactly, and every
product is shifted (Zaragoza 0383: ~+1.85 / -1.5 m against PNOA). Here the displacement is measured again, robustly:
on WIN-metre windows the trial ortho, shifted by the pairs' own median displacement there, is phase-correlated with the
reference on gradient magnitude (sub-pixel, upsample 10); the residual offsets of the reliable windows (ncc >= MIN_NCC)
form a correction field, interpolated (inverse distance, K nearest windows) to every pair, and the trial-side
coordinates are corrected: histo += e. Without enough reliable windows nothing is changed.

Coordinates follow the chain's conventions: the trial tiles' .tfw are read as the pixel CORNER (as Pompei writes its
own tfw and as convert_ori tfw2ori expects; the synthetic test is exact with that), the reference through its GeoTIFF.

  python refine_GCP.py --bdortho pts_bdortho.txt --histo pts_orthomicmac_abs.txt --ref ../metadata/ortho_temp/ORTHO.tif \
      --trial_dir ../Ortho-MEC-Malt-Abs-Ratafia --epsg 25830 [--report refine_GCP_report.txt]
"""
import argparse
import glob
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from skimage.registration import phase_cross_correlation

RES = 0.5


def trial_vrt(trial_dir, out):
    """VRT over the trial tiles with their .tfw read as the pixel corner."""
    parts = []
    for t in sorted(glob.glob(os.path.join(trial_dir, "Orthophotomosaic_Tile_*.tif"))):
        a, _, _, e, x0, y0 = [float(v) for v in open(t[:-4] + ".tfw").read().split()]
        with rasterio.open(t) as d:
            w, h = d.width, d.height
        p = f"{out}_{len(parts)}.vrt"
        with open(p, "w") as f:
            f.write(f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}"><GeoTransform>{x0}, {a}, 0, {y0}, 0, {e}'
                    f'</GeoTransform><VRTRasterBand dataType="Byte" band="1"><SimpleSource><SourceFilename '
                    f'relativeToVRT="0">{os.path.abspath(t)}</SourceFilename><SourceBand>1</SourceBand></SimpleSource>'
                    f'</VRTRasterBand></VRTDataset>')
        parts.append(p)
    if not parts:
        raise SystemExit(f"no Orthophotomosaic_Tile_*.tif in {trial_dir}")
    import subprocess
    subprocess.run(["gdalbuildvrt", "-q", out, *parts], check=True)
    return out


def grid(path, b, crs, win, src_crs=None, band=1):
    n = int(round(win / RES))
    with rasterio.open(path) as d, WarpedVRT(d, src_crs=src_crs or d.crs, crs=crs, transform=from_origin(b[0], b[3], RES, RES),
                                             width=n, height=n, resampling=Resampling.average, nodata=0) as v:
        return v.read(band).astype(np.float32)


def shift(a, ref):
    """Offset of a's content relative to ref's (m; x east, y north) and the aligned-gradient correlation."""
    ga, gr = (np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1)) for x in (a, ref))
    sh, _, _ = phase_cross_correlation(gr, ga, upsample_factor=10)
    m = (slice(40, -40), slice(40, -40))
    q = float(np.corrcoef(gr[m].ravel(), ndimage.shift(ga, sh, order=1)[m].ravel())[0, 1])
    return -sh[1] * RES, sh[0] * RES, q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdortho", required=True)
    ap.add_argument("--histo", required=True, help="trial-ortho side of the pairs; rewritten in place (.orig kept)")
    ap.add_argument("--ref", required=True, help="reference raster (PNOA, band 1 used)")
    ap.add_argument("--trial_dir", required=True)
    ap.add_argument("--epsg", type=int, required=True)
    ap.add_argument("--win", type=float, default=400.0)
    ap.add_argument("--step", type=float, default=1000.0)
    ap.add_argument("--min_pairs", type=int, default=30)
    ap.add_argument("--min_ncc", type=float, default=0.2)
    ap.add_argument("--min_windows", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--report", default="refine_GCP_report.txt")
    a = ap.parse_args()
    crs = CRS.from_epsg(a.epsg)
    trial = trial_vrt(a.trial_dir, "/tmp/refine_trial.vrt")
    bd = np.loadtxt(a.bdortho, ndmin=2)
    om = np.loadtxt(a.histo, ndmin=2)
    d = om - bd
    lines = [f"{len(bd)} pairs; trial minus reference median {np.median(d, axis=0).round(3)} m"]
    win = []
    (x0, y0), (x1, y1) = bd.min(axis=0), bd.max(axis=0)
    for cy in np.arange(y0 + a.win / 2, y1 - a.win / 2 + 1, a.step):
        for cx in np.arange(x0 + a.win / 2, x1 - a.win / 2 + 1, a.step):
            inw = (abs(bd[:, 0] - cx) < a.win / 2) & (abs(bd[:, 1] - cy) < a.win / 2)
            if inw.sum() < a.min_pairs:
                continue
            ddx, ddy = np.median(d[inw], axis=0)
            wb = (cx - a.win / 2, cy - a.win / 2, cx + a.win / 2, cy + a.win / 2)
            t = grid(trial, (wb[0] + ddx, wb[1] + ddy, wb[2] + ddx, wb[3] + ddy), crs, a.win, src_crs=crs)
            r = grid(a.ref, wb, crs, a.win)
            if min((t > 0).mean(), (r > 0).mean()) < 0.95:
                continue
            ex, ey, q = shift(t, r)
            if q >= a.min_ncc and abs(ex) < a.win / 4 and abs(ey) < a.win / 4:
                win.append((cx, cy, ex, ey, q))
    w = np.array(win)
    if len(w) < a.min_windows:
        lines.append(f"only {len(w)} reliable windows (< {a.min_windows}): pairs left unchanged")
        open(a.report, "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    lines.append(f"{len(w)} reliable windows: correction median dx {np.median(w[:, 2]):+.3f} dy {np.median(w[:, 3]):+.3f}"
                 f" m, IQR dx {np.subtract(*np.percentile(w[:, 2], [75, 25])):.2f} dy "
                 f"{np.subtract(*np.percentile(w[:, 3], [75, 25])):.2f}")
    # inverse-distance field over the K nearest reliable windows
    dist = np.hypot(bd[:, None, 0] - w[None, :, 0], bd[:, None, 1] - w[None, :, 1])
    k = min(a.k, len(w))
    idx = np.argsort(dist, axis=1)[:, :k]
    dk = np.take_along_axis(dist, idx, axis=1) + 1.0
    wt = 1.0 / dk ** 2
    e = np.stack([(wt * w[idx, 2]).sum(1) / wt.sum(1), (wt * w[idx, 3]).sum(1) / wt.sum(1)], axis=1)
    shutil.copy(a.histo, a.histo + ".orig")
    np.savetxt(a.histo, om + e, fmt="%.3f")
    lines.append(f"pairs corrected: mean correction {e.mean(axis=0).round(3)} m; trial minus reference now median "
                 f"{np.median(om + e - bd, axis=0).round(3)} m")
    open(a.report, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
