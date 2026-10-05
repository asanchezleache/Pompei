# -*- coding: utf-8 -*-
"""HistCat: IGN-E streaming downloads for Pompei (mode `spain`).

Same tile-loop shape as the vanilla `wms` mode (download_ortho_MNS_wms.py),
pointed at Spanish OGC services verified live 2026-10-05:

- reference orthophoto : WMS  https://www.ign.es/wms-inspire/pnoa-ma
                         layer OI.OrthoimageCoverage, image/tiff, 0.5 m
                         (georeferenced; 3-band uint8 — probe-verified)
- terrain (MNT and MNS): WCS  https://servicios.idee.es/wcs-inspire/mdt
                         coverage Elevacion<EPSG>_5, image/tiff, 5 m
                         (int16 TRUE elevations — the MDT WMS is a uint8
                         colour-ramp and must NOT be used as a DEM)

Extent logic is vanilla's: bbox derived by Pompei from the chantier itself
(metadata/bbox.txt) plus a margin. Ortho margin 500 m (navigation error
only). Terrain margin 2000 m: create_orthos_OIS_Reech projects each photo's
corner rays to the ground and iterates on the DEM; rays landing outside the
terrain raster diverge to NaN (the failure class this mode removes).

Tiles are named ORTHO_dalle_i_j.tif / MNS_dalle_i_j.tif to match the vrt
patterns of the shared tail of download_ortho_MNS.sh. Resumable: existing
non-empty tiles are skipped.
"""
import os
import argparse

import requests

ORTHO_URL = "https://www.ign.es/wms-inspire/pnoa-ma"
ORTHO_LAYER = "OI.OrthoimageCoverage"
ORTHO_RESOL = 0.5      # m, matches vanilla wms mode
TILE_M = 1000.0        # tile size, m, matches vanilla

WCS_URL = "https://servicios.idee.es/wcs-inspire/mdt"
TERRAIN_COVERAGE = "Elevacion{epsg}_5"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 180
RETRIES = 3


def _get(url, params, dest):
    """One tile: GET to dest, resumable, RETRIES attempts; True on success.
    params may be a dict or a list of (k, v) pairs (for repeated keys such as
    WCS subset=, which requests cannot express as a dict)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for _ in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest + ".part", "wb") as out:
                    out.write(r.content)
                os.replace(dest + ".part", dest)
                return True
        except Exception:
            pass
    return False


def _tile_edges(e_min, e_max, n_min, n_max):
    """1 km tile edges + endpoints, same construction as the vanilla loop."""
    es = list(range(int(e_min), int(e_max), int(TILE_M))) + [e_max]
    ns = list(range(int(n_min), int(n_max), int(TILE_M))) + [n_max]
    return es, ns


def fetch_ortho(bbox, metadata="metadata", margin=500.0):
    epsg = int(open(os.path.join(metadata, "EPSG.txt")).read())
    emin, nmin, emax, nmax = bbox
    emin -= margin; nmin -= margin; emax += margin; nmax += margin
    out = os.path.join(metadata, "ortho")
    os.makedirs(out, exist_ok=True)
    es, ns = _tile_edges(emin, emax, nmin, nmax)
    failed = 0
    for i in range(len(es) - 1):
        for j in range(len(ns) - 1):
            w = int((es[i + 1] - es[i]) / ORTHO_RESOL)
            h = int((ns[j + 1] - ns[j]) / ORTHO_RESOL)
            params = {"service": "WMS", "version": "1.3.0", "request": "GetMap",
                      "layers": ORTHO_LAYER, "styles": "", "format": "image/tiff",
                      "crs": f"EPSG:{epsg}",
                      "bbox": f"{es[i]},{ns[j]},{es[i+1]},{ns[j+1]}",
                      "width": w, "height": h}
            dest = os.path.join(out, f"ORTHO_dalle_{i}_{j}.tif")
            if not _get(ORTHO_URL, params, dest):
                failed += 1
    if failed:
        raise RuntimeError(f"ORTHO: {failed} tiles failed ({out})")
    return out


def fetch_terrain(bbox, metadata="metadata", subdir="mns", prefix="MNS",
                  margin=2000.0, epsg=None):
    if epsg is None:
        epsg = int(open(os.path.join(metadata, "EPSG.txt")).read())
    emin, nmin, emax, nmax = bbox
    emin -= margin; nmin -= margin; emax += margin; nmax += margin
    out = os.path.join(metadata, subdir)
    os.makedirs(out, exist_ok=True)
    coverage = TERRAIN_COVERAGE.format(epsg=epsg)
    es, ns = _tile_edges(emin, emax, nmin, nmax)
    failed = 0
    for i in range(len(es) - 1):
        for j in range(len(ns) - 1):
            params = [("service", "WCS"), ("version", "2.0.1"),
                      ("request", "GetCoverage"), ("coverageId", coverage),
                      ("subset", f"Long({es[i]},{es[i+1]})"),
                      ("subset", f"Lat({ns[j]},{ns[j+1]})"),
                      ("format", "image/tiff"), ("responses", "content")]
            dest = os.path.join(out, f"{prefix}_dalle_{i}_{j}.tif")
            if not _get(WCS_URL, params, dest):
                failed += 1
    if failed:
        raise RuntimeError(f"{prefix}: {failed} tiles failed ({out}) — "
                           f"coverage '{coverage}' exists for this zone?")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="metadata")
    args = parser.parse_args()
    bbox = [float(l) for l in open(os.path.join(args.metadata, "bbox.txt"))]
    fetch_ortho(bbox, args.metadata)
    fetch_terrain(bbox, args.metadata, subdir="mns", prefix="MNS")
    # the shared tail skips the French SRTM border-fill when this marker exists
    open(os.path.join(args.metadata, "mns", "PRELOADED_MNS"), "w").close()
    print("spain downloads complete")
