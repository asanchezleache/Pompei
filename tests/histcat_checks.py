# -*- coding: utf-8 -*-
"""L2 preflight battery for the HistCat fork of Pompei.

One check (or group) per ported patch; runs inside the pompei conda env:

    python tests/histcat_checks.py            # offline checks (~seconds)
    python tests/histcat_checks.py --network  # + L1 service probes (real bbox)

Exit code 0 = every patch is present AND behaves. Wired into the Dockerfile
as a build step, so a regressed patch fails the image build, not a 6-hour run.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "pompei", "scripts")
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        FAILS.append(name)


def src(name):
    with open(os.path.join(SCRIPTS, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


def test_equations():
    """Spanish EPSG entries + lon-lat axis order (equations.py commit)."""
    sys.path.insert(0, SCRIPTS)
    import numpy as np
    import equations
    from equations import Shot
    check("equations: 25830 in dict_EPSG", 25830 in equations.dict_EPSG)
    proj = Shot.getProj(25830)
    # Known UTM point in the eastern madrid_0559_00K strip (near Barajas):
    # x=452432 is ~47.6 km west of the zone-30 central meridian -> lon ~ -3.561
    lon, lat = proj["crs_to_crs_geog"].transform(np.array([452432.0]),
                                                 np.array([4476780.0]))
    check("equations: geographic order is LON,LAT (axis fix live)",
          abs(float(lon[0]) + 3.561) < 0.01 and abs(float(lat[0]) - 40.440) < 0.01,
          f"got lon={float(lon[0]):.3f} lat={float(lat[0]):.3f}")
    x, y, z = proj["crs_to_crs_geoc"].transform(np.array([452432.0]),
                                                np.array([4476780.0]),
                                                np.array([0.0]))
    import math
    r = math.sqrt(float(x[0])**2 + float(y[0])**2 + float(z[0])**2)
    check("equations: geocentric is ECEF metres (~6.37e6)",
          6.0e6 < r < 7.0e6, f"|r|={r:.3e}")


def test_auto_mask(tmp):
    """Headless mask on a synthetic scan (black border, bright square)."""
    import numpy as np, rasterio
    img = os.path.join(tmp, "scan.tif")
    a = np.zeros((800, 800), np.uint8)
    a[100:700, 100:700] = 200
    prof = dict(driver="GTiff", dtype="uint8", width=800, height=800,
                count=1, crs="EPSG:25830",
                transform=rasterio.Affine(1, 0, 0, 0, -1, 800))
    with rasterio.open(img, "w", **prof) as d:
        d.write(a, indexes=1)
    out = os.path.join(tmp, "mask.tif")
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "auto_mask_pompei.py"),
                        out, img], capture_output=True, text=True)
    check("auto_mask: runs headless", r.returncode == 0,
          (r.stderr or "")[-80:] if r.returncode else "")
    if r.returncode == 0:
        with rasterio.open(out) as d:
            m = d.read(1)
        check("auto_mask: keeps content, excludes border",
              m[400, 400] > 0 and m[10, 10] == 0 and d.dtypes[0] == "uint8")
        check("auto_mask: uncompressed (Elise requirement)",
              (d.compression or "NONE") in ("NONE", None), str(d.compression))


def test_static_markers():
    """Source-level presence of each structural patch."""
    ini = src("initialize_files.py")
    check("initialize_files: auto-mask wired", "auto_mask_pompei.py" in ini)
    check("initialize_files: tee pipe removed from Tapioca",
          "| tee reports/rapport_Tapioca.txt" not in ini
          and "> reports/rapport_Tapioca.txt" in ini)
    vfp = src("visualize_flight_plan.py")
    check("visualize_flight_plan: EPSG:nnnn accepted", "startswith" in vfp)
    check("visualize_flight_plan: .tif frames movable", '(".jp2", ".tif")' in vfp)
    check("analyze_Tapas: line_residual initialised",
          "line_residual = None" in src("analyze_Tapas.py"))
    dm = src("download_mnt.py")
    check("download_mnt: pre-placed vrt honoured",
          "mnt.vrt" in dm and "preloaded" in dm)
    check("download_ortho_MNS: PRELOADED_MNS guard",
          "PRELOADED_MNS" in src("download_ortho_MNS.sh"))
    check("cut_pleiade_images: mns/MNS.tif fallback",
          '"mns", "MNS.tif"' in src("cut_pleiade_images.py"))
    check("filter_tie_points: empty-Homol guard",
          "Homol empty" in src("filter_tie_points.sh"))
    mos = src("mosaiquage.py")
    check("mosaiquage: collinear + metadata EPSG",
          "hasattr(lines" in mos and "getEPSG(metadata)" in mos)
    with open(os.path.join(HERE, "..", "dockerfile"), encoding="utf-8",
              errors="ignore") as f:
        df = f.read()
    check("dockerfile: conda init (not mamba init)",
          "conda init" in df and "RUN mamba init" not in df)


def test_network():
    """L1 service probes over the real pilot bbox (slow, opt-in)."""
    import numpy as np, rasterio, warnings
    import requests
    warnings.filterwarnings("ignore")
    emin, nmin, te, tn = 428776, 4477003, 430776, 4479003
    wcs = (f"https://servicios.idee.es/wcs-inspire/mdt?service=WCS&version=2.0.1"
           f"&request=GetCoverage&coverageId=Elevacion25830_5"
           f"&subset=Long({emin},{te})&subset=Lat({nmin},{tn})"
           f"&format=image/tiff&responses=content")
    wms = (f"https://www.ign.es/wms-inspire/pnoa-ma?service=WMS&version=1.3.0"
           f"&request=GetMap&layers=OI.OrthoimageCoverage&styles=&crs=EPSG:25830"
           f"&bbox={emin},{nmin},{te},{tn}&width=1000&height=1000&format=image/tiff")
    for label, url, expect in [("WCS", wcs, "int16"), ("WMS", wms, "uint8")]:
        f = f"/tmp/histcat_pf_{label}.tif"
        ok = False
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
            with open(f, "wb") as fh:
                fh.write(r.content)
            with rasterio.open(f) as d:
                a = d.read(1)
                ok = (d.crs.to_epsg() == 25830 and d.dtypes[0] == expect
                      and (a > -30000).mean() > 0.9)
        except Exception:
            pass
        check(f"network: {label} service live + content", ok)


def main():
    import tempfile
    print("HistCat fork preflight (L2)")
    test_equations()
    with tempfile.TemporaryDirectory() as tmp:
        test_auto_mask(tmp)
    test_static_markers()
    if "--network" in sys.argv:
        test_network()
    print(f"\nRESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
