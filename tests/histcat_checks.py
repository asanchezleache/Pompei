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
    check("spain: module present", os.path.exists(os.path.join(SCRIPTS, "download_ortho_MNS_spain.py")))
    check("spain: wired in download_ortho_MNS.sh", '"spain"' in src("download_ortho_MNS.sh"))
    check("spain: wired in download_mnt.py", '"spain"' in src("download_mnt.py"))
    with open(os.path.join(HERE, "..", "dockerfile"), encoding="utf-8",
              errors="ignore") as f:
        df = f.read()
    check("dockerfile: conda init (not mamba init)",
          "conda init" in df and "RUN mamba init" not in df)


def test_network():
    """L1 service probes THROUGH THE SHIPPING MODULE (opt-in)."""
    import numpy as np, rasterio, warnings, tempfile, shutil
    warnings.filterwarnings("ignore")
    sys.path.insert(0, SCRIPTS)
    import download_ortho_MNS_spain as sp
    tmp = tempfile.mkdtemp()
    md = os.path.join(tmp, "metadata"); os.makedirs(md)
    with open(os.path.join(md, "EPSG.txt"), "w") as f:
        f.write("25830\n")
    bbox = [450000, 4477000, 450500, 4477500]      # 0.5 km box, few tiles
    try:
        sp.fetch_ortho(bbox, md, margin=200.0)
        sp.fetch_terrain(bbox, md, subdir="mns", prefix="MNS", margin=500.0)
        import glob
        o = glob.glob(os.path.join(md, "ortho", "ORTHO_dalle_*.tif"))
        t = glob.glob(os.path.join(md, "mns", "MNS_dalle_*.tif"))
        ok_o = ok_t = False
        if o:
            with rasterio.open(o[0]) as d:
                a = d.read()
                ok_o = (d.crs.to_epsg() == 25830 and d.count == 3
                        and d.dtypes[0] == "uint8" and float(a.std()) > 5)
        if t:
            with rasterio.open(t[0]) as d:
                a = d.read(1)
                ok_t = (d.crs.to_epsg() == 25830 and d.dtypes[0] == "int16"
                        and 300 < float(np.median(a[a > -30000])) < 1500)
        check("network: module fetch_ortho live + content", ok_o, f"{len(o)} tiles")
        check("network: module fetch_terrain live + content", ok_t, f"{len(t)} tiles")
    except Exception as e:
        check("network: spain module fetch", False, str(e)[:120])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_supplied(tmp):
    """Production path: chantiers ship with metadata/{ortho,mns,mnt}/*.tif
    cut at staging. The spain module must keep them, skip the stream, and
    clear stale tiles around them. Run with network env poisoned so any
    accidental HTTP call fails loudly."""
    md = os.path.join(tmp, "metadata")
    for sub, name in (("ortho", "ORTHO.tif"), ("mns", "MNS.tif"),
                      ("mnt", "MNT.tif")):
        os.makedirs(os.path.join(md, sub))
        with open(os.path.join(md, sub, name), "wb") as f:
            f.write(b"not-a-tiff-but-nonempty")
    # a stale streamed tile from a previous mode must be cleared
    open(os.path.join(md, "mns", "MNS_dalle_3_4.tif"), "w").close()
    open(os.path.join(md, "EPSG.txt"), "w").write("25830\n")
    open(os.path.join(md, "bbox.txt"), "w").write("1\n1\n2\n2\n")
    env = dict(os.environ, https_proxy="http://127.0.0.1:1",
               http_proxy="http://127.0.0.1:1")
    r = subprocess.run([sys.executable,
                        os.path.join(SCRIPTS, "download_ortho_MNS_spain.py"),
                        "--metadata", md],
                       capture_output=True, text=True, env=env, timeout=60)
    out = r.stdout + r.stderr
    check("supplied: module exits 0 without network", r.returncode == 0,
          out[-150:])
    kept = all(os.path.exists(os.path.join(md, s, n))
               for s, n in (("ortho", "ORTHO.tif"), ("mns", "MNS.tif")))
    check("supplied: shipped rasters survive the purge", kept)
    check("supplied: stale tiles cleared",
          not os.path.exists(os.path.join(md, "mns", "MNS_dalle_3_4.tif")))
    check("supplied: stream skipped (printed)",
          "stream skipped" in out)


def main():
    import tempfile
    print("HistCat fork preflight (L2)")
    test_equations()
    with tempfile.TemporaryDirectory() as tmp:
        test_auto_mask(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_supplied(tmp)
    test_static_markers()
    if "--network" in sys.argv:
        test_network()
    print(f"\nRESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
