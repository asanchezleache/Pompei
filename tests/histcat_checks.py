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


def _scan_with_data_strip(path, transform):
    """Synthetic 1100 x 1200 scan: black surround, textured photo square rows/cols 150-950, and a data strip
    (dark film, two bright dials) attached on the right, cols 950-1150 -- the madrid_0559_00L case."""
    import numpy as np, rasterio
    rng = np.random.default_rng(0)
    a = np.zeros((1100, 1200), np.uint8)
    a[150:950, 150:950] = rng.integers(90, 210, (800, 800))
    a[150:950, 950:1150] = 40
    a[300:340, 1040:1080] = 220
    a[700:740, 1040:1080] = 220
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", width=1200, height=1100, count=1,
                       crs="EPSG:25830", transform=transform) as d:
        d.write(a, indexes=1)


def test_photo_frame(tmp):
    """photo_frame: the data strip is cut (square trim) and the photo box maps to the right ground footprint."""
    import shutil
    import rasterio
    sys.path.insert(0, SCRIPTS)
    import photo_frame
    img = os.path.join(tmp, "scan_strip.tif")
    tf = rasterio.Affine(0.5, 0, 1000.0, 0, -0.5, 5000.0)      # 0.5 m pixels, north-up
    _scan_with_data_strip(img, tf)
    pts, det = photo_frame.footprint(img)
    c0, r0, c1, r1 = det["bbox"]
    check("photo_frame: data strip cut on the right", det["trimmed"] == "right", str(det["trimmed"]))
    check("photo_frame: photo box ~ cols 150-950, rows 150-950 (strip excluded)",
          abs(c0 - 150) <= 16 and abs(r0 - 150) <= 16 and abs(r1 - 950) <= 16 and 900 <= c1 <= 960,
          str(det["bbox"]))
    ll, ur = pts[0], pts[4]
    check("photo_frame: footprint in ground coordinates (8 points, LL/UR)",
          len(pts) == 8 and abs(ll[0] - 1075) <= 10 and abs(ll[1] - 4525) <= 10
          and abs(ur[1] - 4925) <= 10 and ur[0] < 1000 + 960 * 0.5 + 1, f"LL {ll} UR {ur}")
    # per-image masks, as initialize_files.py now calls the script
    work = os.path.join(tmp, "perimage")
    os.makedirs(work)
    for n in ("OIS-Reech_a.tif", "OIS-Reech_b.tif"):
        shutil.copy(img, os.path.join(work, n))
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "auto_mask_pompei.py"), "--per-image", "filtre0.tif"],
                       capture_output=True, text=True, cwd=work)
    made = all(os.path.exists(os.path.join(work, f)) for f in
               ("filtre0.tif", "filtre_OIS-Reech_a.tif", "filtre_OIS-Reech_b.tif"))
    check("auto_mask: --per-image writes filtre0 + one mask per frame", r.returncode == 0 and made,
          (r.stderr or "")[-120:])
    if made:
        with rasterio.open(os.path.join(work, "filtre_OIS-Reech_a.tif")) as d:
            m = d.read(1)
        check("auto_mask: data strip and margin excluded, photo kept",
              m[550, 550] == 255 and m[320, 1060] == 0 and m[550, 152] == 0)


def test_pair_filter(tmp):
    """A pair counts only with a real footprint overlap (>= 10 % of the smaller), before Tapioca and after."""
    import geopandas as gpd
    from shapely.geometry import box
    sys.path.insert(0, SCRIPTS)
    import pair_filter
    fp = {"A": box(0, 0, 10, 10), "B": box(6, 0, 16, 10), "C": box(9.9, 0, 19.9, 10)}
    check("pair_filter: 40 % overlap kept, 1 % sliver dropped",
          pair_filter.overlapping("A", "B", fp, 0.1) and not pair_filter.overlapping("A", "C", fp, 0.1))
    os.makedirs(os.path.join(tmp, "flight_plan"))
    shp = os.path.join(tmp, "flight_plan", "flight_plan.shp")
    gpd.GeoDataFrame({"nom": list(fp)}, geometry=list(fp.values()), crs="EPSG:25830").to_file(shp)
    couples = os.path.join(tmp, "CouplesTA.xml")
    with open(couples, "w") as f:
        f.write('<?xml version="1.0" ?>\n<SauvegardeNamedRel>\n'
                '<Cple>OIS-Reech_A.tif OIS-Reech_B.tif</Cple>\n<Cple>OIS-Reech_A.tif OIS-Reech_C.tif</Cple>\n'
                '<Cple>OIS-Reech_B.tif OIS-Reech_C.tif</Cple>\n</SauvegardeNamedRel>\n')
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "pair_filter.py"), "--couples", couples,
                        "--footprints", shp], capture_output=True, text=True)
    txt = open(couples).read()
    check("pair_filter: CouplesTA.xml keeps A-B and B-C, drops A-C",
          r.returncode == 0 and "OIS-Reech_A.tif OIS-Reech_B.tif" in txt and "OIS-Reech_B.tif OIS-Reech_C.tif" in txt
          and "OIS-Reech_A.tif OIS-Reech_C.tif" not in txt, (r.stdout + r.stderr)[-120:])
    # delete_tie_points.py on the same footprints
    hin = os.path.join(tmp, "HomolMasqFiltered")
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        os.makedirs(os.path.join(hin, f"PastisOIS-Reech_{a}.tif"), exist_ok=True)
        open(os.path.join(hin, f"PastisOIS-Reech_{a}.tif", f"OIS-Reech_{b}.tif.dat"), "w").write("x")
    hout = os.path.join(tmp, "Homol")
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "delete_tie_points.py"), "--homol_input", hin,
                        "--homol_output", hout, "--footprints", shp], capture_output=True, text=True)
    kept = sorted(os.path.relpath(os.path.join(d, f), hout) for d, _, fs in os.walk(hout) for f in fs)
    check("delete_tie_points: keeps A-B and B-C, drops the A-C sliver",
          r.returncode == 0 and len(kept) == 2
          and os.path.join("PastisOIS-Reech_A.tif", "OIS-Reech_C.tif.dat") not in kept,
          f"{kept} {(r.stderr or '')[-80:]}")


def test_static_markers():
    """Source-level presence of each structural patch."""
    ini = src("initialize_files.py")
    check("initialize_files: auto-mask wired", "auto_mask_pompei.py" in ini)
    check("initialize_files: per-image masks (0-fiducial) + MyKeyCalculMasq per frame",
          "auto_mask_pompei.py --per-image" in ini and '"filtre_$1"' in ini)
    check("initialize_files: CouplesTA filtered by footprint overlap before Tapioca", "pair_filter.py" in ini)
    check("create_mask_per_image: per-frame mask preferred for Malt",
          "filtre_OIS-Reech_" in src("create_mask_per_image.py"))
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
    check("mosaiquage: seamlines clipped to the chantier + outline guard + layout check",
          "HISTCAT CLIP" in mos and "HISTCAT GUARD" in mos and "HISTCAT LAYOUT" in mos)
    big = src("create_big_Ortho.py")
    check("create_big_Ortho: polygons burnt as i+1, empty pixels filled, full read window",
          "compte + 1" in big and "HISTCAT FILL" in big and "min(inputds.RasterXSize, max_c)" in big)
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
    with tempfile.TemporaryDirectory() as tmp:
        test_photo_frame(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_pair_filter(tmp)
    test_static_markers()
    if "--network" in sys.argv:
        test_network()
    print(f"\nRESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
