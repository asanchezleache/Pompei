"""Which image pairs really overlap? (HistCat)

Two frames count as overlapping when their footprints (flight_plan/flight_plan.shp, from the TA polygon2d) share at
least MIN_OVERLAP of the smaller footprint's area. The vanilla test was "the footprints touch at all": on
madrid_0559_00L, footprints a few per cent too large made frames i and i+3 touch by 23-106 m, the filter kept those
pairs, and their ~500 noise matches each made Tapas diverge (2.1 px); with only real overlaps it converged (0.85 px).
Real neighbours overlap far more (i+-1 ~60 %, i+-2 ~30 % on a 60 % forward-overlap strip).

Used by delete_tie_points.py (after Tapioca) and, as a script, on CouplesTA.xml before Tapioca so that
non-overlapping pairs are not even matched (fewer noise matches, shorter Tapioca):

  python pair_filter.py --couples CouplesTA.xml --footprints flight_plan/flight_plan.shp [--min_overlap 0.1]

MIN_OVERLAP comes from $POMPEI_MIN_OVERLAP when set (default 0.10).
"""
import argparse
import os

MIN_OVERLAP = float(os.environ.get("POMPEI_MIN_OVERLAP", "0.10"))


def read_footprints(path):
    """{image name as in the TA: geometry} from the flight-plan shapefile ('nom' column)."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    return {str(n).strip(): g for n, g in zip(gdf["nom"], gdf.geometry)}


def overlap_fraction(a, b):
    """Intersection area over the smaller footprint's area (0 when they do not intersect)."""
    if not a.intersects(b):
        return 0.0
    small = min(a.area, b.area)
    return a.intersection(b).area / small if small > 0 else 0.0


def overlapping(name_1, name_2, footprints, min_overlap=None):
    m = MIN_OVERLAP if min_overlap is None else min_overlap
    return overlap_fraction(footprints[name_1], footprints[name_2]) >= m


def ta_name(ois_name):
    """OIS-Reech_<image>.tif -> <image> (the TA / flight-plan name)."""
    n = os.path.basename(ois_name.strip())
    n = n[len("OIS-Reech_"):] if n.startswith("OIS-Reech_") else n
    return n[:-4] if n.endswith(".tif") else n


def filter_couples(couples_path, footprints_path, min_overlap=None):
    """Rewrite CouplesTA.xml (MicMac SauvegardeNamedRel) keeping only overlapping pairs; returns (kept, total)."""
    from lxml import etree
    tree = etree.parse(couples_path)
    root = tree.getroot()
    fp = read_footprints(footprints_path)
    cples = root.findall(".//Cple")
    kept = 0
    for c in cples:
        a, b = c.text.split()
        na, nb = ta_name(a), ta_name(b)
        if na in fp and nb in fp and not overlapping(na, nb, fp, min_overlap):
            c.getparent().remove(c)
        else:
            kept += 1
    tree.write(couples_path, xml_declaration=True, encoding="utf-8")
    return kept, len(cples)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--couples", required=True)
    p.add_argument("--footprints", required=True)
    p.add_argument("--min_overlap", type=float, default=None)
    a = p.parse_args()
    if not os.path.exists(a.footprints):
        print(f"pair_filter: {a.footprints} missing (run visualize_flight_plan.sh first): all pairs kept")
    else:
        kept, total = filter_couples(a.couples, a.footprints, a.min_overlap)
        m = MIN_OVERLAP if a.min_overlap is None else a.min_overlap
        print(f"pair_filter: {kept} of {total} pairs overlap by >= {m:.0%} of the smaller footprint; the rest dropped")
