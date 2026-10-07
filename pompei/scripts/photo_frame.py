"""Find the photo inside a scanned film frame (HistCat, headless).

One detection, two consumers, so they cannot drift apart:
- auto_mask_pompei.py: the image-space masks (tie-point filter and Malt), one per frame;
- HistCat staging: each frame's ground footprint (TA polygon2d), i.e. the photo, not the whole scan.

A scan holds the photo (a square on the film) plus a black film surround and, on one side, the camera's data strip
(instrument dials, clock, frame number). Detection: at 1/STEP resolution, the largest bright connected region, holes
filled. When the data strip touches the photo it joins that region and makes it clearly wider than tall (or taller
than wide); the excess is cut on the darker side (dark film with a few bright dials), keeping SQUARE_KEEP of the
shorter side, so no data-strip edge survives (madrid_0559_00L: 3 of 10 frames needed it).
"""
import numpy as np
from scipy import ndimage

STEP = 8             # work at 1/8 of the scan resolution
THRESH = 25          # grey level separating the photo from the black film surround
SQUARE_TOL = 1.03    # region more than 3 % off square = data strip attached
SQUARE_KEEP = 0.98   # after a trim, keep 98 % of the shorter side
MARGIN = 0.01        # mask safety margin: 1 % of the photo side (~150 px on a 15,700 px photo)


def detect(small, step=STEP, thresh=THRESH):
    """small: 2-D array at 1/step resolution (e.g. a[::step, ::step]).

    Returns dict(keep=bool array at 1/step (photo, no margin), bbox=(c0, r0, c1, r1) in full-resolution pixels
    (end exclusive), trimmed=None or the side cut ("left", "right", "top", "bottom"), step=step)."""
    small = np.asarray(small)
    if small.ndim == 3:
        small = small.mean(axis=0)
    bright = small > thresh
    lab, n = ndimage.label(bright)
    keep = lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else bright
    keep = ndimage.binary_fill_holes(keep)
    trimmed = None
    box = _box(keep)
    if box is not None:
        r0, r1, c0, c1 = box
        h, w = r1 - r0, c1 - c0
        if w > SQUARE_TOL * h:
            e = w - int(SQUARE_KEEP * h)
            if small[r0:r1, c0:c0 + e].mean() < small[r0:r1, c1 - e:c1].mean():
                keep[:, :c0 + e], trimmed = False, "left"
            else:
                keep[:, c1 - e:], trimmed = False, "right"
        elif h > SQUARE_TOL * w:
            e = h - int(SQUARE_KEEP * w)
            if small[r0:r0 + e, c0:c1].mean() < small[r1 - e:r1, c0:c1].mean():
                keep[:r0 + e, :], trimmed = False, "top"
            else:
                keep[r1 - e:, :], trimmed = False, "bottom"
        box = _box(keep)
    if box is None:
        bbox = (0, 0, small.shape[1] * step, small.shape[0] * step)
    else:
        r0, r1, c0, c1 = box
        bbox = (c0 * step, r0 * step, c1 * step, r1 * step)
    return dict(keep=keep, bbox=bbox, trimmed=trimmed, step=step)


def _box(keep):
    rows, cols = np.nonzero(keep.any(1))[0], np.nonzero(keep.any(0))[0]
    if not rows.size or not cols.size:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def mask(det, shape, margin=MARGIN):
    """Full-resolution uint8 mask of shape `shape` (255 = use): the detected photo eroded by margin x photo side."""
    step = det["step"]
    c0, r0, c1, r1 = det["bbox"]
    it = max(1, int(round(margin * min(c1 - c0, r1 - r0) / step)))
    k = ndimage.binary_erosion(det["keep"], ndimage.generate_binary_structure(2, 1), iterations=it)
    m = np.kron(k.astype(np.uint8), np.full((step, step), 255, np.uint8))
    out = np.zeros(shape, np.uint8)
    h, w = min(shape[0], m.shape[0]), min(shape[1], m.shape[1])
    out[:h, :w] = m[:h, :w]
    return out


def read_small(path, step=STEP):
    """Band 1 of a raster at 1/step resolution (rasterio decimated read, ceil like a[::step, ::step])."""
    import rasterio
    with rasterio.open(path) as d:
        shape = (-(-d.height // step), -(-d.width // step))
        return d.read(1, out_shape=shape), d.transform, (d.height, d.width)


def footprint(path, step=STEP):
    """Ground footprint of the photo in a georeferenced scan, as Pompei's TA polygon2d wants it: 8 (x, y) points,
    corners and edge midpoints, counter-clockwise from the lower-left. The detected photo box is mapped through the
    raster's own transform (rotation included). Also returns the detection, for logging."""
    small, transform, _ = read_small(path, step)
    det = detect(small, step)
    c0, r0, c1, r1 = det["bbox"]
    ll, lr = transform * (c0, r1), transform * (c1, r1)
    ur, ul = transform * (c1, r0), transform * (c0, r0)
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    pts = [ll, mid(ll, lr), lr, mid(lr, ur), ur, mid(ur, ul), ul, mid(ul, ll)]
    return pts, det
