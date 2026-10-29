"""
board_detect.py - find the four corners of the chessboard and straighten it.

Pipeline (works for screenshots *and* for slightly rotated / tilted photos):

 1. PROPOSE candidate quadrilaterals
      a. OpenCV contours - the method of the desktop script (Canny -> contours ->
         4-corner polygons), when OpenCV is part of the APK;
      b. the NumPy checkerboard finder (board_finder.py), after undoing any tilt.
 2. REFINE every candidate: the 4 corner points are moved (coarse -> fine) until an
    8x8 grid laid over the quad lands exactly on the squares of the picture.
 3. VALIDATE: only a real board has two colours alternating on an 8x8 grid, so the
    candidate whose grid fits best wins.  This is what stops a big UI panel or the
    phone frame from being mistaken for the board.
 4. WARP the winning quad to an upright square with a perspective transform.

Corner order everywhere: top-left, top-right, bottom-right, bottom-left (as seen in the
picture).  Coordinates are pixels (x, y) of the image they belong to.
"""
import math

import numpy as np
from PIL import Image

import board_finder as bf

try:                                     # OpenCV is optional (adds contour proposals)
    import cv2
    HAVE_CV2 = True
except Exception:                        # noqa
    cv2 = None
    HAVE_CV2 = False

WORK_SIDE = 1000          # detection runs on a copy with at most this many pixels on the long side
MIN_FINAL_SCORE = 2.5     # grid-fit score needed to accept a board
MIN_AREA_FRAC = 0.02      # board must cover at least 2% of the picture


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def order_quad(pts):
    """4 points in any order -> float64 (4,2) as TL, TR, BR, BL."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])        # image coords: increasing = clockwise
    p = p[np.argsort(ang)]
    start = int(np.argmin(p.sum(axis=1)))                    # top-left = smallest x+y
    return np.roll(p, -start, axis=0)


def unit_square_H(quads):
    """Homographies mapping the unit square (0,0),(1,0),(1,1),(0,1) onto each quad.
    quads: (N,4,2) TL,TR,BR,BL  ->  (N,3,3)."""
    q = np.asarray(quads, dtype=np.float64)
    x0, y0, x1, y1, x2, y2, x3, y3 = (q[:, 0, 0], q[:, 0, 1], q[:, 1, 0], q[:, 1, 1],
                                      q[:, 2, 0], q[:, 2, 1], q[:, 3, 0], q[:, 3, 1])
    dx1, dy1, dx2, dy2 = x1 - x2, y1 - y2, x3 - x2, y3 - y2
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    den = dx1 * dy2 - dx2 * dy1
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    g = (sx * dy2 - dx2 * sy) / den
    h = (dx1 * sy - sx * dy1) / den
    H = np.zeros((len(q), 3, 3))
    H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = x1 - x0 + g * x1, x3 - x0 + h * x3, x0
    H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = y1 - y0 + g * y1, y3 - y0 + h * y3, y0
    H[:, 2, 0], H[:, 2, 1], H[:, 2, 2] = g, h, 1.0
    return H


def map_uv(H, uv):
    """H (N,3,3), uv (M,2) in the unit square -> (N,M,2) image points."""
    u, v = uv[:, 0][None, :], uv[:, 1][None, :]
    w = H[:, 2, 0:1] * u + H[:, 2, 1:2] * v + 1.0
    x = (H[:, 0, 0:1] * u + H[:, 0, 1:2] * v + H[:, 0, 2:3]) / w
    y = (H[:, 1, 0:1] * u + H[:, 1, 1:2] * v + H[:, 1, 2:3]) / w
    return np.stack([x, y], axis=-1)


def is_good_quad(q, img_w, img_h, min_area_frac=MIN_AREA_FRAC):
    """Convex, not degenerate, reasonably square-ish and big enough."""
    q = np.asarray(q, dtype=np.float64)
    cross = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False
    area = 0.5 * abs(sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1] for i in range(4)))
    if area < min_area_frac * img_w * img_h:
        return False
    sides = [np.hypot(*(q[(i + 1) % 4] - q[i])) for i in range(4)]
    return min(sides) > 12 and max(sides) / min(sides) < 1.9


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
            img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _box_blur(gray, r):
    if r <= 0:
        return gray
    p = np.pad(gray.astype(np.float64), r + 1, mode="edge")
    ii = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    h, w = gray.shape
    s = (ii[n + 1:n + 1 + h, n + 1:n + 1 + w] - ii[:h, n + 1:n + 1 + w]
         - ii[n + 1:n + 1 + h, :w] + ii[:h, :w])
    return (s / (n * n)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Grid-fit score
# ──────────────────────────────────────────────────────────────────────────────

_PARITY = ((np.arange(8)[:, None] + np.arange(8)[None, :]) % 2).reshape(64)
_UV_CACHE = {}


def _cell_uv(inset):
    if inset not in _UV_CACHE:
        pts = []
        for i in range(8):
            for j in range(8):
                for a, b in ((inset, inset), (1 - inset, inset), (inset, 1 - inset), (1 - inset, 1 - inset)):
                    pts.append(((j + a) / 8.0, (i + b) / 8.0))
        _UV_CACHE[inset] = np.array(pts)
    return _UV_CACHE[inset]


def grid_score(G, quads, inset):
    """How well does an 8x8 checkerboard fit inside each quad?  (N,4,2) -> (N,) score.
    Samples the corners of all 64 cells (pieces sit in the middle), splits the cells into
    the two colour classes by parity and compares: big colour gap, tiny spread = board."""
    quads = np.asarray(quads, dtype=np.float64)
    H = unit_square_H(quads)
    pts = map_uv(H, _cell_uv(inset))                                   # (N,256,2)
    vals = _bilinear(G, pts[..., 0], pts[..., 1]).reshape(len(quads), 64, 4)
    srt = np.sort(vals, axis=-1)
    bg = (srt[..., 1] + srt[..., 2]) / 2.0                             # robust cell colour (N,64)
    a, b = bg[:, _PARITY == 0], bg[:, _PARITY == 1]
    ma, mb = np.median(a, axis=1), np.median(b, axis=1)
    dev = (np.abs(a - ma[:, None]).sum(1) + np.abs(b - mb[:, None]).sum(1)) / 64.0
    d = np.abs(ma - mb)
    score = d / (dev + 1.0)
    score[d < bf.MIN_CONTRAST] = 0.0
    return score


def _moves(q, step):
    """All candidate quads reachable by one move: each corner +-x/+-y, whole-quad shifts,
    and growing/shrinking about the centre."""
    out = [q]
    for k in range(4):
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
            m = q.copy()
            m[k] += (dx, dy)
            out.append(m)
    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
        out.append(q + (dx, dy))
    c = q.mean(axis=0)
    for f in (1 + step / 200.0 * 2, 1 - step / 200.0 * 2):
        out.append(c + (q - c) * f)
    return np.stack(out)


def refine_quad(gray, quad, max_iter=45):
    """Coarse-to-fine corner refinement.  Returns (quad, score)."""
    h, w = gray.shape
    q = np.asarray(quad, dtype=np.float64).copy()
    side = float(np.mean([np.hypot(*(q[(i + 1) % 4] - q[i])) for i in range(4)]))
    c = max(4.0, side / 8.0)                                           # approx cell size in px
    stages = [  # (cell-corner inset, blur radius, first step, last step)
        (0.25, max(1, round(0.20 * c)), 0.30 * c, 0.08 * c),
        (0.14, max(1, round(0.07 * c)), 0.10 * c, 0.03 * c),
        (0.06, max(1, round(0.03 * c)), 0.04 * c, 0.5),
    ]
    cur = 0.0
    for inset, r, step, last in stages:
        G = _box_blur(gray, r)
        cur = float(grid_score(G, q[None], inset)[0])
        it = 0
        while step >= last and it < max_iter:
            it += 1
            cands = _moves(q, step)
            cands[..., 0] = np.clip(cands[..., 0], 0, w - 1)
            cands[..., 1] = np.clip(cands[..., 1], 0, h - 1)
            sc = grid_score(G, cands, inset)
            k = int(sc.argmax())
            if sc[k] > cur + 1e-9:
                q, cur = cands[k], float(sc[k])
            else:
                step *= 0.5
    return q, cur


# ──────────────────────────────────────────────────────────────────────────────
# Final polish: put the 18 grid lines exactly on the edges between squares
# ──────────────────────────────────────────────────────────────────────────────

_LINE_T = np.linspace(0.06, 0.94, 36)
_LINE_W = np.array([0.5] + [1.0] * 7 + [0.5] + [0.5] + [1.0] * 7 + [0.5])      # outer lines count half


def _line_uv():
    """uv sample points of the 9 horizontal + 9 vertical lines: (18,36,2)."""
    rows = [np.stack([_LINE_T, np.full_like(_LINE_T, k / 8.0)], axis=-1) for k in range(9)]
    cols = [np.stack([np.full_like(_LINE_T, k / 8.0), _LINE_T], axis=-1) for k in range(9)]
    return np.stack(rows + cols)


_LINE_UV = _line_uv()


def line_energy(G, quads, delta):
    """Mean edge strength measured *across* each grid line.  Peaks when the lines lie on
    the borders between squares - a much sharper cue than colour patches."""
    quads = np.asarray(quads, dtype=np.float64)
    H = unit_square_H(quads)
    n = len(quads)
    pts = map_uv(H, _LINE_UV.reshape(-1, 2)).reshape(n, 18, len(_LINE_T), 2)
    d = pts[:, :, -1, :] - pts[:, :, 0, :]                                   # line direction (n,18,2)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    nrm = np.stack([-d[..., 1], d[..., 0]], axis=-1)[:, :, None, :]          # unit normals
    a = pts + delta * nrm
    b = pts - delta * nrm
    diff = np.abs(_bilinear(G, a[..., 0], a[..., 1]) - _bilinear(G, b[..., 0], b[..., 1]))
    return (diff.mean(axis=2) * _LINE_W[None, :]).sum(axis=1) / _LINE_W.sum()


def polish_quad(gray, quad, max_iter=40):
    h, w = gray.shape
    q = np.asarray(quad, dtype=np.float64).copy()
    side = float(np.mean([np.hypot(*(q[(i + 1) % 4] - q[i])) for i in range(4)]))
    c = max(4.0, side / 8.0)
    G = _box_blur(gray, 1)
    delta = max(1.5, 0.025 * c)
    cur = float(line_energy(G, q[None], delta)[0])
    step, it = 0.05 * c, 0
    while step >= 0.25 and it < max_iter:
        it += 1
        cands = _moves(q, step)
        cands[..., 0] = np.clip(cands[..., 0], 0, w - 1)
        cands[..., 1] = np.clip(cands[..., 1], 0, h - 1)
        sc = line_energy(G, cands, delta)
        k = int(sc.argmax())
        if sc[k] > cur * 1.0005:
            q, cur = cands[k], float(sc[k])
        else:
            step *= 0.5
    return q


# ──────────────────────────────────────────────────────────────────────────────
# Proposals
# ──────────────────────────────────────────────────────────────────────────────

def _rot_forward(angle_deg, old_shape, new_shape):
    """Matrix/offsets mapping points of the original image into the image rotated by
    PIL's rotate(angle, expand=True)."""
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    c0 = np.array([old_shape[1] / 2.0, old_shape[0] / 2.0])
    c1 = np.array([new_shape[1] / 2.0, new_shape[0] / 2.0])
    return R, c0, c1


def numpy_proposals(work_rgb):
    """Axis-aligned checkerboard finder (after undoing tilt) -> list of quads."""
    out = []
    img, box, angle = bf.find_board_deskewed(work_rgb)
    if box is None:
        return out
    x, y, s = box[:3]
    corners = np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)
    if abs(angle) > 1e-6:                                             # back to the original frame
        R, c0, c1 = _rot_forward(angle, work_rgb.shape, img.shape)
        corners = (corners - c1) @ R + c0                              # inverse of R (orthonormal)
    out.append(order_quad(corners))
    return out


def contour_proposals(work_rgb, max_keep=8):
    """The desktop script's method: Canny -> contours -> polygons with 4 corners."""
    if not HAVE_CV2:
        return []
    h, w = work_rgb.shape[:2]
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    thresholds = [(50, 150), (int(max(10, 0.66 * med)), int(min(255, 1.33 * med + 20))), (20, 60)]
    kernel = np.ones((3, 3), np.uint8)
    found = []
    for lo, hi in thresholds:
        edges = cv2.Canny(blur, lo, hi)
        for variant in (edges, cv2.dilate(edges, kernel)):            # dilate: closes small gaps
            for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
                contours, _ = cv2.findContours(variant, mode, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < MIN_AREA_FRAC * w * h:
                        continue
                    hull = cv2.convexHull(cnt)
                    per = cv2.arcLength(hull, True)
                    for shape in (cnt, hull):
                        for eps in (0.02, 0.03, 0.05):
                            approx = cv2.approxPolyDP(shape, eps * per, True)
                            if len(approx) == 4:
                                found.append(order_quad(approx.reshape(4, 2)))
                                break
                    rect = cv2.minAreaRect(hull)                       # rotated-rectangle proposal
                    found.append(order_quad(cv2.boxPoints(rect)))
    # keep distinct, plausible ones; rank by a quick (unrefined) grid score
    good, seen = [], []
    for q in found:
        if not is_good_quad(q, w, h):
            continue
        if any(np.abs(q - s).max() < 0.02 * max(w, h) for s in seen):
            continue
        seen.append(q)
        good.append(q)
    if not good:
        return []
    side = np.mean([np.hypot(*(good[0][1] - good[0][0]))])
    G = _box_blur(_gray_f(work_rgb), max(1, round(0.07 * side / 8)))
    scores = grid_score(G, np.stack(good), 0.14)
    order = np.argsort(-scores)[:max_keep]
    return [good[i] for i in order]


def _gray_f(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def locate_board(rgb, use_contours=True):
    """Find the board corners in `rgb` (HxWx3 uint8).
    Returns dict(quad=(4,2) array in rgb's pixel coordinates, score, method) or None."""
    H, W = rgb.shape[:2]
    k = min(1.0, WORK_SIDE / max(H, W))
    work = rgb if k >= 1.0 else np.asarray(
        Image.fromarray(rgb).resize((max(16, int(W * k)), max(16, int(H * k))), Image.BOX))
    wh, ww = work.shape[:2]
    gray = _gray_f(work)

    proposals = [("checker", q) for q in numpy_proposals(work)]
    if use_contours:
        proposals += [("contour", q) for q in contour_proposals(work)]

    best = None
    for method, q in proposals:
        if not is_good_quad(q, ww, wh):
            continue
        rq, score = refine_quad(gray, q)
        if not is_good_quad(rq, ww, wh):
            continue
        pq = polish_quad(gray, rq)
        if is_good_quad(pq, ww, wh):
            side = float(np.mean([np.hypot(*(pq[(i + 1) % 4] - pq[i])) for i in range(4)]))
            Gf = _box_blur(gray, max(1, round(0.03 * side / 8)))
            pscore = float(grid_score(Gf, pq[None], 0.06)[0])
            if pscore >= 0.8 * float(grid_score(Gf, rq[None], 0.06)[0]):      # polish must not wreck the fit
                rq, score = pq, max(score, pscore)
        if best is None or score > best["score"]:
            best = {"quad": rq, "score": score, "method": method}
    if best is None or best["score"] < MIN_FINAL_SCORE:
        return None
    quad = best["quad"] / k
    if k < 1.0:                                   # last polish on the full-resolution picture (sub-pixel)
        full = polish_quad(_gray_f(rgb), quad, max_iter=30)
        if is_good_quad(full, W, H):
            quad = full
    best["quad"] = quad
    return best


def warp_board(rgb, quad, out=None):
    """Perspective-warp the quad to an upright square image (out x out x 3 uint8)."""
    quad = order_quad(quad)
    if out is None:
        side = float(np.mean([np.hypot(*(quad[(i + 1) % 4] - quad[i])) for i in range(4)]))
        out = int(min(960, max(320, round(side))))
    H = unit_square_H(quad[None])[0]
    # PIL wants the map output-pixel -> input-pixel: scale the unit square up to `out` pixels
    S = np.diag([1.0 / out, 1.0 / out, 1.0])
    M = H @ S
    M = M / M[2, 2]
    coeffs = (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], M[2, 0], M[2, 1])
    img = Image.fromarray(rgb).transform((out, out), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return np.asarray(img)


def extract_board(rgb, use_contours=True):
    """Picture -> (upright board image, quad, info) or None when no board is found."""
    found = locate_board(rgb, use_contours=use_contours)
    if found is None:
        return None
    return warp_board(rgb, found["quad"]), found["quad"], found


def default_quad(rgb):
    """A centred square, used as the starting point when nothing was detected."""
    H, W = rgb.shape[:2]
    s = 0.8 * min(H, W)
    x0, y0 = (W - s) / 2, (H - s) / 2
    return np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])
