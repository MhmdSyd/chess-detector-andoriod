"""
board_finder.py - locate the chessboard inside a screenshot / photo (NumPy only).

The desktop app used OpenCV (Canny + contours). OpenCV is heavy and fragile to build
for Android, so this version looks for the one thing every digital board has: an 8x8
grid of squares whose colours alternate.

For every candidate square (x, y, size) we sample the *corners* of each of the 64
cells (pieces and highlights live in the middle of a square, so corners are almost
always bare square colour) and measure how cleanly the cells split into two colour
classes by parity.  The true board scores far higher than anything else on screen.
Search is coarse-to-fine using an integral image, so it is fast.
"""
import numpy as np
from PIL import Image

_I, _J = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
_PARITY = ((_I + _J) % 2).reshape(64)
_CORNER_SIGN = [(0, 0), (1, 0), (0, 1), (1, 1)]          # (right?, bottom?)

MIN_SCORE = 3.0        # below this we say "no board found"
MIN_CONTRAST = 10.0    # light vs dark square gray-level difference (0-255)


def _gray(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def _integral(gray):
    ii = np.zeros((gray.shape[0] + 1, gray.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = gray.astype(np.float64).cumsum(0).cumsum(1)
    return ii


_ALL_CELLS = np.arange(64)
# 16 cells (4 rows x 4 cols, both colour parities, including the board edge) for a quick pre-filter
_PROXY_CELLS = np.array([r * 8 + c for r in (0, 3, 4, 7) for c in (0, 3, 4, 7)])


def _score(ii, X, Y, S, pf=0.2, inf=0.08, cells=None):
    """Vectorised checkerboard score for candidates (arrays of x, y, size)."""
    if cells is None:
        cells = _ALL_CELLS
    X = np.asarray(X, dtype=np.float64)[:, None]
    Y = np.asarray(Y, dtype=np.float64)[:, None]
    S = np.asarray(S, dtype=np.float64)[:, None]
    n, m_cells = X.shape[0], len(cells)
    c = S / 8.0
    p = np.maximum(1, np.rint(pf * c)).astype(np.int64)              # patch size
    inset = np.maximum(1, np.rint(inf * c)).astype(np.int64)
    ci = _I.reshape(-1)[cells][None, :]                              # (1,m)
    cj = _J.reshape(-1)[cells][None, :]
    ox = X + cj * c                                                  # cell origins (n,m)
    oy = Y + ci * c
    h, w = ii.shape[0] - 1, ii.shape[1] - 1

    means = []
    for right, bottom in _CORNER_SIGN:
        px = ox + c - inset - p if right else ox + inset
        py = oy + c - inset - p if bottom else oy + inset
        x1 = np.rint(px).astype(np.int64)
        y1 = np.rint(py).astype(np.int64)
        x1c, x2c = np.clip(x1, 0, w), np.clip(x1 + p, 0, w)
        y1c, y2c = np.clip(y1, 0, h), np.clip(y1 + p, 0, h)
        tot = ii[y2c, x2c] - ii[y1c, x2c] - ii[y2c, x1c] + ii[y1c, x1c]
        means.append(tot / (p ** 2))
    srt = np.sort(np.stack(means, axis=-1), axis=-1)                 # (n,m,4)
    bg = (srt[..., 1] + srt[..., 2]) / 2.0                           # robust cell colour (n,m)

    par = _PARITY[cells]
    a = bg[:, par == 0]
    b = bg[:, par == 1]
    med_a = np.median(a, axis=1)
    med_b = np.median(b, axis=1)
    # Mean (not median) deviation: every cell that does not belong to a perfect
    # checkerboard pulls the score down, so boards shifted by whole squares lose.
    dev = (np.abs(a - med_a[:, None]).sum(axis=1) + np.abs(b - med_b[:, None]).sum(axis=1)) / m_cells
    d = np.abs(med_a - med_b)
    score = d / (dev + 1.0)
    score[d < MIN_CONTRAST] = 0.0
    return score, d


def _scan(ii, xs, ys, ss, chunk=3000, pf=0.2, inf=0.08, prefilter_above=20000, keep_frac=0.04):
    """Score the grid xs x ys x ss.  Returns arrays (gx, gy, gs, scores).
    Big grids are first ranked with a cheap 16-cell score; only the best few % are fully scored."""
    gx, gy, gs = np.meshgrid(xs, ys, ss, indexing="ij")
    gx, gy, gs = gx.ravel(), gy.ravel(), gs.ravel()
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    keep = (gx >= 0) & (gy >= 0) & (gx + gs <= W) & (gy + gs <= H)
    gx, gy, gs = gx[keep], gy[keep], gs[keep]
    if len(gx) == 0:
        return gx, gy, gs, np.zeros(0)
    if len(gx) > prefilter_above:
        quick = np.empty(len(gx))
        for s in range(0, len(gx), chunk):
            quick[s:s + chunk], _ = _score(ii, gx[s:s + chunk], gy[s:s + chunk], gs[s:s + chunk],
                                           pf, inf, cells=_PROXY_CELLS)
        k = max(400, int(len(gx) * keep_frac))
        top = np.argpartition(-quick, k - 1)[:k]
        gx, gy, gs = gx[top], gy[top], gs[top]
    scores = np.empty(len(gx))
    for s in range(0, len(gx), chunk):
        scores[s:s + chunk], _ = _score(ii, gx[s:s + chunk], gy[s:s + chunk], gs[s:s + chunk], pf, inf)
    return gx, gy, gs, scores


def _best(ii, xs, ys, ss, pf=0.2, inf=0.08, plateau=0.0, **kw):
    """Best candidate of the grid -> (score, x, y, s).  With plateau>0 the position is the
    average of all candidates scoring >= plateau*best (centres the flat top of the peak)."""
    gx, gy, gs, scores = _scan(ii, xs, ys, ss, pf=pf, inf=inf, **kw)
    if len(scores) == 0:
        return -1.0, 0, 0, 0
    k = int(scores.argmax())
    if plateau > 0:
        sel = scores >= plateau * scores[k]
        return (float(scores[k]), int(round(gx[sel].mean())),
                int(round(gy[sel].mean())), int(round(gs[sel].mean())))
    return float(scores[k]), int(gx[k]), int(gy[k]), int(gs[k])


def _top_seeds(gx, gy, gs, scores, n=10, floor=MIN_SCORE * 0.5):
    """Best n candidates that differ by more than half a square from each other
    (a board shifted by one whole square is a *different* seed on purpose)."""
    order = np.argsort(-scores)
    seeds = []
    for i in order:
        if scores[i] < floor or len(seeds) >= n:
            break
        tol = gs[i] / 16.0
        if all(abs(gx[i] - sx) > tol or abs(gy[i] - sy) > tol or abs(gs[i] - ss_) > tol
               for sx, sy, ss_ in seeds):
            seeds.append((int(gx[i]), int(gy[i]), int(gs[i])))
    return seeds


def find_board(rgb, work_side=300):
    """rgb: HxWx3 uint8.  Returns (x, y, size, score) in full-resolution pixels,
    or None when no convincing board is found."""
    H, W = rgb.shape[:2]
    scale = min(1.0, work_side / max(H, W))
    if scale < 1.0:
        small = np.asarray(Image.fromarray(rgb).resize(
            (max(8, int(W * scale)), max(8, int(H * scale))), Image.BOX))
    else:
        small = rgb
    sh, sw = small.shape[:2]
    ii_small = _integral(_gray(small))

    # ── coarse search ───────────────────────────────────────────────────────
    step = 3
    smin = max(40, int(0.22 * min(sh, sw)))
    smax = min(sh, sw)
    if smax <= smin:
        return None
    sizes = np.arange(smin, smax + 1, step)
    xs = np.arange(0, sw - smin + 1, step)
    ys = np.arange(0, sh - smin + 1, step)
    gx, gy, gs, sc = _scan(ii_small, xs, ys, sizes)
    seeds = _top_seeds(gx, gy, gs, sc)
    if not seeds:
        return None

    # ── refine every seed on the full-resolution image, keep the best ───────
    # Edge-pad first so the search window is symmetric even when the board touches
    # the image border (otherwise the plateau is cut off on one side and biased).
    pad = 24
    ii = _integral(np.pad(_gray(rgb), pad, mode="edge"))
    inv = 1.0 / scale
    best = None
    for sx, sy, ss_ in seeds:
        x, y, s = sx * inv + pad, sy * inv + pad, ss_ * inv
        # Cascade: start at ~half the coarse step and halve it every round until 1 px.
        # Patches get tighter as we get closer, so the score becomes more sensitive to
        # exact alignment.  The last round averages the plateau of near-best candidates.
        st = max(1, int(round(step * inv / 2)))
        span = int(step * inv) + 2
        while True:
            last = st == 1
            score, x, y, s = _best(
                ii,
                np.arange(int(x) - span, int(x) + span + 1, st),
                np.arange(int(y) - span, int(y) + span + 1, st),
                np.arange(int(s) - span, int(s) + span + 1, st),
                pf=0.10 if last else 0.14, inf=0.03 if last else 0.05,
                plateau=0.97 if last else 0.0,
            )
            if last:
                break
            span = st + 1
            st = max(1, st // 2)
        if best is None or score > best[0]:
            best = (score, x, y, s)
    score, x, y, s = best
    if score < MIN_SCORE:
        return None
    x, y = x - pad, y - pad
    # (The size need not be a multiple of 8: the network resizes the crop to 200x200 anyway.)
    s = int(max(8, min(s, H, W)))
    x = int(min(max(x, 0), W - s))
    y = int(min(max(y, 0), H - s))
    return x, y, int(s), float(score)


def crop_board(rgb, box):
    x, y, s = box[:3]
    return rgb[y:y + s, x:x + s]


# ══════════════════════════════════════════════════════════════════════════════
# Tilt correction (photos of a screen are often slightly rotated)
# ══════════════════════════════════════════════════════════════════════════════

TILT_MAX_DEG = 30          # search range +/- degrees
TILT_MIN_DEG = 0.3         # smaller tilts are ignored (keeps clean screenshots untouched)
TILT_MIN_GAIN = 1.10       # tilted version must be this much "sharper" than the original


def _tilt_score(g):
    """How sharply do edges line up with the image axes?  The grid lines of a straight
    board create sharp peaks in the row/column edge profiles; a tilt smears them."""
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    gx[gx < 12] = 0
    gy[gy < 12] = 0
    total = 0.0
    for prof in (gx.sum(axis=0), gy.sum(axis=1)):
        m = prof.mean()
        if m > 0:
            total += prof.var() / (m * m)
    return total


def estimate_tilt(rgb, side=420):
    """Degrees to pass to PIL's rotate() (counter-clockwise) that make the board straight.
    Returns (angle, gain) where gain = sharpness(best) / sharpness(0 degrees)."""
    h, w = rgb.shape[:2]
    k = min(1.0, side / max(h, w))
    pil = Image.fromarray(rgb).convert("L")
    if k < 1.0:
        pil = pil.resize((max(8, int(w * k)), max(8, int(h * k))), Image.BOX)
    fill = int(np.asarray(pil).mean())

    def score(angle):
        g = np.asarray(pil.rotate(angle, resample=Image.BILINEAR, fillcolor=fill), dtype=np.float32)
        gh, gw = g.shape
        g = g[int(.15 * gh):int(.85 * gh), int(.15 * gw):int(.85 * gw)]   # ignore rotated corners
        return _tilt_score(g)

    base = score(0.0)
    coarse = {a: score(float(a)) for a in range(-TILT_MAX_DEG, TILT_MAX_DEG + 1)}
    best = max(coarse, key=coarse.get)
    fine = {round(best + d, 2): None for d in np.arange(-1.0, 1.01, 0.25)}
    for a in fine:
        fine[a] = score(a)
    angle = max(fine, key=fine.get)
    return float(angle), (fine[angle] / base if base > 0 else 1.0)


def rotate_image(rgb, degrees):
    """Rotate counter-clockwise by `degrees`, growing the canvas and filling with the
    picture's border colour so nothing is cut off."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    out = Image.fromarray(rgb).rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    return np.asarray(out)


def find_board_deskewed(rgb):
    """Straighten the picture if it is tilted, then find the board.
    Returns (image, box_or_None, angle_applied).  `image` is what the box refers to."""
    angle, gain = estimate_tilt(rgb)
    if abs(angle) >= TILT_MIN_DEG and gain >= TILT_MIN_GAIN:
        straight = rotate_image(rgb, angle)
        box = find_board(straight)
        if box is not None:
            return straight, box, angle
    box = find_board(rgb)
    return rgb, box, 0.0
