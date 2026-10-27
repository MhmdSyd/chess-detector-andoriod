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


def _score(ii, X, Y, S, pf=0.2, inf=0.08):
    """Vectorised checkerboard score for candidates (arrays of x, y, size)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = len(X)
    c = S / 8.0
    p = np.maximum(1, np.rint(pf * c)).astype(np.int64)              # patch size
    inset = np.maximum(1, np.rint(inf * c)).astype(np.int64)

    ci = _I[None, :, :]                                              # (1,8,8)
    cj = _J[None, :, :]
    ox = X[:, None, None] + cj * c[:, None, None]                    # cell origin
    oy = Y[:, None, None] + ci * c[:, None, None]

    means = []
    for right, bottom in _CORNER_SIGN:
        px = ox + inset[:, None, None] if not right else ox + c[:, None, None] - inset[:, None, None] - p[:, None, None]
        py = oy + inset[:, None, None] if not bottom else oy + c[:, None, None] - inset[:, None, None] - p[:, None, None]
        x1 = np.rint(px).astype(np.int64)
        y1 = np.rint(py).astype(np.int64)
        x2 = x1 + p[:, None, None]
        y2 = y1 + p[:, None, None]
        h, w = ii.shape[0] - 1, ii.shape[1] - 1
        x1c, x2c = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1c, y2c = np.clip(y1, 0, h), np.clip(y2, 0, h)
        tot = ii[y2c, x2c] - ii[y1c, x2c] - ii[y2c, x1c] + ii[y1c, x1c]
        means.append(tot / (p[:, None, None] ** 2))
    m = np.sort(np.stack(means, axis=-1), axis=-1)                   # (n,8,8,4)
    bg = (m[..., 1] + m[..., 2]).reshape(n, 64) / 2.0                # robust cell colour

    a = bg[:, _PARITY == 0]
    b = bg[:, _PARITY == 1]
    med_a = np.median(a, axis=1)
    med_b = np.median(b, axis=1)
    # Mean (not median) deviation: every cell that does not belong to a perfect
    # checkerboard pulls the score down, so boards shifted by whole squares lose.
    dev = (np.abs(a - med_a[:, None]).sum(axis=1) + np.abs(b - med_b[:, None]).sum(axis=1)) / 64.0
    d = np.abs(med_a - med_b)
    score = d / (dev + 1.0)
    score[d < MIN_CONTRAST] = 0.0
    return score, d


def _best(ii, xs, ys, ss, chunk=3000, pf=0.2, inf=0.08, plateau=0.0):
    """Evaluate the grid xs x ys x ss.  Returns (score, x, y, s) of the best candidate;
    with plateau>0 the position is the average of all candidates scoring >= plateau*best."""
    gx, gy, gs = np.meshgrid(xs, ys, ss, indexing="ij")
    gx, gy, gs = gx.ravel(), gy.ravel(), gs.ravel()
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    keep = (gx >= 0) & (gy >= 0) & (gx + gs <= W) & (gy + gs <= H)
    gx, gy, gs = gx[keep], gy[keep], gs[keep]
    if len(gx) == 0:
        return -1.0, 0, 0, 0
    scores = np.empty(len(gx))
    for s in range(0, len(gx), chunk):
        scores[s:s + chunk], _ = _score(ii, gx[s:s + chunk], gy[s:s + chunk], gs[s:s + chunk], pf, inf)
    k = int(scores.argmax())
    if plateau > 0:
        sel = scores >= plateau * scores[k]
        return (float(scores[k]), int(round(gx[sel].mean())),
                int(round(gy[sel].mean())), int(round(gs[sel].mean())))
    return float(scores[k]), int(gx[k]), int(gy[k]), int(gs[k])


def find_board(rgb, work_side=360):
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
    score, x, y, s = _best(ii_small, xs, ys, sizes)
    if score < MIN_SCORE:
        return None

    # ── refine on the full-resolution image ─────────────────────────────────
    # Edge-pad first so the search window is symmetric even when the board touches
    # the image border (otherwise the plateau is cut off on one side and biased).
    pad = 24
    ii = _integral(np.pad(_gray(rgb), pad, mode="edge"))
    inv = 1.0 / scale
    x, y, s = x * inv + pad, y * inv + pad, s * inv
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
        span = 2 * st + 1
        st = max(1, st // 2)
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
