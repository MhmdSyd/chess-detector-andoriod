import os, sys, time, random
sys.path.insert(0, os.path.dirname(__file__))
from synth import *
import numpy as np
import board_detect as bd
from fen_model import FenDetector
det = FenDetector(os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz"))
print("OpenCV available in this run:", bd.HAVE_CV2)

# ---------- unit checks ----------
q = np.array([[100, 80], [520, 120], [480, 560], [60, 500]], float)
H = bd.unit_square_H(q[None])
mapped = bd.map_uv(H, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))[0]
print("homography maps unit square to quad:", np.allclose(mapped, q))
print("order_quad shuffles OK:", np.allclose(bd.order_quad(q[[2, 0, 3, 1]]), q))
# rotation bookkeeping: rotate a marker image with PIL and map its point forward
m = np.zeros((300, 500, 3), np.uint8); m[100:104, 400:404] = 255
for ang in (-17.0, 9.0):
    r = bf = None
    from board_finder import rotate_image
    rot = rotate_image(m, ang)
    R, c0, c1 = bd._rot_forward(ang, m.shape, rot.shape)
    p_new = (np.array([402, 102]) - c0) @ R.T + c1
    ys, xs = np.where(rot[..., 0] > 100)
    print(f"rotation map angle {ang:+5.1f}: predicted {p_new.round(1)}  actual {xs.mean():.1f},{ys.mean():.1f}")
# warp: undo a known distortion
rnd = random.Random(1)
img, bdd, (bx, by, bpx) = screen(1080, 2340, 40, 700, 1000, (38, 36, 33), "green", 30, 42)
true = np.array([[bx, by], [bx + bpx, by], [bx + bpx, by + bpx], [bx, by + bpx]], float)
dimg, dq = distort(img, true, rnd, rotate_deg=8, persp=0.03, blur=False, noise=False, jpeg=0)
warped = bd.warp_board(dimg, dq, out=480)
ref = np.asarray(Image.fromarray(img[by:by + bpx, bx:bx + bpx]).resize((480, 480), Image.BICUBIC)).astype(float)
print("warp with TRUE corners: mean abs diff vs original board = %.1f / 255" % np.abs(warped.astype(float) - ref).mean(),
      "| FEN ok:", det.predict_placement(warped) == bdd.board_fen())

# ---------- detection on distorted 'photos' ----------
def run(cases, use_contours):
    ok_fen = ok_found = 0; errs = []; times = []; rows = []
    for k, (label, kw, dk) in enumerate(cases):
        rnd = random.Random(1000 + k)
        img, bdd, (bx, by, bpx) = screen(seed=500 + k, **kw)
        true = np.array([[bx, by], [bx + bpx, by], [bx + bpx, by + bpx], [bx, by + bpx]], float)
        dimg, dq = distort(img, true, rnd, **dk)
        t0 = time.time(); res = bd.extract_board(dimg, use_contours=use_contours); dt = time.time() - t0; times.append(dt)
        if res is None:
            rows.append(f"  {label:34s} NOT FOUND ({dt:.1f}s)"); continue
        board, quad, info = res; ok_found += 1
        side = np.mean([np.hypot(*(dq[(i + 1) % 4] - dq[i])) for i in range(4)])
        e = np.abs(bd.order_quad(quad) - bd.order_quad(dq)).max() / side; errs.append(e)
        fen = det.predict_placement(board); good = fen == bdd.board_fen(); ok_fen += good
        rows.append(f"  {label:34s} corner err {e*100:5.2f}% of board  via {info['method']:7s} score {info['score']:5.1f} {dt:4.1f}s FEN {'OK' if good else 'WRONG'}")
    return ok_found, ok_fen, errs, times, rows

phone = dict(w=1080, h=2340, bx=40, by=700, bpx=1000, bg=(38, 36, 33), theme="green", moves=30)
cases = [
    ("straight screenshot",               phone, dict(rotate_deg=0, persp=0, scale=1.0, blur=False, noise=False, jpeg=0)),
    ("rotated -6 deg",                    {**phone, "theme": "brown"}, dict(rotate_deg=-6, persp=0, scale=0.92)),
    ("rotated +11 deg",                   {**phone, "moves": 45}, dict(rotate_deg=11, persp=0, scale=0.85)),
    ("rotated 4 deg + mild perspective",  {**phone, "theme": "blue"}, dict(rotate_deg=4, persp=0.02, scale=0.9)),
    ("perspective 4%",                    phone, dict(rotate_deg=0, persp=0.04, scale=0.88)),
    ("rotated -8 + perspective 4%",       {**phone, "theme": "brown", "moves": 55}, dict(rotate_deg=-8, persp=0.04, scale=0.85)),
    ("perspective 6% (strong keystone)",  {**phone, "moves": 20}, dict(rotate_deg=3, persp=0.06, scale=0.85)),
    ("light background photo",            {**phone, "bg": (235, 230, 220), "theme": "green"}, dict(rotate_deg=7, persp=0.03, scale=0.88)),
    ("small board in busy picture",       {**phone, "bx": 190, "by": 900, "bpx": 700, "theme": "blue"}, dict(rotate_deg=-5, persp=0.03, scale=0.9)),
    ("landscape monitor photo",           dict(w=1920, h=1080, bx=300, by=120, bpx=800, bg=(30, 30, 40), theme="brown", moves=35), dict(rotate_deg=6, persp=0.03, scale=0.88)),
]
for uc in ([True, False] if bd.HAVE_CV2 else [False]):
    f, g, errs, times, rows = run(cases, uc)
    print(f"\n=== contours {'ON ' if uc else 'OFF'} (checker finder{' + OpenCV contours' if uc else ' only'}) ===")
    print("\n".join(rows))
    print(f"  board found {f}/{len(cases)}   FEN correct {g}/{len(cases)}   median corner error {np.median(errs)*100:.2f}%   time median {np.median(times):.1f}s max {np.max(times):.1f}s")
