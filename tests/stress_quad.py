import os, sys, time, random, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
from synth import *
import numpy as np
import board_detect as bd
from fen_model import FenDetector
det = FenDetector(os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz"))
def wrong(a, b):
    g = lambda f: "".join("." * int(c) if c.isdigit() else c for c in f.replace("/", ""))
    return sum(x != y for x, y in zip(g(a), g(b)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for uc in ([True, False] if bd.HAVE_CV2 else [False]):
    rnd = random.Random(77); found = ok = 0; errs = []; times = []; wrongs = []
    for k in range(N):
        portrait = rnd.random() < 0.7
        w, h = (1080, 2340) if portrait else (1920, 1080)
        bpx = min(rnd.choice([560, 640, 720, 800, 900, 1000]), w, h)
        bx = rnd.randint(0, w - bpx); by = rnd.randint(0, h - bpx)
        bg = rnd.choice([(38, 36, 33), (49, 46, 43), (245, 245, 245), (20, 20, 30), (230, 225, 215)])
        img, bdd, _ = screen(w, h, bx, by, bpx, bg, rnd.choice(["brown", "green", "blue"]), rnd.choice([10, 25, 40, 70, 100]), 900 + k, lastmove=rnd.random() < .4)
        true = np.array([[bx, by], [bx + bpx, by], [bx + bpx, by + bpx], [bx, by + bpx]], float)
        dimg, dq = distort(img, true, rnd, rotate_deg=rnd.uniform(-12, 12), persp=rnd.choice([0, 0.015, 0.03, 0.045]),
                           scale=rnd.uniform(0.8, 1.0), blur=rnd.random() < .7, noise=rnd.random() < .7, jpeg=rnd.choice([60, 75, 90]))
        t0 = time.time(); res = bd.extract_board(dimg, use_contours=uc); times.append(time.time() - t0)
        if res is None: continue
        board, quad, info = res; found += 1
        side = np.mean([np.hypot(*(dq[(i + 1) % 4] - dq[i])) for i in range(4)])
        errs.append(np.abs(bd.order_quad(quad) - bd.order_quad(dq)).max() / side)
        wsq = wrong(det.predict_placement(board), bdd.board_fen()); wrongs.append(wsq); ok += wsq == 0
    print(f"contours {'ON ' if uc else 'OFF'}: found {found}/{N}  FEN exact {ok}/{N}  median corner err {np.median(errs)*100:.2f}%  worst {np.max(errs)*100:.2f}%  "
          f"wrong-squares avg {np.mean(wrongs):.2f}  time median {np.median(times):.1f}s max {np.max(times):.1f}s")
# negatives: pictures that contain no board
rnd = random.Random(5); false_pos = 0
for k in range(6):
    bg = rnd.choice([(38, 36, 33), (245, 245, 245)]); im = Image.new("RGB", (1080, 2340), bg)
    clutter(ImageDraw.Draw(im), 1080, 2340, sum(bg) / 3, rnd, (-1, -1, -1, -1))
    arr = np.array(im.rotate(rnd.uniform(-10, 10), expand=True, fillcolor=bg))
    for uc in ([True, False] if bd.HAVE_CV2 else [False]):
        false_pos += bd.extract_board(arr, use_contours=uc) is not None
print("pictures with NO board wrongly accepted:", false_pos, "of", 6 * (2 if bd.HAVE_CV2 else 1))
