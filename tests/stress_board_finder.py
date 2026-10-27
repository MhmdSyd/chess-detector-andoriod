import os, sys, io, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageFilter
from test_board_finder import screen, det
from board_finder import find_board, crop_board

rnd = random.Random(2026)
N = 40; found = right = 0; errs = []; times = []; fails = []
for k in range(N):
    portrait = rnd.random() < 0.7
    w, h = (1080, 2340) if portrait else (1920, 1080)
    bpx = rnd.choice([560, 640, 720, 800, 880, 960, 1040]) if portrait else rnd.choice([600, 720, 800, 900])
    bpx = min(bpx, w, h)
    bx = rnd.randint(0, w - bpx); by = rnd.randint(0, h - bpx)
    bg = rnd.choice([(38, 36, 33), (49, 46, 43), (245, 245, 245), (20, 20, 30), (230, 225, 215)])
    theme = rnd.choice(["brown", "green", "blue"])
    img, bd, _ = screen(w, h, bx, by, bpx, bg, theme, rnd.choice([10, 20, 35, 60, 100]), 500 + k, lastmove=rnd.random() < .5)
    pil = Image.fromarray(img)
    if rnd.random() < .5:   # blur
        pil = pil.filter(ImageFilter.GaussianBlur(rnd.choice([0.6, 1.0])))
    if rnd.random() < .5:   # rescale like a screenshot saved at another resolution
        f = rnd.choice([0.5, 0.67, 0.8, 1.3]); pil = pil.resize((int(w*f), int(h*f)), Image.BILINEAR); bx, by, bpx = [v*f for v in (bx, by, bpx)]
    buf = io.BytesIO(); pil.save(buf, "JPEG", quality=rnd.choice([60, 75, 90])); pil = Image.open(buf).convert("RGB")
    arr = np.array(pil)
    t0 = time.time(); box = find_board(arr); times.append(time.time() - t0)
    if box is None:
        fails.append((k, "not found", w, h, bpx, theme)); continue
    found += 1
    x, y, s, sc = box
    errs.append(max(abs(x-bx), abs(y-by), abs(s-bpx)) / bpx)
    fen = det.predict_placement(crop_board(arr, box))
    if fen == bd.board_fen(): right += 1
    else: fails.append((k, "wrong fen", f"err_px=({x-bx:.0f},{y-by:.0f},{s-bpx:.0f})", int(bpx), theme))
print(f"board located: {found}/{N}   FEN exactly right: {right}/{N}")
print(f"position error: median {np.median(errs)*100:.2f}% of board size, worst {np.max(errs)*100:.2f}%")
print(f"time on this PC: median {np.median(times):.2f}s, max {np.max(times):.2f}s")
for f in fails: print("  fail:", f)
