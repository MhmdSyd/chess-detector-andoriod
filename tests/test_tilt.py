import os, sys, io, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageFilter
from synth import screen, det          # (importing runs that file's own quick checks)
import board_finder as bf

print("\n=== TILT TEST ===")
rnd = random.Random(7)
ok = n = 0; errs = []; times = []
def run(label, angle, jpeg=True, **kw):
    global ok, n
    img, bd, _ = screen(seed=rnd.randint(1, 10**6), **kw)
    pil = Image.fromarray(img)
    if angle:
        pil = pil.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=kw["bg"])
    if jpeg:
        b = io.BytesIO(); pil.save(b, "JPEG", quality=85); pil = Image.open(b).convert("RGB")
    arr = np.array(pil)
    t0 = time.time(); out, box, applied = bf.find_board_deskewed(arr); dt = time.time() - t0; times.append(dt)
    n += 1
    if box is None:
        print(f"{label:34s} tilt={angle:+5.1f}  NOT FOUND"); return
    fen = det.predict_placement(bf.crop_board(out, box)); good = fen == bd.board_fen(); ok += good
    errs.append(abs(applied + angle) if angle else abs(applied))       # residual tilt
    print(f"{label:34s} tilt={angle:+5.1f} applied={applied:+6.2f} residual={abs(applied+angle):4.2f} {dt:4.1f}s FEN {'OK' if good else 'WRONG'}")

base = dict(w=1080, h=2340, bx=40, by=700, bpx=1000, bg=(38, 36, 33), theme="green", moves=30)
for a in (0, 0, 0):
    run("straight (must stay untouched)", 0, **{**base, "moves": rnd.choice([10, 30, 60])})
for a in (-14, -9, -5, -3, -1.5, 2, 4, 7, 11, 15):
    run("phone dark green", a, **{**base, "moves": rnd.choice([12, 30, 55])})
for a in (-8, 6):
    run("phone light brown", a, **{**base, "bg": (245, 245, 245), "theme": "brown", "bx": 80, "bpx": 920})
for a in (-6, 9):
    run("desktop-ish landscape", a, w=1920, h=1080, bx=300, by=100, bpx=800, bg=(38, 36, 33), theme="blue", moves=35)
print(f"\nFEN correct after straightening: {ok}/{n}")
print(f"time: median {np.median(times):.1f}s max {np.max(times):.1f}s (this PC)")
