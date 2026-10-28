import os, sys, io, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from test_board_finder import screen, det, clutter
import board_finder as bf

print("\n=== HARD TILT TEST (blur + noise + uneven light + JPEG) ===")
rnd = random.Random(11); nprng = np.random.default_rng(3)
ok = n = 0
def degrade(pil, angle, bg):
    pil = pil.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=bg)
    pil = pil.filter(ImageFilter.GaussianBlur(rnd.choice([0.8, 1.2])))
    a = np.asarray(pil).astype(np.float32)
    h, w = a.shape[:2]
    gx = np.linspace(rnd.uniform(0.75, 0.9), rnd.uniform(1.0, 1.1), w)[None, :, None]       # uneven lighting
    gy = np.linspace(rnd.uniform(0.9, 1.0), rnd.uniform(0.95, 1.1), h)[:, None, None]
    a = a * gx * gy + nprng.normal(0, 4, a.shape)                                           # sensor noise
    pil = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    b = io.BytesIO(); pil.save(b, "JPEG", quality=70); return np.array(Image.open(b).convert("RGB"))

cases = [(-12, 960), (-7, 800), (-4, 1000), (3, 900), (6, 700), (10, 960), (13, 800), (-9, 600), (5, 560), (8, 1040)]
for angle, bpx in cases:
    img, bd, _ = screen(1080, 2340, rnd.randint(0, 1080 - bpx), rnd.randint(300, 2340 - bpx - 100), bpx,
                        rnd.choice([(38, 36, 33), (235, 230, 220)]), rnd.choice(["brown", "green", "blue"]),
                        rnd.choice([14, 30, 55]), rnd.randint(1, 10**6))
    arr = degrade(Image.fromarray(img), angle, tuple(int(v) for v in img[0, 0]))
    out, box, applied = bf.find_board_deskewed(arr)
    n += 1
    if box is None: print(f"tilt {angle:+3d} board {bpx}px: NOT FOUND"); continue
    fen = det.predict_placement(bf.crop_board(out, box)); good = fen == bd.board_fen(); ok += good
    print(f"tilt {angle:+3d} board {bpx:4d}px: applied {applied:+6.2f} (residual {abs(applied+angle):4.2f}) FEN {'OK' if good else 'WRONG'}")
print(f"FEN correct: {ok}/{n}")

# no board, only clutter, tilted: must not be 'straightened' weirdly and must return no board
neg = Image.new("RGB", (1080, 2340), (38, 36, 33)); clutter(ImageDraw.Draw(neg), 1080, 2340, 40, rnd, (-1, -1, -1, -1))
neg = np.array(neg.rotate(8, expand=True, fillcolor=(38, 36, 33)))
o, b, a = bf.find_board_deskewed(neg); print("no-board tilted screenshot ->", b, "angle applied", round(a, 2))
