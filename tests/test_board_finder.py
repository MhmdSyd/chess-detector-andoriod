import os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, chess
from PIL import Image, ImageDraw, ImageFont
from render import random_position, render
from board_finder import find_board, crop_board
from fen_model import FenDetector

W = os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz")
det = FenDetector(W)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def clutter(draw, w, h, bg_lum, rnd, region_skip):
    """Fake UI: bars, buttons, text-like stripes, avatars - avoiding the board region."""
    fg = (230, 230, 230) if bg_lum < 128 else (40, 40, 40)
    f = ImageFont.truetype(FONT, max(14, w // 40))
    for _ in range(60):
        x, y = rnd.randint(0, w - 100), rnd.randint(0, h - 40)
        bw, bh = rnd.randint(40, w // 3), rnd.randint(10, 60)
        if not (x + bw < region_skip[0] or x > region_skip[2] or y + bh < region_skip[1] or y > region_skip[3]):
            continue
        col = tuple(rnd.randint(30, 220) for _ in range(3))
        draw.rectangle([x, y, x + bw, y + bh], fill=col)
    for i in range(25):
        x, y = rnd.randint(0, w - 200), rnd.randint(0, h - 30)
        if not (x + 200 < region_skip[0] or x > region_skip[2] or y + 30 < region_skip[1] or y > region_skip[3]):
            continue
        draw.text((x, y), "Magnus_C 2850  e4 e5 Nf3 Nc6", fill=fg, font=f)

def screen(w, h, bx, by, bpx, bg, theme, moves, seed, lastmove=False):
    rnd = random.Random(seed)
    bd = random_position(moves, seed)
    lm = bd.peek() if (lastmove and bd.move_stack) else None
    board_img = render(bd, px=bpx, theme=theme, lastmove=lm)
    canvas = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(canvas)
    clutter(d, w, h, sum(bg) / 3, rnd, (bx - 5, by - 5, bx + bpx + 5, by + bpx + 5))
    canvas.paste(board_img, (bx, by))
    return np.array(canvas), bd, (bx, by, bpx)

cases = [
    ("phone dark full-width",  dict(w=1080, h=2340, bx=0,   by=520,  bpx=1080, bg=(38, 36, 33),  theme="green", moves=24)),
    ("phone inset brown",      dict(w=1080, h=2400, bx=40,  by=700,  bpx=1000, bg=(49, 46, 43),  theme="brown", moves=40)),
    ("phone light, blue",      dict(w=1080, h=2340, bx=60,  by=420,  bpx=960,  bg=(245, 245, 245), theme="blue", moves=60)),
    ("phone small board",      dict(w=1080, h=2340, bx=190, by=900,  bpx=700,  bg=(38, 36, 33),  theme="brown", moves=16)),
    ("desktop 1080p",          dict(w=1920, h=1080, bx=120, by=140,  bpx=800,  bg=(38, 36, 33),  theme="green", moves=34)),
    ("desktop light + lastmv", dict(w=1920, h=1080, bx=560, by=100,  bpx=880,  bg=(250, 250, 250), theme="brown", moves=28, lastmove=True)),
    ("low-res screenshot",     dict(w=720,  h=1560, bx=0,   by=300,  bpx=720,  bg=(38, 36, 33),  theme="blue",  moves=30)),
    ("endgame (few pieces)",   dict(w=1080, h=2340, bx=0,   by=600,  bpx=1080, bg=(38, 36, 33),  theme="green", moves=120)),
]
good = 0
for k, (name, kw) in enumerate(cases):
    img, bd, (bx, by, bpx) = screen(seed=100 + k, **kw)
    t0 = time.time(); box = find_board(img); dt = time.time() - t0
    if box is None:
        print(f"{name:26s} NOT FOUND ({dt:.1f}s)"); continue
    x, y, s, sc = box
    crop = crop_board(img, box)
    fen = det.predict_placement(crop)
    okfen = fen == bd.board_fen()
    good += okfen
    print(f"{name:26s} truth=({bx},{by},{bpx}) found=({x},{y},{s}) err=({x-bx:+d},{y-by:+d},{s-bpx:+d}) score={sc:5.1f} {dt:4.1f}s FEN {'OK' if okfen else 'WRONG: '+fen}")
print(f"\nFEN correct end-to-end: {good}/{len(cases)}")

# negative: screenshot with no board
rnd = random.Random(5)
neg = Image.new("RGB", (1080, 2340), (38, 36, 33)); d = ImageDraw.Draw(neg)
clutter(d, 1080, 2340, 40, rnd, (-1, -1, -1, -1))
t0 = time.time(); r = find_board(np.array(neg)); print("no-board screenshot ->", r, f"({time.time()-t0:.1f}s)")
