"""Shared synthetic-data helpers for the board-detection tests."""
import io, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from render import random_position, render

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def clutter(draw, w, h, bg_lum, rnd, skip):
    fg = (230, 230, 230) if bg_lum < 128 else (40, 40, 40)
    f = ImageFont.truetype(FONT, max(14, w // 40))
    for _ in range(60):
        x, y = rnd.randint(0, w - 100), rnd.randint(0, h - 40)
        bw, bh = rnd.randint(40, w // 3), rnd.randint(10, 60)
        if not (x + bw < skip[0] or x > skip[2] or y + bh < skip[1] or y > skip[3]):
            continue
        draw.rectangle([x, y, x + bw, y + bh], fill=tuple(rnd.randint(30, 220) for _ in range(3)))
    for _ in range(25):
        x, y = rnd.randint(0, w - 200), rnd.randint(0, h - 30)
        if not (x + 200 < skip[0] or x > skip[2] or y + 30 < skip[1] or y > skip[3]):
            continue
        draw.text((x, y), "Magnus_C 2850  e4 e5 Nf3 Nc6", fill=fg, font=f)

def screen(w, h, bx, by, bpx, bg, theme, moves, seed, lastmove=False):
    rnd = random.Random(seed)
    bd = random_position(moves, seed)
    lm = bd.peek() if (lastmove and bd.move_stack) else None
    board_img = render(bd, px=bpx, theme=theme, lastmove=lm)
    canvas = Image.new("RGB", (w, h), bg)
    clutter(ImageDraw.Draw(canvas), w, h, sum(bg) / 3, rnd, (bx - 5, by - 5, bx + bpx + 5, by + bpx + 5))
    canvas.paste(board_img, (bx, by))
    return np.array(canvas), bd, (bx, by, bpx)

def solve_homography(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)

def project(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]

def distort(arr, board_quad, rnd, rotate_deg=0.0, persp=0.0, scale=0.9, fill=None, blur=True, noise=True, jpeg=70):
    """Photograph-like distortion of a whole picture.  Returns (new_array, true_board_quad)."""
    h, w = arr.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    c = np.array([w / 2, h / 2])
    t = np.radians(rotate_deg); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    dst = ((src - c) * scale) @ R.T + c
    dst += np.array([[rnd.uniform(-1, 1) * persp * w, rnd.uniform(-1, 1) * persp * h] for _ in range(4)])
    Hf = solve_homography(src, dst); Hi = np.linalg.inv(Hf); Hi /= Hi[2, 2]
    fill = fill or tuple(int(v) for v in arr[0, 0])
    im = Image.fromarray(arr).transform((w, h), Image.PERSPECTIVE, tuple(Hi.reshape(-1)[:8]), Image.BICUBIC, fillcolor=fill)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(rnd.choice([0.7, 1.0])))
    a = np.asarray(im).astype(np.float32)
    if noise:
        gx = np.linspace(rnd.uniform(0.8, 0.95), rnd.uniform(1.0, 1.08), w)[None, :, None]
        gy = np.linspace(rnd.uniform(0.92, 1.0), rnd.uniform(0.97, 1.06), h)[:, None, None]
        a = a * gx * gy + np.random.default_rng(rnd.randint(0, 10**6)).normal(0, 3.5, a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if jpeg:
        b = io.BytesIO(); im.save(b, "JPEG", quality=jpeg); im = Image.open(b).convert("RGB")
    return np.array(im), project(Hf, board_quad)
