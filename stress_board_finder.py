import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, chess
from skimage.transform import resize
from render import random_position, render
from fen_model import FenDetector, resize_like_skimage, split_tiles
from tensorflow.keras.models import load_model

W = os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz")
det = FenDetector(W)
keras = load_model("/tmp/m.keras")

# 1) resize parity vs skimage, several crop sizes
for px in (200, 333, 480, 720, 1000):
    img = np.array(render(random_position(30, 1), px=px))
    ref = resize(img, (200, 200), mode="constant", anti_aliasing=True).astype("float32")
    mine = resize_like_skimage(img)
    print(f"resize {px:4d}px -> max|diff| = {np.abs(ref-mine).max():.2e}")

# 2) keras vs numpy network + 3) accuracy vs ground truth
ok = total = mism = 0
t_np = []
for seed, (moves, theme, px) in enumerate([(n, t, p) for n in (0, 12, 30, 55, 90) for t in ("brown", "green", "blue") for p in (480, 720)]):
    bd = random_position(moves, seed)
    img = np.array(render(bd, px=px, theme=theme))
    truth = bd.board_fen()
    t0 = time.time(); fen = det.predict_placement(img); t_np.append(time.time() - t0)
    # reference = original library path (skimage resize + keras)
    tiles = resize(img, (200, 200), mode="constant", anti_aliasing=True).astype("float32")
    tiles = split_tiles(tiles)
    kpred = keras.predict(tiles, verbose=0).argmax(1)
    from fen_model import classes_to_placement
    kfen = classes_to_placement(kpred)
    mism += (kfen != fen)
    ok += (fen == truth); total += 1
print(f"numpy == keras on {total-mism}/{total} boards")
print(f"matches ground-truth FEN on {ok}/{total} boards")
print(f"numpy inference time (this PC): mean {np.mean(t_np):.2f}s max {np.max(t_np):.2f}s")
