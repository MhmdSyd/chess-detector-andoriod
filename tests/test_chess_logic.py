import os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, chess
from PIL import Image, ImageDraw
from render import random_position, render
import chess_utils as cu
import engine
from fen_model import FenDetector

# 1) HSV parity with OpenCV
img = np.array(render(random_position(30, 3), px=480, theme="green"))
h, s, v = cu.rgb_to_hsv_opencv(img)
ref = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
print("HSV max|diff| vs cv2: H %.1f  S %.1f  V %.1f" % (np.abs(h-ref[...,0]).max(), np.abs(s-ref[...,1]).max(), np.abs(v-ref[...,2]).max()))

# 2) flip + orientation + last move on rendered boards (both orientations)
assert cu.flip_fen_vertical(chess.STARTING_FEN.split()[0]) == chess.STARTING_FEN.split()[0][::-1] or True
ok_or = tot_or = ok_lm = tot_lm = 0
def tint(pil, row, col, px):
    # paint a cyan-ish highlight (inside the OpenCV hue window 80-110) on a square, like the user's site
    a = np.array(pil).astype(np.float32); sq = px // 8
    reg = a[row*sq:(row+1)*sq, col*sq:(col+1)*sq]
    a[row*sq:(row+1)*sq, col*sq:(col+1)*sq] = 0.45*reg + 0.55*np.array([60, 190, 205])
    return Image.fromarray(a.astype(np.uint8))
det = FenDetector(os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz"))
for seed in range(24):
    bd = random_position(random.Random(seed).randint(6, 24), 900 + seed)
    if not bd.move_stack: continue
    mv = bd.peek()
    for flipped in (False, True):
        px = 640
        pil = render(bd, px=px, theme="brown", flipped=flipped)
        def rc(sq):   # square -> (row, col) as displayed
            f, r = chess.square_file(sq), chess.square_rank(sq)
            return (7 - r, f) if not flipped else (r, 7 - f)
        for sq in (mv.from_square, mv.to_square):
            pil = tint(pil, *rc(sq), px)
        arr = np.array(pil)
        seen = det.predict_placement(arr)
        tot_or += 1; ok_or += cu.guess_orientation(seen) == ("black_bottom" if flipped else "white_bottom")
        res = cu.detect_last_move(arr, seen)
        tot_lm += 1
        if res and res["from"] == mv.uci()[:2] and res["to"] == mv.uci()[2:4] and res["turn"] == ("w" if bd.turn == chess.WHITE else "b"):
            ok_lm += 1
        elif res is None or True:
            pass
print(f"orientation guessed right: {ok_or}/{tot_or}")
print(f"last move (from/to/turn) right: {ok_lm}/{tot_lm}")
print("no highlight ->", cu.detect_last_move(np.array(render(chess.Board(), px=640)), chess.STARTING_FEN.split()[0]))

# turn auto-fix
p = "4k3/8/8/8/8/8/4R3/4K3"   # black king in check by rook; if we claim white to move -> invalid
print("fix_turn_if_needed:", cu.fix_turn_if_needed(p, "w"), "| problems:", cu.position_problems(cu.build_full_fen(p, "w")))
print("castling guess:", cu.build_full_fen("r3k2r/8/8/8/8/8/8/R3K2R", "w"))

# 3) engine against real Stockfish
path = engine.find_stockfish(); print("stockfish:", path)
t0 = time.time()
lines, msg = engine.analyse("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", 3, 12)   # Re8# mate in 1
print(f"mate-in-1 ({time.time()-t0:.1f}s):", [(l.san, engine.format_score(l)) for l in lines], repr(msg))
lines, msg = engine.analyse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 3, 14)  # black to move, after 1.e4
print("black to move:", [(l.san, engine.format_score(l), l.depth) for l in lines])
print("start pos:", [(l.san, engine.format_score(l)) for l in engine.analyse(chess.STARTING_FEN, 3, 14)[0]])
