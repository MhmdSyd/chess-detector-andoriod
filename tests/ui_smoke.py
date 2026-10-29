"""Headless end-to-end test of the AUTOMATIC flow:  xvfb-run -a python tests/ui_smoke.py"""
import os, sys, random
os.environ["KIVY_NO_ARGS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, chess
from PIL import Image
from synth import screen, distort
from render import random_position, render

bd = random_position(21, 77)                     # white has just moved (21 plies) -> black to move
mv = bd.peek(); px = 1000
pil = render(bd, px=px, theme="green"); a = np.array(pil).astype(np.float32); sq = px // 8
for s_ in (mv.from_square, mv.to_square):        # highlight the last move in the colour the app looks for
    r, c = 7 - chess.square_rank(s_), chess.square_file(s_)
    a[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 0.45 * a[r*sq:(r+1)*sq, c*sq:(c+1)*sq] + 0.55 * np.array([60, 190, 205])
base, _, _ = screen(1080, 2340, 40, 620, px, (38, 36, 33), "green", 21, 77)
base = np.array(Image.fromarray(base)); base[620:620 + px, 40:40 + px] = a.astype(np.uint8)
true = np.array([[40, 620], [40 + px, 620], [40 + px, 620 + px], [40, 620 + px]], float)
rot = float(os.environ.get("ROT", 7)); persp = float(os.environ.get("PERSP", 0.03))
photo, dq = distort(base, true, random.Random(3), rotate_deg=rot, persp=persp, scale=0.9)
path = "/tmp/fake_photo.jpg"; Image.fromarray(photo).save(path, quality=88)
print(f"photo: rotated {rot} deg, perspective {persp*100:.0f}% | expected placement:", bd.board_fen(), "| side to move:", "w" if bd.turn else "b")

sys.argv = [sys.argv[0], path]                   # the app auto-processes this picture on start (dev helper)
from kivy.clock import Clock
from kivy.core.window import Window
import main as M
import board_detect
if os.environ.get('NO_CV2'): board_detect.HAVE_CV2 = False; print('OpenCV disabled for this run')

class T(M.ChessAnalyzerApp):
    def on_start(self):
        super().on_start()
        Clock.schedule_once(lambda dt: self.report(), float(os.environ.get("WAIT", 22)))
    def report(self):
        Window.screenshot("/tmp/ui_auto.png")
        fen = self.board.get_fen()
        print("screen:", self.sm.current, "| app FEN:", fen)
        print("placement matches:", fen.split()[0] == bd.board_fen(), "| turn:", fen.split()[1], "| flipped:", self.board.flipped)
        print("best moves card:", self.main.result.text.replace("\n", " | ")[:110])
        print("status:", self.main.status.text)
        print("detector message shown:", [t for t in ("contour detector", "grid finder", "OpenCV not in this build") if t in self.main.status.text] or "(replaced by best-move text)")
        print("quad found:", None if self.quad is None else np.round(self.quad).astype(int).tolist())
        print("true quad :", np.round(dq).astype(int).tolist())
        self.stop()
T().run()
