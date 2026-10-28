"""Headless end-to-end UI test:  xvfb-run -a python tests/ui_smoke.py"""
import os, sys, random
os.environ["KIVY_NO_ARGS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np, chess
from PIL import Image, ImageDraw
from render import render, random_position

# ---- build a fake phone screenshot: black just played, last move highlighted (cyan) ----
bd = random_position(21, 77)               # 21 plies -> black has just moved (odd) => white to move
mv = bd.peek(); px = 1000
pil = render(bd, px=px, theme="green")
a = np.array(pil).astype(np.float32); sq = px // 8
for s_ in (mv.from_square, mv.to_square):
    r, c = 7 - chess.square_rank(s_), chess.square_file(s_)
    reg = a[r*sq:(r+1)*sq, c*sq:(c+1)*sq]; a[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 0.45*reg + 0.55*np.array([60, 190, 205])
canvas = Image.new("RGB", (1080, 2340), (38, 36, 33)); d = ImageDraw.Draw(canvas)
rnd = random.Random(1)
for _ in range(14):
    y = rnd.choice(list(range(80, 540, 60)) + list(range(1700, 2300, 70))); d.rectangle([40, y, 40 + rnd.randint(200, 900), y + 30], fill=(90, 88, 84))
canvas.paste(Image.fromarray(a.astype(np.uint8)), (40, 620))
if os.environ.get("TILT"): canvas = canvas.rotate(float(os.environ["TILT"]), resample=Image.BILINEAR, expand=True, fillcolor=(38, 36, 33))
path = "/tmp/fake_screenshot.png"; canvas.save(path)
print("expected FEN placement:", bd.board_fen(), "| side to move:", "w" if bd.turn else "b", "| last move:", mv.uci())

# ---- run the app and script it ----
sys.argv = [sys.argv[0]]
from kivy.clock import Clock
from kivy.core.window import Window
import main as M

class TestApp(M.ChessAnalyzerAppBase if hasattr(M, "ChessAnalyzerAppBase") else M.ChessAnalyzerApp):
    def on_start(self):
        super().on_start()
        Clock.schedule_once(lambda dt: self.on_image_ready(path, None), 1.0)
        Clock.schedule_once(lambda dt: Window.screenshot("/tmp/ui_1_crop.png"), 10)
        Clock.schedule_once(lambda dt: (print("box:", self.cropper.crop.get_box()), self.on_crop_confirm()), 10.5)
        Clock.schedule_once(lambda dt: Window.screenshot("/tmp/ui_2_main.png"), 18)
        Clock.schedule_once(lambda dt: self.report(), 18.5)
    def report(self):
        print("app FEN :", self.board.get_fen())
        print("expected:", bd.fen().rsplit(" ", 2)[0].split(" ")[0], "(placement)")
        print("placement match:", self.board.get_fen().split()[0] == bd.board_fen(), "| turn:", self.board.get_fen().split()[1])
        print("flipped:", self.board.flipped, "| last_move squares:", [chess.square_name(s) for s in self.board.last_move])
        print("result label:\n", self.main.result.text)
        print("status:", self.main.status.text)
        self.stop()

TestApp().run()
