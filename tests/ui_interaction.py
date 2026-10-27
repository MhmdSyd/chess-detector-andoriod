"""Headless touch-interaction test:  xvfb-run -a python tests/ui_interaction.py"""
import os, sys
os.environ["KIVY_NO_ARGS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
sys.argv = [sys.argv[0]]
import numpy as np, chess
from kivy.clock import Clock
import main as M

class Touch:                       # minimal stand-in for a Kivy touch
    def __init__(self, x, y): self.pos = (x, y); self.grab_current = None
    def grab(self, w): self.grab_current = w
    def ungrab(self, w): self.grab_current = None

def tap(board, sq):
    x, y, s = board._sq_rect(sq); board.on_touch_down(Touch(x + s/2, y + s/2))

results = []
def check(name, cond): results.append(cond); print(("PASS " if cond else "FAIL ") + name)

class T(M.ChessAnalyzerApp):
    def on_start(self):
        super().on_start()
        Clock.schedule_once(self.step1, 1.5)
    def step1(self, dt):
        b = self.board
        # --- play 1.e4 by tapping e2 then e4 ---
        tap(b, chess.E2); check("select e2 shows 2 targets", len(b.targets) == 2)
        tap(b, chess.E4)
        check("move e2e4 applied", b.get_fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b"))
        check("turn toggle switched to Black", self.main.btn_b.state == "down")
        check("last-move squares set", list(b.last_move) == [chess.E2, chess.E4])
        # --- tapping opponent / empty squares does nothing harmful ---
        tap(b, chess.E4); check("cannot select a piece of the side NOT to move", b.selected is None)
        # --- flipped board maps taps correctly ---
        self.on_flip(); b.flipped = True
        tap(b, chess.E7); tap(b, chess.E5)
        check("flipped board: e7e5 played", b.get_fen().split()[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR")
        Clock.schedule_once(self.step2, 5)
    def step2(self, dt):
        check("analysis result shown after user move", "1." in self.main.result.text and "Stockfish" in self.main.result.text)
        check("3 arrows drawn", len(self.board.arrows) == 3)
        # --- FEN entry + invalid FEN ---
        self.main.fen_input.text = "8/8/8/8/8/8/8/8 w - - 0 1"; self.on_fen_enter()
        check("empty board is rejected for analysis (no kings)", "no white king" in self.main.status.text)
        self.main.fen_input.text = "garbage"; self.on_fen_enter()
        check("garbage FEN reports an error", "Invalid" in self.main.status.text)
        self.on_reset(); check("reset restores start position", self.board.get_fen() == chess.STARTING_FEN)
        # --- crop widget gestures ---
        c = self.cropper.crop; c.size = (400, 600); c.pos = (0, 0)
        img = np.zeros((1000, 800, 3), np.uint8); c.set_image(img)           # 800x1000 image
        c.box = [100, 200, 400]
        k, ox, oy = c._layout()
        def at(ix, iy): return c._to_widget(ix, iy)
        t = Touch(*at(500, 600)); c.on_touch_down(t)                         # bottom-right corner (BR)
        check("grabbing BR corner enters resize mode", c._mode == 3)
        t.pos = at(700, 800); c.on_touch_move(t); c.on_touch_up(t)
        check("BR resize keeps top-left fixed and stays square", c.box[0] == 100 and c.box[1] == 200 and c.box[2] == 600)
        t = Touch(*at(100, 200)); c.on_touch_down(t); t.pos = at(300, 300); c.on_touch_move(t); c.on_touch_up(t)
        x, y, s = c.get_box(); check("TL resize keeps bottom-right fixed", x + s == 700 and y + s == 800)
        t = Touch(*at(400, 600)); c.on_touch_down(t)
        check("touch inside = move mode", c._mode == "move")
        t.pos = at(2000, 2000); c.on_touch_move(t); c.on_touch_up(t)
        x, y, s = c.get_box(); check("dragging far outside is clamped to image", x + s <= 800 and y + s <= 1000 and x >= 0 and y >= 0)
        t = Touch(*at(x + s - 10, y + s - 10)); c.on_touch_down(t); t.pos = at(5000, 5000); c.on_touch_move(t); c.on_touch_up(t)
        x, y, s = c.get_box(); check("resize beyond image is clamped", x + s <= 800 and y + s <= 1000)
        print(f"\n{sum(results)}/{len(results)} checks passed")
        self.stop()

T().run()
