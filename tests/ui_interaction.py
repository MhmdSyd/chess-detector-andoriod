"""Headless touch/history test:  xvfb-run -a python tests/ui_interaction.py"""
import os, sys
os.environ["KIVY_NO_ARGS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
import numpy as np, chess
from kivy.clock import Clock
import main as M

class Touch:
    def __init__(self, x, y): self.pos = (x, y); self.x, self.y = x, y; self.grab_current = None
    def grab(self, w): self.grab_current = w
    def ungrab(self, w): self.grab_current = None

def tap(board, sq):
    x, y, s = board._sq_rect(sq); board.on_touch_down(Touch(x + s/2, y + s/2))

results = []
def check(name, cond): results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

class T(M.ChessAnalyzerApp):
    def on_start(self):
        super().on_start(); self.set_depth(10, analyse=False); Clock.schedule_once(self.step1, 1.5)
    def step1(self, dt):
        b, m = self.board, self.main
        check("history starts with 1 entry, arrows disabled", len(self.history) == 1 and not m.btn_back.enabled and not m.btn_fwd.enabled)
        tap(b, chess.E2); check("select e2 shows 2 targets", len(b.targets) == 2)
        tap(b, chess.E4)
        check("move e2e4 applied", b.get_fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b"))
        check("turn toggle switched to Black", m.turn_toggle.value == "b")
        tap(b, chess.E7); tap(b, chess.E5)
        check("black reply e7e5 applied", b.get_fen().split()[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR")
        check("history has 3 entries, Back enabled, Forward disabled", len(self.history) == 3 and m.btn_back.enabled and not m.btn_fwd.enabled)
        # ---- back / forward ----
        self.on_back()
        check("Back -> position after 1.e4", b.get_fen().split()[0] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR" and m.turn_toggle.value == "b")
        check("Forward now enabled", m.btn_fwd.enabled and m.btn_back.enabled)
        check("last-move highlight restored on Back", list(b.last_move) == [chess.E2, chess.E4])
        self.on_back()
        check("Back again -> start, Back disabled", b.get_fen() == chess.STARTING_FEN and not m.btn_back.enabled)
        self.on_back(); check("Back at the start does nothing", b.get_fen() == chess.STARTING_FEN)
        self.on_forward(); self.on_forward()
        check("Forward twice -> after 1...e5, Forward disabled", b.get_fen().split()[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR" and not m.btn_fwd.enabled)
        self.on_forward(); check("Forward at the end does nothing", len(self.history) == 3 and self.hindex == 2)
        # play a different move after going back -> old future is dropped
        self.on_back(); tap(b, chess.C7); tap(b, chess.C5)
        check("new move after Back drops the old future", len(self.history) == 3 and not m.btn_fwd.enabled and b.get_fen().split()[0] == "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR")
        # ---- turn toggle goes in the history too ----
        n = len(self.history); cur = b.get_fen().split()[1]; other = "b" if cur == "w" else "w"
        self.on_turn(other)
        check("side-to-move change is a history step", len(self.history) == n + 1 and b.get_fen().split()[1] == other)
        self.on_back(); check("Back undoes the side change", b.get_fen().split()[1] == cur)
        # ---- flipped board still maps taps correctly ----
        self.on_reset(); b.flipped = True
        tap(b, chess.E2); tap(b, chess.E4); check("flipped board: e2e4 played", b.get_fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b"))
        Clock.schedule_once(self.step2, 5)
    def step2(self, dt):
        check("analysis result card shown", "1." in self.main.result.text and self.main.card.opacity == 1)
        check("3 arrows drawn", len(self.board.arrows) == 3)
        self.on_back(); self.on_forward()
        check("cached analysis re-used instantly after Back/Forward", len(self.board.arrows) == 3)
        self.main.fen_input.text = "8/8/8/8/8/8/8/8 w - - 0 1"; self.on_fen_enter()
        check("empty board rejected for analysis (no kings)", "no white king" in self.main.status.text)
        self.main.fen_input.text = "garbage"; self.on_fen_enter()
        check("garbage FEN reports an error", "Invalid" in self.main.status.text)
        self.main.fen_input.text = chess.STARTING_FEN + "\n"          # Enter key on the on-screen keyboard
        check("Enter (newline) in the FEN box applies it", self.board.get_fen() == chess.STARTING_FEN and "\n" not in self.main.fen_input.text)
        # ---- 4-corner editor gestures ----
        c = self.cropper.crop; c.size = (400, 600); c.pos = (0, 0)
        c.set_image(np.zeros((1000, 800, 3), np.uint8), np.array([[100, 200], [500, 200], [500, 600], [100, 600]], float))
        at = lambda ix, iy: c._to_widget(ix, iy)
        t = Touch(*at(500, 600)); c.on_touch_down(t)
        check("grabbing a corner selects it (BR)", c._mode == 2 and c.dragging == 2)
        t.pos = at(560, 640); c.on_touch_move(t)
        q = c.get_quad(); check("only that corner moves", np.allclose(q[2], [560, 640], atol=1) and np.allclose(q[0], [100, 200]) and np.allclose(q[1], [500, 200]) and np.allclose(q[3], [100, 600]))
        c.on_touch_up(t); check("loupe state cleared after release", c.dragging == -1)
        t = Touch(*at(300, 400)); c.on_touch_down(t); check("touch inside = move whole quad", c._mode == "move")
        t.pos = at(5000, 5000); c.on_touch_move(t); c.on_touch_up(t)
        q = c.get_quad(); check("moving far outside is clamped to the picture", q[:, 0].min() >= 0 and q[:, 1].min() >= 0 and q[:, 0].max() <= 799 and q[:, 1].max() <= 999)
        t = Touch(*at(q[0][0], q[0][1])); c.on_touch_down(t); t.pos = at(-300, -300); c.on_touch_move(t); c.on_touch_up(t)
        q = c.get_quad(); check("corner dragged outside is clamped", q[0][0] >= 0 and q[0][1] >= 0)
        # ---- Stockfish depth control ----
        self.set_depth(15, analyse=False)
        self.on_depth_step(+1); check("depth + button: 15 -> 16 (field shows 16)", self.depth == 16 and self.main.depth_input.text == "16")
        self.on_depth_step(-1); self.on_depth_step(-1); check("depth - button: 16 -> 14", self.depth == 14 and self.main.depth_input.text == "14")
        self.set_depth(99, analyse=False); check("depth is capped at 30", self.depth == 30 and self.main.depth_input.text == "30")
        self.set_depth(0, analyse=False); check("depth is floored at 1", self.depth == 1 and self.main.depth_input.text == "1")
        self.main.depth_input.text = "22"; self.on_depth_entered(); check("typing 22 + Enter sets depth 22", self.depth == 22)
        self.main.depth_input.text = ""; self.on_depth_entered(); check("empty field restores the old value", self.depth == 22 and self.main.depth_input.text == "22")
        import json
        saved = json.load(open(self._settings_file()))["depth"]; check("depth is saved to settings (%s)" % saved, saved == 22)
        check("a restarted app would load the saved depth", self._load_depth() == 22)
        check("depth field accepts digits only", self.main.depth_input.input_filter == "int")
        self.on_reset(); self.set_depth(6, analyse=False); self.on_user_move_demo = True
        b = self.board; tap(b, chess.E2); tap(b, chess.E4)                  # a move -> analysis at the chosen depth
        Clock.schedule_once(self.step3, 5)
    def step3(self, dt):
        txt = self.main.result.text
        check("engine really ran at depth 6 (card says: %s)" % txt.split("depth")[-1][:4].strip(), "depth 6" in txt)
        n_cached = len([k for k in self._cache if k[1] == 6]); self.set_depth(7)      # changing depth re-analyses (new cache key)
        check("cache is keyed by depth (depth-6 entry kept, depth-7 requested)", n_cached >= 1 and self.depth == 7)
        print(f"\n{sum(results)}/{len(results)} checks passed"); self.stop()

T().run()
