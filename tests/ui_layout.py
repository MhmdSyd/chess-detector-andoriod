"""Layout test: every control visible, nothing scrolls, title/FEN on one line, on several phone sizes.
   CHESS_WIN=360x640 xvfb-run -a python tests/ui_layout.py"""
import os, sys
os.environ["KIVY_NO_ARGS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.text import Label as CL
from kivy.metrics import dp, sp
from kivy.uix.scrollview import ScrollView
import main as M

class T(M.ChessAnalyzerApp):
    def on_start(self):
        super().on_start(); self.set_depth(15, analyse=False)
        Clock.schedule_once(self.check, 1.5)
    def walk(self, w):
        yield w
        for c in w.children: yield from self.walk(c)
    def check(self, dt):
        m, W, H = self.main, Window.width, Window.height
        res = []; ok = lambda n, c: (res.append(bool(c)), print(("PASS " if c else "FAIL ") + n))
        ok("no ScrollView anywhere on the main screen", not any(isinstance(w, ScrollView) for w in self.walk(m)))
        key = {"title": m.children[0].children[-1].children[-1] if False else None}
        outer = m.children[0]; rows = list(reversed(outer.children))         # top -> bottom
        names = ["header", "board", "history", "side", "actions", "depth", "fen", "card", "status"]
        for n, w in zip(names, rows):
            x, y = w.to_window(w.x, w.y); inside = x >= -1 and y >= -1 and x + w.width <= W + 1 and y + w.height <= H + 1 and w.height > 0
            ok(f"{n:8s} inside the {W}x{H} window (y={y:.0f}..{y + w.height:.0f}, h={w.height:.0f})", inside)
        board_side = min(m.board.width, m.board.height)
        ok(f"board is big enough to play ({board_side:.0f}px >= 200)", board_side >= 200)
        title = rows[0].children[-1]
        need = CL(font_size=sp(17), bold=True).get_extents("Chess Analyzer")[0]
        ok(f"title fits on ONE line without being cut ({need:.0f}px needed, {title.width:.0f}px available)", need <= title.width)
        ok("FEN field is single-line", not m.fen_input.multiline)
        ok("depth field is single-line + numeric", not m.depth_input.multiline and m.depth_input.input_filter == "int")
        # rows must not overlap each other
        tops = [(w.to_window(w.x, w.y)[1], w.height) for w in rows]
        ok("rows do not overlap", all(tops[i][0] >= tops[i + 1][0] + tops[i + 1][1] - 1 for i in range(len(tops) - 1)))
        print(f"{sum(res)}/{len(res)} checks passed @ {W}x{H}")
        Window.screenshot(f"/tmp/layout_{W}x{H}.png"); self.stop()
T().run()
