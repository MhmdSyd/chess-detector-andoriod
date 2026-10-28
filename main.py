"""
Chess Analyzer for Android (Kivy)
=================================
Pick a chess screenshot from the gallery (or take a photo), adjust the square around
the board if needed, and the app reads the position (FEN), works out whose turn it is
and shows Stockfish's best moves as arrows on an interactive board.

Android port of the desktop Tk application:
    screen capture        ->  gallery / camera
    OpenCV board crop     ->  board_finder.py (NumPy) + manual crop box
    chess-fen-detector    ->  fen_model.py   (same network, NumPy only)
    detect_last_move      ->  chess_utils.py
    Stockfish             ->  engine.py      (bundled binary, Lichess fallback)
"""
import os
import sys
import threading

import chess
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.utils import platform

import android_utils
import board_finder
import chess_utils as cu
import engine
from board_widget import BoardWidget
from crop_view import CropView
from fen_model import FenDetector

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FONT = os.path.join(ASSETS, "fonts", "DejaVuSans.ttf")
WEIGHTS = os.path.join(ASSETS, "piece_model.npz")

BG, BAR, INDIGO, TEAL, GREY = "#1a1a2e", "#16213e", "#4f46e5", "#0f766e", "#374151"
ARROW_COLORS = ["#0A70AC", "#FF8800", "#FFCC00"]
STATUS_COLORS = {"info": "#7dd3fc", "ok": "#a3e635", "warn": "#fbbf24", "err": "#f87171"}
SF_DEPTH, SF_MOVES = 15, 3


def rgba(h, a=1.0):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a


def flat_button(text, color, **kw):
    return Button(text=text, background_normal="", background_down="", background_color=rgba(color),
                  color=(1, 1, 1, 1), font_size=sp(14), bold=True, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# The reading pipeline (runs in a worker thread)
# ──────────────────────────────────────────────────────────────────────────────

def read_position(detector, crop_rgb):
    """Cropped board picture -> dict with full FEN, orientation, last move, notes."""
    seen = detector.predict_placement(crop_rgb)                  # as printed on screen
    orientation = cu.guess_orientation(seen)
    last = cu.detect_last_move(crop_rgb, seen, orientation)
    placement = seen if orientation == "white_bottom" else cu.flip_fen_vertical(seen)

    notes = []
    if last:
        turn = last["turn"]                                      # side to move next
    else:
        # No highlighted last move: assume the user wants advice for the side at the bottom
        turn = "w" if orientation == "white_bottom" else "b"
        notes.append("Last-move highlight not found - check whose turn it is.")
    turn, fixed = cu.fix_turn_if_needed(placement, turn)
    if fixed:
        notes.append("Turn corrected (the other king was in check).")
    return {
        "fen": cu.build_full_fen(placement, turn),
        "orientation": orientation,
        "last": (last["from"], last["to"]) if last else None,
        "notes": notes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Screens
# ──────────────────────────────────────────────────────────────────────────────

class MainScreen(Screen):
    def __init__(self, app, **kw):
        super().__init__(name="main", **kw)
        self.app = app
        root = BoxLayout(orientation="vertical")

        # top bar
        top = BoxLayout(size_hint_y=None, height=dp(52), padding=dp(6), spacing=dp(6))
        top.add_widget(Label(text="Chess Analyzer", font_size=sp(18), bold=True,
                             color=rgba("#e2e8f0"), halign="left", valign="middle"))
        top.children[0].bind(size=lambda w, s: setattr(w, "text_size", s))
        top.add_widget(flat_button("Gallery", INDIGO, size_hint_x=None, width=dp(86), on_release=app.on_gallery))
        top.add_widget(flat_button("Camera", INDIGO, size_hint_x=None, width=dp(86), on_release=app.on_camera))
        root.add_widget(top)

        # board
        self.board = BoardWidget(FONT, size_hint=(1, None))
        self.board.bind(width=lambda w, v: setattr(w, "height", v))
        self.board.on_user_move(app.on_user_move)
        root.add_widget(self.board)

        # side to move
        row = BoxLayout(size_hint_y=None, height=dp(42), padding=(dp(6), dp(3)), spacing=dp(6))
        row.add_widget(Label(text="Side to move:", size_hint_x=None, width=dp(100),
                             font_size=sp(13), color=rgba("#94a3b8")))
        self.btn_w = ToggleButton(text="White", group="turn", state="down", allow_no_selection=False,
                                  background_normal="", background_down="", font_size=sp(14))
        self.btn_b = ToggleButton(text="Black", group="turn", allow_no_selection=False,
                                  background_normal="", background_down="", font_size=sp(14))
        for b, ch in ((self.btn_w, "w"), (self.btn_b, "b")):
            b.bind(on_release=lambda _w, c=ch: app.on_turn(c), state=self._style_toggles)
            row.add_widget(b)
        self._style_toggles()
        root.add_widget(row)

        # actions
        row = BoxLayout(size_hint_y=None, height=dp(42), padding=(dp(6), dp(3)), spacing=dp(6))
        row.add_widget(flat_button("Analyse", TEAL, on_release=app.on_analyse))
        row.add_widget(flat_button("Flip", GREY, on_release=app.on_flip))
        row.add_widget(flat_button("Reset", GREY, on_release=app.on_reset))
        root.add_widget(row)

        # FEN
        row = BoxLayout(size_hint_y=None, height=dp(44), padding=(dp(6), dp(3)), spacing=dp(6))
        self.fen_input = TextInput(multiline=False, font_size=sp(11), foreground_color=rgba("#a3d7f0"),
                                   background_color=rgba("#0f172a"), cursor_color=(1, 1, 1, 1),
                                   write_tab=False, padding=(dp(6), dp(10)))
        self.fen_input.bind(on_text_validate=lambda *_: app.on_fen_enter())
        row.add_widget(self.fen_input)
        row.add_widget(flat_button("Set", GREY, size_hint_x=None, width=dp(52), on_release=lambda *_: app.on_fen_enter()))
        row.add_widget(flat_button("Copy", GREY, size_hint_x=None, width=dp(56), on_release=app.on_copy))
        root.add_widget(row)

        # results
        scroll = ScrollView(do_scroll_x=False, bar_width=dp(3))
        self.result = Label(text="", markup=True, font_size=sp(17), color=rgba("#a3e635"),
                            size_hint_y=None, halign="left", valign="top", padding=(dp(10), dp(6)))
        self.result.bind(width=lambda w, v: setattr(w, "text_size", (v, None)),
                         texture_size=lambda w, s: setattr(w, "height", max(s[1] + dp(8), dp(60))))
        scroll.add_widget(self.result)
        root.add_widget(scroll)

        # status bar
        self.status = Label(text="", size_hint_y=None, height=dp(34), font_size=sp(12),
                            halign="left", valign="middle", padding=(dp(8), 0), color=rgba(STATUS_COLORS["info"]))
        self.status.bind(size=lambda w, s: setattr(w, "text_size", s))
        root.add_widget(self.status)
        self.add_widget(root)

    def _style_toggles(self, *_):
        for b in (self.btn_w, self.btn_b):
            b.background_color = rgba(INDIGO) if b.state == "down" else rgba(GREY)

    def set_turn_toggle(self, turn):
        (self.btn_w if turn == "w" else self.btn_b).state = "down"
        (self.btn_b if turn == "w" else self.btn_w).state = "normal"
        self._style_toggles()


class CropScreen(Screen):
    def __init__(self, app, **kw):
        super().__init__(name="crop", **kw)
        self.app = app
        root = BoxLayout(orientation="vertical")
        self.hint = Label(text="", size_hint_y=None, height=dp(44), font_size=sp(13),
                          color=rgba("#e2e8f0"), halign="center", valign="middle")
        self.hint.bind(size=lambda w, s: setattr(w, "text_size", (s[0] - dp(12), s[1])))
        root.add_widget(self.hint)
        self.crop = CropView()
        root.add_widget(self.crop)
        row = BoxLayout(size_hint_y=None, height=dp(52), padding=dp(6), spacing=dp(6))
        row.add_widget(flat_button("Back", GREY, on_release=lambda *_: app.show_main()))
        row.add_widget(flat_button("Auto-detect", INDIGO, on_release=app.on_redetect))
        row.add_widget(flat_button("Analyse board", TEAL, on_release=app.on_crop_confirm))
        root.add_widget(row)
        self.add_widget(root)


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

class ChessAnalyzerApp(App):
    title = "Chess Analyzer"

    def build(self):
        Window.clearcolor = rgba(BG)
        if platform not in ("android", "ios"):
            Window.size = (400, 800)                     # phone-like window on a desktop
        self.detector = None
        self.image = None                                 # working picture (numpy)
        self._analysis_id = 0
        self._busy_view = None
        self.turn = "w"

        self.sm = ScreenManager(transition=NoTransition())
        self.main = MainScreen(self)
        self.cropper = CropScreen(self)
        self.sm.add_widget(self.main)
        self.sm.add_widget(self.cropper)
        Window.bind(on_keyboard=self._on_key)
        return self.sm

    def on_start(self):
        self.board = self.main.board
        self.apply_fen(chess.STARTING_FEN)
        self.status("Loading the piece-recognition model...", "info")
        threading.Thread(target=self._load_model, daemon=True).start()
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):          # desktop testing helper
            Clock.schedule_once(lambda dt: self.on_image_ready(sys.argv[1], None), 0.5)

    def _load_model(self):
        try:
            self.detector = FenDetector(WEIGHTS)
            Clock.schedule_once(lambda dt: self.status("Ready - pick a screenshot or take a photo of a board.", "info"))
        except Exception as e:                                       # noqa
            Clock.schedule_once(lambda dt: self.status("Model failed to load: %s" % e, "err"))

    # -- small UI helpers -------------------------------------------------------
    def status(self, msg, level="info"):
        self.main.status.text = msg
        self.main.status.color = rgba(STATUS_COLORS.get(level, "#e2e8f0"))

    def busy(self, msg):
        self.unbusy()
        v = ModalView(size_hint=(0.8, None), height=dp(90), auto_dismiss=False,
                      background_color=rgba("#0f172a", 0.95))
        v.add_widget(Label(text=msg, font_size=sp(16), color=(1, 1, 1, 1)))
        v.open()
        self._busy_view = v

    def unbusy(self):
        if self._busy_view is not None:
            self._busy_view.dismiss()
            self._busy_view = None

    def show_main(self, *_):
        self.sm.current = "main"

    def _on_key(self, window, key, *args):
        if key == 27 and self.sm.current == "crop":              # Android back button
            self.show_main()
            return True
        return False

    # -- getting a picture ------------------------------------------------------
    def on_gallery(self, *_):
        android_utils.pick_image(self.on_image_ready)

    def on_camera(self, *_):
        android_utils.take_photo(self.on_image_ready)

    def on_image_ready(self, path, message):
        if not path:
            if message:
                self.status(message, "err")
            return
        self.busy("Loading picture...")

        def work():
            try:
                img = android_utils.load_image(path)
                img, box, angle = board_finder.find_board_deskewed(img)     # straightens tilted photos
                Clock.schedule_once(lambda dt: self._show_crop(img, box, angle))
            except Exception as e:                               # noqa
                Clock.schedule_once(lambda dt: (self.unbusy(), self.status("Could not open picture: %s" % e, "err")))

        threading.Thread(target=work, daemon=True).start()

    def _show_crop(self, img, box, angle=0.0):
        self.unbusy()
        self.image = img
        self.cropper.crop.set_image(img)
        tilt = " Photo was tilted %.1f deg - straightened." % abs(angle) if abs(angle) >= 0.3 else ""
        if box:
            self.cropper.crop.box = list(box[:3])
            self.cropper.hint.text = "Board found." + tilt + " Drag the square or its corners if the grid is not on the squares."
        else:
            self.cropper.hint.text = "Board not found automatically." + tilt + " Move and resize the square over the board."
        self.sm.current = "crop"

    def on_redetect(self, *_):
        if self.image is None:
            return
        self.busy("Looking for the board...")
        src = self.image

        def work():
            img, box, angle = board_finder.find_board_deskewed(src)

            def done(dt):
                self.unbusy()
                if img is not src:                                   # a residual tilt was corrected
                    self.image = img
                    self.cropper.crop.set_image(img)
                if box:
                    self.cropper.crop.box = list(box[:3])
                    self.cropper.hint.text = "Board found." + (" Straightened %.1f deg." % abs(angle) if abs(angle) >= 0.3 else "")
                else:
                    self.cropper.hint.text = "No board found - place the square by hand."
            Clock.schedule_once(done)

        threading.Thread(target=work, daemon=True).start()

    # -- reading the board ------------------------------------------------------
    def on_crop_confirm(self, *_):
        if self.image is None:
            return
        x, y, s = self.cropper.crop.get_box()
        crop = self.image[y:y + s, x:x + s].copy()
        self.busy("Reading the position...")

        def work():
            try:
                while self.detector is None:                      # model still loading
                    threading.Event().wait(0.1)
                res = read_position(self.detector, crop)
                Clock.schedule_once(lambda dt: self._position_read(res))
            except Exception as e:                               # noqa
                Clock.schedule_once(lambda dt: (self.unbusy(), self.status("Reading failed: %s" % e, "err")))

        threading.Thread(target=work, daemon=True).start()

    def _position_read(self, res):
        self.unbusy()
        self.show_main()
        self.board.flipped = res["orientation"] == "black_bottom"
        self.apply_fen(res["fen"])
        if res["last"]:
            self.board.last_move = [chess.parse_square(res["last"][0]), chess.parse_square(res["last"][1])]
        msg = "Position read."
        if res["notes"]:
            msg += " " + " ".join(res["notes"])
        problems = cu.position_problems(res["fen"])
        if problems:
            self.status("Position looks wrong (%s). Tap a square/edit the FEN to fix it." % ", ".join(problems), "warn")
            self.main.result.text = ""
            return
        self.status(msg, "warn" if res["notes"] else "ok")
        self.on_analyse()

    # -- board state -----------------------------------------------------------
    def apply_fen(self, fen):
        if not self.board.set_fen(fen):
            self.status("Invalid FEN", "err")
            return False
        self.main.fen_input.text = fen
        self.turn = fen.split()[1]
        self.main.set_turn_toggle(self.turn)
        return True

    def on_user_move(self):
        self.main.fen_input.text = self.board.get_fen()
        self.turn = "w" if self.board.board.turn == chess.WHITE else "b"
        self.main.set_turn_toggle(self.turn)
        self.on_analyse()

    def on_turn(self, turn):
        parts = self.board.get_fen().split()
        parts[1] = turn
        last = list(self.board.last_move)
        flipped = self.board.flipped
        if self.apply_fen(" ".join(parts)):
            self.board.flipped = flipped
            self.board.last_move = last
            self.on_analyse()

    def on_fen_enter(self, *_):
        fen = self.main.fen_input.text.strip()
        if self.apply_fen(fen):
            self.main.result.text = ""
            self.on_analyse()

    def on_flip(self, *_):
        self.board.flipped = not self.board.flipped

    def on_reset(self, *_):
        self.board.flipped = False
        self.apply_fen(chess.STARTING_FEN)
        self.main.result.text = ""
        self.status("Board reset to the starting position.", "info")

    def on_copy(self, *_):
        Clipboard.copy(self.board.get_fen())
        self.status("FEN copied to the clipboard.", "ok")

    # -- analysis ---------------------------------------------------------------
    def on_analyse(self, *_):
        fen = self.board.get_fen()
        problems = cu.position_problems(fen)
        if problems:
            self.status("Cannot analyse: %s." % ", ".join(problems), "warn")
            return
        self._analysis_id += 1
        my_id = self._analysis_id
        self.main.result.text = "Analysing..."
        self.status("Running Stockfish...", "info")

        def work():
            lines, note = engine.analyse(fen, SF_MOVES, SF_DEPTH)
            Clock.schedule_once(lambda dt: self._show_lines(my_id, fen, lines, note))

        threading.Thread(target=work, daemon=True).start()

    def _show_lines(self, my_id, fen, lines, note):
        if my_id != self._analysis_id:
            return                                               # a newer request replaced this one
        if not lines:
            self.main.result.text = "No result."
            self.status(note or "Analysis failed.", "err")
            return
        out, arrows = [], []
        for i, ln in enumerate(lines):
            col = ARROW_COLORS[i % len(ARROW_COLORS)]
            out.append("[color=%s][b]%d.[/b][/color]  [b]%s[/b]   %s" % (col, i + 1, ln.san, engine.format_score(ln)))
            arrows.append((chess.parse_square(ln.uci[:2]), chess.parse_square(ln.uci[2:4]), col))
        out.append("[size=%d][color=#94a3b8]%s, depth %d. Score: + = White better.[/color][/size]"
                   % (int(sp(12)), lines[0].source, lines[0].depth))
        self.main.result.text = "\n".join(out)
        self.board.arrows = arrows
        self.status((note + " " if note else "") + "Best move: %s" % lines[0].san, "warn" if note else "ok")


if __name__ == "__main__":
    ChessAnalyzerApp().run()
