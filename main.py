"""
Chess Analyzer for Android (Kivy)
=================================
Pick a chess screenshot from the gallery (or take a photo).  The app does everything
automatically:

    find the 4 corners of the board  ->  straighten it (perspective warp)
    ->  read the position (FEN)  ->  work out whose turn it is  ->  Stockfish best moves

If the board cannot be found, a 4-corner editor opens so you can outline it by hand.
Back / forward arrows step through the positions you have played.

    screen capture        ->  gallery / camera
    OpenCV board crop     ->  board_detect.py (contours + grid fit, perspective warp)
    chess-fen-detector    ->  fen_model.py   (same network, NumPy only)
    detect_last_move      ->  chess_utils.py
    Stockfish             ->  engine.py      (bundled binary, Lichess fallback)
"""
import os
import sys
import threading

import chess
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.utils import platform

import android_utils
import board_detect
import chess_utils as cu
import engine
from board_widget import BoardWidget
from crop_view import CropView
from fen_model import FenDetector
from theme import (GREEN, PANEL_BORDER, PANEL_FILL, STATUS, TEXT, TEXT_MUTED, IconButton, PillToggle,
                   RoundedBox, icon_path, label, rgba, set_gradient)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FONT = os.path.join(ASSETS, "fonts", "DejaVuSans.ttf")
WEIGHTS = os.path.join(ASSETS, "piece_model.npz")

ARROW_COLORS = ["#0A70AC", "#FF8800", "#FFCC00"]
SF_DEPTH, SF_MOVES = 15, 3
HISTORY_MAX = 400


class TapImage(ButtonBehavior, Image):
    pass


def to_texture(rgb):
    """numpy HxWx3 uint8 -> Kivy texture (upright)."""
    rgb = np.ascontiguousarray(rgb)
    tex = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt="rgb")
    tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    tex.flip_vertical()
    return tex


# ──────────────────────────────────────────────────────────────────────────────
# The reading pipeline (runs in a worker thread)
# ──────────────────────────────────────────────────────────────────────────────

def read_position(detector, board_rgb):
    """Upright board picture -> dict with full FEN, orientation, last move, notes."""
    seen = detector.predict_placement(board_rgb)                 # as printed on screen
    orientation = cu.guess_orientation(seen)
    last = cu.detect_last_move(board_rgb, seen, orientation)
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
        outer = BoxLayout(orientation="vertical")
        set_gradient(outer)

        scroll = ScrollView(do_scroll_x=False, bar_width=0)
        col = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10),
                        padding=(dp(14), dp(6), dp(14), dp(8)))
        col.bind(minimum_height=col.setter("height"))

        # ── header ──
        head = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10))
        head.add_widget(label("Chess Analyzer", size=21, color=TEXT))
        head.add_widget(IconButton("Gallery", icon="image", glow=True, size_hint_x=None, width=dp(104),
                                   on_release=app.on_gallery))
        head.add_widget(IconButton("Camera", icon="camera", glow=True, size_hint_x=None, width=dp(104),
                                   on_release=app.on_camera))
        col.add_widget(head)

        # ── board ──
        self.board = BoardWidget(FONT, size_hint=(1, None))
        self.board.bind(width=lambda w, v: setattr(w, "height", v))
        self.board.on_user_move(app.on_user_move)
        col.add_widget(self.board)

        # ── back / forward ──
        hist = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(10))
        self.btn_back = IconButton(icon="arrow-left", size_hint_x=None, width=dp(64), on_release=app.on_back)
        self.btn_fwd = IconButton(icon="arrow-right", size_hint_x=None, width=dp(64), on_release=app.on_forward)
        self.hist_label = label("", size=13, color=TEXT_MUTED, halign="center")
        hist.add_widget(self.btn_back)
        hist.add_widget(self.hist_label)
        hist.add_widget(self.btn_fwd)
        col.add_widget(hist)

        # ── side to move ──
        row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        row.add_widget(label("Side to move", size=14, color=TEXT_MUTED, size_hint_x=None, width=dp(104)))
        self.turn_toggle = PillToggle(on_change=app.on_turn)
        row.add_widget(self.turn_toggle)
        col.add_widget(row)

        # ── actions ──
        row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        row.add_widget(IconButton("Analyse", icon="flask-conical", icon_right=True, icon_variant="green",
                                  on_release=app.on_analyse))
        row.add_widget(IconButton("Flip", icon="refresh-cw", icon_right=True, icon_variant="green",
                                  on_release=app.on_flip))
        row.add_widget(IconButton("Reset", icon="rotate-ccw", icon_right=True, icon_variant="green",
                                  on_release=app.on_reset))
        col.add_widget(row)

        # ── FEN ──
        row = BoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))
        box = RoundedBox(fill=PANEL_FILL, border=PANEL_BORDER, radius=12, padding=(dp(4), dp(2), dp(8), dp(2)))
        self.fen_input = TextInput(multiline=True, font_size=sp(14), foreground_color=rgba("#cfd4d5"),
                                   background_normal="", background_active="", background_color=(0, 0, 0, 0),
                                   cursor_color=(1, 1, 1, 1), write_tab=False, padding=(dp(10), dp(9)))
        self.fen_input.bind(text=app.on_fen_text)
        box.add_widget(self.fen_input)
        copy_icon = TapImage(source=icon_path("copy", "green"), size_hint=(None, None), size=(dp(22), dp(22)),
                             pos_hint={"center_y": 0.5}, fit_mode="contain")
        copy_icon.bind(on_release=app.on_copy)
        side = BoxLayout(size_hint=(None, 1), width=dp(26), padding=(0, dp(18), 0, dp(18)))
        side.add_widget(copy_icon)
        box.add_widget(side)
        row.add_widget(box)
        row.add_widget(IconButton("Set", size_hint_x=None, width=dp(58), on_release=app.on_fen_enter))
        row.add_widget(IconButton("Copy", size_hint_x=None, width=dp(68), on_release=app.on_copy))
        col.add_widget(row)

        # ── results card (shows the straightened board that was read + the best moves) ──
        self.card = RoundedBox(fill="#1a252d", border="#2c3b46", radius=12, size_hint_y=None, height=dp(80),
                               padding=dp(10), spacing=dp(10), opacity=0)
        self.thumb = TapImage(size_hint=(None, None), size=(dp(70), dp(70)), fit_mode="contain",
                              on_release=app.on_adjust)
        self.result = Label(text="", markup=True, font_size=sp(16), color=rgba("#a3e635"), halign="left",
                            valign="top", size_hint_y=None)
        self.result.bind(width=lambda w, v: setattr(w, "text_size", (v, None)),
                         texture_size=self._fit_card)
        thumb_col = BoxLayout(orientation="vertical", size_hint=(None, 1), width=dp(70), spacing=dp(2))
        thumb_col.add_widget(self.thumb)
        thumb_col.add_widget(label("tap: adjust", size=10, color=TEXT_MUTED, halign="center",
                                   size_hint_y=None, height=dp(14)))
        self.thumb_col = thumb_col
        self.card.add_widget(thumb_col)
        self.card.add_widget(self.result)
        col.add_widget(self.card)

        scroll.add_widget(col)
        outer.add_widget(scroll)

        # ── status line (pinned to the bottom) ──
        self.status = label("", size=12, color=STATUS["info"], size_hint_y=None, height=dp(34),
                            padding=(dp(16), 0))
        outer.add_widget(self.status)
        self.add_widget(outer)
        self.thumb_col.opacity = 0
        self.thumb_col.width = 0

    def _fit_card(self, w, size):
        w.height = max(size[1], dp(54))
        self.card.height = w.height + dp(20) + (dp(8) if self.thumb_col.width else 0)
        self.card.opacity = 1 if w.text else 0

    def show_thumb(self, tex):
        self.thumb.texture = tex
        self.thumb_col.opacity = 1
        self.thumb_col.width = dp(70)

    def set_turn_toggle(self, turn):
        self.turn_toggle.set_value(turn)


class CropScreen(Screen):
    def __init__(self, app, **kw):
        super().__init__(name="crop", **kw)
        root = BoxLayout(orientation="vertical", padding=(dp(14), dp(8)), spacing=dp(8))
        set_gradient(root)
        self.hint = label("", size=13, color=TEXT, halign="center", size_hint_y=None, height=dp(46))
        root.add_widget(self.hint)
        self.crop = CropView()
        root.add_widget(self.crop)
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        row.add_widget(IconButton("Back", icon="arrow-left", on_release=lambda *_: app.show_main()))
        row.add_widget(IconButton("Detect", icon="crop", icon_variant="green", on_release=app.on_redetect))
        row.add_widget(IconButton("Analyse", icon="flask-conical", icon_variant="green", glow=True,
                                  on_release=app.on_crop_confirm))
        root.add_widget(row)
        self.add_widget(root)


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

class ChessAnalyzerApp(App):
    title = "Chess Analyzer"
    icon = os.path.join(ASSETS, "icon.png")

    def build(self):
        if platform not in ("android", "ios"):
            Window.size = (400, 800)                     # phone-like window on a desktop
        else:
            Window.softinput_mode = "below_target"       # keyboard must not hide the FEN box
        self.detector = None
        self.image = None                                 # last picture (numpy) and its board corners
        self.quad = None
        self._analysis_id = 0
        self._busy_view = None
        self._busy_label = None
        self.turn = "w"
        self.history, self.hindex = [], -1
        self._cache = {}                                  # fen -> (lines, note)
        self._engine_lock = threading.Lock()              # one Stockfish at a time

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
        self.hist_reset()
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
        self.main.status.color = rgba(STATUS.get(level, "#e2e8f0"))

    def busy(self, msg):
        if self._busy_view is not None:
            self._busy_label.text = msg
            return
        box = RoundedBox(fill="#141d22", border="#2d6a3d", radius=14, padding=dp(12))
        self._busy_label = Label(text=msg, font_size=sp(16), color=rgba(TEXT))
        box.add_widget(self._busy_label)
        v = ModalView(size_hint=(0.8, None), height=dp(90), auto_dismiss=False,
                      background="", background_color=(0, 0, 0, 0.35))
        v.add_widget(box)
        v.open()
        self._busy_view = v

    def unbusy(self):
        if self._busy_view is not None:
            self._busy_view.dismiss()
            self._busy_view = None

    def ui(self, fn, *a):
        """Run fn(*a) on the Kivy thread (safe to call from worker threads)."""
        Clock.schedule_once(lambda dt: fn(*a), 0)

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
        threading.Thread(target=self._process_picture, args=(path,), daemon=True).start()

    def _wait_for_model(self):
        while self.detector is None:
            threading.Event().wait(0.1)
        return self.detector

    def _process_picture(self, path):
        """Worker: load -> find corners -> straighten -> read -> (UI) analyse.  Fully automatic."""
        try:
            img = android_utils.load_image(path)
            self.ui(self.busy, "Finding the board...")
            res = board_detect.extract_board(img)
            if res is None:
                self.ui(self._open_editor, img, None,
                        "Board not found automatically. Drag the 4 corners onto the board corners, then tap Analyse.")
                return
            board, quad, info = res
            self.ui(self.busy, "Reading the position...")
            read = read_position(self._wait_for_model(), board)
            self.ui(self._position_read, read, img, quad, board, info["method"])
        except Exception as e:                                   # noqa - never crash the app
            self.ui(self.unbusy)
            self.ui(self.status, "Could not process the picture: %s" % e, "err")

    def _open_editor(self, img, quad, hint):
        self.unbusy()
        self.image = img
        self.quad = quad
        self.cropper.crop.set_image(img, quad)
        self.cropper.hint.text = hint
        self.sm.current = "crop"

    def on_adjust(self, *_):
        """Re-open the corner editor on the last picture (tap the small board preview)."""
        if self.image is None:
            return
        self._open_editor(self.image, self.quad, "Drag the corners so the grid lines sit on the board squares, then tap Analyse.")

    def on_redetect(self, *_):
        if self.image is None:
            return
        self.busy("Looking for the board...")
        img = self.image

        def work():
            res = board_detect.extract_board(img)
            def done():
                self.unbusy()
                if res:
                    self.cropper.crop.set_quad(res[1])
                    self.cropper.hint.text = "Board found. Check the corners, then tap Analyse."
                else:
                    self.cropper.hint.text = "No board found - drag the 4 corners onto the board by hand."
            self.ui(done)

        threading.Thread(target=work, daemon=True).start()

    def on_crop_confirm(self, *_):
        if self.image is None:
            return
        quad = self.cropper.crop.get_quad()
        img = self.image
        self.busy("Reading the position...")

        def work():
            try:
                board = board_detect.warp_board(img, quad)
                read = read_position(self._wait_for_model(), board)
                self.ui(self._position_read, read, img, quad, board)
            except Exception as e:                               # noqa
                self.ui(self.unbusy)
                self.ui(self.status, "Reading failed: %s" % e, "err")

        threading.Thread(target=work, daemon=True).start()

    def _position_read(self, res, img, quad, board_rgb, method=None):
        self.unbusy()
        self.image, self.quad = img, np.asarray(quad, dtype=float)
        self.show_main()
        self.main.show_thumb(to_texture(board_rgb))
        self.board.flipped = res["orientation"] == "black_bottom"
        self.apply_fen(res["fen"])
        if res["last"]:
            self.board.last_move = [chess.parse_square(res["last"][0]), chess.parse_square(res["last"][1])]
        self.hist_reset()
        how = {"contour": "contour detector", "checker": "grid finder"}.get(method, "your corners")
        msg = "Board found with the %s." % how if method else "Position read from your corners."
        if method == "checker" and not board_detect.HAVE_CV2:
            msg += " (OpenCV not in this build.)"
        if res["notes"]:
            msg += " " + " ".join(res["notes"])
        problems = cu.position_problems(res["fen"])
        if problems:
            self.status("Position looks wrong (%s). Tap the preview to adjust the corners, or edit the FEN." % ", ".join(problems), "warn")
            self.main.result.text = ""
            return
        self.status(msg, "warn" if res["notes"] else "ok")
        self.on_analyse()

    # -- board state -----------------------------------------------------------
    def apply_fen(self, fen):
        if not self.board.set_fen(fen):
            self.status("Invalid FEN", "err")
            return False
        self._set_fen_text(fen)
        self.turn = fen.split()[1]
        self.main.set_turn_toggle(self.turn)
        return True

    def _set_fen_text(self, fen):
        ti = self.main.fen_input
        ti.text = fen
        Clock.schedule_once(lambda dt: (setattr(ti, "cursor", (0, 0)), setattr(ti, "scroll_y", 0)), 0.05)

    def on_user_move(self):
        self._set_fen_text(self.board.get_fen())
        self.turn = "w" if self.board.board.turn == chess.WHITE else "b"
        self.main.set_turn_toggle(self.turn)
        self.hist_push()
        self.on_analyse()

    def on_turn(self, turn):
        if turn == self.turn:
            return
        parts = self.board.get_fen().split()
        parts[1] = turn
        last, flipped = list(self.board.last_move), self.board.flipped
        if self.apply_fen(" ".join(parts)):
            self.board.flipped = flipped
            self.board.last_move = last
            self.hist_push()
            self.on_analyse()

    def on_fen_text(self, ti, text):
        """The FEN box is multi-line (so long FENs wrap); the keyboard's Enter key applies it."""
        if "\n" in text:
            ti.text = " ".join(text.split())
            ti.focus = False
            self.on_fen_enter()

    def on_fen_enter(self, *_):
        fen = " ".join(self.main.fen_input.text.split())
        flipped = self.board.flipped
        if self.apply_fen(fen):
            self.board.flipped = flipped
            self.main.result.text = ""
            self.hist_push()
            self.on_analyse()

    def on_flip(self, *_):
        self.board.flipped = not self.board.flipped

    def on_reset(self, *_):
        self.board.flipped = False
        self.apply_fen(chess.STARTING_FEN)
        self.hist_reset()
        self.main.result.text = ""
        self.status("Board reset to the starting position.", "info")

    def on_copy(self, *_):
        Clipboard.copy(self.board.get_fen())
        self.status("FEN copied to the clipboard.", "ok")

    # -- history (back / forward) -------------------------------------------------
    def _state(self):
        return {"fen": self.board.get_fen(), "last": list(self.board.last_move)}

    def hist_reset(self):
        self.history, self.hindex = [self._state()], 0
        self._hist_ui()

    def hist_push(self):
        self.history = self.history[:self.hindex + 1]            # playing from the past drops the old future
        self.history.append(self._state())
        if len(self.history) > HISTORY_MAX:
            self.history = self.history[-HISTORY_MAX:]
        self.hindex = len(self.history) - 1
        self._hist_ui()

    def _hist_ui(self):
        self.main.btn_back.enabled = self.hindex > 0
        self.main.btn_fwd.enabled = self.hindex < len(self.history) - 1
        self.main.hist_label.text = "Position %d of %d" % (self.hindex + 1, len(self.history))

    def _goto(self, index):
        self.hindex = index
        st = self.history[index]
        flipped = self.board.flipped
        self.apply_fen(st["fen"])
        self.board.flipped = flipped
        self.board.last_move = list(st["last"])
        self._hist_ui()
        self.on_analyse()

    def on_back(self, *_):
        if self.hindex > 0:
            self._goto(self.hindex - 1)

    def on_forward(self, *_):
        if self.hindex < len(self.history) - 1:
            self._goto(self.hindex + 1)

    # -- analysis ---------------------------------------------------------------
    def on_analyse(self, *_):
        fen = self.board.get_fen()
        problems = cu.position_problems(fen)
        if problems:
            self.main.result.text = ""
            self.status("Cannot analyse: %s." % ", ".join(problems), "warn")
            return
        self._analysis_id += 1
        my_id = self._analysis_id
        if fen in self._cache:                                   # visited before: instant
            lines, note = self._cache[fen]
            self._show_lines(my_id, fen, lines, note)
            return
        self.main.result.text = "Analysing..."
        self.status("Running Stockfish...", "info")

        def work():
            stale = lambda: my_id != self._analysis_id            # a newer request replaces this one
            with self._engine_lock:
                if stale():
                    return
                lines, note = engine.analyse(fen, SF_MOVES, SF_DEPTH, should_stop=stale)
            if lines:
                self._cache[fen] = (lines, note)
            self.ui(self._show_lines, my_id, fen, lines, note)

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
        out.append("[size=%d][color=#8c94a3]%s, depth %d. Score: + = White better.[/color][/size]"
                   % (int(sp(12)), lines[0].source, lines[0].depth))
        self.main.result.text = "\n".join(out)
        self.board.arrows = arrows
        self.status((note + " " if note else "") + "Best move: %s" % lines[0].san, "warn" if note else "ok")


if __name__ == "__main__":
    ChessAnalyzerApp().run()
