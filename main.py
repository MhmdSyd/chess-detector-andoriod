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
import json
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
from kivy.uix.textinput import TextInput
from kivy.utils import platform

import android_utils
import board_detect
import chess_utils as cu
import engine
from board_widget import BoardWidget
from crop_view import CropView
from fen_model import FenDetector
from theme import (GREEN, PANEL_BORDER, PANEL_FILL, STATUS, TEXT, TEXT_DIM, TEXT_MUTED, IconButton, PillToggle,
                   RoundedBox, icon_path, label, rgba, set_gradient)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
PIECES = os.path.join(ASSETS, "pieces")
WEIGHTS = os.path.join(ASSETS, "piece_model.npz")

ARROW_COLORS = ["#0A70AC", "#FF8800", "#FFCC00"]
SF_MOVES = 3
DEPTH_MIN, DEPTH_MAX, DEPTH_DEFAULT = 1, 30, 15
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

PLACEHOLDER = "[color=#5d6672]Best moves will appear here[/color]"


def centered_input(ti):
    """Vertically centre the text of a single-line TextInput inside its box."""
    def fix(*_):
        pad = max(0, (ti.height - ti.line_height) / 2.0)
        ti.padding_y = [pad, pad]
    ti.bind(size=fix)
    fix()


class MainScreen(Screen):
    """Fixed (non-scrolling) layout: every row has a set height, the board takes what is left."""

    def __init__(self, app, **kw):
        super().__init__(name="main", **kw)
        self.app = app
        outer = BoxLayout(orientation="vertical", padding=(dp(10), dp(6), dp(10), dp(4)), spacing=dp(6))
        self.outer = outer
        set_gradient(outer)

        # ── header: title on ONE line + two compact buttons ──
        head = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        self.title = label("Chess Analyzer", size=17, color=TEXT, bold=True, shorten=True, max_lines=1)
        head.add_widget(self.title)
        head.add_widget(IconButton("Gallery", icon="image", glow=True, font_size=12, radius=10,
                                   size_hint_x=None, width=dp(88), on_release=app.on_gallery))
        head.add_widget(IconButton("Camera", icon="camera", glow=True, font_size=12, radius=10,
                                   size_hint_x=None, width=dp(88), on_release=app.on_camera))
        outer.add_widget(head)
        self.head_buttons = [w for w in head.children if isinstance(w, IconButton)]

        # ── board: gets all the free height, drawn as the biggest centred square that fits ──
        self.board = BoardWidget(PIECES, size_hint=(1, 1))
        self.board.on_user_move(app.on_user_move)
        outer.add_widget(self.board)

        # ── back / forward ──
        hist = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(8))
        self.btn_back = IconButton(icon="arrow-left", radius=10, size_hint_x=None, width=dp(60), on_release=app.on_back)
        self.btn_fwd = IconButton(icon="arrow-right", radius=10, size_hint_x=None, width=dp(60), on_release=app.on_forward)
        self.hist_label = label("", size=12, color=TEXT_MUTED, halign="center")
        hist.add_widget(self.btn_back)
        hist.add_widget(self.hist_label)
        hist.add_widget(self.btn_fwd)
        outer.add_widget(hist)

        # ── side to move ──
        row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
        row.add_widget(label("Side to move", size=13, color=TEXT_MUTED, size_hint_x=None, width=dp(94)))
        self.turn_toggle = PillToggle(on_change=app.on_turn)
        row.add_widget(self.turn_toggle)
        outer.add_widget(row)

        # ── actions ──
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        for text, icon, cb in (("Analyse", "flask-conical", app.on_analyse), ("Flip", "refresh-cw", app.on_flip),
                               ("Reset", "rotate-ccw", app.on_reset)):
            row.add_widget(IconButton(text, icon=icon, icon_right=True, icon_variant="green", font_size=13,
                                      radius=10, on_release=cb))
        actions_row = row
        outer.add_widget(row)

        # ── Stockfish depth:  [-]  [ 15 ]  [+]  ──
        row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        row.add_widget(label("Stockfish depth", size=13, color=TEXT_MUTED, size_hint_x=None, width=dp(106)))
        row.add_widget(IconButton("-", font_size=20, radius=10, size_hint_x=None, width=dp(42),
                                  on_release=lambda *_: app.on_depth_step(-1)))
        dbox = RoundedBox(fill=PANEL_FILL, border=PANEL_BORDER, radius=10, size_hint_x=None, width=dp(58),
                          padding=(dp(2), 0))
        self.depth_input = TextInput(text=str(DEPTH_DEFAULT), multiline=False, input_filter="int", halign="center",
                                     font_size=sp(16), foreground_color=rgba("#e6eef0"), background_normal="",
                                     background_active="", background_color=(0, 0, 0, 0), cursor_color=(1, 1, 1, 1),
                                     write_tab=False)
        centered_input(self.depth_input)
        self.depth_input.bind(on_text_validate=app.on_depth_entered,
                              focus=lambda w, f: None if f else app.on_depth_entered())
        dbox.add_widget(self.depth_input)
        row.add_widget(dbox)
        row.add_widget(IconButton("+", font_size=20, radius=10, size_hint_x=None, width=dp(42),
                                  on_release=lambda *_: app.on_depth_step(+1)))
        self.depth_hint = label("range 1-30", size=11, color=TEXT_DIM, shorten=True, shorten_from="right", max_lines=1)
        row.add_widget(self.depth_hint)
        outer.add_widget(row)

        # ── FEN: one line ──
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        box = RoundedBox(fill=PANEL_FILL, border=PANEL_BORDER, radius=10, padding=(dp(2), 0, dp(8), 0))
        self.fen_input = TextInput(multiline=False, font_size=sp(12), foreground_color=rgba("#cfd4d5"),
                                   background_normal="", background_active="", background_color=(0, 0, 0, 0),
                                   cursor_color=(1, 1, 1, 1), write_tab=False, padding=(dp(8), 0))
        centered_input(self.fen_input)
        self.fen_input.bind(text=app.on_fen_text, on_text_validate=app.on_fen_enter)
        box.add_widget(self.fen_input)
        copy_icon = TapImage(source=icon_path("copy", "green"), size_hint=(None, 1), width=dp(20), fit_mode="contain")
        copy_icon.bind(on_release=app.on_copy)
        box.add_widget(copy_icon)
        row.add_widget(box)
        row.add_widget(IconButton("Set", font_size=13, radius=10, size_hint_x=None, width=dp(50), on_release=app.on_fen_enter))
        row.add_widget(IconButton("Copy", font_size=13, radius=10, size_hint_x=None, width=dp(58), on_release=app.on_copy))
        outer.add_widget(row)

        # ── results card: preview of the straightened board + best moves ──
        self.card = RoundedBox(fill="#1a252d", border="#2c3b46", radius=12, size_hint_y=None, height=dp(98),
                               padding=dp(8), spacing=dp(8))
        thumb_col = BoxLayout(orientation="vertical", size_hint=(None, 1), width=0, spacing=dp(2), opacity=0)
        self.thumb = TapImage(fit_mode="contain", on_release=app.on_adjust)
        thumb_col.add_widget(self.thumb)
        thumb_col.add_widget(label("tap: adjust", size=9, color=TEXT_MUTED, halign="center",
                                   size_hint_y=None, height=dp(12)))
        self.thumb_col = thumb_col
        self.result = Label(text=PLACEHOLDER, markup=True, font_size=sp(15), color=rgba("#a3e635"), halign="left",
                            valign="middle")
        self.result.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.card.add_widget(thumb_col)
        self.card.add_widget(self.result)
        outer.add_widget(self.card)

        # ── status line ──
        self.status = label("", size=11, color=STATUS["info"], size_hint_y=None, height=dp(32),
                            shorten=True, max_lines=2)
        outer.add_widget(self.status)
        self.add_widget(outer)

        # (widget, normal height dp, compact height dp) for every fixed-height row
        self._rows = [(head, 38, 34), (hist, 34, 30), (self.turn_toggle.parent, 38, 34), (actions_row, 40, 36),
                      (self.depth_input.parent.parent, 38, 34), (self.fen_input.parent.parent, 40, 36),
                      (self.card, 98, 80), (self.status, 32, 28)]
        Window.bind(size=lambda *_: self.apply_density())
        self.apply_density()

    def apply_density(self):
        """Tighten the layout on small screens so the whole UI still fits without scrolling."""
        w_dp, h_dp = Window.width / dp(1), Window.height / dp(1)
        compact = h_dp < 640
        for widget, normal, small in self._rows:
            widget.height = dp(small if compact else normal)
        self.outer.spacing = dp(4 if compact else 6)
        self.result.font_size = sp(13 if compact else 15)
        narrow = w_dp < 345
        self.title.font_size = sp(15 if narrow else 17)
        for b in self.head_buttons:
            b.width = dp(76 if narrow else 88)
        self.depth_hint.opacity = 0 if narrow else 1          # no room for the hint on very narrow phones

    def set_result(self, text):
        self.result.text = text or PLACEHOLDER

    def show_thumb(self, tex):
        self.thumb.texture = tex
        self.thumb_col.opacity = 1
        self.thumb_col.width = dp(78)

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
            w, h = 400, 800                              # phone-like window on a desktop
            if os.environ.get("CHESS_WIN"):              # e.g. CHESS_WIN=360x640 (for layout tests)
                w, h = [int(v) for v in os.environ["CHESS_WIN"].split("x")]
            Window.size = (w, h)
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
        self._cache = {}                                  # (fen, depth) -> (lines, note)
        self.depth = self._load_depth()
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
        self.main.depth_input.text = str(self.depth)
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
            self.main.set_result("")
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
            self.main.set_result("")
            self.hist_push()
            self.on_analyse()

    def on_flip(self, *_):
        self.board.flipped = not self.board.flipped

    def on_reset(self, *_):
        self.board.flipped = False
        self.apply_fen(chess.STARTING_FEN)
        self.hist_reset()
        self.main.set_result("")
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

    # -- Stockfish depth setting ---------------------------------------------------
    def _settings_file(self):
        return os.path.join(self.user_data_dir, "settings.json")

    def _load_depth(self):
        try:
            with open(self._settings_file()) as f:
                return max(DEPTH_MIN, min(DEPTH_MAX, int(json.load(f)["depth"])))
        except Exception:                                        # first run / unreadable file
            return DEPTH_DEFAULT

    def _save_depth(self):
        try:
            with open(self._settings_file(), "w") as f:
                json.dump({"depth": self.depth}, f)
        except Exception:                                        # saving is a convenience only
            pass

    def set_depth(self, value, analyse=True):
        value = max(DEPTH_MIN, min(DEPTH_MAX, int(value)))
        changed = value != self.depth
        self.depth = value
        self.main.depth_input.text = str(value)
        if changed:
            self._save_depth()
            self.status("Stockfish depth set to %d%s." % (value, " - deeper is slower" if value > 18 else ""), "info")
            if analyse:
                self.on_analyse()

    def on_depth_step(self, delta):
        self.set_depth(self.depth + delta)

    def on_depth_entered(self, *_):
        text = self.main.depth_input.text.strip()
        try:
            self.set_depth(int(text))
        except ValueError:                                       # empty / not a number: put the old value back
            self.main.depth_input.text = str(self.depth)

    # -- analysis ---------------------------------------------------------------
    def on_analyse(self, *_):
        fen = self.board.get_fen()
        problems = cu.position_problems(fen)
        if problems:
            self.main.set_result("")
            self.status("Cannot analyse: %s." % ", ".join(problems), "warn")
            return
        self._analysis_id += 1
        my_id = self._analysis_id
        depth = self.depth
        if (fen, depth) in self._cache:                          # visited before: instant
            lines, note = self._cache[(fen, depth)]
            self._show_lines(my_id, fen, lines, note)
            return
        self.main.set_result("Analysing...")
        self.status("Running Stockfish...", "info")

        def work():
            stale = lambda: my_id != self._analysis_id            # a newer request replaces this one
            with self._engine_lock:
                if stale():
                    return
                lines, note = engine.analyse(fen, SF_MOVES, depth, should_stop=stale)
            if lines:
                self._cache[(fen, depth)] = (lines, note)
            self.ui(self._show_lines, my_id, fen, lines, note)

        threading.Thread(target=work, daemon=True).start()

    def _show_lines(self, my_id, fen, lines, note):
        if my_id != self._analysis_id:
            return                                               # a newer request replaced this one
        if not lines:
            self.main.set_result("No result.")
            self.status(note or "Analysis failed.", "err")
            return
        out, arrows = [], []
        for i, ln in enumerate(lines):
            col = ARROW_COLORS[i % len(ARROW_COLORS)]
            out.append("[color=%s][b]%d.[/b][/color]  [b]%s[/b]   %s" % (col, i + 1, ln.san, engine.format_score(ln)))
            arrows.append((chess.parse_square(ln.uci[:2]), chess.parse_square(ln.uci[2:4]), col))
        out.append("[size=%d][color=#8c94a3]%s, depth %d. Score: + = White better.[/color][/size]"
                   % (int(sp(10)), lines[0].source, lines[0].depth))
        self.main.set_result("\n".join(out))
        self.board.arrows = arrows
        self.status((note + " " if note else "") + "Best move: %s" % lines[0].san, "warn" if note else "ok")


if __name__ == "__main__":
    ChessAnalyzerApp().run()
