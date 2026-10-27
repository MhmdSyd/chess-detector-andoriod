"""
board_widget.py - the interactive chessboard (port of the Tk BoardCanvas).

Tap one of your pieces, then a highlighted square, to play a move.  Best-move arrows
and the last move are drawn on top.  Pieces are drawn from the bundled DejaVu Sans font
(Android's own fonts do not reliably contain chess glyphs).
"""
import math

import chess
from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Ellipse, Line, Rectangle, Triangle
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty
from kivy.uix.widget import Widget

LIGHT_SQ = "#ffffdd"
DARK_SQ = "#86a666"
SELECT_SQ = "#4fa28e"
LAST_MOVE = (1.0, 0.92, 0.2, 0.40)
MOVE_DOT = (0.15, 0.35, 0.35, 0.55)

# filled glyphs for both colours; colour + outline tell the sides apart
GLYPH = {"p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A"}


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a


class BoardWidget(Widget):
    flipped = BooleanProperty(False)
    arrows = ListProperty([])            # [(from_sq, to_sq, '#rrggbb'), ...]
    last_move = ListProperty([])         # [from_sq, to_sq] or []

    def __init__(self, font_path, **kw):
        super().__init__(**kw)
        self.font_path = font_path
        self.board = chess.Board()
        self.selected = None
        self.targets = []
        self._tex_cache = {}
        self._tex_size = None
        self._callbacks = []
        self.bind(pos=self.redraw, size=self.redraw, flipped=self.redraw,
                  arrows=self.redraw, last_move=self.redraw)

    # -- tiny event system (kept simple instead of EventDispatcher registration) ----
    def on_user_move(self, fn):
        self._callbacks.append(fn)

    # -- public API ------------------------------------------------------------
    def set_fen(self, fen):
        try:
            self.board = chess.Board(fen)
        except Exception:
            return False
        self.selected, self.targets = None, []
        self.arrows, self.last_move = [], []
        self.redraw()
        return True

    def get_fen(self):
        return self.board.fen()

    # -- geometry --------------------------------------------------------------
    def _metrics(self):
        s = min(self.width, self.height) / 8.0
        ox = self.x + (self.width - 8 * s) / 2
        oy = self.y + (self.height - 8 * s) / 2
        return s, ox, oy

    def _sq_rect(self, sq):
        s, ox, oy = self._metrics()
        col, rank = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            col, rank = 7 - col, 7 - rank
        return ox + col * s, oy + rank * s, s            # kivy y grows upward

    def _xy_to_sq(self, x, y):
        s, ox, oy = self._metrics()
        col, rank = int((x - ox) // s), int((y - oy) // s)
        if not (0 <= col < 8 and 0 <= rank < 8):
            return None
        if self.flipped:
            col, rank = 7 - col, 7 - rank
        return chess.square(col, rank)

    # -- textures ---------------------------------------------------------------
    def _piece_texture(self, symbol, size_px):
        if self._tex_size != size_px:
            self._tex_cache.clear()
            self._tex_size = size_px
        if symbol not in self._tex_cache:
            white = symbol.isupper()
            lbl = CoreLabel(
                text=GLYPH[symbol.lower()], font_size=size_px, font_name=self.font_path,
                color=(1, 1, 1, 1) if white else (0.08, 0.08, 0.08, 1),
                outline_width=max(2, int(size_px / 22)),
                outline_color=(0.05, 0.05, 0.05) if white else (0.9, 0.9, 0.9),
            )
            lbl.refresh()
            self._tex_cache[symbol] = lbl.texture
        return self._tex_cache[symbol]

    def _text_texture(self, text, size_px, color):
        key = ("t", text, size_px, color)
        if key not in self._tex_cache:
            lbl = CoreLabel(text=text, font_size=size_px, bold=True, color=color)
            lbl.refresh()
            self._tex_cache[key] = lbl.texture
        return self._tex_cache[key]

    # -- drawing ---------------------------------------------------------------
    def redraw(self, *_):
        if self.width < 10:
            return
        s, ox, oy = self._metrics()
        self.canvas.clear()
        with self.canvas:
            for sq in chess.SQUARES:
                x, y, _ = self._sq_rect(sq)
                light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
                Color(*hex_rgba(LIGHT_SQ if light else DARK_SQ))
                Rectangle(pos=(x, y), size=(s, s))
                if sq in self.last_move:
                    Color(*LAST_MOVE)
                    Rectangle(pos=(x, y), size=(s, s))
                if sq == self.selected:
                    Color(*hex_rgba(SELECT_SQ, 0.85))
                    Rectangle(pos=(x, y), size=(s, s))

            # coordinates
            fs = max(8, int(s * 0.17))
            files, ranks = "abcdefgh", "12345678"
            for i in range(8):
                f = files[i] if not self.flipped else files[7 - i]
                r = ranks[i] if not self.flipped else ranks[7 - i]
                col = (0.35, 0.35, 0.3, 1)
                t = self._text_texture(f, fs, col)
                Color(1, 1, 1, 1)
                Rectangle(texture=t, size=t.size, pos=(ox + i * s + s - t.width - 2, oy + 2))
                t = self._text_texture(r, fs, col)
                Rectangle(texture=t, size=t.size, pos=(ox + 2, oy + i * s + s - t.height - 1))

            # legal-move markers
            for sq in self.targets:
                x, y, _ = self._sq_rect(sq)
                Color(*MOVE_DOT)
                if self.board.piece_at(sq):
                    Line(circle=(x + s / 2, y + s / 2, s * 0.44), width=max(2, s / 18))
                else:
                    r = s * 0.16
                    Ellipse(pos=(x + s / 2 - r, y + s / 2 - r), size=(2 * r, 2 * r))

            # pieces
            px = int(s * 0.82)
            for sq, piece in self.board.piece_map().items():
                x, y, _ = self._sq_rect(sq)
                tex = self._piece_texture(piece.symbol(), px)
                Color(1, 1, 1, 1)
                Rectangle(texture=tex, size=tex.size,
                          pos=(x + (s - tex.width) / 2, y + (s - tex.height) / 2))

            # arrows
            for fr, to, colr in self.arrows:
                self._draw_arrow(fr, to, colr, s)

    def _draw_arrow(self, fr, to, colr, s):
        x1, y1, _ = self._sq_rect(fr)
        x2, y2, _ = self._sq_rect(to)
        fx, fy, tx, ty = x1 + s / 2, y1 + s / 2, x2 + s / 2, y2 + s / 2
        dx, dy = tx - fx, ty - fy
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        head = s * 0.42
        bx, by = tx - ux * head * 0.9, ty - uy * head * 0.9          # base of the head
        Color(*hex_rgba(colr, 0.85))
        Line(points=[fx, fy, bx, by], width=max(dp(3), s / 9), cap="round")
        nx, ny = -uy, ux
        w = s * 0.24
        Triangle(points=[tx - ux * s * 0.05, ty - uy * s * 0.05, bx + nx * w, by + ny * w, bx - nx * w, by - ny * w])

    # -- interaction -----------------------------------------------------------
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        sq = self._xy_to_sq(*touch.pos)
        if sq is None:
            return True
        if self.selected is None:
            self._select(sq)
        else:
            move = self._find_move(self.selected, sq)
            if move:
                self.board.push(move)
                self.selected, self.targets = None, []
                self.arrows = []
                self.last_move = [move.from_square, move.to_square]
                self.redraw()
                for fn in self._callbacks:
                    fn()
            elif not self._select(sq):
                self.selected, self.targets = None, []
                self.redraw()
        return True

    def _select(self, sq):
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn:
            self.selected = sq
            self.targets = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
            self.arrows = []
            self.redraw()
            return True
        return False

    def _find_move(self, fr, to):
        for m in self.board.legal_moves:
            if m.from_square == fr and m.to_square == to:
                return chess.Move(fr, to, chess.QUEEN) if m.promotion else m     # auto-queen
        return None
