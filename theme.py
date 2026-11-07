"""
theme.py - the look of the app (dark slate + green), matching the design mock-up.

Contains the colour palette and a few small custom widgets:
    IconButton   rounded dark button with an optional icon (left or right of the text)
    PillToggle   "White | Black" segmented switch with a green selected pill
    RoundedBox   container with a rounded, bordered background
    set_gradient vertical gradient behind a widget
"""
import os

from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(HERE, "assets", "icons")


def rgba(h, a=1.0):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a


def icon_path(name, variant="light"):
    return os.path.join(ICON_DIR, "%s_%s.png" % (name, variant))


# ── palette (sampled from the mock-up) ────────────────────────────────────────
BG_TOP, BG_BOTTOM = "#232d37", "#2a343e"
TEXT, TEXT_MUTED, TEXT_DIM = "#d3d9e2", "#8c94a3", "#5d6672"
BTN_FILL, BTN_FILL_DOWN, BTN_BORDER = "#222f39", "#18222a", "#3b4856"
GLOW_BORDER = "#355f45"
PANEL_FILL, PANEL_BORDER = "#141d22", "#2d6a3d"
TOGGLE_TRACK, GREEN, GREEN_DARK = "#1b262c", "#468644", "#3a7439"
STATUS = {"info": "#969daa", "ok": "#8fd07a", "warn": "#e6b955", "err": "#ef7b7b"}
BOARD_LIGHT, BOARD_DARK = "#e6e5c7", "#739151"


def set_gradient(widget, top=BG_TOP, bottom=BG_BOTTOM):
    """Vertical gradient behind `widget` (a 1x2 texture stretched with linear filtering)."""
    tex = Texture.create(size=(1, 2), colorfmt="rgb")
    t, b = rgba(top), rgba(bottom)
    tex.blit_buffer(bytes([int(b[0] * 255), int(b[1] * 255), int(b[2] * 255),
                           int(t[0] * 255), int(t[1] * 255), int(t[2] * 255)]), colorfmt="rgb", bufferfmt="ubyte")
    tex.mag_filter = "linear"
    with widget.canvas.before:
        Color(1, 1, 1, 1)
        rect = Rectangle(texture=tex, pos=widget.pos, size=widget.size)
    widget.bind(pos=lambda w, v: setattr(rect, "pos", v), size=lambda w, v: setattr(rect, "size", v))


class RoundedBox(BoxLayout):
    """BoxLayout with a rounded, optionally bordered background."""

    def __init__(self, fill=PANEL_FILL, border=None, radius=12, border_w=1.2, **kw):
        super().__init__(**kw)
        self._r = dp(radius)
        with self.canvas.before:
            self._fill_c = Color(*rgba(fill))
            self._fill = RoundedRectangle(pos=self.pos, size=self.size, radius=[self._r])
            self._border_c = Color(*(rgba(border) if border else (0, 0, 0, 0)))
            self._border = Line(rounded_rectangle=(self.x, self.y, self.width, self.height, self._r), width=dp(border_w))
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *_):
        self._fill.pos, self._fill.size = self.pos, self.size
        self._border.rounded_rectangle = (self.x, self.y, self.width, self.height, self._r)

    def set_colors(self, fill=None, border=None):
        if fill:
            self._fill_c.rgba = rgba(fill)
        if border:
            self._border_c.rgba = rgba(border)


class IconButton(ButtonBehavior, AnchorLayout):
    """Rounded dark button: [icon] text   or   text [icon]   or icon only."""
    enabled = BooleanProperty(True)

    def __init__(self, text="", icon=None, icon_right=False, glow=False, icon_variant="light",
                 font_size=14, radius=12, **kw):
        super().__init__(anchor_x="center", anchor_y="center", **kw)
        self._icon_name, self._variant, self._glow = icon, icon_variant, glow
        self._r = dp(radius)
        with self.canvas.before:
            self._fill_c = Color(*rgba(BTN_FILL))
            self._fill = RoundedRectangle(pos=self.pos, size=self.size, radius=[self._r])
            self._border_c = Color(*rgba(GLOW_BORDER if glow else BTN_BORDER))
            self._border = Line(rounded_rectangle=(self.x, self.y, self.width, self.height, self._r), width=dp(1.1))
        self.bind(pos=self._sync, size=self._sync, state=self._restyle, enabled=self._restyle)

        self._inner = BoxLayout(size_hint=(None, None), spacing=dp(8), height=dp(24))
        self._img = Image(size_hint=(None, 1), width=dp(18), fit_mode="contain") if icon else None
        self._lbl = Label(text=text, size_hint=(None, 1), font_size=sp(font_size), bold=True,
                          color=rgba(TEXT)) if text else None
        parts = [w for w in ((self._lbl, self._img) if icon_right else (self._img, self._lbl)) if w is not None]
        for w in parts:
            self._inner.add_widget(w)
        if self._lbl:
            self._lbl.bind(texture_size=self._resize)
        self._resize()
        self.add_widget(self._inner)
        self._restyle()

    def _resize(self, *_):
        w = 0
        if self._lbl:
            self._lbl.width = self._lbl.texture_size[0]
            w += self._lbl.width
        if self._img:
            w += self._img.width
        if self._lbl and self._img:
            w += self._inner.spacing
        self._inner.width = w

    def _sync(self, *_):
        self._fill.pos, self._fill.size = self.pos, self.size
        self._border.rounded_rectangle = (self.x, self.y, self.width, self.height, self._r)

    def _restyle(self, *_):
        self._fill_c.rgba = rgba(BTN_FILL_DOWN if (self.state == "down" and self.enabled) else BTN_FILL)
        self._border_c.rgba = rgba(GLOW_BORDER if self._glow and self.enabled else BTN_BORDER)
        if self._img:
            self._img.source = icon_path(self._icon_name, self._variant if self.enabled else "dim")
        if self._lbl:
            self._lbl.color = rgba(TEXT if self.enabled else TEXT_DIM)
        self.opacity = 1.0 if self.enabled else 0.75

    def set_icon(self, name):
        self._icon_name = name
        self._restyle()

    def on_touch_down(self, touch):
        if not self.enabled:
            return self.collide_point(*touch.pos)          # swallow the tap, do nothing
        return super().on_touch_down(touch)


class PillToggle(Widget):
    """Two-option segmented switch.  `value` is the option's value ('w' / 'b')."""
    value = StringProperty("w")

    def __init__(self, options=(("White", "w"), ("Black", "b")), on_change=None, **kw):
        super().__init__(**kw)
        self.options = list(options)
        self._on_change = on_change
        with self.canvas.before:
            Color(*rgba(TOGGLE_TRACK))
            self._track = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(20)])
            Color(*rgba("#2a3640"))
            self._track_line = Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(20)), width=dp(1))
            Color(*rgba(GREEN))
            self._pill = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(17)])
        self._labels = []
        for text, _v in self.options:
            lbl = Label(text=text, font_size=sp(14), bold=True)
            self._labels.append(lbl)
            self.add_widget(lbl)
        self.bind(pos=self._layout, size=self._layout, value=self._layout)
        self._layout()

    def _idx(self):
        return [v for _t, v in self.options].index(self.value)

    def _layout(self, *_):
        n, pad = len(self.options), dp(4)
        seg = (self.width - 2 * pad) / n
        self._track.pos, self._track.size = self.pos, self.size
        r = self.height / 2.0
        self._track.radius = [r]
        self._track_line.rounded_rectangle = (self.x, self.y, self.width, self.height, r)
        i = self._idx()
        self._pill.pos = (self.x + pad + i * seg, self.y + pad)
        self._pill.size = (seg, self.height - 2 * pad)
        self._pill.radius = [(self.height - 2 * pad) / 2]
        for k, lbl in enumerate(self._labels):
            lbl.pos = (self.x + pad + k * seg, self.y)
            lbl.size = (seg, self.height)
            lbl.color = rgba("#eaffe8") if k == i else rgba(TEXT_DIM)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        n, pad = len(self.options), dp(4)
        seg = (self.width - 2 * pad) / n
        k = int(min(max((touch.x - self.x - pad) // seg, 0), n - 1))
        new = self.options[k][1]
        if new != self.value:
            self.value = new
        if self._on_change:
            self._on_change(new)                              # also fires when re-tapping the same side
        return True

    def set_value(self, v):
        self.value = v


def label(text="", size=14, color=TEXT, bold=False, halign="left", **kw):
    lb = Label(text=text, font_size=sp(size), color=rgba(color), bold=bold,
               halign=halign, valign="middle", **kw)
    lb.bind(size=lambda w, s: setattr(w, "text_size", s))
    return lb
