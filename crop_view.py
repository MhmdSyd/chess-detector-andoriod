"""
crop_view.py - picture viewer with an adjustable 4-corner board outline.

Used when the board could not be found automatically, or when the user wants to fine-tune
the detected corners.  Each corner can be dragged on its own, so a tilted or
perspective-distorted board can be outlined exactly; the 8x8 grid drawn inside follows
the perspective.  While a corner is dragged a magnifier shows the area under the finger.
"""
import numpy as np
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import ListProperty, NumericProperty, ObjectProperty
from kivy.uix.widget import Widget
from PIL import Image

import board_detect as bd

TEX_MAX = 2048               # keep the display texture within safe GPU limits


class CropView(Widget):
    quad = ListProperty([0.0] * 8)               # x0,y0,...,x3,y3 in image pixels (TL,TR,BR,BL)
    texture = ObjectProperty(None, allownone=True)
    dragging = NumericProperty(-1)               # index of the handle being dragged, -1 = none

    def __init__(self, **kw):
        super().__init__(**kw)
        self.img_w = self.img_h = 1
        self._tex_scale = 1.0
        self._disp = None                        # display copy of the picture (numpy)
        self._mode = None                        # None | 'move' | corner index
        self._grab = (0.0, 0.0)
        self.bind(pos=self._redraw, size=self._redraw, quad=self._redraw,
                  texture=self._redraw, dragging=self._redraw)

    # -- public ----------------------------------------------------------------
    def set_image(self, rgb, quad=None):
        h, w = rgb.shape[:2]
        self.img_w, self.img_h = w, h
        disp, self._tex_scale = rgb, 1.0
        if max(h, w) > TEX_MAX:
            self._tex_scale = TEX_MAX / max(h, w)
            disp = np.asarray(Image.fromarray(rgb).resize(
                (int(w * self._tex_scale), int(h * self._tex_scale)), Image.BILINEAR))
        tex = Texture.create(size=(disp.shape[1], disp.shape[0]), colorfmt="rgb")
        tex.blit_buffer(np.ascontiguousarray(disp).tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        tex.flip_vertical()
        self.texture = tex
        self._disp = np.ascontiguousarray(disp)
        self.set_quad(quad if quad is not None else bd.default_quad(rgb))

    def set_quad(self, q):
        self.quad = [float(v) for v in np.asarray(q, dtype=float).reshape(-1)]

    def get_quad(self):
        return np.array(self.quad, dtype=float).reshape(4, 2)

    # -- coordinate mapping ----------------------------------------------------
    def _layout(self):
        k = min(self.width / self.img_w, self.height / self.img_h)
        return k, self.x + (self.width - self.img_w * k) / 2, self.y + (self.height - self.img_h * k) / 2

    def _to_widget(self, ix, iy):
        k, ox, oy = self._layout()
        return ox + ix * k, oy + (self.img_h - iy) * k

    def _to_image(self, wx, wy):
        k, ox, oy = self._layout()
        return (wx - ox) / k, self.img_h - (wy - oy) / k

    # -- drawing ---------------------------------------------------------------
    def _redraw(self, *_):
        self.canvas.clear()
        if self.texture is None or self.width < 10:
            return
        k, ox, oy = self._layout()
        q = self.get_quad()
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=self.texture, pos=(ox, oy), size=(self.img_w * k, self.img_h * k))
            H = bd.unit_square_H(q[None])
            for col, width in (((0, 0, 0, 0.55), 3), ((1, 1, 1, 0.85), 1)):      # grid readable on any colour
                Color(*col)
                for i in range(1, 8):
                    for uv in ([[0, i / 8.0], [1, i / 8.0]], [[i / 8.0, 0], [i / 8.0, 1]]):
                        p = bd.map_uv(H, np.array(uv, dtype=float))[0]
                        pts = [c for x, y in p for c in self._to_widget(x, y)]
                        Line(points=pts, width=width)
            Color(1.0, 0.85, 0.1, 1)
            outline = [c for x, y in q for c in self._to_widget(x, y)]
            Line(points=outline, width=dp(2), close=True)
            for i, (x, y) in enumerate(q):
                wx, wy = self._to_widget(x, y)
                rad = dp(13) if i == self.dragging else dp(11)
                Color(0, 0, 0, 0.6)
                Ellipse(pos=(wx - rad - 1.5, wy - rad - 1.5), size=(2 * rad + 3, 2 * rad + 3))
                Color(1.0, 0.85, 0.1, 1)
                Ellipse(pos=(wx - rad, wy - rad), size=(2 * rad, 2 * rad))
            if self.dragging >= 0:
                self._draw_loupe(q[self.dragging])

    def _draw_loupe(self, pt):
        """Magnifier in the top corner away from the finger (built from the picture data)."""
        if self._disp is None:
            return
        size = dp(120)
        half = max(14.0, 0.07 * min(self.img_w, self.img_h))              # image px shown on each side
        s = self._tex_scale
        dh, dw = self._disp.shape[:2]
        span = max(8, int(2 * half * s))
        x0 = int(max(0, min(dw - span, pt[0] * s - span / 2)))
        y0 = int(max(0, min(dh - span, pt[1] * s - span / 2)))
        crop = self._disp[y0:y0 + span, x0:x0 + span]
        zoom = np.asarray(Image.fromarray(crop).resize((160, 160), Image.BILINEAR))
        tex = Texture.create(size=(160, 160), colorfmt="rgb")
        tex.blit_buffer(np.ascontiguousarray(zoom).tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        tex.flip_vertical()
        wx, _ = self._to_widget(*pt)
        lx = self.x + dp(6) if wx > self.center_x else self.right - size - dp(6)
        ly = self.top - size - dp(6)
        Color(0, 0, 0, 0.8)
        Rectangle(pos=(lx - 2, ly - 2), size=(size + 4, size + 4))
        Color(1, 1, 1, 1)
        Rectangle(texture=tex, pos=(lx, ly), size=(size, size))
        cx = lx + (pt[0] * s - x0) / span * size                          # where the corner is inside the loupe
        cy = ly + size - (pt[1] * s - y0) / span * size
        Color(1.0, 0.2, 0.2, 0.95)
        Line(points=[cx - dp(9), cy, cx + dp(9), cy], width=1.3)
        Line(points=[cx, cy - dp(9), cx, cy + dp(9)], width=1.3)

    # -- touch -----------------------------------------------------------------
    @staticmethod
    def _inside(q, p):
        sign = None
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if abs(c) < 1e-9:
                continue
            if sign is None:
                sign = c > 0
            elif (c > 0) != sign:
                return False
        return True

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) or self.texture is None:
            return super().on_touch_down(touch)
        k, _, _ = self._layout()
        ix, iy = self._to_image(*touch.pos)
        q = self.get_quad()
        reach = dp(40) / k
        d = np.hypot(q[:, 0] - ix, q[:, 1] - iy)
        i = int(d.argmin())
        if d[i] < reach:
            self._mode = i
            self.dragging = i
        elif self._inside(q, (ix, iy)):
            self._mode = "move"
            self._grab = (ix, iy)
        else:
            self._mode = None
            return True
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self or self._mode is None:
            return super().on_touch_move(touch)
        ix, iy = self._to_image(*touch.pos)
        q = self.get_quad()
        if self._mode == "move":
            dx, dy = ix - self._grab[0], iy - self._grab[1]
            dx = min(max(dx, -q[:, 0].min()), self.img_w - 1 - q[:, 0].max())
            dy = min(max(dy, -q[:, 1].min()), self.img_h - 1 - q[:, 1].max())
            q = q + (dx, dy)
            self._grab = (self._grab[0] + dx, self._grab[1] + dy)
        else:
            q[self._mode] = (min(max(ix, 0), self.img_w - 1), min(max(iy, 0), self.img_h - 1))
        self.set_quad(q)
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._mode = None
            self.dragging = -1
            return True
        return super().on_touch_up(touch)
