"""
crop_view.py - shows the picture with a square crop box the user can drag / resize.

The box is always square (a chessboard is).  An 8x8 grid is drawn inside it so the
user can see whether the squares line up before analysing.
"""
import numpy as np
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.widget import Widget
from PIL import Image

MIN_BOX = 64                 # px in image coordinates
TEX_MAX = 2048               # keep display texture within safe GPU limits


class CropView(Widget):
    box = ListProperty([0, 0, 100])            # x, y, size  (image pixel coordinates)
    texture = ObjectProperty(None, allownone=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.img_w = self.img_h = 1
        self._mode = None                      # None | 'move' | corner index 0..3
        self._anchor = None
        self._grab_off = (0, 0)
        self.bind(pos=self._redraw, size=self._redraw, box=self._redraw, texture=self._redraw)

    # -- public ----------------------------------------------------------------
    def set_image(self, rgb):
        """rgb: HxWx3 uint8 numpy array (the working image; crop coordinates refer to it)."""
        h, w = rgb.shape[:2]
        self.img_w, self.img_h = w, h
        disp = rgb
        if max(h, w) > TEX_MAX:
            k = TEX_MAX / max(h, w)
            disp = np.asarray(Image.fromarray(rgb).resize((int(w * k), int(h * k)), Image.BILINEAR))
        tex = Texture.create(size=(disp.shape[1], disp.shape[0]), colorfmt="rgb")
        tex.blit_buffer(np.ascontiguousarray(disp).tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        tex.flip_vertical()
        self.texture = tex
        s = int(min(w, h) * 0.8)
        self.box = [(w - s) // 2, (h - s) // 2, s]

    def get_box(self):
        x, y, s = [int(v) for v in self.box]
        return x, y, s

    # -- coordinate mapping ----------------------------------------------------
    def _layout(self):
        k = min(self.width / self.img_w, self.height / self.img_h)
        ox = self.x + (self.width - self.img_w * k) / 2
        oy = self.y + (self.height - self.img_h * k) / 2
        return k, ox, oy

    def _to_widget(self, ix, iy):
        k, ox, oy = self._layout()
        return ox + ix * k, oy + (self.img_h - iy) * k

    def _to_image(self, wx, wy):
        k, ox, oy = self._layout()
        return (wx - ox) / k, self.img_h - (wy - oy) / k

    def _corners(self):
        x, y, s = self.box
        return [(x, y), (x + s, y), (x, y + s), (x + s, y + s)]      # TL TR BL BR (image coords)

    # -- drawing ---------------------------------------------------------------
    def _redraw(self, *_):
        self.canvas.clear()
        if self.texture is None or self.width < 10:
            return
        k, ox, oy = self._layout()
        iw, ih = self.img_w * k, self.img_h * k
        bx, by, bs = self.box
        l, t = self._to_widget(bx, by)          # top-left in widget coords
        r, b = self._to_widget(bx + bs, by + bs)
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=self.texture, pos=(ox, oy), size=(iw, ih))
            Color(0, 0, 0, 0.55)                                    # dim everything outside the box
            Rectangle(pos=(ox, t), size=(iw, oy + ih - t))          # above
            Rectangle(pos=(ox, oy), size=(iw, b - oy))              # below
            Rectangle(pos=(ox, b), size=(l - ox, t - b))            # left
            Rectangle(pos=(r, b), size=(ox + iw - r, t - b))        # right
            for col, w in (((0, 0, 0, 0.55), 3), ((1, 1, 1, 0.8), 1)):    # 8x8 grid: dark+light = visible on any colour
                Color(*col)
                for i in range(1, 8):
                    gx = l + (r - l) * i / 8
                    gy = b + (t - b) * i / 8
                    Line(points=[gx, b, gx, t], width=w)
                    Line(points=[l, gy, r, gy], width=w)
            Color(1.0, 0.85, 0.1, 1)
            Line(rectangle=(l, b, r - l, t - b), width=dp(1.8))
            for cx, cy in self._corners():
                wx, wy = self._to_widget(cx, cy)
                rad = dp(11)
                Ellipse(pos=(wx - rad, wy - rad), size=(2 * rad, 2 * rad))

    # -- touch -----------------------------------------------------------------
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) or self.texture is None:
            return super().on_touch_down(touch)
        k, _, _ = self._layout()
        ix, iy = self._to_image(*touch.pos)
        hit = dp(34) / k
        corners = self._corners()
        best, best_d = None, hit
        for i, (cx, cy) in enumerate(corners):
            d = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = i, d
        x, y, s = self.box
        if best is not None:
            self._mode = best
            self._anchor = corners[3 - best]                        # opposite corner stays put
        elif x <= ix <= x + s and y <= iy <= y + s:
            self._mode = "move"
            self._grab_off = (ix - x, iy - y)
        else:
            self._mode = None
            return True
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self or self._mode is None:
            return super().on_touch_move(touch)
        ix, iy = self._to_image(*touch.pos)
        W, H = self.img_w, self.img_h
        if self._mode == "move":
            x, y, s = self.box
            nx = min(max(ix - self._grab_off[0], 0), W - s)
            ny = min(max(iy - self._grab_off[1], 0), H - s)
            self.box = [int(nx), int(ny), s]
        else:
            ax, ay = self._anchor
            right = self._mode in (1, 3)           # dragging a right-hand corner
            bottom = self._mode in (2, 3)          # dragging a bottom corner
            size = max(abs(ix - ax), abs(iy - ay))
            max_w = (W - ax) if right else ax
            max_h = (H - ay) if bottom else ay
            size = int(max(MIN_BOX, min(size, max_w, max_h)))
            nx = ax if right else ax - size
            ny = ay if bottom else ay - size
            self.box = [int(nx), int(ny), size]
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._mode = None
            return True
        return super().on_touch_up(touch)
