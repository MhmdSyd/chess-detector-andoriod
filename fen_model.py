"""
fen_model.py - board image -> FEN piece placement, using NumPy only.

This is a port of the `chess-fen-detector` package used by the desktop app.
The original needs TensorFlow + scikit-image, which cannot run on Android, so:

  * the Keras network (3 x Conv3x3+ReLU -> Dense128+ReLU -> Dense13) was exported
    to `assets/piece_model.npz` (see tools/export_weights.py) and is evaluated
    here with plain NumPy;
  * `resize_like_skimage()` reproduces skimage.transform.resize(anti_aliasing=True)
    so the network sees the same pixels it saw in the desktop version.
"""
import numpy as np

PIECE_SYMBOLS = "prbnkqPRBNKQ"      # class index 0..11, index 12 == empty square
BOARD_PX = 200                      # board is resized to 200x200 -> 64 tiles of 25x25
TILE_PX = BOARD_PX // 8


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(sigma):
    radius = int(4.0 * sigma + 0.5)                 # scipy: truncate=4.0
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), radius


def _blur_axis(a, sigma, axis):
    """Separable Gaussian along one axis, zero padding (== scipy mode='constant')."""
    if sigma <= 0:
        return a
    k, r = _gaussian_kernel(sigma)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="constant")
    n = a.shape[axis]
    out = np.zeros_like(a)
    for j, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(j, j + n)
        out += w * p[tuple(sl)]
    return out


def _linear_resample_axis(a, out_n, axis):
    """Bilinear resampling with pixel-centre alignment (scipy zoom grid_mode=True)."""
    in_n = a.shape[axis]
    f = in_n / out_n
    x = (np.arange(out_n, dtype=np.float32) + 0.5) * f - 0.5
    x = np.clip(x, 0, in_n - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, in_n - 1)
    w = (x - i0).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = out_n
    w = w.reshape(shape)
    return a.take(i0, axis=axis) * (1 - w) + a.take(i1, axis=axis) * w


def resize_like_skimage(img_u8, out_h=BOARD_PX, out_w=BOARD_PX):
    """uint8 HxWx3 -> float32 out_h x out_w x 3 in [0,1] (anti-aliased)."""
    a = img_u8.astype(np.float32) / 255.0
    h, w = a.shape[:2]
    a = _blur_axis(a, max(0.0, (h / out_h - 1) / 2), 0)
    a = _blur_axis(a, max(0.0, (w / out_w - 1) / 2), 1)
    a = _linear_resample_axis(a, out_h, 0)
    a = _linear_resample_axis(a, out_w, 1)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def split_tiles(board_f32):
    """200x200x3 -> (64, 25, 25, 3), row-major (a8..h8, a7..h7, ...)."""
    t = board_f32.reshape(8, TILE_PX, 8, TILE_PX, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(t.reshape(64, TILE_PX, TILE_PX, 3))


# ──────────────────────────────────────────────────────────────────────────────
# The network
# ──────────────────────────────────────────────────────────────────────────────

class PieceClassifier:
    def __init__(self, weights_path):
        z = np.load(weights_path)
        self.convs = []
        for i in range(3):
            w = z[f"w{i}"]                                   # (3,3,Cin,Cout)
            self.convs.append((w.reshape(-1, w.shape[-1]), z[f"b{i}"]))
        self.w3, self.b3 = z["w3"], z["b3"]
        self.w4, self.b4 = z["w4"], z["b4"]

    @staticmethod
    def _conv3x3_relu(x, w2d, b):
        n, h, w, c = x.shape
        ho, wo = h - 2, w - 2
        cols = np.concatenate(
            [x[:, dy:dy + ho, dx:dx + wo, :] for dy in range(3) for dx in range(3)],
            axis=-1,
        )                                                     # (n,ho,wo,9c) order (dy,dx,c)
        out = cols.reshape(-1, 9 * c) @ w2d + b
        np.maximum(out, 0, out=out)
        return out.reshape(n, ho, wo, -1)

    def predict_classes(self, tiles, batch=16):
        """tiles (64,25,25,3) float32 -> int array (64,) with values 0..12."""
        result = []
        for s in range(0, len(tiles), batch):
            x = tiles[s:s + batch]
            for w2d, b in self.convs:
                x = self._conv3x3_relu(x, w2d, b)
            x = x.reshape(len(x), -1)
            x = np.maximum(x @ self.w3 + self.b3, 0)
            logits = x @ self.w4 + self.b4
            result.append(logits.argmax(axis=1))
        return np.concatenate(result)


def classes_to_placement(classes):
    """64 class ids -> 'rnbqkbnr/pppppppp/8/...' piece-placement string."""
    rows = []
    for r in range(8):
        row, empty = "", 0
        for c in range(8):
            k = int(classes[r * 8 + c])
            if k == 12:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += PIECE_SYMBOLS[k]
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


class FenDetector:
    """Drop-in replacement for chess_fen_detector.ChessFENDetector."""

    def __init__(self, weights_path):
        self.net = PieceClassifier(weights_path)

    def predict_placement(self, board_rgb_u8):
        """Cropped board (HxWx3 uint8 RGB) -> FEN piece placement (as seen on screen,
        i.e. top row of the picture first)."""
        board = resize_like_skimage(board_rgb_u8)
        classes = self.net.predict_classes(split_tiles(board))
        return classes_to_placement(classes)
