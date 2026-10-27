"""
chess_utils.py - FEN helpers, board orientation and last-move detection.

Ported from the desktop script (flip_fen_vertical, guess_castling, build_full_fen,
detect_last_move).  OpenCV is replaced by NumPy and a few failure modes that used to
raise exceptions now return None so the app can ask the user instead.
"""
import numpy as np
import chess


# ──────────────────────────────────────────────────────────────────────────────
# FEN helpers (same behaviour as the desktop version)
# ──────────────────────────────────────────────────────────────────────────────

def flip_fen_vertical(placement):
    """Rotate a piece-placement string by 180 degrees (reverse ranks and files).
    Used when the picture shows the board from Black's side."""
    placement = placement.split(" ")[0]
    return "/".join(row[::-1] for row in placement.split("/")[::-1])


def guess_castling(board):
    """Guess castling rights from king/rook positions."""
    rights = ""
    wk, bk = chess.Piece(chess.KING, chess.WHITE), chess.Piece(chess.KING, chess.BLACK)
    wr, br = chess.Piece(chess.ROOK, chess.WHITE), chess.Piece(chess.ROOK, chess.BLACK)
    if board.piece_at(chess.E1) == wk and board.piece_at(chess.H1) == wr:
        rights += "K"
    if board.piece_at(chess.E1) == wk and board.piece_at(chess.A1) == wr:
        rights += "Q"
    if board.piece_at(chess.E8) == bk and board.piece_at(chess.H8) == br:
        rights += "k"
    if board.piece_at(chess.E8) == bk and board.piece_at(chess.A8) == br:
        rights += "q"
    return rights or "-"


def build_full_fen(placement, turn):
    """Piece placement + side to move -> complete FEN with guessed castling rights."""
    try:
        castling = guess_castling(chess.Board(f"{placement} {turn} KQkq - 0 1"))
    except Exception:
        castling = "-"
    return f"{placement} {turn} {castling} - 0 1"


def position_problems(fen):
    """Human-readable list of reasons why python-chess considers the FEN illegal."""
    try:
        board = chess.Board(fen)
    except Exception as e:                      # malformed FEN
        return [str(e)]
    st = board.status()
    names = {
        chess.STATUS_NO_WHITE_KING: "no white king",
        chess.STATUS_NO_BLACK_KING: "no black king",
        chess.STATUS_TOO_MANY_KINGS: "too many kings",
        chess.STATUS_TOO_MANY_WHITE_PAWNS: "more than 8 white pawns",
        chess.STATUS_TOO_MANY_BLACK_PAWNS: "more than 8 black pawns",
        chess.STATUS_PAWNS_ON_BACKRANK: "pawn on the first/last rank",
        chess.STATUS_TOO_MANY_WHITE_PIECES: "more than 16 white pieces",
        chess.STATUS_TOO_MANY_BLACK_PIECES: "more than 16 black pieces",
        chess.STATUS_OPPOSITE_CHECK: "the side not to move is in check",
        chess.STATUS_EMPTY: "empty board",
    }
    return [txt for flag, txt in names.items() if st & flag]


def fix_turn_if_needed(placement, turn):
    """If the guessed side to move leaves the *other* king in check (impossible),
    the guess was wrong - use the opposite side.  Returns (turn, changed)."""
    try:
        b = chess.Board(f"{placement} {turn} - - 0 1")
        if b.status() & chess.STATUS_OPPOSITE_CHECK:
            other = "b" if turn == "w" else "w"
            b2 = chess.Board(f"{placement} {other} - - 0 1")
            if not (b2.status() & chess.STATUS_OPPOSITE_CHECK):
                return other, True
    except Exception:
        pass
    return turn, False


# ──────────────────────────────────────────────────────────────────────────────
# Orientation
# ──────────────────────────────────────────────────────────────────────────────

def guess_orientation(placement):
    """'white_bottom' or 'black_bottom' for a placement read top-row-first from a picture.

    The desktop version only looked for upper-case letters on the bottom rank, which
    fails in endgames (empty back rank).  Here we compare where each side's pieces
    sit on average; the original rule is the tie-breaker."""
    rows = placement.split("/")
    white_rows, black_rows = [], []
    for r, row in enumerate(rows):
        for ch in row:
            if ch.isalpha():
                (white_rows if ch.isupper() else black_rows).append(r)
    if white_rows and black_rows:
        mw, mb = float(np.mean(white_rows)), float(np.mean(black_rows))
        if abs(mw - mb) > 0.25:
            return "white_bottom" if mw > mb else "black_bottom"
    bottom = rows[7] if len(rows) == 8 else ""
    return "white_bottom" if any(c.isupper() for c in bottom if c.isalpha()) else "black_bottom"


# ──────────────────────────────────────────────────────────────────────────────
# Last-move detection from highlighted squares
# ──────────────────────────────────────────────────────────────────────────────

# Same HSV window as the desktop script (OpenCV scale: H 0-179, S/V 0-255)
HL_LOWER = (80, 40, 40)
HL_UPPER = (110, 255, 255)
HL_RATIO = 0.15


def rgb_to_hsv_opencv(rgb):
    """uint8 HxWx3 RGB -> float32 H(0-179), S(0-255), V(0-255) like cv2.COLOR_BGR2HSV."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    v = a.max(axis=-1)
    mn = a.min(axis=-1)
    delta = v - mn
    s = np.where(v > 0, 255.0 * delta / np.maximum(v, 1e-6), 0.0)
    safe = np.maximum(delta, 1e-6)
    h = np.where(v == r, 60.0 * (g - b) / safe,
        np.where(v == g, 120.0 + 60.0 * (b - r) / safe,
                         240.0 + 60.0 * (r - g) / safe))
    h = np.where(delta == 0, 0.0, h)
    h = np.where(h < 0, h + 360.0, h) / 2.0
    return h, s, v


def highlighted_cells(board_rgb):
    """List of (row, col) squares (top-left = (0,0)) that carry the highlight colour."""
    h, s, v = rgb_to_hsv_opencv(board_rgb)
    mask = ((h >= HL_LOWER[0]) & (h <= HL_UPPER[0]) &
            (s >= HL_LOWER[1]) & (s <= HL_UPPER[1]) &
            (v >= HL_LOWER[2]) & (v <= HL_UPPER[2]))
    H, W = mask.shape
    sq_h, sq_w = H // 8, W // 8
    cells = []
    for row in range(8):
        for col in range(8):
            m = mask[row * sq_h:(row + 1) * sq_h, col * sq_w:(col + 1) * sq_w]
            if m.mean() > HL_RATIO:
                cells.append((row, col))
    return cells


def detect_last_move(board_rgb, placement, orientation=None):
    """Infer the last move from the two highlighted squares.

    Returns {'from','to','turn','orientation'} or None when the picture does not
    contain exactly two highlighted squares (the desktop version raised ValueError).
    `turn` is the side that has to move NEXT."""
    if orientation is None:
        orientation = guess_orientation(placement)
    cells = highlighted_cells(board_rgb)
    if len(cells) != 2:
        return None

    grid = []
    for row in placement.split("/"):
        grid.append("".join("." * int(c) if c.isdigit() else c for c in row))
    if len(grid) != 8 or any(len(r) != 8 for r in grid):
        return None

    files = "abcdefgh" if orientation == "white_bottom" else "hgfedcba"

    def to_alg(row, col):
        rank = 8 - row if orientation == "white_bottom" else row + 1
        return files[col] + str(rank)

    (r1, c1), (r2, c2) = cells
    p1, p2 = grid[r1][c1], grid[r2][c2]
    if (p1 == ".") == (p2 == "."):          # both empty or both occupied -> ambiguous
        return None
    if p1 != ".":
        (tr, tc), (fr, fc), piece = (r1, c1), (r2, c2), p1
    else:
        (tr, tc), (fr, fc), piece = (r2, c2), (r1, c1), p2
    return {
        "from": to_alg(fr, fc),
        "to": to_alg(tr, tc),
        "turn": "b" if piece.isupper() else "w",     # white just moved -> black next
        "orientation": orientation,
    }
