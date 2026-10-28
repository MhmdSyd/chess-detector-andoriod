"""Test helper: render chess positions to PNG with python-chess (lichess 'cburnett' pieces)."""
import io, random
import chess, chess.svg, cairosvg
from PIL import Image

THEMES = {
    "brown": {"square light": "#f0d9b5", "square dark": "#b58863"},
    "green": {"square light": "#eeeed2", "square dark": "#769656"},
    "blue":  {"square light": "#dee3e6", "square dark": "#8ca2ad"},
}

def random_position(n_moves, seed):
    rnd = random.Random(seed)
    b = chess.Board()
    for _ in range(n_moves):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            break
        b.push(rnd.choice(moves))
    return b

def render(board, px=640, theme="brown", flipped=False, lastmove=None):
    colors = dict(THEMES[theme])
    svg = chess.svg.board(board, size=px, coordinates=False, flipped=flipped,
                          colors=colors, lastmove=lastmove)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=px, output_height=px)
    return Image.open(io.BytesIO(png)).convert("RGB")
