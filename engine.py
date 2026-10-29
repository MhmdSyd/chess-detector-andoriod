"""
engine.py - best-move analysis.

1. Local Stockfish (preferred, works offline)
     * Android: the arm64 binary is shipped inside the APK as `libstockfish.so`
       (Android only allows executing files from the app's native-library folder).
     * Desktop (for testing): `stockfish` on PATH, STOCKFISH_PATH env var, or the
       locations the original script searched.
2. Lichess cloud evaluation (needs internet) as an automatic fallback.

The UCI protocol is spoken directly over a subprocess pipe (no asyncio), which is far
more predictable on Android than python-chess's engine module.
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import namedtuple

import chess

# score is always from WHITE's point of view: cp in centipawns, or mate = moves to mate
Line = namedtuple("Line", "uci san cp mate depth source")

DESKTOP_CANDIDATES = [
    "/usr/bin/stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    r"stockfish\stockfish-windows-x86-64.exe",
]


def _android_native_dir():
    try:
        from jnius import autoclass
        activity = autoclass("org.kivy.android.PythonActivity").mActivity
        return str(activity.getApplicationInfo().nativeLibraryDir)
    except Exception:
        return None


def find_stockfish():
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.isfile(env):
        return env
    native = _android_native_dir()
    if native:
        p = os.path.join(native, "libstockfish.so")
        if os.path.isfile(p):
            return p
        return None                      # on Android never fall back to desktop paths
    found = shutil.which("stockfish")
    if found:
        return found
    for p in DESKTOP_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Local Stockfish over UCI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_info(line):
    """'info depth 15 multipv 2 score cp 34 pv e2e4 e7e5' -> dict or None."""
    t = line.split()
    if "score" not in t or "pv" not in t or "multipv" not in t:
        return None
    if "lowerbound" in t or "upperbound" in t:
        return None
    try:
        i = t.index("score")
        kind, val = t[i + 1], int(t[i + 2])
        return {
            "depth": int(t[t.index("depth") + 1]),
            "multipv": int(t[t.index("multipv") + 1]),
            "cp": val if kind == "cp" else None,
            "mate": val if kind == "mate" else None,
            "pv": t[t.index("pv") + 1:],
        }
    except (ValueError, IndexError):
        return None


def analyse_local(fen, path, num_moves=3, depth=15, timeout=60, should_stop=None):
    """Returns a list of Lines, or None if `should_stop()` became true (request cancelled)."""
    board = chess.Board(fen)
    proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    q = queue.Queue()

    def reader():
        for ln in proc.stdout:
            q.put(ln.strip())
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(token, limit):
        end = time.time() + limit
        while time.time() < end:
            try:
                ln = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if ln is None:
                raise RuntimeError("engine closed unexpectedly")
            if ln.startswith(token):
                return ln
        raise TimeoutError(f"engine did not answer '{token}'")

    best = {}
    cancelled = False
    try:
        send("uci")
        wait_for("uciok", 15)
        send(f"setoption name MultiPV value {num_moves}")
        send(f"setoption name Threads value {max(1, min(4, os.cpu_count() or 1))}")
        send("setoption name Hash value 64")
        send("isready")
        wait_for("readyok", 15)
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        end = time.time() + timeout
        while True:
            if should_stop and should_stop():
                cancelled = True
                break
            if time.time() > end:
                send("stop")
                end = time.time() + 5
            try:
                ln = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if ln is None:
                break
            if ln.startswith("info"):
                info = _parse_info(ln)
                if info and (info["multipv"] not in best or info["depth"] >= best[info["multipv"]]["depth"]):
                    best[info["multipv"]] = info
            elif ln.startswith("bestmove"):
                break
    finally:
        try:
            send("quit")
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    if cancelled:
        return None
    sign = 1 if board.turn == chess.WHITE else -1       # engine reports side-to-move POV
    lines = []
    for k in sorted(best):
        info = best[k]
        if not info["pv"]:
            continue
        uci = info["pv"][0]
        try:
            san = board.san(chess.Move.from_uci(uci))
        except Exception:
            san = uci
        lines.append(Line(uci, san,
                          None if info["cp"] is None else sign * info["cp"],
                          None if info["mate"] is None else sign * info["mate"],
                          info["depth"], "Stockfish"))
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Online fallback: Lichess cloud evaluation
# ──────────────────────────────────────────────────────────────────────────────

def analyse_online(fen, num_moves=3, timeout=15):
    board = chess.Board(fen)
    url = "https://lichess.org/api/cloud-eval?" + urllib.parse.urlencode({"fen": fen, "multiPv": num_moves})
    req = urllib.request.Request(url, headers={"User-Agent": "ChessAnalyzerAndroid"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    sign = 1 if board.turn == chess.WHITE else -1
    lines = []
    for pv in data.get("pvs", [])[:num_moves]:
        uci = pv["moves"].split()[0]
        try:
            san = board.san(chess.Move.from_uci(uci))
        except Exception:
            san = uci
        lines.append(Line(uci, san,
                          pv.get("cp"), pv.get("mate"),      # cloud-eval is already White POV
                          data.get("depth", 0), "Lichess cloud"))
    return lines


def analyse(fen, num_moves=3, depth=15, should_stop=None):
    """Returns (lines, message).  `message` explains fallbacks / failures.
    `should_stop` is polled while the engine thinks; when it returns True the work is
    abandoned and ([], 'cancelled') is returned."""
    note = ""
    path = find_stockfish()
    if path:
        try:
            lines = analyse_local(fen, path, num_moves, depth, should_stop=should_stop)
            if lines is None:
                return [], "cancelled"
            if lines:
                return lines, ""
            note = "Stockfish returned no moves."
        except Exception as e:
            note = f"Stockfish failed ({e})."
    else:
        note = "Stockfish binary not found."
    try:
        lines = analyse_online(fen, num_moves)
        if lines:
            return lines, note + " Used Lichess cloud analysis instead."
        note += " Lichess had no result."
    except Exception as e:
        note += f" Online fallback failed ({e.__class__.__name__})."
    return [], note


def format_score(line):
    if line.mate is not None:
        return f"M{line.mate}" if line.mate > 0 else f"-M{abs(line.mate)}"
    if line.cp is None:
        return "?"
    return f"{line.cp / 100:+.2f}"
