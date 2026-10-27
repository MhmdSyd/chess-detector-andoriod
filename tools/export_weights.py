"""One-off helper (desktop only): export the Keras model weights from the
`chess-fen-detector` pip package into a plain .npz file so the Android app can
run the network with NumPy only (no TensorFlow on the phone).

    pip install tensorflow chess-fen-detector
    python tools/export_weights.py
"""
import os
import numpy as np
import chess_fen_detector
from tensorflow.keras.models import load_model

pkg = os.path.dirname(chess_fen_detector.__file__)
model = load_model(os.path.join(pkg, "models", "chess_best_model_mini.keras"))

arrays = {}
layers = [l for l in model.layers if l.get_weights()]
for i, layer in enumerate(layers):
    w, b = layer.get_weights()
    arrays[f"w{i}"] = w.astype(np.float32)
    arrays[f"b{i}"] = b.astype(np.float32)

out = os.path.join(os.path.dirname(__file__), "..", "assets", "piece_model.npz")
np.savez_compressed(out, **arrays)
print("saved", os.path.abspath(out), {k: v.shape for k, v in arrays.items()})
