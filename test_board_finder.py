#!/bin/sh
# For local builds on Linux/WSL: put the Android Stockfish binary where buildozer expects it.
set -e
mkdir -p libs/arm64-v8a
curl -L -o /tmp/sf.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-android-armv8.tar
tar xf /tmp/sf.tar -C /tmp stockfish/stockfish-android-armv8
cp /tmp/stockfish/stockfish-android-armv8 libs/arm64-v8a/libstockfish.so
echo "Stockfish ready in libs/arm64-v8a/"
