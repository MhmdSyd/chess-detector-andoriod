# Chess Analyzer for Android

Pick a chess screenshot from the gallery (or take a photo), the app reads the position
(FEN), finds whose turn it is and shows Stockfish's best moves on the board.

## Build the APK (easiest: GitHub builds it for you)
1. Create a free GitHub account and a new repository.
2. Upload ALL files of this folder (keep the `.github` folder).
3. Open the **Actions** tab -> "Build APK" -> Run workflow. First build takes 30-60 min.
4. Download the artifact `ChessAnalyzer-apk`, unzip it, copy the .apk to your phone.
5. On the phone allow "Install unknown apps" for your file manager, open the APK.

## Build locally (Linux or Windows WSL)
    pip install buildozer cython
    sh scripts/fetch_stockfish.sh
    buildozer android debug        # APK appears in ./bin

## Test on a PC
    pip install -r requirements-desktop.txt
    python main.py [optional_screenshot.png]      # needs Stockfish on PATH

## Changes in 1.5
- The board now draws your own piece images (assets/pieces/wk.png ... bp.png) instead of font symbols.
  To change the look later, just replace those 12 PNG files (150x150, transparent background).

## Changes in 1.4
- New look (dark slate + green) following the design mock-up, icons by Lucide (ISC licence).
- Fully automatic: pick/take a photo -> board corners found -> straightened -> FEN read -> best moves.
  Board detection = OpenCV contours (like the desktop script) + grid fit + perspective warp; a NumPy
  detector is the fallback when OpenCV is not in the APK.
- 4-corner editor (tap the small board preview) with magnifier, for photos that need a manual fix.
- Back / forward arrows to step through positions; analysis results are cached per position.
- Only one Stockfish runs at a time; a newer request cancels the older one.
- Build: Actions -> Run workflow has an "OpenCV" checkbox (on by default). If the build fails because
  of OpenCV, run it again with the box unticked.

## Changes in 1.1
- Camera: fixed a crash when pressing Camera (nested Java class bug), the app now asks for the
  CAMERA permission and shows a message instead of closing if anything goes wrong.
- Tilted photos: the picture is automatically straightened before the board is cropped.

## Notes
- Needs a 64-bit phone with Android 10+. APK is ~50 MB (Stockfish inside).
- Not tested on a real phone yet: gallery/camera and running the bundled Stockfish.
  If Stockfish can't start, the app falls back to Lichess cloud analysis (needs internet).
- If the board is not found automatically, drag the yellow square over the board.
- Tests are in `tests/` (they need `chess cairosvg tensorflow chess-fen-detector`).
