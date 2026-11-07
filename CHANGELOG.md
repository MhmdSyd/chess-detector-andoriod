# Changelog

All notable changes to **Chess Analyzer for Android**.

## 1.6
- Fixed one-screen layout with no scrolling. The board uses the free space; small phones get a compact layout automatically.
- "Chess Analyzer" title stays on one line; header and row buttons are shorter; the FEN box is a single line.
- New **Stockfish depth** control (1-30) with - / + buttons and a number field. The value is remembered between launches.

## 1.5
- The board now draws the project's own piece images (`assets/pieces`) instead of font symbols.

## 1.4
- New dark design with a green accent, icons from Lucide.
- Fully automatic flow: pick or take a picture, the app finds and straightens the board, reads the position and analyses it.
- Board detection with OpenCV contours plus a grid fit and perspective correction (NumPy fallback when OpenCV is not in the APK).
- 4-corner editor with a magnifier (tap the small board preview).
- Back / forward arrows to step through positions; analysis results are remembered per position.
- Only one Stockfish runs at a time; a newer request cancels an older one.
- The workflow has an "Include OpenCV" switch.

## 1.3
- App icon, adaptive icon, splash screen and logo.

## 1.2
- Build fix: test files are no longer packaged into the app.

## 1.1
- Camera fix (the app closed when pressing Camera) and a camera permission request.
- Tilted photos are straightened before the board is cut out.

## 1.0
- First Android version: gallery picker, board reading, Stockfish best moves.
