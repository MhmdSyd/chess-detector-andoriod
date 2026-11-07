[app]
title = Chess Analyzer
package.name = chessanalyzer
package.domain = org.chessanalyzer
source.dir = .
source.include_exts = py,png,jpg,npz,ttf
source.exclude_dirs = tests,tools,scripts,.github,bin,.buildozer
source.exclude_patterns = test_*.py,stress_*.py,ui_*.py,render.py,synth.py,export_weights.py,*.md,requirements-desktop.txt
version = 1.6
icon.filename = %(source.dir)s/assets/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/assets/icon_fg.png
icon.adaptive_background.filename = %(source.dir)s/assets/icon_bg.png
presplash.filename = %(source.dir)s/assets/presplash.png
android.presplash_color = #1a1a2e
requirements = python3,kivy,numpy,pillow,chess,pyjnius,android,opencv
orientation = portrait
fullscreen = 0
android.permissions = INTERNET,CAMERA
android.minapi = 29
android.archs = arm64-v8a
android.accept_sdk_license = True
android.add_libs_arm64_v8a = libs/arm64-v8a/*.so

[buildozer]
log_level = 2
warn_on_root = 1
