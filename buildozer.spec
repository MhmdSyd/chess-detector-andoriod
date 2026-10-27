[app]
title = Chess Analyzer
package.name = chessanalyzer
package.domain = org.chessanalyzer
source.dir = .
source.include_exts = py,png,jpg,npz,ttf
source.exclude_dirs = tests,tools,scripts,.github,bin,.buildozer
version = 1.0
requirements = python3,kivy,numpy,pillow,chess,pyjnius,android
orientation = portrait
fullscreen = 0
android.permissions = INTERNET
android.minapi = 29
android.archs = arm64-v8a
android.accept_sdk_license = True
android.add_libs_arm64_v8a = libs/arm64-v8a/*.so

[buildozer]
log_level = 2
warn_on_root = 1
