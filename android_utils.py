"""
android_utils.py - pick a picture from the gallery, take one with the camera, load it.

On Android both actions use the system's own screens through Intents, so the app needs
NO storage / camera permission:
  * gallery  -> ACTION_GET_CONTENT (the system file picker, returns a content:// URI)
  * camera   -> ACTION_IMAGE_CAPTURE writing into a MediaStore entry we create
The picture is copied into the app cache, then loaded with Pillow.

On a desktop (used only for development) a simple file chooser pop-up is shown.
"""
import os
import threading
import time

import numpy as np
from kivy.clock import Clock
from kivy.utils import platform
from PIL import Image, ImageOps

IS_ANDROID = platform == "android"
REQ_PICK, REQ_CAMERA = 7411, 7412
RESULT_OK = -1
MAX_SIDE = 2200          # larger pictures are shrunk: plenty for a board, saves memory/time


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path, max_side=MAX_SIDE):
    """File -> HxWx3 uint8 RGB array (EXIF rotation applied, long side <= max_side)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(im)


# ──────────────────────────────────────────────────────────────────────────────
# Android implementation
# ──────────────────────────────────────────────────────────────────────────────

class _Android:
    def __init__(self):
        self._bound = False
        self._callback = None          # callback(path_or_None, message_or_None) on Kivy thread
        self._camera_uri = None

    # -- helpers ---------------------------------------------------------------
    def _activity(self):
        from jnius import autoclass
        return autoclass("org.kivy.android.PythonActivity").mActivity

    def _bind(self):
        if not self._bound:
            from android import activity
            activity.bind(on_activity_result=self._on_result)
            self._bound = True

    def _finish(self, path, msg=None):
        cb = self._callback
        self._callback = None
        if cb:
            Clock.schedule_once(lambda dt: cb(path, msg), 0)

    def _copy_uri(self, uri, dest):
        """Copy a content:// URI into a normal file."""
        from jnius import autoclass
        resolver = self._activity().getContentResolver()
        FileOutputStream = autoclass("java.io.FileOutputStream")
        try:
            ins = resolver.openInputStream(uri)
            out = FileOutputStream(dest)
            autoclass("android.os.FileUtils").copy(ins, out)       # API 29+
            out.close()
            ins.close()
        except Exception:
            # Fallback: let Android decode and re-encode the picture
            BitmapFactory = autoclass("android.graphics.BitmapFactory")
            Bitmap = autoclass("android.graphics.Bitmap")
            CompressFormat = autoclass("android.graphics.Bitmap$CompressFormat")   # nested class: use '$'
            ins = resolver.openInputStream(uri)
            bmp = BitmapFactory.decodeStream(ins)
            out = FileOutputStream(dest)
            bmp.compress(CompressFormat.JPEG, 95, out)
            out.close()
            ins.close()

    def _cache_file(self, name):
        d = str(self._activity().getCacheDir().getAbsolutePath())
        return os.path.join(d, name)

    # -- public ----------------------------------------------------------------
    def pick_image(self, callback):
        self._callback = callback
        try:
            from jnius import autoclass, cast
            self._bind()
            Intent = autoclass("android.content.Intent")
            String = autoclass("java.lang.String")
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            title = cast("java.lang.CharSequence", String("Choose a chess screenshot"))
            self._activity().startActivityForResult(Intent.createChooser(intent, title), REQ_PICK)
        except Exception as e:                                   # noqa - never crash the app
            self._finish(None, "Gallery error: %s" % e)

    def take_photo(self, callback):
        """Ask for the CAMERA permission (first time), then open the camera app."""
        self._callback = callback
        try:
            from android.permissions import Permission, check_permission, request_permissions
        except Exception:                                        # very old p4a: just try
            self._launch_camera()
            return
        try:
            if check_permission(Permission.CAMERA):
                self._launch_camera()
                return

            def on_permission(permissions, grants):
                if grants and all(grants):
                    Clock.schedule_once(lambda dt: self._launch_camera(), 0)
                else:
                    self._finish(None, "Camera permission denied. Enable it in Settings > Apps > "
                                       "Chess Analyzer > Permissions.")
            request_permissions([Permission.CAMERA], on_permission)
        except Exception as e:                                   # noqa
            self._finish(None, "Camera permission error: %s" % e)

    def _launch_camera(self):
        try:
            from jnius import autoclass, cast
            self._bind()
            Intent = autoclass("android.content.Intent")
            MediaStore = autoclass("android.provider.MediaStore")
            # Nested Java classes must be loaded with '$' in pyjnius (MediaStore.Images.Media
            # as an attribute does not exist and used to crash the app).
            ImagesMedia = autoclass("android.provider.MediaStore$Images$Media")
            ContentValues = autoclass("android.content.ContentValues")
            values = ContentValues()
            values.put("_display_name", "chess_board_%d.jpg" % int(time.time()))
            values.put("mime_type", "image/jpeg")
            values.put("relative_path", "Pictures/ChessAnalyzer")
            resolver = self._activity().getContentResolver()
            self._camera_uri = resolver.insert(ImagesMedia.EXTERNAL_CONTENT_URI, values)
            if self._camera_uri is None:
                self._finish(None, "Could not prepare the camera.")
                return
            intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            intent.putExtra(MediaStore.EXTRA_OUTPUT, cast("android.os.Parcelable", self._camera_uri))
            self._activity().startActivityForResult(intent, REQ_CAMERA)
        except Exception as e:                                   # noqa - never crash the app
            self._delete_camera_entry()
            self._finish(None, "Camera error: %s" % e)

    # -- result ----------------------------------------------------------------
    def _on_result(self, request_code, result_code, intent):
        if request_code not in (REQ_PICK, REQ_CAMERA):
            return
        if request_code == REQ_PICK:
            if result_code != RESULT_OK or intent is None or intent.getData() is None:
                self._finish(None, None)
                return
            uri = intent.getData()
            name = "picked.jpg"
        else:
            uri = self._camera_uri
            name = "camera.jpg"
            if result_code != RESULT_OK:
                self._delete_camera_entry()
                self._finish(None, None)
                return

        def work():
            try:
                dest = self._cache_file(name)
                self._copy_uri(uri, dest)
                self._finish(dest)
            except Exception as e:                       # noqa
                self._finish(None, "Could not read the picture: %s" % e)
            finally:
                if request_code == REQ_CAMERA:
                    self._delete_camera_entry()          # keep the gallery tidy

        threading.Thread(target=work, daemon=True).start()

    def _delete_camera_entry(self):
        try:
            if self._camera_uri is not None:
                self._activity().getContentResolver().delete(self._camera_uri, None, None)
        except Exception:
            pass
        self._camera_uri = None


# ──────────────────────────────────────────────────────────────────────────────
# Desktop stand-in (development only)
# ──────────────────────────────────────────────────────────────────────────────

class _Desktop:
    def pick_image(self, callback):
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        chooser = FileChooserListView(path=os.path.expanduser("~"),
                                      filters=["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"])
        box = BoxLayout(orientation="vertical")
        box.add_widget(chooser)
        btn = Button(text="Open", size_hint_y=None, height=48)
        box.add_widget(btn)
        pop = Popup(title="Choose a picture", content=box, size_hint=(0.95, 0.9))

        def ok(*_):
            pop.dismiss()
            callback(chooser.selection[0] if chooser.selection else None, None)
        btn.bind(on_release=ok)
        pop.open()

    def take_photo(self, callback):
        callback(None, "The camera is only available on Android.")


_impl = _Android() if IS_ANDROID else _Desktop()
pick_image = _impl.pick_image
take_photo = _impl.take_photo
