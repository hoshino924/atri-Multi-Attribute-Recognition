"""Project-relative assets shared by the Tk interfaces and release entry points."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER_WEIGHT = str(PROJECT_ROOT / "models" / "classifier" / "atri_net_best.pth")
DEFAULT_LOCATOR_WEIGHT = str(PROJECT_ROOT / "models" / "locator" / "face_locator_best.pth")


def configure_taskbar(application):
    """Set the Windows process identity before Tk creates any windows."""
    if sys.platform != "win32":
        return
    import ctypes
    import warnings

    try:
        setter = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long  # HRESULT remains 32-bit on Windows.
        result = setter(f"Hoshino924.ATRI.{application}")
        if result != 0:
            raise OSError(f"SetCurrentProcessExplicitAppUserModelID returned {result & 0xffffffff:#010x}")
    except (AttributeError, OSError) as exc:
        warnings.warn(f"Could not set the Windows taskbar identity: {exc}", RuntimeWarning, stacklevel=2)


def set_app_icon(root):
    """Use the authored size variants without resizing; keep CLI imports Tk-free."""
    import tkinter as tk

    directory = PROJECT_ROOT / "icons"
    if root.tk.call("tk", "windowingsystem") == "win32":
        try:
            root.iconbitmap(default=str(directory / "atri.ico"))
            return
        except tk.TclError:
            pass

    images = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        try:
            images.append(tk.PhotoImage(master=root, file=str(directory / f"icon-{size}.png")))
        except tk.TclError:
            continue
    if images:
        try:
            root.iconphoto(True, *images)
        except tk.TclError:
            return
        root._atri_icon_images = images
