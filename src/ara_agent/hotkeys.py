"""Global hotkey for ara-agent via Carbon's RegisterEventHotKey.

Why Carbon (instead of Cocoa's NSEvent global monitor)?
  * Cocoa's `addGlobalMonitorForEventsMatchingMask` for key events
    requires the user to grant macOS Input Monitoring permission. On
    VS Code in particular, that permission often fails to propagate
    from the host app to Python subprocesses run via the integrated
    terminal, leaving the monitor silently dead. We confirmed this
    empirically — diagnostic logging showed zero key events arriving
    in the Python process even after enabling VS Code in both the
    Accessibility and Input Monitoring panes.
  * Carbon's RegisterEventHotKey, by contrast, registers a single
    specific key combination with the system. The OS delivers events
    only when *that* combination fires — there's no monitoring of
    other input, no privacy concern, no TCC entry needed. It's the
    same API every menu-bar utility (e.g. Alfred, Raycast) uses for
    their main toggle.
  * Carbon is "deprecated" but RegisterEventHotKey and the surrounding
    HIToolbox event handler functions remain functional on modern
    macOS (Sequoia 15.x verified) and Apple has not announced removal.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_int32,
    c_uint32,
    c_void_p,
)
from typing import Callable, Dict, Optional


# ----- Carbon framework loading -----

_carbon_path = ctypes.util.find_library("Carbon")
_carbon = ctypes.CDLL(_carbon_path) if _carbon_path else None


def _fourcc(s: bytes) -> int:
    """Pack 4 ASCII bytes into a uint32 OSType (big-endian)."""
    return (s[0] << 24) | (s[1] << 16) | (s[2] << 8) | s[3]


# ----- Carbon constants -----

kEventClassKeyboard = _fourcc(b"keyb")
kEventHotKeyPressed = 5

# Carbon modifier flag bits (NOT the same as NSEventModifierFlag*).
cmdKey = 1 << 8       # 0x0100
shiftKey = 1 << 9     # 0x0200
optionKey = 1 << 11   # 0x0800
controlKey = 1 << 12  # 0x1000

# Virtual keycodes — same as Cocoa's hardware virtual keys (kHIDUsage_*).
KEY_A = 0x00
KEY_S = 0x01
KEY_D = 0x02
KEY_ESCAPE = 0x35
KEY_SPACE = 0x31


# ----- Carbon struct types -----

class EventHotKeyID(Structure):
    _fields_ = [
        ("signature", c_uint32),
        ("id", c_uint32),
    ]


class EventTypeSpec(Structure):
    _fields_ = [
        ("eventClass", c_uint32),
        ("eventKind", c_uint32),
    ]


# Carbon event handler signature:
#   OSStatus handler(EventHandlerCallRef, EventRef, void *userData);
_EventHandlerProc = CFUNCTYPE(c_int32, c_void_p, c_void_p, c_void_p)


# ----- Function signatures -----

if _carbon is not None:
    _carbon.GetApplicationEventTarget.argtypes = []
    _carbon.GetApplicationEventTarget.restype = c_void_p

    _carbon.RegisterEventHotKey.argtypes = [
        c_uint32,                # inHotKeyCode
        c_uint32,                # inHotKeyModifiers
        EventHotKeyID,           # inHotKeyID (struct, passed by value)
        c_void_p,                # inTarget
        c_uint32,                # inOptions
        POINTER(c_void_p),       # outRef
    ]
    _carbon.RegisterEventHotKey.restype = c_int32

    _carbon.UnregisterEventHotKey.argtypes = [c_void_p]
    _carbon.UnregisterEventHotKey.restype = c_int32

    _carbon.InstallEventHandler.argtypes = [
        c_void_p,
        c_void_p,                # EventHandlerProcPtr
        c_uint32,                # numTypes
        POINTER(EventTypeSpec),
        c_void_p,                # userData
        POINTER(c_void_p),       # outRef
    ]
    _carbon.InstallEventHandler.restype = c_int32


# ----- Module-level dispatcher state -----

_registry: Dict[int, Callable[[], None]] = {}
_handler_proc = None  # strong reference so ctypes doesn't GC the C closure
_handler_ref: Optional[c_void_p] = None
_next_id: int = 1


def _dispatch(_next_handler, _event_ref, _user_data):
    """Carbon C callback. Fires for any of our registered hotkeys.
    We currently only register one, so this just invokes every callback."""
    for callback in list(_registry.values()):
        try:
            callback()
        except Exception as e:
            print(f"Hotkey callback error: {type(e).__name__}: {e}")
    return 0  # noErr


def _ensure_handler_installed() -> bool:
    """One-time install of the Carbon event handler. Subsequent
    RegisterEventHotKey calls route events through this handler."""
    global _handler_proc, _handler_ref
    if _handler_ref is not None:
        return True
    if _carbon is None:
        return False

    # Hold a strong reference — ctypes would otherwise GC the C closure
    # and Carbon would call freed memory the next time the hotkey fires.
    _handler_proc = _EventHandlerProc(_dispatch)

    event_type = EventTypeSpec(
        eventClass=kEventClassKeyboard,
        eventKind=kEventHotKeyPressed,
    )
    handler_ref = c_void_p()
    status = _carbon.InstallEventHandler(
        _carbon.GetApplicationEventTarget(),
        ctypes.cast(_handler_proc, c_void_p),
        1,
        byref(event_type),
        None,
        byref(handler_ref),
    )
    if status != 0:
        print(f"⚠️  InstallEventHandler failed: status={status}")
        return False
    _handler_ref = handler_ref
    return True


# ----- Public API -----

class GlobalHotkey:
    """Register a global hotkey via Carbon. No user permission needed —
    the system only delivers events when this exact combination fires.

    Defaults to ⌘⇧A. To customize, pass keycode (one of the KEY_*
    module constants) and modifiers as a bitwise OR of cmdKey, shiftKey,
    optionKey, controlKey.
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        keycode: int = KEY_A,
        modifiers: int = cmdKey | shiftKey,
    ):
        global _next_id
        self._on_press = on_press
        self._keycode = keycode
        self._modifiers = modifiers
        self._hotkey_ref = c_void_p()
        self._id = _next_id
        _next_id += 1

    def install(self) -> bool:
        if _carbon is None:
            print("⚠️  Carbon framework not found — hotkey disabled.")
            return False
        if not _ensure_handler_installed():
            return False

        hk_id = EventHotKeyID(signature=_fourcc(b"AraA"), id=self._id)
        status = _carbon.RegisterEventHotKey(
            self._keycode,
            self._modifiers,
            hk_id,
            _carbon.GetApplicationEventTarget(),
            0,
            byref(self._hotkey_ref),
        )
        if status != 0:
            print(f"⚠️  RegisterEventHotKey failed: status={status}")
            return False
        _registry[self._id] = self._on_press
        return True

    def uninstall(self) -> None:
        if _carbon is not None and self._hotkey_ref.value:
            _carbon.UnregisterEventHotKey(self._hotkey_ref)
            self._hotkey_ref = c_void_p()
        _registry.pop(self._id, None)
