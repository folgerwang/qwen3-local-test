"""Memory figures for the status bar: system RAM, a process's RAM, and VRAM of the first NVIDIA GPU."""
import ctypes
import os
import subprocess

_nvml = None  # the loaded library and device handle, or False once loading failed


def ram():
    """(used, total) bytes of physical memory, or None."""
    try:
        if os.name == "nt":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]
            s = Status()
            s.length = ctypes.sizeof(s)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return None
            return s.total - s.avail, s.total
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return total - avail, total
    except (OSError, ValueError, AttributeError):
        return None


def process_ram(pid):
    """Bytes of physical memory the process holds (its working set), or None. Windows only."""
    if os.name != "nt":
        return None
    try:
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + \
                       [(n, ctypes.c_size_t) for n in ("peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged",
                                                       "nonpaged", "pagefile", "peak_pagefile")]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.OpenProcess.restype = ctypes.c_void_p
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        c = Counters()
        c.cb = ctypes.sizeof(c)
        ok = psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(c), c.cb)
        k32.CloseHandle(ctypes.c_void_p(h))
        return c.ws if ok else None
    except (OSError, AttributeError):
        return None


def vram():
    """(used, total) bytes of the first NVIDIA GPU, counting every program on it, or None."""
    global _nvml
    if _nvml is None:
        _nvml = False
        try:
            lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
            handle = ctypes.c_void_p()
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                _nvml = (lib, handle)
        except OSError:
            pass
    if _nvml:
        class Memory2(ctypes.Structure):  # the figures nvidia-smi shows: "used" without the driver's reserve
            _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        m2 = Memory2()
        m2.version = ctypes.sizeof(m2) | (2 << 24)
        try:
            if _nvml[0].nvmlDeviceGetMemoryInfo_v2(_nvml[1], ctypes.byref(m2)) == 0:
                return m2.used, m2.total
        except AttributeError:  # an older driver without the v2 call
            pass

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        m = Memory()
        if _nvml[0].nvmlDeviceGetMemoryInfo(_nvml[1], ctypes.byref(m)) == 0:
            return m.used, m.total
        return None
    try:  # no NVML library: ask nvidia-smi
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split("\n")[0].split(",")
        return int(out[0]) << 20, int(out[1]) << 20
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
