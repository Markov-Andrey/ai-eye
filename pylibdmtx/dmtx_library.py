import platform
from ctypes import cdll
from ctypes.util import find_library
from pathlib import Path

__all__ = ['load']


def load():
    if 'Windows' == platform.system():
        fname = 'libdmtx-64.dll'
        try:
            libdmtx = cdll.LoadLibrary(fname)
        except OSError:
            libdmtx = cdll.LoadLibrary(str(Path(__file__).parent.joinpath(fname)))
    else:
        path = find_library('dmtx')
        if not path:
            raise ImportError('Unable to find dmtx shared library')
        libdmtx = cdll.LoadLibrary(path)

    return libdmtx
