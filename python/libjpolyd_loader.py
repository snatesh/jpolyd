import ctypes
import os
import sys 

_here = os.path.dirname(__file__)
if sys.platform.startswith("linux"):
  _libname = "libjpolyd.so"
elif sys.platform == "darwin":
  _libname = "libjpolyd.dylib"
elif sys.platform == "win32":
  _libname = "jpolyd.dll"
else:
  _libname = "libjpolyd.so"

libjpolyd = ctypes.CDLL(os.path.join(_here, _libname))

