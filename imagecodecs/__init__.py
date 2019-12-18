# -*- coding: utf-8 -*-
# imagecodecs/__init__.py

from ._imagecodecs import __doc__, __version__
from ._imagecodecs import *
from ._utils import *

# keep for older versions of tifffile and czifile
j2k_encode = jpeg2k_encode
j2k_decode = jpeg2k_decode
jxr_encode = jpegxr_encode
jxr_decode = jpegxr_decode
