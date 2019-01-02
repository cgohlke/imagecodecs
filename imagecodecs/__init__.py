# -*- coding: utf-8 -*-
# imagecodecs/__init__.py

try:
    from ._imagecodecs import __doc__, __version__
    from ._imagecodecs import *
except ImportError as error:
    import warnings
    warnings.warn("""

%s

*******************************************************************

The _imagecodecs Cython extension module could not be loaded.
Using a fallback module with limited functionality and performance.

*******************************************************************
        """ % str(error))
    from .imagecodecs import __doc__, __version__
    from .imagecodecs import *
