# -*- coding: utf-8 -*-
# imagecodecs/__init__.py

try:
    from ._imagecodecs import __doc__
    from ._imagecodecs import __version__
    from ._imagecodecs import *
except ImportError as error:
    import warnings
    warnings.warn(
        str(error) +
        '\n\nThe _imagecodecs Cython extension module could not be found.\n'
        'Using a fallback module with limited functionality and performance.\n'
    )
    from .imagecodecs import __doc__
    from .imagecodecs import __version__
    from .imagecodecs import *
