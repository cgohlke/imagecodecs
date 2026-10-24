# imagecodecs/__init__.py

# Copyright (c) Christoph Gohlke
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE file in the project root for details.

from .imagecodecs import (
    DelayedImportError,
    __all__,
    __dir__,
    __doc__,
    __getattr__,
    __version__,
    _codecs,
    _extensions,
    imagefileext,
    iminfo,
    imread,
    imwrite,
    version,
)

# constants are repeated for documentation

__version__ = __version__
"""Imagecodecs version string."""
