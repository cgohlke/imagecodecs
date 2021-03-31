# imagecodecs/_shared.pxi
# cython: language_level = 3

# Include file for imagecodecs extensions.

import numpy
cimport numpy
cimport cython

from ._shared import _log_warning, _set_attributes

from ._shared cimport (
    _parse_output,
    _create_output,
    _return_output,
    _create_array,
    _readable_input,
    _writable_input,
    _inplace_input,
    _default_value
)

from libc.string cimport memset, memcpy, memmove
from libc.stdlib cimport malloc, free, realloc

from libc.stdint cimport (
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    UINT16_MAX,
    UINT32_MAX,
    UINT64_MAX,
    SIZE_MAX,
)

numpy.import_array()
