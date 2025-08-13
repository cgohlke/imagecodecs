# imagecodecs/_shared.pxi
# cython: language_level = 3

# Include file for imagecodecs extensions.

import enum

import numpy

cimport cython
cimport numpy

from ._shared import _log_warning

from libc.stdint cimport (
    SIZE_MAX,
    UINT16_MAX,
    UINT32_MAX,
    UINT64_MAX,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.stdlib cimport calloc, free, malloc, realloc
from libc.string cimport memcpy, memmove, memset

from ._shared cimport (
    _create_array,
    _create_output,
    _default_threads,
    _default_value,
    _inplace_input,
    _parse_output,
    _readable_input,
    _return_output,
    _squeeze_shape,
    _writable_input,
)

numpy.import_array()
