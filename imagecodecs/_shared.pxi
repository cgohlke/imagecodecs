# imagecodecs/_shared.pxi

# Include file for imagecodecs extensions.

import enum

import numpy

cimport cython
cimport numpy

from ._shared import _log_warning

from libc.limits cimport ULONG_MAX
from libc.stdint cimport (
    INT16_MAX,
    INT32_MAX,
    INT64_MAX,
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


cdef inline size_t _align_size_t(size_t size) noexcept nogil:
    """Return size_t increased to next multiple of 64KB."""
    return (size + 65536 - 1) // 65536 * 65536


cdef inline size_t _align_ssize_t(ssize_t size) noexcept nogil:
    """Return ssize_t increased to next multiple of 64KB."""
    return (size + 65536 - 1) // 65536 * 65536
