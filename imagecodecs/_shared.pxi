# imagecodecs/_shared.pxi

# Include file for imagecodecs extensions.

import enum

import numpy

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
    IC_ALPHA,
    IC_BOOL,
    IC_BPS,
    IC_CIELAB,
    IC_CMYK,
    IC_COMPLEX,
    IC_DEPTH,
    IC_EXTRA,
    IC_EXTRA_ASSOCALPHA,
    IC_EXTRA_UNASSALPHA,
    IC_EXTRA_UNSPECIFIED,
    IC_FLOAT,
    IC_FRAMES,
    IC_GRAY,
    IC_ICCLAB,
    IC_PALETTE,
    IC_PHOTO_CIELAB,
    IC_PHOTO_CMYK,
    IC_PHOTO_GRAY,
    IC_PHOTO_ICCLAB,
    IC_PHOTO_PALETTE,
    IC_PHOTO_RGB,
    IC_PHOTO_UNSPECIFIED,
    IC_PHOTO_YCBCR,
    IC_PLANAR,
    IC_RGB,
    IC_SF_BOOL,
    IC_SF_COMPLEX,
    IC_SF_FLOAT,
    IC_SF_SINT,
    IC_SF_UINT,
    IC_SINT,
    IC_SZ1,
    IC_SZ2,
    IC_SZ4,
    IC_SZ8,
    IC_SZ16,
    IC_UINT,
    IC_YCBCR,
    _create_array,
    _create_output,
    _default_threads,
    _default_value,
    _enum_value,
    _image_layout,
    _inplace_input,
    _parse_output,
    _photo_cap,
    _photo_samples,
    _readable_input,
    _return_output,
    _squeeze_shape,
    _writable_input,
    imagecaps_t,
    imagelayout_t,
)


cdef inline size_t _align_size_t(size_t size) noexcept nogil:
    """Return size_t increased to next multiple of 64KB."""
    return (size + 65536 - 1) // 65536 * 65536


cdef inline size_t _align_ssize_t(ssize_t size) noexcept nogil:
    """Return ssize_t increased to next multiple of 64KB."""
    return (size + 65536 - 1) // 65536 * 65536
