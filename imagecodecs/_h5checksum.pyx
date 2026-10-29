# imagecodecs/_h5checksum.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2023-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""H5CHECKSUM codec for the imagecodecs package."""

include '_shared.pxi'

from h5checksum cimport *
from libc.string cimport memchr

H5_checksum_init()


class H5CHECKSUM:
    """H5CHECKSUM codec constants."""

    available = True


def h5checksum_version():
    """Return h5checksum library version string."""
    return f'h5checksum {H5_VERS_MAJOR}.{H5_VERS_MINOR}.{H5_VERS_RELEASE}'


def h5checksum_fletcher32(
    data,
    /,
    value=None,
):
    """Return Fletcher-32 checksum of data.

    The checksum cannot be continued; value must be None.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t val

    if value is not None:
        raise ValueError('fletcher32 checksum cannot be continued')

    with nogil:
        val = H5_checksum_fletcher32(<const void*> src._data, srcsize)
    return int(val)


def h5checksum_lookup3(
    data,
    /,
    value=None,
):
    """Return Jenkins lookup3 checksum of data.

    Only the lower 32 bits of the data size affect the checksum.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t val = _uint32(value)

    with nogil:
        val = H5_checksum_lookup3(<const void*> src._data, srcsize, val)
    return int(val)


def h5checksum_crc(
    data,
    /,
    value=None,
):
    """Return HDF5 CRC checksum of data.

    The HDF5 variant differs from the standard CRC-32 used by zlib.
    Pass a previous result as value to continue the checksum.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t val = _uint32(value)

    with nogil:
        val = H5_checksum_crc_value(<const void*> src._data, srcsize, val)
    return int(val)


def h5checksum_metadata(
    data,
    /,
    value=None,
):
    """Return checksum of metadata (same as lookup3)."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t val = _uint32(value)

    with nogil:
        val = H5_checksum_metadata(<const void*> src._data, srcsize, val)
    return int(val)


def h5checksum_hash_string(
    data,
    /,
    value=None,
):
    """Return djb2 hash of string, which must not contain NUL bytes.

    Pass a previous result as value to continue the hash.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        size_t i
        uint32_t val = 5381 if value is None else _uint32(value)

    if srcsize > 0 and memchr(<const void*> src._data, 0, srcsize) != NULL:
        raise ValueError('string contains NUL byte')

    with nogil:
        for i in range(srcsize):
            # sign-extend like H5_hash_string on platforms with signed char
            val = val * 33 + <uint32_t> <int> (<const char*> src._data)[i]
    return int(val)


cdef uint32_t _uint32(value):
    return 0 if value is None else <uint32_t> (int(value) & 0xFFFFFFFF)
