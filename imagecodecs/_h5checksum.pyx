# imagecodecs/_h5checksum.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2023-2025, Christoph Gohlke
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

"""H5checksum codec for the imagecodecs package."""

include '_shared.pxi'

from h5checksum cimport *


class H5CHECKSUM:
    """H5checksum codec constants."""

    available = True


def h5checksum_version():
    """Return h5checksum library version string."""
    return f'h5checksum {H5_VERS_MAJOR}.{H5_VERS_MINOR}.{H5_VERS_RELEASE}'


def h5checksum_fletcher32(data, value=None):
    """Return fletcher32 checksum of data (value is ignored)."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t val = 0  # if value is None else value

    with nogil:
        val = H5_checksum_fletcher32(<const void *> &src[0], srcsize)
    return int(val)


def h5checksum_lookup3(data, value=None):
    """Return Jenkins lookup3 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t val = 0 if value is None else value

    with nogil:
        val = H5_checksum_lookup3(<const void *> &src[0], srcsize, val)
    return int(val)


def h5checksum_crc(data, value=None):
    """Return crc checksum of data (value is ignored)."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t val = 0  # if value is None else value

    with nogil:
        val = H5_checksum_crc(<const void *> &src[0], srcsize)
    return int(val)


def h5checksum_metadata(data, value=None):
    """Return checksum of metadata."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t val = 0 if value is None else value

    with nogil:
        val = H5_checksum_metadata(<const void *> &src[0], srcsize, val)
    return int(val)


def h5checksum_hash_string(data, value=None):
    """Return hash of bytes string (value is ignored)."""
    cdef:
        const char* str = data
        uint32_t val = 0  # if value is None else value

    with nogil:
        val = H5_hash_string(str)
    return int(val)
