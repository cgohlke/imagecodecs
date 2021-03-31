# imagecodecs/_szip.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""SZIP codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from szlib cimport *


class SZIP:
    """SZIP Constants."""


class SzipError(RuntimeError):
    """SZIP Exceptions."""

    def __init__(self, func, err):
        msg = {
            SZ_OK: 'SZ_OK',
            SZ_OUTBUFF_FULL: 'SZ_OUTBUFF_FULL',
            SZ_NO_ENCODER_ERROR: 'SZ_NO_ENCODER_ERROR',
            SZ_PARAM_ERROR: 'SZ_PARAM_ERROR',
            SZ_MEM_ERROR: 'SZ_MEM_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def szip_version():
    """Return SZIP library version string."""
    return 'libsz n/a'


def szip_check(data):
    """Return True if data likely contains SZIP data."""
    return False


def szip_encode(data, level=None, bitspersample=None, flags=None, out=None):
    """Compress SZIP.

    """
    raise NotImplementedError('szip_encode')


def szip_decode(data, out=None):
    """Decompress SZIP.

    """
    raise NotImplementedError('szip_decode')
