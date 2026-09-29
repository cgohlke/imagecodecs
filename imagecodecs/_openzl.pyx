# imagecodecs/_openzl.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""OpenZL codec for the imagecodecs package."""

include '_shared.pxi'

from openzl cimport *


class OPENZL:
    """OPENZL codec constants."""

    available = True


class OpenzlError(RuntimeError):
    """OPENZL codec exceptions."""

    def __init__(self, func, err):
        cdef:
            const char* error = ZL_ErrorCode_toString(<ZL_ErrorCode> err)

        msg = f'unknown error {err!r}' if error == NULL else error.decode()
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def openzl_version():
    """Return openzl library version string."""
    return (
        f'openzl '
        f'{ZL_LIBRARY_VERSION_MAJOR}.'
        f'{ZL_LIBRARY_VERSION_MINOR}.'
        f'{ZL_LIBRARY_VERSION_PATCH}'
    )


def openzl_check(const uint8_t[::1] data, /):
    """Return whether data is OpenZL encoded or None if unknown."""
    cdef:
        ZL_FrameInfo* fi
        ZL_Report ret

    if data.shape[0] == 0:
        return False

    fi = ZL_FrameInfo_create(<const void*> data._data, <size_t> data.shape[0])
    if fi == NULL:
        return False
    try:
        ret = ZL_FrameInfo_getFormatVersion(fi)
        return not ZL_isError(ret)
    finally:
        ZL_FrameInfo_free(fi)


def openzl_encode(
    data,
    /,
    level=None,
    *,
    out=None,
):
    """Return OpenZL encoded data (not implemented)."""
    raise NotImplementedError('openzl_encode')


def openzl_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded OpenZL data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ZL_Report res

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            with nogil:
                res = ZL_getDecompressedSize(
                    <const void*> src._data, <size_t> srcsize,
                )
            if ZL_isError(res):
                raise OpenzlError('ZL_getDecompressedSize', ZL_errorCode(res))
            dstsize = <ssize_t> ZL_validResult(res)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        res = ZL_decompress(
            <void*> dst._data,
            <size_t> dstsize,
            <const void*> src._data,
            <size_t> srcsize
        )
    if ZL_isError(res):
        raise OpenzlError('ZL_decompress', ZL_errorCode(res))

    dstsize = <ssize_t> ZL_validResult(res)

    del dst
    return _return_output(out, dstsize, dstsize, outgiven)
