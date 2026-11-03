# imagecodecs/_lzo.pyx
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

"""LZO (Lempel-Ziv-Oberhumer) codec for the imagecodecs package."""

include '_shared.pxi'

from lzokay cimport *


class LZO:
    """LZO codec constants."""

    available = True


class LzoError(RuntimeError):
    """LZO codec exceptions."""

    def __init__(self, func, err):
        if isinstance(err, str):
            super().__init__(f'{func} returned {err!r}')
            return
        msg = {
            EResult_LookbehindOverrun: 'LookbehindOverrun',
            EResult_OutputOverrun: 'OutputOverrun',
            EResult_InputOverrun: 'InputOverrun',
            EResult_Error: 'Error',
            EResult_Success: 'Success',
            EResult_InputNotConsumed: 'InputNotConsumed',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        if err == EResult_OutputOverrun:
            msg += ' (output too small)'
        super().__init__(msg)


def lzo_version():
    """Return lzokay library version string."""
    return 'lzokay unknown'


def lzo_check(const uint8_t[::1] data not None, /):
    """Return whether data is LZO encoded or None if unknown."""
    cdef:
        ssize_t size = data.shape[0]

    # python-lzo header followed by at least an end-of-stream marker
    if (
        size >= 8
        and (data[0] == 0xf0 or data[0] == 0xf1)
        and 0 <= _lzo_header_size(data) <= _lzo_max_size(size - 5)
    ):
        return True
    return None


def lzo_encode(
    data,
    /,
    level=None,
    *,
    header=False,
    out=None,
):
    """Return LZO encoded data (not implemented)."""
    raise NotImplementedError('lzo_encode')


def lzo_decode(
    data,
    /,
    *,
    header=False,
    out=None,
):
    """Return decoded LZO data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t hdrsize = -1
        ssize_t offset = 5 if header else 0
        size_t output_len
        lzokay_EResult ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if header:
        if srcsize < offset:
            raise ValueError(f'invalid data size {srcsize} < 5')
        if src[0] != 0xf0 and src[0] != 0xf1:
            raise ValueError(f'invalid LZO header {src[0]:#x}')
        hdrsize = _lzo_header_size(src)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if not header:
                raise TypeError(
                    'lzo_decode() requires header=True or out=<output size>'
                )
            if hdrsize < 0 or hdrsize > _lzo_max_size(srcsize - offset):
                raise ValueError(f'invalid LZO header size {hdrsize}')
            dstsize = hdrsize
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]
    output_len = <size_t> dstsize

    with nogil:
        ret = lzokay_decompress(
            <const uint8_t*> (src._data + offset),
            <size_t> (srcsize - offset),
            <uint8_t*> dst._data,
            &output_len
        )
    if ret != EResult_Success:
        raise LzoError('lzokay_decompress', ret)
    if header and <ssize_t> output_len != hdrsize:
        raise LzoError(
            'lzokay_decompress',
            f'decoded size {output_len} does not match header size {hdrsize}'
        )

    del dst
    return _return_output(out, dstsize, <ssize_t> output_len, outgiven)


cdef inline ssize_t _lzo_header_size(const uint8_t[::1] src) noexcept nogil:
    """Return big-endian uncompressed size from python-lzo header."""
    return <ssize_t> (
        (<uint32_t> src[1] << 24)
        | (<uint32_t> src[2] << 16)
        | (<uint32_t> src[3] << 8)
        | (<uint32_t> src[4])
    )


cdef inline ssize_t _lzo_max_size(ssize_t srcsize) noexcept nogil:
    """Return upper bound of LZO1X decoded size."""
    # match lengths grow by 255 per extension byte
    if srcsize > (PY_SSIZE_T_MAX - 64) // 255:
        return PY_SSIZE_T_MAX
    return srcsize * 255 + 64
