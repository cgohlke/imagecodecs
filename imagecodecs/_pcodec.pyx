# imagecodecs/_pcodec.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2024-2026, Christoph Gohlke
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

"""PCODEC codec for the imagecodecs package.

See Pcodec C API issues: https://github.com/pcodec/pcodec/issues/247

"""

include '_shared.pxi'

from pcodec cimport *


class PCODEC:
    """PCODEC codec constants."""

    available = True


class PcodecError(RuntimeError):
    """PCODEC codec exceptions."""

    def __init__(self, func, err):
        msg = {
            PcoSuccess: 'PcoSuccess',
            PcoInvalidType: 'PcoInvalidType',
            PcoCompressionError: 'PcoCompressionError',
            PcoDecompressionError: 'PcoDecompressionError',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def pcodec_version():
    """Return pcodec library version string."""
    # TODO: use version from header when available
    return 'pcodec 0.4.7'


def pcodec_check(const uint8_t[::1] data, /):
    """Return whether data is PCODEC encoded or None if unknown."""


def pcodec_encode(
    data,
    /,
    level=None,
    *,
    out=None,
):
    """Return PCODEC encoded data."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, pcovec_len
        unsigned int pcolevel = _default_value(level, 8, 0, 12)
        unsigned char pcotype
        PcoFfiVec pcovec
        PcoError ret

    if src.size > INT32_MAX:
        raise ValueError(f'invalid {src.size=}')

    try:
        pcotype = PCO_TYPE[(src.dtype.kind, src.dtype.itemsize)]
    except KeyError:
        raise ValueError(f'{src.dtype=} not supported')

    with nogil:
        memset(<void*> &pcovec, 0, sizeof(PcoFfiVec))

        ret = pco_simpler_compress(
            <const void*> src.data,
            <size_t> src.size,
            pcotype,
            pcolevel,
            &pcovec
        )
        if ret != PcoSuccess:
            raise PcodecError('pco_simpler_compress', ret)

        pcovec_len = <ssize_t> pcovec.len

    try:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            dstsize = pcovec_len
            out = _create_output(
                outtype, dstsize, <const char*> pcovec.ptr
            )
        else:
            dst = out
            dstsize = dst.nbytes
            if dstsize < pcovec_len:
                raise ValueError(
                    f'output buffer too small {dstsize=} < {pcovec.len=}'
                )
            memcpy(<void*> &dst[0], pcovec.ptr, pcovec.len)
            del dst
    finally:
        pco_free_pcovec(&pcovec)

    return _return_output(out, dstsize, pcovec_len, outgiven)


def pcodec_decode(
    data,
    /,
    shape=None,
    dtype=None,
    *,
    out=None,
):
    """Return decoded PCODEC data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t src_len = src.size
        unsigned char pcotype
        PcoFfiVec pcovec
        PcoError ret

    if isinstance(out, numpy.ndarray):
        dtype = out.dtype
        shape = out.shape
    elif dtype is not None:
        dtype = numpy.dtype(dtype)
    else:
        raise ValueError('dtype not specified')

    try:
        dtype = numpy.dtype(dtype)
        pcotype = PCO_TYPE[(dtype.kind, dtype.itemsize)]
    except KeyError:
        raise ValueError(f'{dtype=} not supported')

    with nogil:
        memset(<void*> &pcovec, 0, sizeof(PcoFfiVec))

        ret = pco_simple_decompress(
            <const void*> &src[0], src_len, pcotype, &pcovec
        )
        if ret != PcoSuccess:
            raise PcodecError('pco_simple_decompress', ret)

    try:
        if shape is None:
            shape = (int(pcovec.len),)
        out = _create_array(out, shape, dtype)
        dst = out
        if out.size != pcovec.len:
            raise ValueError(
                f'invalid output size {out.size=} != {pcovec.len=}'
            )
        memcpy(<void*> &dst.data[0], pcovec.ptr, dst.nbytes)
    finally:
        pco_free_pcovec(&pcovec)

    return out


PCO_TYPE = {
    ('u', 4): PCO_TYPE_U32,
    ('u', 8): PCO_TYPE_U64,
    ('i', 4): PCO_TYPE_I32,
    ('i', 8): PCO_TYPE_I64,
    ('f', 4): PCO_TYPE_F32,
    ('f', 8): PCO_TYPE_F64,
    ('u', 2): PCO_TYPE_U16,
    ('i', 2): PCO_TYPE_I16,
    ('f', 2): PCO_TYPE_F16,
}
