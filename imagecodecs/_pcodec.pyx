# imagecodecs/_pcodec.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2024-2025, Christoph Gohlke
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

"""Pcodec codec for the imagecodecs package."""

include '_shared.pxi'

from pcodec cimport *


class PCODEC:
    """Pcodec codec constants."""

    available = True


class PcodecError(RuntimeError):
    """Pcodec codec exceptions."""

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
    return 'pcodec 0.3.1'


def pcodec_check(const uint8_t[::1] data):
    """Return whether data is pcodec encoded."""
    return None


def pcodec_encode(data, level=None, out=None):
    """Return pcodec encoded data."""
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, dst_len
        unsigned int len = <unsigned int> src.size
        unsigned int pcolevel = _default_value(level, 8, 0, 12)
        unsigned char pcotype
        PcoFfiVec pcovec
        PcoError ret

    if src.size > 2147483647:
        raise ValueError(f'invalid {src.size=}')

    try:
        pcotype = PCO_TYPE[(src.dtype.kind, src.dtype.itemsize)]
    except KeyError:
        raise ValueError(f'{src.dtype=} not supported')

    with nogil:
        ret = pco_simpler_compress(
            <const void*> src.data,
            len,
            pcotype,
            pcolevel,
            &pcovec
        )
        if ret != PcoSuccess:
            raise PcodecError('pco_simpler_compress', ret)

    try:
        dst_len = <ssize_t> pcovec.len

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            dstsize = dst_len
            out = _create_output(outtype, dstsize, <const char*> pcovec.ptr)
        else:
            dst = out
            dstsize = dst.nbytes
            if dstsize < dst_len:
                raise ValueError(
                    f'output buffer too small {dstsize=} < {dst_len=}'
                )
            memcpy(<void*> &dst[0], pcovec.ptr, dst_len)
            del dst
    finally:
        pco_free_pcovec(&pcovec)

    return _return_output(out, dstsize, dst_len, outgiven)


def pcodec_decode(data, shape=None, dtype=None, out=None):
    """Return decoded pcodec data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        unsigned int src_len = <unsigned int> src.size
        unsigned char pcotype
        PcoFfiVec pcovec
        PcoError ret

    if src.size > 2147483647:
        raise ValueError(f'invalid {src.size=}')

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
