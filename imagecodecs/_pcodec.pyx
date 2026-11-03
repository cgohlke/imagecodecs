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
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def pcodec_version():
    """Return pcodec library version string."""
    # TODO: use version from header when available
    return 'pcodec 1.1.0'


def pcodec_check(const uint8_t[::1] data not None, /):
    """Return whether data is PCODEC encoded or None if unknown."""
    return _startswith(data, b'pco!')


def pcodec_encode(
    data,
    /,
    level=None,
    *,
    pagesize=None,
    out=None
):
    """Return PCODEC encoded data."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        PcoChunkConfig config
        unsigned char pcotype
        size_t size = <size_t> src.size
        size_t max_page_n = 0
        size_t bound, pagebound, npages, n_written
        PcoError ret

    try:
        pcotype = PCO_TYPE[(src.dtype.kind, src.dtype.itemsize)]
    except KeyError:
        raise ValueError(f'{src.dtype=} not supported')

    if pagesize is not None:
        if pagesize < 0:
            raise ValueError(f'invalid {pagesize=}')
        max_page_n = pagesize

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            bound = pco_standalone_guarantee_file_size(size, pcotype)
            if bound == 0:
                raise PcodecError(
                    'pco_standalone_guarantee_file_size', PcoInvalidType
                )
            if 0 < max_page_n < size:
                # library bound assumes default page size
                npages = (size + max_page_n - 1) // max_page_n
                pagebound = pco_standalone_guarantee_file_size(
                    max_page_n, pcotype
                )
                if pagebound == 0 or npages > SIZE_MAX // 2 // pagebound:
                    raise ValueError(f'invalid {pagesize=}')
                bound = max(bound, npages * pagebound)
            if bound > SIZE_MAX // 2:
                raise ValueError(f'invalid {src.size=}')
            dstsize = <ssize_t> bound
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    config.compression_level = _default_value(level, 8, 0, 12)
    config.max_page_n = max_page_n

    with nogil:
        ret = pco_standalone_simple_compress_into(
            <const void*> src.data,
            size,
            pcotype,
            &config,
            <void*> dst._data,
            <size_t> dstsize,
            &n_written,
        )

    if ret != PcoSuccess:
        raise PcodecError('pco_standalone_simple_compress_into', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> n_written, outgiven)


def pcodec_decode(
    data,
    /,
    shape=None,
    dtype=None,
    *,
    out=None
):
    """Return decoded PCODEC data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t src_len = <size_t> src.shape[0]
        unsigned char pcotype
        size_t n_written
        PcoError ret

    if data is None:
        raise ValueError('input data is None')

    if out is not None and data is out:
        raise ValueError('cannot transform in-place')

    if isinstance(out, numpy.ndarray):
        dtype = out.dtype
        shape = out.shape
    elif dtype is not None:
        dtype = numpy.dtype(dtype)
    else:
        raise ValueError('dtype not specified')

    try:
        pcotype = PCO_TYPE[(dtype.kind, dtype.itemsize)]
    except KeyError:
        raise ValueError(f'{dtype=} not supported')

    if shape is None:
        raise ValueError('shape not specified')
    if isinstance(shape, int | numpy.integer):
        shape = (int(shape),)
    else:
        shape = tuple(shape)

    out = _create_array(out, shape, dtype)
    dst = out

    with nogil:
        ret = pco_standalone_simple_decompress_into(
            <const void*> src._data,
            src_len,
            pcotype,
            <void*> dst.data,
            <size_t> dst.size,
            &n_written,
        )

    if ret != PcoSuccess:
        raise PcodecError('pco_standalone_simple_decompress_into', ret)

    if n_written != <size_t> dst.size:
        raise ValueError(
            f'decompressed element count mismatch: {n_written=} != {dst.size=}'
        )

    if not dtype.isnative:
        # library writes native byte order
        out.byteswap(inplace=True)

    return out


PCO_TYPE = {
    ('u', 1): PCO_TYPE_U8,
    ('u', 2): PCO_TYPE_U16,
    ('u', 4): PCO_TYPE_U32,
    ('u', 8): PCO_TYPE_U64,
    ('i', 1): PCO_TYPE_I8,
    ('i', 2): PCO_TYPE_I16,
    ('i', 4): PCO_TYPE_I32,
    ('i', 8): PCO_TYPE_I64,
    ('f', 2): PCO_TYPE_F16,
    ('f', 4): PCO_TYPE_F32,
    ('f', 8): PCO_TYPE_F64,
}
