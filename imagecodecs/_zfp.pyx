# imagecodecs/_zfp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2019-2026, Christoph Gohlke
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

"""ZFP codec for the imagecodecs package."""

include '_shared.pxi'

from zfp cimport *


class ZFP:
    """ZFP codec constants."""

    available = True

    class EXEC(enum.IntEnum):
        """ZFP codec execution policies."""

        SERIAL = zfp_exec_serial
        OMP = zfp_exec_omp
        CUDA = zfp_exec_cuda

    class MODE(enum.IntEnum):
        """ZFP codec compression modes."""

        NONE = zfp_mode_null
        EXPERT = zfp_mode_expert  # c
        FIXED_RATE = zfp_mode_fixed_rate  # r
        FIXED_PRECISION = zfp_mode_fixed_precision  # p
        FIXED_ACCURACY = zfp_mode_fixed_accuracy  # a
        REVERSIBLE = zfp_mode_reversible  # R
        # alias
        RATE = zfp_mode_fixed_rate
        PRECISION = zfp_mode_fixed_precision
        ACCURACY = zfp_mode_fixed_accuracy

    class HEADER(enum.IntEnum):
        """ZFP codec header types."""

        MAGIC = ZFP_HEADER_MAGIC
        META = ZFP_HEADER_META
        MODE = ZFP_HEADER_MODE
        FULL = ZFP_HEADER_FULL


class ZfpError(RuntimeError):
    """ZFP codec exceptions."""


def zfp_version():
    """Return ZFP library version string."""
    return 'zfp ' + ZFP_VERSION_STRING.decode()


def zfp_check(const uint8_t[::1] data not None, /):
    """Return whether data is ZFP encoded or None if unknown."""
    return _startswith(data, b'zfp')


def zfp_encode(
    data,
    /,
    level=None,
    *,
    mode=None,
    execution=None,
    chunksize=None,
    header=True,
    numthreads=None,
    out=None,
):
    """Return ZFP encoded data."""
    cdef:
        numpy.ndarray src = _array_input(data, out, dtype=None, contig=False)
        const uint8_t[::1] dst  # must be const to write to bytes
        void* dstptr = NULL
        size_t byteswritten
        size_t maxsize
        ssize_t dstsize
        bitstream* stream = NULL
        zfp_stream* zfp = NULL
        zfp_field* field = NULL
        zfp_type ztype
        zfp_mode zmode
        zfp_exec_policy zexec
        uint ndim = src.ndim
        ssize_t itemsize = src.itemsize
        uint precision = 0
        uint threads = _default_threads(numthreads)
        uint chunk_size = 0
        uint minbits = 0
        uint maxbits = 0
        uint maxprec = 0
        int minexp = 0
        size_t nx, ny, nz, nw
        ptrdiff_t sx, sy, sz, sw
        zfp_bool ret
        double tolerance = 0.0
        double rate = 0.0
        bint bheader = header

    if src.size == 0:
        raise ValueError('src is empty')

    if src.dtype == numpy.int32:
        ztype = zfp_type_int32
    elif src.dtype == numpy.int64:
        ztype = zfp_type_int64
    elif src.dtype == numpy.float32:
        ztype = zfp_type_float
    elif src.dtype == numpy.float64:
        ztype = zfp_type_double
    else:
        raise ValueError('data type not supported by ZFP')

    if ndim == 1:
        nx = <size_t> src.shape[0]
        sx = <ptrdiff_t> (src.strides[0] // itemsize)
    elif ndim == 2:
        ny = <size_t> src.shape[0]
        nx = <size_t> src.shape[1]
        sy = <ptrdiff_t> (src.strides[0] // itemsize)
        sx = <ptrdiff_t> (src.strides[1] // itemsize)
    elif ndim == 3:
        nz = <size_t> src.shape[0]
        ny = <size_t> src.shape[1]
        nx = <size_t> src.shape[2]
        sz = <ptrdiff_t> (src.strides[0] // itemsize)
        sy = <ptrdiff_t> (src.strides[1] // itemsize)
        sx = <ptrdiff_t> (src.strides[2] // itemsize)
    elif ndim == 4:
        nw = <size_t> src.shape[0]
        nz = <size_t> src.shape[1]
        ny = <size_t> src.shape[2]
        nx = <size_t> src.shape[3]
        sw = <ptrdiff_t> (src.strides[0] // itemsize)
        sz = <ptrdiff_t> (src.strides[1] // itemsize)
        sy = <ptrdiff_t> (src.strides[2] // itemsize)
        sx = <ptrdiff_t> (src.strides[3] // itemsize)
    else:
        raise ValueError('data shape not supported by ZFP')

    zmode = _enum_value(mode, ZFP.MODE, zfp_mode_reversible)

    if zmode == zfp_mode_fixed_precision:
        precision = _default_value(level, ZFP_MAX_PREC, 0, ZFP_MAX_PREC)
    elif zmode == zfp_mode_fixed_rate:
        if level is None:
            raise ValueError('fixed-rate mode requires level')
        rate = level
    elif zmode == zfp_mode_fixed_accuracy:
        if level is None:
            raise ValueError('fixed-accuracy mode requires level')
        tolerance = level
    elif zmode == zfp_mode_expert:
        if level is None:
            raise ValueError(
                'expert mode requires '
                'level=(minbits, maxbits, maxprec, minexp)'
            )
        minbits, maxbits, maxprec, minexp = level

    if execution is None:
        zexec = zfp_exec_omp if threads > 1 else zfp_exec_serial
    else:
        zexec = _enum_value(execution, ZFP.EXEC)

    if zexec == zfp_exec_omp:
        chunk_size = <uint> _default_value(chunksize, 0, 0, None)

    try:
        zfp = zfp_stream_open(NULL)
        if zfp == NULL:
            raise ZfpError('zfp_stream_open failed')

        if ndim == 1:
            field = zfp_field_1d(<void*> src.data, ztype, nx)
            if field == NULL:
                raise ZfpError('zfp_field_1d failed')
            zfp_field_set_stride_1d(field, sx)
        elif ndim == 2:
            field = zfp_field_2d(<void*> src.data, ztype, nx, ny)
            if field == NULL:
                raise ZfpError('zfp_field_2d failed')
            zfp_field_set_stride_2d(field, sx, sy)
        elif ndim == 3:
            field = zfp_field_3d(<void*> src.data, ztype, nx, ny, nz)
            if field == NULL:
                raise ZfpError('zfp_field_3d failed')
            zfp_field_set_stride_3d(field, sx, sy, sz)
        elif ndim == 4:
            field = zfp_field_4d(<void*> src.data, ztype, nx, ny, nz, nw)
            if field == NULL:
                raise ZfpError('zfp_field_4d failed')
            zfp_field_set_stride_4d(field, sx, sy, sz, sw)

        if zmode == zfp_mode_reversible:
            zfp_stream_set_reversible(zfp)
        elif zmode == zfp_mode_fixed_precision:
            precision = zfp_stream_set_precision(zfp, precision)
        elif zmode == zfp_mode_fixed_rate:
            rate = zfp_stream_set_rate(zfp, rate, ztype, ndim, 0)
        elif zmode == zfp_mode_fixed_accuracy:
            tolerance = zfp_stream_set_accuracy(zfp, tolerance)
        elif zmode == zfp_mode_expert:
            ret = zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp)
            if ret == zfp_false:
                raise ZfpError('zfp_stream_set_params failed')

        maxsize = _zfp_maximum_size(zfp, field)
        if maxsize == 0:
            raise ZfpError('zfp_stream_maximum_size failed')

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            if dstsize < 0:
                dstsize = <ssize_t> maxsize
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]

        # zfp writes whole 64-bit words without bounds checking
        if dstsize < <ssize_t> maxsize:
            raise ValueError(f'output too small {dstsize=} < {maxsize=}')
        dstptr = <void*> &dst[0]
        if (<size_t> dstptr) % 8 != 0:
            raise ValueError('output is not 8-byte aligned')

        with nogil:
            stream = stream_open(dstptr, maxsize)
            if stream == NULL:
                raise ZfpError('stream_open failed')

            zfp_stream_set_bit_stream(zfp, stream)
            zfp_stream_rewind(zfp)

            ret = zfp_stream_set_execution(zfp, zexec)
            if ret == zfp_false:
                raise ZfpError('zfp_stream_set_execution failed')

            if zexec == zfp_exec_omp:
                if threads > 1:
                    ret = zfp_stream_set_omp_threads(zfp, threads)
                    if ret == zfp_false:
                        raise ZfpError('zfp_stream_set_omp_threads failed')
                if chunk_size > 0:
                    ret = zfp_stream_set_omp_chunk_size(zfp, chunk_size)
                    if ret == zfp_false:
                        raise ZfpError('zfp_stream_set_omp_chunk_size failed')

            byteswritten = zfp_write_header(
                zfp,
                field,
                ZFP_HEADER_FULL
                if bheader else ZFP_HEADER_MAGIC | ZFP_HEADER_MODE
            )
            if byteswritten == 0:
                raise ZfpError('zfp_write_header failed')

            byteswritten = zfp_compress(zfp, field)
            if byteswritten == 0:
                raise ZfpError('zfp_compress failed')

    finally:
        if field != NULL:
            zfp_field_free(field)
        if zfp != NULL:
            zfp_stream_close(zfp)
        if stream != NULL:
            stream_close(stream)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def zfp_decode(
    data,
    /,
    *,
    shape=None,
    dtype=None,
    strides=None,
    numthreads=None,
    out=None,
    bint _info=False,
):
    """Return decoded ZFP data."""
    cdef:
        numpy.ndarray dst
        uint8_t* buffer = NULL
        const uint8_t[::1] src = data
        const uint8_t* srcptr = NULL
        ssize_t srcsize = src.shape[0]
        uint64_t[4] header  # ZFP_HEADER_MAX_BITS fit in 3 words
        zfp_stream* zfp = NULL
        bitstream* stream = NULL
        zfp_field* field = NULL
        zfp_type ztype
        ssize_t ndim
        size_t size, headerbits, maxsize, bufsize
        # uint threads = _default_threads(numthreads)
        size_t nx, ny, nz, nw
        ptrdiff_t sx = 0
        ptrdiff_t sy = 0
        ptrdiff_t sz = 0
        ptrdiff_t sw = 0

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize == 0:
        raise ValueError('src is empty')

    if dtype is None:
        ztype = zfp_type_none
    else:
        dtype = numpy.dtype(dtype)
        if dtype == numpy.int32:
            ztype = zfp_type_int32
        elif dtype == numpy.int64:
            ztype = zfp_type_int64
        elif dtype == numpy.float32:
            ztype = zfp_type_float
        elif dtype == numpy.float64:
            ztype = zfp_type_double
        else:
            raise ValueError('dtype not supported by ZFP')

    count = 0
    if shape is None:
        ndim = -1
        if strides is not None:
            raise ValueError('strides require shape')
    else:
        shape = tuple(int(i) for i in shape)
        ndim = len(shape)
        if ndim < 1 or ndim > 4:
            raise ValueError('shape not supported by ZFP')
        if any(i < 1 for i in shape):
            raise ValueError(f'invalid {shape=!r}')
        count = 1
        for i in shape:
            count *= i
        if strides is not None:
            strides = tuple(int(i) for i in strides)
            if len(strides) != ndim:
                raise ValueError(f'{strides=!r} do not match {shape=!r}')
            # element strides into contiguous output; 0 means default
            lo = hi = 0
            n = 1
            for i, s in zip(reversed(shape), reversed(strides)):
                if s == 0:
                    s = n
                if s > 0:
                    hi += (i - 1) * s
                else:
                    lo += (i - 1) * s
                n *= i
            if lo < 0 or hi >= count:
                raise ValueError(f'{strides=!r} exceed output array')

    if ndim == -1:
        pass
    elif ndim == 1:
        nx = <size_t> shape[0]
        if strides is not None:
            sx = <ptrdiff_t> strides[0]
    elif ndim == 2:
        nx = <size_t> shape[1]
        ny = <size_t> shape[0]
        if strides is not None:
            sx = <ptrdiff_t> strides[1]
            sy = <ptrdiff_t> strides[0]
    elif ndim == 3:
        nx = <size_t> shape[2]
        ny = <size_t> shape[1]
        nz = <size_t> shape[0]
        if strides is not None:
            sx = <ptrdiff_t> strides[2]
            sy = <ptrdiff_t> strides[1]
            sz = <ptrdiff_t> strides[0]
    elif ndim == 4:
        nx = <size_t> shape[3]
        ny = <size_t> shape[2]
        nz = <size_t> shape[1]
        nw = <size_t> shape[0]
        if strides is not None:
            sx = <ptrdiff_t> strides[3]
            sy = <ptrdiff_t> strides[2]
            sz = <ptrdiff_t> strides[1]
            sw = <ptrdiff_t> strides[0]

    # TODO: enable execution mode or threading when supported
    # if execution is None:
    #     zexec = zfp_exec_omp if threads > 1 else zfp_exec_serial
    # elif execution == zfp_exec_serial or execution == 'serial':
    #     zexec = zfp_exec_serial
    #     threads = 1
    # elif execution == zfp_exec_omp or execution == 'omp':
    #     zexec = zfp_exec_serial if threads == 1 else zfp_exec_omp
    # elif execution == zfp_exec_cuda or execution == 'cuda':
    #     zexec = zfp_exec_cuda
    # else:
    #     raise ValueError('invalid ZFP execution policy')

    try:
        zfp = zfp_stream_open(NULL)
        if zfp == NULL:
            raise ZfpError('zfp_stream_open failed')

        field = zfp_field_alloc()
        if field == NULL:
            raise ZfpError('zfp_field_alloc failed')

        # read header from zero-padded copy to not read past end of src
        memset(<void*> header, 0, 32)
        memcpy(<void*> header, <const void*> &src[0], min(srcsize, 32))
        stream = stream_open(<void*> header, 32)
        if stream == NULL:
            raise ZfpError('stream_open failed')

        zfp_stream_set_bit_stream(zfp, stream)
        zfp_stream_rewind(zfp)

        # ret = zfp_stream_set_execution(zfp, zexec)
        # if ret == zfp_false:
        #     raise ZfpError('zfp_stream_set_execution failed')

        headerbits = zfp_read_header(zfp, field, ZFP_HEADER_FULL)
        if ztype != zfp_type_none and ndim != -1 and (
            headerbits == 0
            or field.dtype != ztype
            or zfp_field_size(field, NULL) != count
        ):
            # data were probably encoded without metadata
            zfp_stream_rewind(zfp)
            headerbits = zfp_read_header(
                zfp, field, ZFP_HEADER_MAGIC | ZFP_HEADER_MODE
            )
        if headerbits == 0:
            raise ZfpError('zfp_read_header failed')

        stream_close(stream)
        stream = NULL

        if ztype == zfp_type_none:
            ztype = field.dtype
            if ztype == zfp_type_float:
                dtype = numpy.float32
            elif ztype == zfp_type_double:
                dtype = numpy.float64
            elif ztype == zfp_type_int32:
                dtype = numpy.int32
            elif ztype == zfp_type_int64:
                dtype = numpy.int64
            else:
                raise ZfpError('invalid zfp_field type')
        else:
            ztype = zfp_field_set_type(field, ztype)
            if ztype == zfp_type_none:
                raise ZfpError('zfp_field_set_type failed')

        if ndim == -1:
            if field.nx == 0:
                raise ZfpError('invalid zfp_field nx')
            elif field.ny == 0:
                shape = (field.nx,)
            elif field.nz == 0:
                shape = (field.ny, field.nx)
            elif field.nw == 0:
                shape = (field.nz, field.ny, field.nx)
            else:
                shape = (field.nw, field.nz, field.ny, field.nx)
        elif ndim == 1:
            zfp_field_set_size_1d(field, nx)
            zfp_field_set_stride_1d(field, sx)
        elif ndim == 2:
            zfp_field_set_size_2d(field, nx, ny)
            zfp_field_set_stride_2d(field, sx, sy)
        elif ndim == 3:
            zfp_field_set_size_3d(field, nx, ny, nz)
            zfp_field_set_stride_3d(field, sx, sy, sz)
        elif ndim == 4:
            zfp_field_set_size_4d(field, nx, ny, nz, nw)
            zfp_field_set_stride_4d(field, sx, sy, sz, sw)

        maxsize = _zfp_maximum_size(zfp, field)
        if maxsize == 0:
            raise ZfpError('zfp_stream_maximum_size failed')

        if _info:
            return {
                '_codec': 'zfp',
                'shape': shape,
                'dtype': numpy.dtype(dtype),
                'maxsize': maxsize,
                **_zfp_info(zfp, field),
            }

        out = _create_array(out, shape, dtype)
        dst = out

        with nogil:
            # zfp reads whole 64-bit words without bounds checking
            if (
                <size_t> srcsize >= maxsize
                and (<size_t> <const void*> &src[0]) % 8 == 0
            ):
                srcptr = &src[0]
                bufsize = <size_t> srcsize
            else:
                bufsize = (max(maxsize, <size_t> srcsize) + 7) // 8 * 8
                buffer = <uint8_t*> malloc(bufsize)
                if buffer == NULL:
                    raise MemoryError('failed to allocate buffer')
                memcpy(<void*> buffer, <const void*> &src[0], srcsize)
                memset(<void*> (buffer + srcsize), 0, bufsize - srcsize)
                srcptr = buffer

            stream = stream_open(<void*> srcptr, bufsize)
            if stream == NULL:
                raise ZfpError('stream_open failed')

            zfp_stream_set_bit_stream(zfp, stream)
            zfp_stream_rewind(zfp)
            stream_skip(stream, headerbits)

            zfp_field_set_pointer(field, <void*> dst.data)
            size = zfp_decompress(zfp, field)

            if size == 0:
                raise ZfpError('zfp_decompress failed')

    finally:
        if field != NULL:
            zfp_field_free(field)
        if zfp != NULL:
            zfp_stream_close(zfp)
        if stream != NULL:
            stream_close(stream)
        free(buffer)

    return out


def zfp_info(
    data,
    /,
    *,
    shape=None,
    dtype=None,
    strides=None,
):
    """Return information about ZFP encoded data."""
    return zfp_decode(
        data,
        shape=shape,
        dtype=dtype,
        strides=strides,
        _info=True,
    )


cdef dict _zfp_info(const zfp_stream* zfp, const zfp_field* field):
    """Return compression parameters of ZFP stream."""
    cdef:
        zfp_mode zmode = zfp_stream_compression_mode(zfp)
        uint minbits, maxbits, maxprec
        int minexp

    zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp)
    if zmode == zfp_mode_fixed_rate:
        level = zfp_stream_rate(zfp, zfp_field_dimensionality(field))
    elif zmode == zfp_mode_fixed_precision:
        level = zfp_stream_precision(zfp)
    elif zmode == zfp_mode_fixed_accuracy:
        level = zfp_stream_accuracy(zfp)
    elif zmode == zfp_mode_expert:
        level = (minbits, maxbits, maxprec, minexp)
    else:
        level = None
    return {
        'mode': ZFP.MODE(zmode),
        'level': level,
        'minbits': minbits,
        'maxbits': maxbits,
        'maxprec': maxprec,
        'minexp': minexp,
    }


cdef size_t _zfp_maximum_size(
    const zfp_stream* zfp,
    const zfp_field* field
) noexcept nogil:
    """Return zfp_stream_maximum_size or 0 if it overflows size_t."""
    # zfp computes blocks * maxbits in size_t, which overflows on 32-bit
    cdef:
        uint64_t maxbits = (
            (<uint64_t> 1 << (2 * zfp_field_dimensionality(field))) * 66 + 20
        )

    if maxbits > zfp.maxbits:
        maxbits = zfp.maxbits
    if maxbits < zfp.minbits:
        maxbits = zfp.minbits
    if (
        <uint64_t> zfp_field_blocks(field) * maxbits + ZFP_HEADER_MAX_BITS + 64
        > <uint64_t> SIZE_MAX
    ):
        return 0
    return zfp_stream_maximum_size(zfp, field)
