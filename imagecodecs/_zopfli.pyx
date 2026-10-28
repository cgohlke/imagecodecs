# imagecodecs/_zopfli.pyx
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

"""ZOPFLI codec for the imagecodecs package."""

include '_shared.pxi'

from zlib cimport adler32, crc32, uInt, uLong

from cpython.exc cimport PyErr_CheckSignals
from zopfli cimport *


class ZOPFLI:
    """ZOPFLI codec constants."""

    available = True

    class FORMAT(enum.IntEnum):
        """ZOPFLI codec formats."""

        GZIP = ZOPFLI_FORMAT_GZIP
        ZLIB = ZOPFLI_FORMAT_ZLIB
        DEFLATE = ZOPFLI_FORMAT_DEFLATE


class ZopfliError(RuntimeError):
    """ZOPFLI codec exceptions."""


def zopfli_version():
    """Return Zopfli library version string."""
    return 'zopfli 1.0.3'


# zopfli_check = zlib_check
# zopfli_decode = zlib_decode


cdef size_t ZOPFLI_MASTER_BLOCK_SIZE = 1000000


def zopfli_encode(
    data,
    /,
    level=None,  # numiterations
    *,
    format=None,
    blocksplitting=None,
    blocksplittingmax=None,
    verbose=None,
    verbose_more=None,
    out=None,
):
    """Return ZOPFLI encoded data.

    Only the ZLIB format can be decoded with ``zopfli_decode``.
    Zopfli terminates the process if memory allocation fails.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        const unsigned char* srcptr = <const unsigned char*> src._data
        size_t srcsize = <size_t> src.shape[0]
        ssize_t dstsize
        size_t outsize, pos, end
        size_t deflatesize = 0
        size_t headersize = 0
        size_t trailersize = 0
        uLong checksum
        uInt n
        unsigned char header[10]
        unsigned char trailer[8]
        unsigned char bp = 0
        unsigned char* buffer = NULL
        int final
        ZopfliOptions options
        ZopfliFormat format_

    format_ = <ZopfliFormat> <int> _enum_value(
        format, ZOPFLI.FORMAT, ZOPFLI.FORMAT.ZLIB
    )
    checksum = 1 if format_ == ZOPFLI_FORMAT_ZLIB else 0

    out, dstsize, outgiven, outtype = _parse_output(out)

    ZopfliInitOptions(&options)
    if level is not None:
        options.numiterations = _default_value(level, 15, 1, INT_MAX)
    if blocksplitting is not None:
        options.blocksplitting = bool(blocksplitting)
    if blocksplittingmax is not None:
        options.blocksplittingmax = _default_value(
            blocksplittingmax, 15, 0, INT_MAX
        )
    if verbose is not None:
        options.verbose = bool(verbose)
    if verbose_more is not None:
        options.verbose_more = bool(verbose_more)

    try:
        with nogil:
            if format_ != ZOPFLI_FORMAT_DEFLATE:
                # zlib checksum functions take 32-bit lengths
                pos = 0
                while pos < srcsize:
                    n = <uInt> min(srcsize - pos, <size_t> UINT32_MAX)
                    if format_ == ZOPFLI_FORMAT_ZLIB:
                        checksum = adler32(checksum, srcptr + pos, n)
                    else:
                        checksum = crc32(checksum, srcptr + pos, n)
                    pos += n

            if format_ == ZOPFLI_FORMAT_ZLIB:
                # CMF=0x78, FLEVEL=3, as in ZopfliZlibCompress
                header[0] = 0x78
                header[1] = 0xDA
                headersize = 2
                trailer[0] = (checksum >> 24) & 0xFF
                trailer[1] = (checksum >> 16) & 0xFF
                trailer[2] = (checksum >> 8) & 0xFF
                trailer[3] = checksum & 0xFF
                trailersize = 4
            elif format_ == ZOPFLI_FORMAT_GZIP:
                # no MTIME, XFL=2, OS=3, as in ZopfliGzipCompress
                memcpy(
                    header, b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03', 10
                )
                headersize = 10
                trailer[0] = checksum & 0xFF
                trailer[1] = (checksum >> 8) & 0xFF
                trailer[2] = (checksum >> 16) & 0xFF
                trailer[3] = (checksum >> 24) & 0xFF
                trailer[4] = srcsize & 0xFF
                trailer[5] = (srcsize >> 8) & 0xFF
                trailer[6] = (srcsize >> 16) & 0xFF
                trailer[7] = (srcsize >> 24) & 0xFF
                trailersize = 8

            # replaces ZopfliCompress to allow KeyboardInterrupt between blocks
            pos = 0
            while True:
                if srcsize - pos <= ZOPFLI_MASTER_BLOCK_SIZE:
                    end = srcsize
                else:
                    end = pos + ZOPFLI_MASTER_BLOCK_SIZE
                final = end == srcsize
                ZopfliDeflatePart(
                    &options,
                    2,  # dynamic block
                    final,
                    srcptr,
                    pos,
                    end,
                    &bp,
                    &buffer,
                    &deflatesize
                )
                if final:
                    break
                pos = end
                with gil:
                    # check KeyboardInterrupt
                    PyErr_CheckSignals()

            if buffer == NULL:
                raise ZopfliError('ZopfliDeflatePart returned NULL')

        outsize = headersize + deflatesize + trailersize
        if out is None:
            if dstsize >= 0 and dstsize < <ssize_t> outsize:
                raise ZopfliError('output too small')
            dstsize = <ssize_t> outsize
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]
        if dstsize < <ssize_t> outsize:
            raise ZopfliError('output too small')
        memcpy(<void*> dst._data, <const void*> header, headersize)
        memcpy(
            <void*> (dst._data + headersize),
            <const void*> buffer,
            deflatesize
        )
        memcpy(
            <void*> (dst._data + headersize + deflatesize),
            <const void*> trailer,
            trailersize
        )
        del dst
    finally:
        free(buffer)

    return _return_output(out, dstsize, outsize, outgiven)
