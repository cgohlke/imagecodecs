# -*- coding: utf-8 -*-
# _imagecodecs.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2008-2019, Christoph Gohlke
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

"""Image transformation, compression, and decompression codecs.

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in the
tifffile, czifile, and other scientific imaging modules.

Decode and/or encode functions are implemented for Zlib (DEFLATE),
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZW, LZF, ZFP, AEC,
NPY, PNG, WebP, JPEG 8-bit, JPEG 12-bit, JPEG SOF3, JPEG 2000, JPEG LS,
JPEG XR, JPEG XL, PackBits, Packed Integers, Delta, XOR Delta,
Floating Point Predictor, Bitorder reversal, and Bitshuffle.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.16

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 2.7.17, 3.5.4, 3.6.8, 3.7.5, 3.8.0 64-bit <https://www.python.org>`_
* `Numpy 1.16.5 <https://www.numpy.org>`_
* `Cython 0.29.14 <https://cython.org>`_
* `zlib 1.2.11 <https://github.com/madler/zlib>`_
* `lz4 1.9.2 <https://github.com/lz4/lz4>`_
* `zstd 1.4.4 <https://github.com/facebook/zstd>`_
* `blosc 1.17.1 <https://github.com/Blosc/c-blosc>`_
* `bzip2 1.0.8 <https://sourceware.org/bzip2>`_
* `liblzma 5.2.4 <https://github.com/xz-mirror/xz>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
* `libwebp 1.0.3 <https://github.com/webmproject/libwebp>`_
* `libjpeg-turbo 2.0.3 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
  (8 and 12-bit)
* `charls 2.1.0 <https://github.com/team-charls/charls>`_
* `openjpeg 2.3.1 <https://github.com/uclouvain/openjpeg>`_
* `jxrlib 1.1 <https://packages.debian.org/source/sid/jxrlib>`_
* `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `libaec 1.0.4 <https://gitlab.dkrz.de/k202009/libaec>`_
* `snappy 1.1.7 <https://github.com/google/snappy>`_
* `zopfli-1.0.3 <https://github.com/google/zopfli>`_
* `brotli 1.0.7 <https://github.com/google/brotli>`_
* `brunsli 0.1 <https://github.com/google/brunsli>`_
* `lcms 2.9 <https://github.com/mm2/Little-CMS>`_

Required Python packages for testing (other versions may work):

* `tifffile 2019.7.26 <https://pypi.org/project/tifffile/>`_
* `czifile 2019.7.2 <https://pypi.org/project/czifile/>`_
* `python-blosc 1.8.3 <https://github.com/Blosc/python-blosc>`_
* `python-lz4 2.2.1 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd 1.4.4 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
* `python-brotli 1.0.7 <https://github.com/google/brotli/tree/master/python>`_
* `python-snappy 0.5.4 <https://github.com/andrix/python-snappy>`_
* `zopflipy 1.3 <https://github.com/hattya/zopflipy>`_
* `backports.lzma 0.0.14 <https://github.com/peterjc/backports.lzma>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_

Notes
-----
The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 2.7, 3.5, and 32-bit are deprecated.

The `Microsoft Visual C++ Redistributable Packages
<https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ are required on Windows.

Refer to the imagecodecs/licenses folder for 3rd party library licenses.

This software is based in part on the work of the Independent JPEG Group.

This software includes modified versions of `dcm2niix's jpg_0XC3.cpp
<https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_
and `OpenJPEG's color.c
<https://github.com/uclouvain/openjpeg/blob/master/src/bin/common/color.c>`_.

Build instructions and wheels for manylinux and macOS courtesy of
`Grzegorz Bokota <https://github.com/Czaki>`_.

To install the requirements for building imagecodecs from source code on
latest Ubuntu Linux distributions, run:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy
    python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-turbo8-dev libjxr-dev
    liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev
    libzopfli-dev``

The imagecodecs package can be challenging to build from source code. Consider
using the `imagecodecs-lite <https://pypi.org/project/imagecodecs-lite/>`_
package instead, which does not depend on external third-party C libraries
and provides a subset of image codecs for the tifffile library:
LZW, PackBits, Delta, XOR Delta, Packed Integers, Floating Point Predictor,
and Bitorder reversal.

Other Python packages providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `packbits <https://github.com/psd-tools/packbits>`_
* `fpzip <https://github.com/seung-lab/fpzip>`_

Revisions
---------
2019.12.16
    Pass 3287 tests.
    Add Zopfli codec.
    Add Snappy codec.
    Rename j2k codec to jpeg2k.
    Rename jxr codec to jpegxr.
    Use Debian's jxrlib.
    Support pathlib and binary streams in imread and imwrite.
    Move external C declarations to pxd files.
    Move shared code to pxi file.
    Update copyright notices.
2019.12.10
    Add version functions.
    Add Brotli codec (WIP).
    Add optional JPEG XL codec via Brunsli repacker (WIP).
2019.12.3
    Sync with imagecodecs-lite.
2019.11.28
    Add AEC codec via libaec (WIP).
    Do not require scikit-image for testing.
    Require CharLS 2.1.
2019.11.18
    Add bitshuffle codec.
    Fix formatting of unknown error numbers.
    Fix test failures with official python-lzf.
2019.11.5
    Rebuild with updated dependencies.
2019.5.22
    Add optional YCbCr chroma subsampling to JPEG encoder.
    Add default reversible mode to ZFP encoder.
    Add imread and imwrite helper functions.
2019.4.20
    Fix setup requirements.
2019.2.22
    Move codecs without 3rd-party C library dependencies to imagecodecs_lite.
2019.2.20
    Rebuild with updated dependencies.
2019.1.20
    Add more pixel formats to JPEG XR codec.
    Add JPEG XR encoder.
2019.1.14
    Add optional ZFP codec via zfp library (WIP).
    Add numpy NPY and NPZ codecs.
    Fix some static codechecker errors.
2019.1.1
    Update copyright year.
    Do not install package if Cython extension fails to build.
    Fix compiler warnings.
2018.12.16
    Reallocate LZW buffer on demand.
    Ignore integer type output arguments for codecs returning images.
2018.12.12
    Enable decoding of subsampled J2K images via conversion to RGB.
    Enable decoding of large JPEG using patched libjpeg-turbo.
    Switch to Cython 0.29, language_level=3.
2018.12.1
    Add J2K encoder (WIP).
    Use ZStd content size 1 MB if it cannot be determined.
    Use logging.warning instead of warnings.warn or print.
2018.11.8
    Decode LSB style LZW.
    Fix last byte not written by LZW decoder (bug fix).
    Permit unknown colorspaces in JPEG codecs (e.g. CFA used in TIFF).
2018.10.30
    Add JPEG 8-bit and 12-bit encoders.
    Improve color space handling in JPEG codecs.
2018.10.28
    Rename jpeg0xc3 to jpegsof3.
    Add optional JPEG LS codec via CharLS.
    Fix missing alpha values in jxr_decode.
    Fix decoding JPEG SOF3 with multiple DHTs.
2018.10.22
    Add Blosc codec via libblosc.
2018.10.21
    Builds on Ubuntu 18.04 WSL.
    Include liblzf in srcdist.
    Do not require CreateDecoderFromBytes patch to jxrlib.
2018.10.18
    Improve jpeg_decode wrapper.
2018.10.17
    Add JPEG SOF3 decoder based on jpg_0XC3.cpp.
2018.10.10
    Add PNG codec via libpng.
    Add option to specify output colorspace in JPEG decoder.
    Fix Delta codec for floating point numbers.
    Fix XOR Delta codec.
2018.9.30
    Add LZF codec via liblzf.
2018.9.22
    Add WebP codec via libwebp.
2018.8.29
    Add PackBits encoder.
2018.8.22
    Add link library version information.
    Add option to specify size of LZW buffer.
    Add JPEG 2000 decoder via OpenJPEG.
    Add XOR Delta codec.
2018.8.16
    Link to libjpeg-turbo.
    Support Python 2.7 and Visual Studio 2008.
2018.8.10
    Initial alpha release.
    Add LZW, PackBits, PackInts and FloatPred decoders from tifffile.c module.
    Add JPEG and JPEG XR decoders from czifile.pyx module.

"""

__version__ = '2019.12.16'

include '_imagecodecs.pxi'

from cython.operator cimport dereference as deref

from libc.math cimport ceil
from libc.string cimport memset, memcpy, memmove
from libc.stdlib cimport malloc, free, realloc
from libc.setjmp cimport setjmp, longjmp, jmp_buf
from libc.stdint cimport (
    int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
    UINT64_MAX)

from numpy cimport (
    PyArray_DescrNewFromType, NPY_BOOL, NPY_INTP,
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, NPY_INT128,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128)


###############################################################################

def version(astype=None):
    """Return detailed version information."""
    versions = (
        'imagecodecs %s' % __version__,
        'cython %s' % cython.__version__,
        numpy_version(),
        icd_version(),
        zlib_version(),
        lzma_version(),
        zstd_version(),
        brotli_version(),
        snappy_version(),
        zopfli_version(),
        blosc_version(),
        bz2_version(),
        lz4_version(),
        lzf_version(),
        aec_version(),
        zfp_version(),
        bitshuffle_version(),
        png_version(),
        webp_version(),
        jpeg_turbo_version(),
        jpeg8_version(),
        jpeg12_version(),
        jpegsof3_version(),
        jpegls_version(),
        jpegxl_version(),
        jpegxr_version(),
        jpeg2k_version(),
    )
    if astype is None or astype is str:
        return ', '.join(ver.replace(' ', '-') for ver in versions)
    elif astype is dict:
        return dict(ver.split(' ') for ver in versions)
    else:
        return versions


# Bitshuffle ##################################################################

from bitshuffle cimport *


class BitshuffleError(RuntimeError):
    """Bitshuffle Exceptions."""
    def __init__(self, func, err):
        msg = {
            0: 'No Error',
            -1: 'Failed to allocate memory',
            -11: 'Missing SSE',
            -12: 'Missing AVX',
            -80: 'Input size not a multiple of 8',
            -81: 'Block size not a multiple of 8',
            -91: 'Decompression error, wrong number of bytes processed',
        }.get(err, 'internal error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def bitshuffle_encode(data, level=None, itemsize=1, blocksize=0, out=None):
    """Bitshuffle.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray ndarr
        ssize_t srcsize
        ssize_t dstsize
        size_t elem_size
        size_t block_size = blocksize
        int64_t ret

    if data is out:
        raise ValueError('cannot encode in-place')

    if isinstance(data, numpy.ndarray):
        out = _create_array(out, data.shape, data.dtype)
        ndarr = out
        srcsize = data.size
        elem_size = <size_t>data.itemsize
        with nogil:
            ret = bshuf_bitshuffle(
                <void *>&src[0],
                <void *>ndarr.data,
                <size_t>srcsize,
                elem_size,
                block_size
            )
        if ret < 0:
            raise BitshuffleError('bshuf_bitshuffle', ret)
        return out

    srcsize = src.size
    elem_size = itemsize

    if elem_size != 1 and elem_size != 2 and elem_size != 4 and elem_size != 8:
        raise ValueError('invalid item size')
    if srcsize % elem_size != 0:
        raise ValueError('data size not a multiple of item size')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize < srcsize:
        raise RuntimeError('output too small')

    with nogil:
        ret = bshuf_bitshuffle(
            <void *>&src[0],
            <void *>&dst[0],
            <size_t>srcsize / elem_size,
            elem_size,
            block_size
        )
    if ret < 0:
        raise BitshuffleError('bshuf_bitshuffle', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def bitshuffle_decode(data, itemsize=1, blocksize=0, out=None):
    """Un-Bitshuffle.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray ndarr
        ssize_t srcsize
        ssize_t dstsize
        size_t elem_size
        size_t block_size = blocksize
        int64_t ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if isinstance(data, numpy.ndarray):
        out = _create_array(out, data.shape, data.dtype)
        ndarr = out
        srcsize = data.size
        elem_size = <size_t>data.itemsize
        with nogil:
            ret = bshuf_bitunshuffle(
                <void *>&src[0],
                <void *>ndarr.data,
                <size_t>srcsize,
                elem_size,
                block_size
            )
        if ret < 0:
            raise BitshuffleError('bshuf_bitunshuffle', ret)
        return out

    srcsize = src.size
    elem_size = itemsize

    if elem_size != 1 and elem_size != 2 and elem_size != 4 and elem_size != 8:
        raise ValueError('invalid item size')
    if srcsize % elem_size != 0:
        raise ValueError('data size not a multiple of item size')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize < srcsize:
        raise RuntimeError('output too small')

    with nogil:
        ret = bshuf_bitunshuffle(
            <void *>&src[0],
            <void *>&dst[0],
            <size_t>srcsize / elem_size,
            elem_size,
            block_size
        )
    if ret < 0:
        raise BitshuffleError('bshuf_bitunshuffle', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def bitshuffle_version():
    """Return Bitshuffle version string."""
    return 'bitshuffle %i.%i.%i' % (
        BSHUF_VERSION_MAJOR, BSHUF_VERSION_MINOR, BSHUF_VERSION_POINT)


# Zlib DEFLATE ################################################################

from zlib cimport *


class ZlibError(RuntimeError):
    """Zlib Exceptions."""
    def __init__(self, func, err):
        msg = {
            Z_OK: 'Z_OK',
            Z_MEM_ERROR: 'Z_MEM_ERROR',
            Z_BUF_ERROR: 'Z_BUF_ERROR',
            Z_DATA_ERROR: 'Z_DATA_ERROR',
            Z_STREAM_ERROR: 'Z_STREAM_ERROR',
            Z_NEED_DICT: 'Z_NEED_DICT',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def zlib_encode(data, level=None, out=None):
    """Compress Zlib.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)  # TODO: non-contiguous
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret
        int compresslevel = _default_value(level, 6, 0, 9)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib
        return zlib.compress(data, compresslevel)
        # TODO: use zlib streaming API
        # return _zlib_compress(src, compresslevel, outtype)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError()
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long>dstsize
    srclen = <unsigned long>srcsize

    with nogil:
        ret = compress2(
            <Bytef *>&dst[0],
            &dstlen,
            &src[0],
            srclen,
            compresslevel
        )
    if ret != Z_OK:
        raise ZlibError('compress2', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def zlib_decode(data, out=None):
    """Decompress Zlib.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib
        return zlib.decompress(data)
        # TODO: use zlib streaming API
        # return _zlib_decompress(src, outtype)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError()
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long>dstsize
    srclen = <unsigned long>srcsize

    with nogil:
        ret = uncompress2(
            <Bytef *>&dst[0],
            &dstlen,
            &src[0],
            &srclen
        )
    if ret != Z_OK:
        raise ZlibError('uncompress2', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def zlib_crc32(data):
    """Return cyclic redundancy checksum CRC-32 of data."""
    cdef:
        const uint8_t[::1] src = _parse_input(data)  # TODO: non-contiguous
        uInt srcsize = <uInt>src.size
        uLong crc = 0
    with nogil:
        crc = crc32(crc, NULL, 0)
        crc = crc32(crc, <Bytef *>&src[0], srcsize)
    return int(crc)


def zlib_version():
    """Return zlib version string."""
    return 'zlib ' + zlibVersion().decode('utf-8')



# Zopfli ######################################################################

from zopfli cimport *

# _add_globals(
#     ZOPFLI_FORMAT_GZIP=ZOPFLI_FORMAT_GZIP,
#     ZOPFLI_FORMAT_ZLIB=ZOPFLI_FORMAT_ZLIB,
#     ZOPFLI_FORMAT_DEFLATE=ZOPFLI_FORMAT_DEFLATE)


class ZopfliError(RuntimeError):
    """Zopfli Exceptions."""


def zopfli_encode(data, level=None, out=None, **kwargs):
    """Compress Zlib format using Zopfli.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)  # TODO: non-contiguous
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t outsize = 0
        ZopfliOptions options
        ZopfliFormat format = ZOPFLI_FORMAT_ZLIB
        unsigned char* buffer = NULL

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    ZopfliInitOptions(&options)
    if kwargs:
        if 'format' in kwargs:
            format = <ZopfliFormat><int>(
                _default_value(kwargs['format'], 1, 0, 2))
        if 'verbose' in kwargs:
            options.verbose = bool(kwargs['verbose'])
        if 'verbose_more' in kwargs:
            options.verbose_more = bool(kwargs['verbose_more'])
        if 'numiterations' in kwargs:
            options.numiterations = _default_value(
                kwargs['numiterations'], 15, 1, 255)
        if 'blocksplitting' in kwargs:
            options.blocksplitting = bool(kwargs['blocksplitting'])
        if 'blocksplittingmax' in kwargs:
            options.blocksplittingmax = _default_value(
                kwargs['blocksplittingmax'], 15, 0, 2**15)

    with nogil:
        ZopfliCompress(
            &options,
            format,
            <const unsigned char*>&src[0],
            <size_t>srcsize,
            &buffer,
            &outsize
        )
    if buffer == NULL:
        raise ZopfliError('ZopfliCompress returned NULL')

    try:
        if out is None:
            if dstsize >= 0 and dstsize < <ssize_t>outsize:
                raise RuntimeError('output too small')
            dstsize = <ssize_t>outsize
            out = _create_output(outtype, dstsize, <const char*>buffer)
        else:
            dst = out
            dstsize = dst.size
            if dstsize < <ssize_t>outsize:
                raise RuntimeError('output too small')
            memcpy(<void *>&dst[0], <const void *>buffer, outsize)
            del dst
    finally:
        free(buffer)

    return _return_output(out, dstsize, outsize, outgiven)


zopfli_decode = zlib_decode


def zopfli_version():
    """Return Zopfli version string."""
    return 'zopfli 1.0.3'


# ZStandard ###################################################################

from zstd cimport *


class ZstdError(RuntimeError):
    """ZStandard Exceptions."""
    def __init__(self, func, msg='', err=0):
        cdef const char *errmsg
        if msg:
            RuntimeError.__init__(self, "%s returned '%s'" % (func, msg))
        else:
            errmsg = ZSTD_getErrorName(err)
            RuntimeError.__init__(
                self,
                u"%s returned '%s'" % (func, errmsg.decode('utf-8'))
            )


def zstd_encode(data, level=None, out=None):
    """Compress ZStandard.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = src.size
        ssize_t dstsize
        size_t ret
        int compresslevel = _default_value(level, 5, 1, 22)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = <ssize_t>ZSTD_compressBound(srcsize)
            if dstsize < 0:
                raise ZstdError('ZSTD_compressBound', '%i' % dstsize)
        if dstsize < 64:
            dstsize = 64
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = ZSTD_compress(
            <void *>&dst[0],
            <size_t>dstsize,
            <void *>&src[0],
            srcsize,
            compresslevel
        )
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_compress', err=ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def zstd_decode(data, out=None):
    """Decompress ZStandard.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = <size_t>src.size
        ssize_t dstsize
        size_t ret
        uint64_t cntsize

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            cntsize = ZSTD_getFrameContentSize(<void *>&src[0], srcsize)
            if (
                cntsize == ZSTD_CONTENTSIZE_UNKNOWN or
                cntsize == ZSTD_CONTENTSIZE_ERROR
            ):
                # 1 MB; arbitrary
                cntsize = max(<uint64_t>1048576, <uint64_t>(srcsize*2))
            # TODO: use stream interface
            # if cntsize == ZSTD_CONTENTSIZE_UNKNOWN:
            #     raise ZstdError('ZSTD_getFrameContentSize',
            #                     'ZSTD_CONTENTSIZE_UNKNOWN')
            # if cntsize == ZSTD_CONTENTSIZE_ERROR:
            #     raise ZstdError('ZSTD_getFrameContentSize',
            #                     'ZSTD_CONTENTSIZE_ERROR')
            dstsize = cntsize
            if dstsize < 0:
                raise ZstdError('ZSTD_getFrameContentSize', '%i' % dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <size_t>dst.size

    with nogil:
        ret = ZSTD_decompress(
            <void *>&dst[0],
            <size_t>dstsize,
            <void *>&src[0],
            srcsize
        )
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_decompress', err=ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def zstd_version():
    """Return Zstd version string."""
    return 'zstd %i.%i.%i' % (
        ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE)


# LZ4 #########################################################################

from lz4 cimport *


class Lz4Error(RuntimeError):
    """LZ4 Exceptions."""


def lz4_encode(data, level=None, header=False, out=None):
    """Compress LZ4.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = src.size
        int dstsize
        int offset = 4 if header else 0
        int ret
        uint8_t *pdst
        int acceleration = _default_value(level, 1, 1, 1000)

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.size > LZ4_MAX_INPUT_SIZE:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(srcsize) + offset
            if dstsize < 0:
                raise Lz4Error('LZ4_compressBound returned %i' % dstsize)
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int>dst.size - offset

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_compress_fast(
            <char *>&src[0],
            <char *>&dst[offset],
            srcsize,
            dstsize,
            acceleration
        )
    if ret <= 0:
        raise Lz4Error('LZ4_compress_fast returned %i' % ret)

    if header:
        pdst = <uint8_t *>&dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def lz4_decode(data, header=False, out=None):
    """Decompress LZ4.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int>src.size
        int dstsize
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.size > 2**31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None:
        if dstsize < 0:
            dstsize = max(24, 24 + 255 * (srcsize - offset - 10))  # ugh
            if dstsize < 0:
                raise Lz4Error('invalid output size %i' % dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int>dst.size

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_decompress_safe(
            <char *>&src[offset],
            <char *>&dst[0],
            srcsize - offset,
            dstsize
        )
    if ret < 0:
        raise Lz4Error('LZ4_decompress_safe returned %i' % ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def lz4_version():
    """Return LZ4 version string."""
    return 'lz4 %i.%i.%i' % (
        LZ4_VERSION_MAJOR, LZ4_VERSION_MINOR, LZ4_VERSION_RELEASE)


# LZF #########################################################################

from liblzf cimport *


class LzfError(RuntimeError):
    """LZF Exceptions."""


def lzf_encode(data, level=None, header=False, out=None):
    """Compress LZF.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned int ret
        uint8_t *pdst
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if srcsize > 2**31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # dstsize = ((srcsize * 33) >> 5 ) + 1 + offset
            dstsize = srcsize + srcsize // 20 + 32
        else:
            dstsize += 1  # bug in liblzf ?
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size - offset

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_compress(
            <void *>&src[0],
            <unsigned int>srcsize,
            <void *>&dst[offset],
            <unsigned int>dstsize
        )
    if ret == 0:
        raise LzfError('lzf_compress returned 0')

    if header:
        pdst = <uint8_t *>&dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def lzf_decode(data, header=False, out=None):
    """Decompress LZF.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        unsigned int ret
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2**31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int>dst.size

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_decompress(
            <void *>&src[offset],
            <unsigned int>(srcsize - offset),
            <void *>&dst[0],
            <unsigned int>dstsize
        )
    if ret == 0:
        raise LzfError('lzf_decompress returned %i' % ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def lzf_version():
    """Return LibLZF version string."""
    return 'liblzf %i.%i' % (LZF_VERSION >> 8, LZF_VERSION & 255)


# LZMA ########################################################################

from liblzma cimport *


class LzmaError(RuntimeError):
    """LZMA Exceptions."""
    def __init__(self, func, err):
        msg = {
            LZMA_OK: 'LZMA_OK',
            LZMA_STREAM_END: 'LZMA_STREAM_END',
            LZMA_NO_CHECK: 'LZMA_NO_CHECK',
            LZMA_UNSUPPORTED_CHECK: 'LZMA_UNSUPPORTED_CHECK',
            LZMA_GET_CHECK: 'LZMA_GET_CHECK',
            LZMA_MEM_ERROR: 'LZMA_MEM_ERROR',
            LZMA_MEMLIMIT_ERROR: 'LZMA_MEMLIMIT_ERROR',
            LZMA_FORMAT_ERROR: 'LZMA_FORMAT_ERROR',
            LZMA_OPTIONS_ERROR: 'LZMA_OPTIONS_ERROR',
            LZMA_DATA_ERROR: 'LZMA_DATA_ERROR',
            LZMA_BUF_ERROR: 'LZMA_BUF_ERROR',
            LZMA_PROG_ERROR: 'LZMA_PROG_ERROR',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def _lzma_uncompressed_size(const uint8_t[::1] data, ssize_t size):
    """Return size of decompressed LZMA data."""
    cdef:
        lzma_ret ret
        lzma_index *index
        lzma_stream_flags options
        lzma_vli usize = 0
        uint64_t memlimit = UINT64_MAX
        ssize_t offset
        size_t pos = 0

    if size < LZMA_STREAM_HEADER_SIZE:
        raise ValueError('invalid LZMA data')
    try:
        index = lzma_index_init(NULL)
        offset = size - LZMA_STREAM_HEADER_SIZE
        ret = lzma_stream_footer_decode(&options, &data[offset])
        if ret != LZMA_OK:
            raise LzmaError('lzma_stream_footer_decode', ret)
        offset -= options.backward_size
        ret = lzma_index_buffer_decode(
            &index,
            &memlimit,
            NULL,
            &data[offset],
            &pos,
            options.backward_size
        )
        if ret != LZMA_OK or pos != options.backward_size:
            raise LzmaError('lzma_index_buffer_decode', ret)
        usize = lzma_index_uncompressed_size(index)
    finally:
        lzma_index_end(index, NULL)
    return <ssize_t>usize


def lzma_decode(data, out=None):
    """Decompress LZMA.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        lzma_ret ret
        lzma_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = _lzma_uncompressed_size(src, srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(lzma_stream))
    ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED)
    if ret != LZMA_OK:
        raise LzmaError('lzma_stream_decoder', ret)

    try:
        with nogil:
            strm.next_in = &src[0]
            strm.avail_in = <size_t>srcsize
            strm.next_out = <uint8_t*>&dst[0]
            strm.avail_out = <size_t>dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != LZMA_OK and ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzma_encode(data, level=None, out=None):
    """Compress LZMA.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        uint32_t preset = _default_value(level, 6, 0, 9)
        lzma_stream strm
        lzma_ret ret

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = lzma_stream_buffer_bound(srcsize)
            if dstsize == 0:
                raise LzmaError('lzma_stream_buffer_bound', '0')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(lzma_stream))
    ret = lzma_easy_encoder(&strm, preset, LZMA_CHECK_CRC64)
    if ret != LZMA_OK:
        raise LzmaError('lzma_easy_encoder', ret)

    try:
        with nogil:
            strm.next_in = &src[0]
            strm.avail_in = <size_t>srcsize
            strm.next_out = <uint8_t*>&dst[0]
            strm.avail_out = <size_t>dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            if ret == LZMA_OK or ret == LZMA_STREAM_END:
                ret = lzma_code(&strm, LZMA_FINISH)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzma_version():
    """Return liblzma version string."""
    return 'liblzma %i.%i.%i' % (
        LZMA_VERSION_MAJOR, LZMA_VERSION_MINOR, LZMA_VERSION_PATCH)


# BZ2 #########################################################################

from libbzip2 cimport *


class Bz2Error(RuntimeError):
    """BZ2 Exceptions."""
    def __init__(self, func, err):
        msg = {
            BZ_OK: 'BZ_OK',
            BZ_RUN_OK: 'BZ_RUN_OK',
            BZ_FLUSH_OK: 'BZ_FLUSH_OK',
            BZ_FINISH_OK: 'BZ_FINISH_OK',
            BZ_STREAM_END: 'BZ_STREAM_END',
            BZ_SEQUENCE_ERROR: 'BZ_SEQUENCE_ERROR',
            BZ_PARAM_ERROR: 'BZ_PARAM_ERROR',
            BZ_MEM_ERROR: 'BZ_MEM_ERROR',
            BZ_DATA_ERROR: 'BZ_DATA_ERROR',
            BZ_DATA_ERROR_MAGIC: 'BZ_DATA_ERROR_MAGIC',
            BZ_IO_ERROR: 'BZ_IO_ERROR',
            BZ_UNEXPECTED_EOF: 'BZ_UNEXPECTED_EOF',
            BZ_OUTBUFF_FULL: 'BZ_OUTBUFF_FULL',
            BZ_CONFIG_ERROR: 'BZ_CONFIG_ERROR',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def bz2_encode(data, level=None, out=None):
    """Compress BZ2.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm
        int compresslevel = _default_value(level, 9, 1, 9)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2
        return bz2.compress(data, compresslevel)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError()
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzCompressInit(&strm, compresslevel, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzCompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char *>&src[0]
            strm.avail_in = <unsigned int>srcsize
            strm.next_out = <char *>&dst[0]
            strm.avail_out = <unsigned int>dstsize
            # while True
            ret = BZ2_bzCompress(&strm, BZ_FINISH)
            #    if ret == BZ_STREAM_END:
            #        break
            #    elif ret != BZ_OK:
            #        break
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzCompress', ret)
    finally:
        ret = BZ2_bzCompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def bz2_decode(data, out=None):
    """Decompress BZ2.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2
        return bz2.decompress(data)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError()
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzDecompressInit(&strm, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzDecompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char *>&src[0]
            strm.avail_in = <unsigned int>srcsize
            strm.next_out = <char *>&dst[0]
            strm.avail_out = <unsigned int>dstsize
            ret = BZ2_bzDecompress(&strm)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret == BZ_OK:
            pass  # output buffer too small
        elif ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzDecompress', ret)
    finally:
        ret = BZ2_bzDecompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def bz2_version():
    """Return libbzip2 version string."""
    return 'libbzip2 ' + str(BZ2_bzlibVersion().decode('utf-8')).split(',')[0]


# Blosc #######################################################################

from blosc cimport *


class BloscError(RuntimeError):
    """Blosc Exceptions."""


def blosc_decode(data, numthreads=1, out=None):
    """Decode Blosc.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        size_t nbytes, cbytes, blocksize
        int numinternalthreads = numthreads
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            blosc_cbuffer_sizes(
                <const void *>&src[0],
                &nbytes,
                &cbytes,
                &blocksize
            )
            if nbytes == 0 and blocksize == 0:
                raise BloscError('invalid blosc data')
            dstsize = <ssize_t>nbytes
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = blosc_decompress_ctx(
            <const void *>&src[0],
            <void *>&dst[0],
            dstsize,
            numinternalthreads
        )
    if ret < 0:
        raise BloscError('blosc_decompress_ctx returned %i' % ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc_encode(data, level=None, compressor='blosclz', typesize=8,
                 blocksize=0, shuffle=None, numthreads=1, out=None):
    """Encode Blosc.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t blocksize_ = blocksize
        size_t typesize_ = typesize
        char *compressor_ = NULL
        int clevel = _default_value(level, 9, 0, 9)
        int doshuffle = BLOSC_SHUFFLE
        int numinternalthreads = numthreads
        int ret

    if data is out:
        raise ValueError('cannot encode in-place')

    compressor = compressor.encode('utf-8')
    compressor_ = compressor

    if shuffle is not None:
        if shuffle == 'noshuffle' or shuffle == BLOSC_NOSHUFFLE:
            doshuffle = BLOSC_NOSHUFFLE
        elif shuffle == 'bitshuffle' or shuffle == BLOSC_BITSHUFFLE:
            doshuffle = BLOSC_BITSHUFFLE
        else:
            doshuffle = BLOSC_SHUFFLE

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + BLOSC_MAX_OVERHEAD
        if dstsize < 17:
            dstsize = 17
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = blosc_compress_ctx(
            clevel,
            doshuffle,
            typesize_,
            <size_t>srcsize,
            <const void *>&src[0],
            <void *>&dst[0],
            <size_t>dstsize,
            <const char*>compressor_,
            blocksize_,
            numinternalthreads
        )
    if ret <= 0:
        raise BloscError('blosc_compress_ctx returned %i' % ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc_version():
    """Return Blosc version string."""
    return 'blosc ' + BLOSC_VERSION_STRING.decode('utf-8')


# Snappy ######################################################################

from snappy cimport *


class SnappyError(RuntimeError):
    """Snappy Exceptions."""
    def __init__(self, func, err):
        msg = {
            SNAPPY_OK: 'SNAPPY_OK',
            SNAPPY_INVALID_INPUT: 'SNAPPY_INVALID_INPUT',
            SNAPPY_BUFFER_TOO_SMALL: 'SNAPPY_BUFFER_TOO_SMALL',
        }.get(err, 'internal error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def snappy_decode(data, numthreads=1, out=None):
    """Decode Snappy.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        snappy_status ret
        size_t output_length
        size_t result

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            ret = snappy_uncompressed_length(
                <const char*>&src[0],
                <size_t>srcsize,
                &result
            )
            if ret != SNAPPY_OK:
                raise SnappyError('snappy_uncompressed_length', ret)
            dstsize = <ssize_t>result
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    output_length = <size_t>dstsize

    with nogil:
        ret = snappy_uncompress(
            <const char*>&src[0],
            <size_t>srcsize,
            <char*>&dst[0],
            &output_length
        )
    if ret != SNAPPY_OK:
        raise SnappyError('snappy_uncompress', ret)

    del dst
    return _return_output(out, dstsize, output_length, outgiven)


def snappy_encode(data, level=None, out=None):
    """Encode Snappy.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t output_length = snappy_max_compressed_length(<size_t>srcsize)
        snappy_status ret
        char* buffer = NULL

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        # override any provided output size
        if dstsize < 0:
            dstsize = <ssize_t>output_length
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if <size_t>dstsize < output_length:
        # snappy_compress requires at least (32+len(data)+len(data)/6) bytes
        with nogil:
            buffer = <char*>malloc(output_length)
            if buffer == NULL:
                raise MemoryError('failed to allocate buffer')
            ret = snappy_compress(
                <const char*>&src[0],
                <size_t>srcsize,
                buffer,
                &output_length
            )
            if ret != SNAPPY_OK:
                free(buffer)
                raise SnappyError('snappy_compress', ret)
            if <size_t>dstsize < output_length:
                free(buffer)
                raise SnappyError('snappy_compress', SNAPPY_BUFFER_TOO_SMALL)
            memcpy(<void *>&dst[0], buffer, output_length)
            free(buffer)
    else:
        with nogil:
            output_length = <size_t>dstsize
            ret = snappy_compress(
                <const char*>&src[0],
                <size_t>srcsize,
                <char*>&dst[0],
                &output_length
            )
        if ret != SNAPPY_OK:
            raise SnappyError('snappy_compress', ret)

    del dst
    return _return_output(out, dstsize, output_length, outgiven)


def snappy_version():
    """Return Snappy version string."""
    return 'snappy 1.1.7'


# AEC #########################################################################

from libaec cimport *

_add_globals(
    AEC_DATA_SIGNED=AEC_DATA_SIGNED,
    AEC_DATA_3BYTE=AEC_DATA_3BYTE,
    AEC_DATA_PREPROCESS=AEC_DATA_PREPROCESS,
    AEC_RESTRICTED=AEC_RESTRICTED,
    AEC_PAD_RSI=AEC_PAD_RSI,
    AEC_NOT_ENFORCE=AEC_NOT_ENFORCE,
)


class AecError(RuntimeError):
    """AEC Exceptions."""
    def __init__(self, func, err):
        msg = {
            AEC_OK: 'AEC_OK',
            AEC_CONF_ERROR: 'AEC_CONF_ERROR',
            AEC_STREAM_ERROR: 'AEC_STREAM_ERROR',
            AEC_DATA_ERROR: 'AEC_DATA_ERROR',
            AEC_MEM_ERROR: 'AEC_MEM_ERROR',
        }.get(err, 'internal error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def aec_encode(data, level=None, bitspersample=None, flags=None,
               blocksize=None, rsi=None, out=None):
    """Compress AEC.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm

    if data is out:
        raise ValueError('cannot encode in-place')

    if flags is None:
        flags_ = AEC_DATA_PREPROCESS
    else:
        flags_ = flags

    if isinstance(data, numpy.ndarray):
        if bitspersample is None:
            bitspersample = data.itemsize * 8
        elif bitspersample > data.itemsize * 8:
            raise ValueError('invalid bitspersample')
        if data.dtype.char == 'i':
            flags_ |= AEC_DATA_SIGNED

    if bitspersample:
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError('invalid bits_per_sample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize  # ? TODO: use dynamic destination buffer
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char *>&src[0]
            strm.avail_in = srcsize
            strm.next_out = <unsigned char *>&dst[0]
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_encode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_encode_init', ret)

            ret = aec_encode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_encode', ret)

            byteswritten = <ssize_t>strm.total_out
            if strm.total_in != <size_t>srcsize:
                raise ValueError('output buffer too small')
    finally:
        ret = aec_encode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_encode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def aec_decode(data, bitspersample=None, flags=None, blocksize=None, rsi=None,
               out=None):
    """Decompress AEC.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

    if flags is None:
        flags_ = AEC_DATA_PREPROCESS
    else:
        flags_ = flags

    if isinstance(out, numpy.ndarray):
        if not numpy.PyArray_ISCONTIGUOUS(out):
            # TODO: handle this
            raise ValueError('output is not contiguous')
        if bitspersample is None:
            bitspersample = out.itemsize * 8
        elif bitspersample > out.itemsize * 8:
            raise ValueError('invalid bitspersample')
        if out.dtype.char == 'i':
            flags_ |= AEC_DATA_SIGNED

    if bitspersample:
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError('invalid bits_per_sample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 8  # ? TODO: use dynamic destination buffer
        out = _create_output(outtype, dstsize)

    try:
        dst = out
    except ValueError:
        dst = numpy.ravel(out).view('uint8')
    dstsize = <int>dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char *>&src[0]
            strm.avail_in = srcsize
            strm.next_out = <unsigned char *>&dst[0]
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_decode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_decode_init', ret)

            ret = aec_decode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_decode', ret)

            byteswritten = <ssize_t>strm.total_out
            if strm.total_in != <size_t>srcsize:
                raise ValueError('output buffer too small')
    finally:
        ret = aec_decode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_decode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def aec_version():
    """Return libaec version string."""
    return 'libaec 1.0.4'


# SZIP ########################################################################

from szlib cimport *


class SzipError(RuntimeError):
    """SZIP Exceptions."""
    def __init__(self, func, err):
        msg = {
            SZ_OK: 'SZ_OK',
            SZ_OUTBUFF_FULL: 'SZ_OUTBUFF_FULL',
            SZ_NO_ENCODER_ERROR: 'SZ_NO_ENCODER_ERROR',
            SZ_PARAM_ERROR: 'SZ_PARAM_ERROR',
            SZ_MEM_ERROR: 'SZ_MEM_ERROR',
        }.get(err, 'internal error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def _szip_decode(data, out=None):
    """Decompress SZIP.

    """
    raise NotImplementedError()


def _szip_encode(data, level=None, bitspersample=None, flags=None, out=None):
    """Compress SZIP.

    """
    raise NotImplementedError()


def szip_version():
    """Return SZIP version string."""
    return 'libsz n/a'


# Brotli ######################################################################

from brotli cimport *


class BrotliError(RuntimeError):
    """Brotli Exceptions."""
    def __init__(self, func, err):
        err = {
            True: 'True',
            False: 'False',
            BROTLI_DECODER_RESULT_ERROR: 'BROTLI_DECODER_RESULT_ERROR',
            BROTLI_DECODER_RESULT_SUCCESS: 'BROTLI_DECODER_RESULT_SUCCESS',
            BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT:
                'BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT',
            BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
                'BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, err)
        RuntimeError.__init__(self, msg)


def brotli_encode(data, level=None, mode=None, lgwin=None, out=None):
    """Compress Brotli.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t encoded_size
        BROTLI_BOOL ret = BROTLI_FALSE
        BrotliEncoderMode mode_ = BROTLI_MODE_GENERIC if mode is None else mode
        int quality_ = _default_value(level, 11, 0, 11)
        int lgwin_ = _default_value(lgwin, 22, 10, 24)
        # int lgblock_ = _default_value(lgblock, 0, 16, 24)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming interface with dynamic buffer
            dstsize = <ssize_t>BrotliEncoderMaxCompressedSize(<size_t>srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    encoded_size = <size_t>dstsize

    with nogil:
        ret = BrotliEncoderCompress(
            quality_,
            lgwin_,
            mode_,
            <size_t>srcsize,
            <const uint8_t*>&src[0],
            &encoded_size,
            <uint8_t*>&dst[0]
        )
    if ret != BROTLI_TRUE:
        raise BrotliError('BrotliEncoderCompress', bool(ret))

    del dst
    return _return_output(out, dstsize, encoded_size, outgiven)


def brotli_decode(data, out=None):
    """Decompress Brotli.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        size_t decoded_size
        BrotliDecoderResult ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming API with dynamic buffer
            dstsize = srcsize * 4
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    decoded_size = <size_t>dstsize

    with nogil:
        ret = BrotliDecoderDecompress(
            <size_t>srcsize,
            <const uint8_t*>&src[0],
            &decoded_size,
            <uint8_t*>&dst[0]
        )
    if ret != BROTLI_DECODER_RESULT_SUCCESS:
        raise BrotliError('BrotliDecoderDecompress', ret)

    del dst
    return _return_output(out, dstsize, decoded_size, outgiven)


def brotli_version():
    """Return Brotli version string."""
    cdef uint32_t ver = BrotliDecoderVersion()
    return 'brotli %i.%i.%i' % (ver >> 24, (ver >> 12) & 4095, ver & 4095)


# PNG #########################################################################

from libpng cimport *


cdef void png_error_callback(png_structp png_ptr,
                             png_const_charp msg) with gil:
    raise PngError(msg.decode('utf8').strip())


cdef void png_warn_callback(png_structp png_ptr,
                            png_const_charp msg) with gil:
    import logging
    logging.warning('PNG %s' % msg.decode('utf8').strip())


ctypedef struct png_memstream_t:
    png_bytep data
    png_size_t size
    png_size_t offset


cdef void png_read_data_fn(png_structp png_ptr,
                           png_bytep dst,
                           png_size_t size) nogil:
    """PNG read callback function."""
    cdef:
        png_memstream_t *memstream = <png_memstream_t *>png_get_io_ptr(png_ptr)
    if memstream == NULL:
        return
    if memstream.offset >= memstream.size:
        return
    if size > memstream.size - memstream.offset:
        # size = memstream.size - memstream.offset
        raise PngError('PNG input stream too small %i' % memstream.size)
    memcpy(
        <void *>dst,
        <const void *>&(memstream.data[memstream.offset]),
        size)
    memstream.offset += size


cdef void png_write_data_fn(png_structp png_ptr,
                            png_bytep src,
                            png_size_t size) nogil:
    """PNG write callback function."""
    cdef:
        png_memstream_t *memstream = <png_memstream_t *>png_get_io_ptr(png_ptr)
    if memstream == NULL:
        return
    if memstream.offset >= memstream.size:
        return
    if size > memstream.size - memstream.offset:
        # size = memstream.size - memstream.offset
        raise PngError('PNG output stream too small %i' % memstream.size)
    memcpy(
        <void *>&(memstream.data[memstream.offset]),
        <const void *>src,
        size)
    memstream.offset += size


cdef void png_output_flush_fn(png_structp png_ptr) nogil:
    """PNG flush callback function."""
    pass


cdef ssize_t png_size_max(ssize_t size):
    """Return upper bound size of PNG stream from uncompressed image size."""
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12  # sig IHDR gAMA cHRM IEND
    return size


class PngError(RuntimeError):
    """PNG Exceptions."""


def png_decode(data, out=None):
    """Decode PNG image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size * src.itemsize
        int samples = 0
        png_memstream_t memstream
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_uint_32 ret = 0
        png_uint_32 width = 0
        png_uint_32 height = 0
        png_uint_32 row
        int bit_depth = 0
        int color_type = -1
        png_bytepp image = NULL  # row pointers
        png_bytep rowptr
        ssize_t rowstride

    if data is out:
        raise ValueError('cannot decode in-place')

    if png_sig_cmp(&src[0], 0, 8) != 0:
        raise ValueError('not a PNG image')

    try:
        with nogil:
            memstream.data = <png_bytep>&src[0]
            memstream.size = srcsize
            memstream.offset = 8

            png_ptr = png_create_read_struct(
                PNG_LIBPNG_VER_STRING, NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise PngError('png_create_read_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise PngError('png_create_info_struct returned NULL')

            png_set_read_fn(png_ptr, <png_voidp>&memstream, png_read_data_fn)
            png_set_sig_bytes(png_ptr, 8)
            png_read_info(png_ptr, info_ptr)
            ret = png_get_IHDR(
                png_ptr,
                info_ptr,
                &width,
                &height,
                &bit_depth,
                &color_type,
                NULL,
                NULL,
                NULL
            )
            if ret != 1:
                raise PngError('png_get_IHDR returned %i' % ret)

            if bit_depth > 8:
                png_set_swap(png_ptr)

            if color_type == PNG_COLOR_TYPE_GRAY:
                samples = 1
                if bit_depth < 8:
                    png_set_expand_gray_1_2_4_to_8(png_ptr)
            elif color_type == PNG_COLOR_TYPE_GRAY_ALPHA:
                samples = 2
            elif color_type == PNG_COLOR_TYPE_RGB:
                samples = 3
            elif color_type == PNG_COLOR_TYPE_PALETTE:
                samples = 3
                png_set_palette_to_rgb(png_ptr)
            elif color_type == PNG_COLOR_TYPE_RGB_ALPHA:
                samples = 4
            else:
                raise ValueError(
                'PNG color type not supported %i' % color_type)

        dtype = numpy.dtype('u%i' % (1 if bit_depth//8 < 2 else 2))
        if samples > 1:
            shape = int(height), int(width), int(samples)
            strides = None, shape[2] * dtype.itemsize, dtype.itemsize
        else:
            shape = int(height), int(width)
            strides = None, dtype.itemsize

        out = _create_array(out, shape, dtype, strides)
        dst = out
        rowptr = <png_bytep>&dst.data[0]
        rowstride = dst.strides[0]

        with nogil:
            image = <png_bytepp>malloc(sizeof(png_bytep) * height)
            if image == NULL:
                raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = <png_bytep>rowptr
                rowptr += rowstride
            png_read_update_info(png_ptr, info_ptr)
            png_read_image(png_ptr, image)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


def png_encode(data, level=None, out=None):
    """Encode numpy array to PNG image.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        ssize_t rowstride = src.strides[0]
        png_bytep rowptr = <png_bytep>&src.data[0]
        int color_type
        int bit_depth = src.itemsize * 8
        int samples = <int>src.shape[2] if src.ndim == 3 else 1
        int compresslevel = _default_value(level, 5, 0, 10)
        png_memstream_t memstream
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp image = NULL  # row pointers
        png_uint_32 width = <png_uint_32>src.shape[1]
        png_uint_32 height = <png_uint_32>src.shape[0]
        png_uint_32 row

    if not (
        data.dtype in (numpy.uint8, numpy.uint16)
        and data.ndim in (2, 3)
        and data.shape[0] < 2**31 - 1
        and data.shape[1] < 2**31 - 1
        and samples <= 4
        and data.strides[data.ndim-1] == data.itemsize
        and (data.ndim == 2 or data.strides[1] == samples*data.itemsize)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = png_size_max(srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size * dst.itemsize

    try:
        with nogil:

            memstream.data = <png_bytep>&dst[0]
            memstream.size = <png_size_t>dstsize
            memstream.offset = 0

            if samples == 1:
                color_type = PNG_COLOR_TYPE_GRAY
            elif samples == 2:
                color_type = PNG_COLOR_TYPE_GRAY_ALPHA
            elif samples == 3:
                color_type = PNG_COLOR_TYPE_RGB
            elif samples == 4:
                color_type = PNG_COLOR_TYPE_RGB_ALPHA
            else:
                raise ValueError('PNG color type not supported')

            png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING,
                NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise PngError('png_create_write_struct returned NULL')

            png_set_write_fn(
                png_ptr,
                <png_voidp>&memstream,
                png_write_data_fn,
                png_output_flush_fn
            )

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise PngError('png_create_info_struct returned NULL')

            png_set_IHDR(
                png_ptr,
                info_ptr,
                width,
                height,
                bit_depth,
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT
            )

            png_write_info(png_ptr, info_ptr)
            png_set_compression_level(png_ptr, compresslevel)
            if bit_depth > 8:
                png_set_swap(png_ptr)

            image = <png_bytepp>malloc(sizeof(png_bytep) * height)
            if image == NULL:
                raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = rowptr
                rowptr += rowstride

            png_write_image(png_ptr, image)
            png_write_end(png_ptr, info_ptr)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        elif png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, NULL)

    del dst
    return _return_output(out, dstsize, memstream.offset, outgiven)


def png_version():
    """Return PNG version string."""
    return 'libpng ' + PNG_LIBPNG_VER_STRING.decode('utf-8')


# WebP ########################################################################

from libwebp cimport *


class WebpError(RuntimeError):
    """WebP Exceptions."""
    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            VP8_STATUS_OK: 'VP8_STATUS_OK',
            VP8_STATUS_OUT_OF_MEMORY: 'VP8_STATUS_OUT_OF_MEMORY',
            VP8_STATUS_INVALID_PARAM: 'VP8_STATUS_INVALID_PARAM',
            VP8_STATUS_BITSTREAM_ERROR: 'VP8_STATUS_BITSTREAM_ERROR',
            VP8_STATUS_UNSUPPORTED_FEATURE: 'VP8_STATUS_UNSUPPORTED_FEATURE',
            VP8_STATUS_SUSPENDED: 'VP8_STATUS_SUSPENDED',
            VP8_STATUS_USER_ABORT: 'VP8_STATUS_USER_ABORT',
            VP8_STATUS_NOT_ENOUGH_DATA: 'VP8_STATUS_NOT_ENOUGH_DATA',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def webp_encode(data, level=None, out=None):
    """Encode numpy array to WebP image.

    """
    cdef:
        const uint8_t[:, :, :] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t *srcptr = <uint8_t*>&src[0, 0, 0]
        uint8_t *output
        ssize_t dstsize
        size_t ret = 0
        int width, height, stride
        float quality_factor = _default_value(level, 75.0, -1.0, 100.0)
        int lossless = quality_factor < 0.0
        int rgba = data.shape[2] == 4

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        data.ndim == 3
        and data.shape[0] < WEBP_MAX_DIMENSION
        and data.shape[1] < WEBP_MAX_DIMENSION
        and data.shape[2] in (3, 4)
        and data.strides[2] == 1
        and data.strides[1] in (3, 4)
        and data.strides[0] >= data.strides[1] * data.strides[2]
        and data.dtype == numpy.uint8
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    height, width = data.shape[:2]
    stride = data.strides[0]

    with nogil:
        if lossless:
            if rgba:
                ret = WebPEncodeLosslessRGBA(
                    <const uint8_t*>srcptr,
                    width,
                    height,
                    stride,
                    &output)
            else:
                ret = WebPEncodeLosslessRGB(
                    <const uint8_t*>srcptr,
                    width,
                    height,
                    stride,
                    &output)
        elif rgba:
            ret = WebPEncodeRGBA(
                <const uint8_t*>srcptr,
                width,
                height,
                stride,
                quality_factor,
                &output)
        else:
            ret = WebPEncodeRGB(
                <const uint8_t*>srcptr,
                width,
                height,
                stride,
                quality_factor,
                &output)

    if ret <= 0:
        raise WebpError('WebPEncode', ret)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = ret
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if <size_t>dstsize < ret:
        raise RuntimeError('output too small')

    with nogil:
        memcpy(<void *>&dst[0], <const void *>output, ret)
        WebPFree(<void *>output)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def webp_decode(data, out=None):
    """Decode WebP image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        int output_stride
        size_t data_size
        WebPBitstreamFeatures features
        int ret = VP8_STATUS_OK
        uint8_t *pout

    if data is out:
        raise ValueError('cannot decode in-place')

    ret = <int>WebPGetFeatures(&src[0], <size_t>srcsize, &features)
    if ret != VP8_STATUS_OK:
        raise WebpError('WebPGetFeatures', ret)

    if features.has_alpha:
        shape = features.height, features.width, 4
    else:
        shape = features.height, features.width, 3

    out = _create_array(out, shape, numpy.uint8, strides=(None, shape[2], 1))
    dst = out
    dstsize = dst.shape[0] * dst.strides[0]
    output_stride = <int>dst.strides[0]

    with nogil:
        if features.has_alpha:
            pout = WebPDecodeRGBAInto(
                &src[0],
                <size_t> srcsize,
                <uint8_t*> dst.data,
                <size_t> dstsize,
                output_stride
            )
        else:
            pout = WebPDecodeRGBInto(
                &src[0],
                <size_t> srcsize,
                <uint8_t*> dst.data,
                <size_t> dstsize,
                output_stride
            )
    if pout == NULL:
        raise WebpError('WebPDecodeRGBAInto', None)

    return out


def webp_version():
    """Return WebP version string."""
    cdef int ver = WebPGetDecoderVersion()
    return 'libwebp %i.%i.%i' % (ver >> 24, (ver >> 12) & 4095, ver & 4095)


# JPEG 8-bit ##################################################################

from libjpeg_turbo cimport *


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct *cinfo):
    cdef my_error_mgr *error = <my_error_mgr*> deref(cinfo).err
    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct *cinfo):
    pass


def _jcs_colorspace(colorspace):
    """Return JCS colorspace value from user input."""
    return {
        'GRAY': JCS_GRAYSCALE,
        'GRAYSCALE': JCS_GRAYSCALE,
        'MINISWHITE': JCS_GRAYSCALE,
        'MINISBLACK': JCS_GRAYSCALE,
        'RGB': JCS_RGB,
        'RGBA': JCS_EXT_RGBA,
        'CMYK': JCS_CMYK,
        'YCCK': JCS_YCCK,
        'YCBCR': JCS_YCbCr,
        'UNKNOWN': JCS_UNKNOWN,
        None: JCS_UNKNOWN,
        JCS_UNKNOWN: JCS_UNKNOWN,
        JCS_GRAYSCALE: JCS_GRAYSCALE,
        JCS_RGB: JCS_RGB,
        JCS_YCbCr: JCS_YCbCr,
        JCS_CMYK: JCS_CMYK,
        JCS_YCCK: JCS_YCCK,
        JCS_EXT_RGB: JCS_EXT_RGB,
        JCS_EXT_RGBX: JCS_EXT_RGBX,
        JCS_EXT_BGR: JCS_EXT_BGR,
        JCS_EXT_BGRX: JCS_EXT_BGRX,
        JCS_EXT_XBGR: JCS_EXT_XBGR,
        JCS_EXT_XRGB: JCS_EXT_XRGB,
        JCS_EXT_RGBA: JCS_EXT_RGBA,
        JCS_EXT_BGRA: JCS_EXT_BGRA,
        JCS_EXT_ABGR: JCS_EXT_ABGR,
        JCS_EXT_ARGB: JCS_EXT_ARGB,
        JCS_RGB565: JCS_RGB565,
        }.get(colorspace, JCS_UNKNOWN)


def _jcs_colorspace_samples(colorspace):
    """Return expected number of samples in colorspace."""
    three = (3,)
    four = (4,)
    return {
        JCS_UNKNOWN: (1, 2, 3, 4),
        JCS_GRAYSCALE: (1,),
        JCS_RGB: three,
        JCS_YCbCr: three,
        JCS_CMYK: four,
        JCS_YCCK: four,
        JCS_EXT_RGB: three,
        JCS_EXT_RGBX: four,
        JCS_EXT_BGR: three,
        JCS_EXT_BGRX: four,
        JCS_EXT_XBGR: four,
        JCS_EXT_XRGB: four,
        JCS_EXT_RGBA: four,
        JCS_EXT_BGRA: four,
        JCS_EXT_ABGR: four,
        JCS_EXT_ARGB: four,
        JCS_RGB565: three,
        }[colorspace]


class Jpeg8Error(RuntimeError):
    """JPEG Exceptions."""


def jpeg8_encode(data, level=None, colorspace=None, outcolorspace=None,
                 subsampling=None, optimize=None, smoothing=None, out=None):
    """Return JPEG 8-bit image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        ssize_t rowstride = src.strides[0]
        int samples = <int>src.shape[2] if src.ndim == 3 else 1
        int quality = _default_value(level, 90, 0, 100)
        my_error_mgr err
        jpeg_compress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE in_color_space = JCS_UNKNOWN
        J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN
        unsigned long outsize = 0
        unsigned char *outbuffer = NULL
        const char *msg
        int h_samp_factor = 0
        int v_samp_factor = 0
        int smoothing_factor = _default_value(smoothing, -1, 0, 100)
        int optimize_coding = -1 if optimize is None else 1 if optimize else 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        data.dtype == numpy.uint8
        and data.ndim in (2, 3)
        # and data.size * data.itemsize < 2**31-1  # limit to 2 GB
        and samples in (1, 3, 4)
        and data.strides[data.ndim-1] == data.itemsize
        and (data.ndim == 2 or data.strides[1] == samples*data.itemsize)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    if colorspace is None:
        if samples == 1:
            in_color_space = JCS_GRAYSCALE
        elif samples == 3:
            in_color_space = JCS_RGB
        # elif samples == 4:
        #     in_color_space = JCS_CMYK
        else:
            in_color_space = JCS_UNKNOWN
    else:
        in_color_space = _jcs_colorspace(colorspace)
        if samples not in _jcs_colorspace_samples(in_color_space):
            raise ValueError('invalid input shape')

    jpeg_color_space = _jcs_colorspace(outcolorspace)

    if jpeg_color_space == JCS_YCbCr and subsampling is not None:
        if subsampling in ('444', (1, 1)):
            h_samp_factor = 1
            v_samp_factor = 1
        elif subsampling in ('422', (2, 1)):
            h_samp_factor = 2
            v_samp_factor = 1
        elif subsampling in ('420', (2, 2)):
            h_samp_factor = 2
            v_samp_factor = 2
        elif subsampling in ('411', (4, 1)):
            h_samp_factor = 4
            v_samp_factor = 1
        elif subsampling in ('440', (1, 2)):
            h_samp_factor = 1
            v_samp_factor = 2
        else:
            raise ValueError('invalid subsampling')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize > 0:
        out = _create_output(outtype, dstsize)

    if out is not None:
        dst = out
        dstsize = dst.size * dst.itemsize
        outsize = <unsigned long>dstsize
        outbuffer = <unsigned char*>&dst[0]

    with nogil:
        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message

        if setjmp(err.setjmp_buffer):
            jpeg_destroy_compress(&cinfo)
            msg = err.pub.jpeg_message_table[err.pub.msg_code]
            raise Jpeg8Error(msg.decode('utf-8'))

        jpeg_create_compress(&cinfo)

        cinfo.image_height = <JDIMENSION>src.shape[0]
        cinfo.image_width = <JDIMENSION>src.shape[1]
        cinfo.input_components = samples

        if in_color_space != JCS_UNKNOWN:
            cinfo.in_color_space = in_color_space
        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space

        jpeg_set_defaults(&cinfo)
        jpeg_mem_dest(&cinfo, &outbuffer, &outsize)  # must call after defaults
        jpeg_set_quality(&cinfo, quality, 1)

        if smoothing_factor >= 0:
            cinfo.smoothing_factor = smoothing_factor
        if optimize_coding >= 0:
            cinfo.optimize_coding = <boolean>optimize_coding
        if h_samp_factor != 0:
            cinfo.comp_info[0].h_samp_factor = h_samp_factor
            cinfo.comp_info[0].v_samp_factor = v_samp_factor
            cinfo.comp_info[1].h_samp_factor = 1
            cinfo.comp_info[1].v_samp_factor = 1
            cinfo.comp_info[2].h_samp_factor = 1
            cinfo.comp_info[2].v_samp_factor = 1

        # TODO: add option to use or return JPEG tables

        jpeg_start_compress(&cinfo, 1)

        while cinfo.next_scanline < cinfo.image_height:
            rowpointer = <JSAMPROW>(
                <char*>src.data + cinfo.next_scanline * rowstride)
            jpeg_write_scanlines(&cinfo, &rowpointer, 1)

        jpeg_finish_compress(&cinfo)
        jpeg_destroy_compress(&cinfo)

    if out is None or outbuffer != <unsigned char*>&dst[0]:
        # outbuffer was allocated in jpeg_mem_dest
        out = _create_output(outtype, <ssize_t>outsize, <const char*>outbuffer)
        free(outbuffer)
        return out

    del dst
    return _return_output(out, dstsize, outsize, outgiven)


def jpeg8_decode(data, tables=None, colorspace=None, outcolorspace=None,
                 shape=None, out=None):
    """Decode JPEG 8-bit image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t[::1] tables_
        unsigned long tablesize = 0
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t rowstride
        int numlines
        my_error_mgr err
        jpeg_decompress_struct cinfo
        JSAMPROW rowpointer
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        JDIMENSION width = 0
        JDIMENSION height = 0
        const char *msg

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2**32 - 1:
        # limit to 4 GB
        raise ValueError('data too large')

    jpeg_color_space = _jcs_colorspace(colorspace)
    if outcolorspace is None:
        out_color_space = jpeg_color_space
    else:
        out_color_space = _jcs_colorspace(outcolorspace)

    if tables is not None:
        tables_ = tables
        tablesize = tables_.size

    if shape is not None and (shape[0] >= 65500 or shape[1] >= 65500):
        # enable decoding of large (JPEG_MAX_DIMENSION <= 2^20) JPEG
        # when using a patched jibjpeg-turbo
        height = <JDIMENSION>shape[0]
        width = <JDIMENSION>shape[1]

    with nogil:

        cinfo.err = jpeg_std_error(&err.pub)
        err.pub.error_exit = my_error_exit
        err.pub.output_message = my_output_message
        if setjmp(err.setjmp_buffer):
            jpeg_destroy_decompress(&cinfo)
            msg = err.pub.jpeg_message_table[err.pub.msg_code]
            raise Jpeg8Error(msg.decode('utf-8'))

        jpeg_create_decompress(&cinfo)
        cinfo.do_fancy_upsampling = True
        if width > 0:
            cinfo.image_width = width
            cinfo.image_height = height

        if tablesize > 0:
            jpeg_mem_src(&cinfo, &tables_[0], tablesize)
            jpeg_read_header(&cinfo, 0)

        jpeg_mem_src(&cinfo, &src[0], <unsigned long>srcsize)
        jpeg_read_header(&cinfo, 1)

        if jpeg_color_space != JCS_UNKNOWN:
            cinfo.jpeg_color_space = jpeg_color_space
        if out_color_space != JCS_UNKNOWN:
            cinfo.out_color_space = out_color_space

        jpeg_start_decompress(&cinfo)

        with gil:
            # if (cinfo.output_components not in
            #         _jcs_colorspace_samples(out_color_space)):
            #     raise ValueError('invalid output shape')

            shape = cinfo.output_height, cinfo.output_width
            if cinfo.output_components > 1:
                shape += cinfo.output_components,

            out = _create_array(out, shape, numpy.uint8)  # TODO: allow strides
            dst = out
            dstsize = dst.size * dst.itemsize
            rowstride = dst.strides[0]

        memset(<void *>dst.data, 0, dstsize)
        rowpointer = <JSAMPROW>dst.data
        while cinfo.output_scanline < cinfo.output_height:
            jpeg_read_scanlines(&cinfo, &rowpointer, 1)
            rowpointer += rowstride

        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


def jpeg8_version():
    """Return JPEG 8-bit version string."""
    return 'libjpeg %.1f' % (JPEG_LIB_VERSION / 10.0)


def jpeg_turbo_version():
    """Return libjpeg-turbo version string."""
    ver = str(LIBJPEG_TURBO_VERSION_NUMBER)
    return 'libjpeg_turbo %i.%i.%i' % (
        int(ver[:1]), int(ver[3:4]), int(ver[6:]))


# JPEG SOF3 ###############################################################

# The "JPEG Lossless, Nonhierarchical, First Order Prediction" format is
# described at <http://www.w3.org/Graphics/JPEG/itu-t81.pdf>.
# The format is identified by a Start of Frame (SOF) code 0xC3.

from jpeg_sof3 cimport *


class JpegSof3Error(RuntimeError):
    """JPEG SOF3 Exceptions."""
    def __init__(self, err):
        msg = {
            JPEG_SOF3_INVALID_OUTPUT:
                'output array is too small',
            JPEG_SOF3_INVALID_SIGNATURE:
                'JPEG signature 0xFFD8FF not found',
            JPEG_SOF3_INVALID_HEADER_TAG:
                'header tag must begin with 0xFF',
            JPEG_SOF3_SEGMENT_GT_IMAGE:
                'segment larger than image',
            JPEG_SOF3_INVALID_ITU_T81:
                'not a lossless (sequential) JPEG image (SoF must be 0xC3)',
            JPEG_SOF3_INVALID_BIT_DEPTH:
                'data must be 2..16 bit, 1..4 frames',
            JPEG_SOF3_TABLE_CORRUPTED:
                'Huffman table corrupted',
            JPEG_SOF3_TABLE_SIZE_CORRUPTED:
                'Huffman size array corrupted',
            JPEG_SOF3_INVALID_RESTART_SEGMENTS:
                'unsupported Restart Segments',
            JPEG_SOF3_NO_TABLE:
                'no Huffman tables',
            }.get(err, 'unknown error %i' % err)
        msg = "jpeg_0x3c_decode returned '%s'" % msg
        RuntimeError.__init__(self, msg)


def jpegsof3_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JPEG SOF3 encoding
    raise NotImplementedError('jpegsof3_encode')


def jpegsof3_decode(data, out=None):
    """Decode JPEG SOF3 image to numpy array.

    Beware, the input data is modified!

    RGB images are returned as non-contiguous arrays as samples are decoded
    into separate frames first (RRGGBB).

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        int dimX, dimY, bits, frames
        int ret = JPEG_SOF3_OK

    if data is out:
        raise ValueError('cannot decode in-place')

    with nogil:
        ret = jpeg_sof3_decode(
            <unsigned char *>&src[0],
            srcsize,
            NULL,
            0,
            &dimX,
            &dimY,
            &bits,
            &frames
        )
    if ret != JPEG_SOF3_OK:
        raise JpegSof3Error(ret)

    if frames > 1:
        shape = frames, dimY, dimX
    else:
        shape = dimY, dimX

    if bits > 8:
        dtype = numpy.uint16
    else:
        dtype = numpy.uint8

    out = _create_array(out, shape, dtype)
    dst = out
    dstsize = dst.size * dst.itemsize

    with nogil:
        ret = jpeg_sof3_decode(
            <unsigned char *>&src[0],
            srcsize,
            <unsigned char *>dst.data,
            dstsize,
            &dimX,
            &dimY,
            &bits,
            &frames
        )
    if ret != JPEG_SOF3_OK:
        raise JpegSof3Error(ret)

    if frames > 1:
        out = numpy.moveaxis(out, 0, -1)

    return out


def jpegsof3_version():
    """Return JPEG SOF3 version string."""
    return 'jpegsof3 ' + JPEG_SOF3_VERSION.decode('utf-8')


# JPEG Wrapper ################################################################

def jpeg_decode(data, bitspersample=None, tables=None, colorspace=None,
                outcolorspace=None, shape=None, out=None):
    """Decode JPEG 8-bit, 12-bit, SOF3, LS, or XL.

    """
    if bitspersample is None:
        try:
            return jpeg8_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        except Jpeg8Error as exception:
            msg = str(exception)
            if 'Empty JPEG image' in msg:
                # TODO: handle Hamamatsu NDPI slides with dimensions > 65500
                raise exception
            if 'Unsupported JPEG data precision' in msg:
                return jpeg12_decode(
                    data, tables=tables, colorspace=colorspace,
                    outcolorspace=outcolorspace, shape=shape, out=out)
            if 'SOF type' in msg:
                return jpegsof3_decode(data, out=out)
            # Unsupported marker type
            try:
                return jpegls_decode(data, out=out)
            except Exception:
                return jpegxl_decode(data, out=out)
    try:
        if bitspersample == 8:
            return jpeg8_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        if bitspersample == 12:
            return jpeg12_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        try:
            return jpegls_decode(data, out=out)
        except Exception:
            return jpegsof3_decode(data, out=out)
    except (Jpeg8Error, Jpeg12Error, NotImplementedError) as exception:
        msg = str(exception)
        if 'Empty JPEG image' in msg:
            raise exception
        if 'SOF type' in msg:
            return jpegsof3_decode(data, out=out)
        try:
            return jpegls_decode(data, out=out)
        except Exception:
            return jpegxl_decode(data, out=out)


def jpeg_encode(data, level=None, colorspace=None, outcolorspace=None,
                subsampling=None, optimize=None, smoothing=None, out=None):
    """Encode 8-bit or 12-bit JPEG.

    """
    if data.dtype == numpy.uint8:
        func = jpeg8_encode
    elif data.dtype == numpy.uint16:
        func = jpeg12_encode
    else:
        raise ValueError('invalid data dtype %s' % data.dtype)
    return func(data, level=level, colorspace=colorspace,
                outcolorspace=outcolorspace, subsampling=subsampling,
                optimize=optimize, smoothing=smoothing, out=out)


# JPEG 2000 ###################################################################

from openjpeg cimport *


ctypedef struct opj_memstream_t:
    OPJ_UINT8 *data
    OPJ_UINT64 size
    OPJ_UINT64 offset
    OPJ_UINT64 written


cdef OPJ_SIZE_T opj_mem_read(void *dst, OPJ_SIZE_T size, void *data) nogil:
    """opj_stream_set_read_function."""
    cdef:
        opj_memstream_t *memstream = <opj_memstream_t*>data
        OPJ_SIZE_T count = size
    if memstream.offset >= memstream.size:
        return <OPJ_SIZE_T>-1
    if size > memstream.size - memstream.offset:
        count = memstream.size - memstream.offset
    memcpy(
        <void *>dst,
        <const void *>&(memstream.data[memstream.offset]),
        count)
    memstream.offset += count
    return count


cdef OPJ_SIZE_T opj_mem_write(void *dst, OPJ_SIZE_T size, void *data) nogil:
    """opj_stream_set_write_function."""
    cdef:
        opj_memstream_t *memstream = <opj_memstream_t*>data
        OPJ_SIZE_T count = size
    if memstream.offset >= memstream.size:
        return <OPJ_SIZE_T>-1
    if size > memstream.size - memstream.offset:
        count = memstream.size - memstream.offset
        memstream.written = memstream.size + 1  # indicates error
    memcpy(
        <void *>&(memstream.data[memstream.offset]),
        <const void *>dst,
        count)
    memstream.offset += count
    if memstream.written < memstream.offset:
        memstream.written = memstream.offset
    return count


cdef OPJ_BOOL opj_mem_seek(OPJ_OFF_T size, void *data) nogil:
    """opj_stream_set_seek_function."""
    cdef:
        opj_memstream_t *memstream = <opj_memstream_t *>data
    if size < 0 or size >= <OPJ_OFF_T>memstream.size:
        return OPJ_FALSE
    memstream.offset = <OPJ_SIZE_T>size
    return OPJ_TRUE


cdef OPJ_OFF_T opj_mem_skip(OPJ_OFF_T size, void *data) nogil:
    """opj_stream_set_skip_function."""
    cdef:
        opj_memstream_t *memstream = <opj_memstream_t *>data
        OPJ_SIZE_T count
    if size < 0:
        return -1
    count = <OPJ_SIZE_T>size
    if count > memstream.size - memstream.offset:
        count = memstream.size - memstream.offset
    memstream.offset += count
    return count


cdef void opj_mem_nop(void *data) nogil:
    """opj_stream_set_user_data."""


cdef opj_stream_t* opj_memstream_create(opj_memstream_t *memstream,
                                        OPJ_BOOL isinput) nogil:
    """Return OPJ stream using memory as input or output."""
    cdef:
        opj_stream_t *stream = opj_stream_default_create(isinput)
    if stream == NULL:
        return NULL
    if isinput:
        opj_stream_set_read_function(stream, <opj_stream_read_fn>opj_mem_read)
    else:
        opj_stream_set_write_function(
            stream,
            <opj_stream_write_fn>opj_mem_write)
    opj_stream_set_seek_function(stream, <opj_stream_seek_fn>opj_mem_seek)
    opj_stream_set_skip_function(stream, <opj_stream_skip_fn>opj_mem_skip)
    opj_stream_set_user_data(
        stream,
        memstream,
        <opj_stream_free_user_data_fn>opj_mem_nop)
    opj_stream_set_user_data_length(stream, memstream.size)
    return stream


class Jpeg2kError(RuntimeError):
    """OpenJPEG Exceptions."""


cdef void j2k_error_callback(char *msg, void *client_data) with gil:
    raise Jpeg2kError(msg.decode('utf8').strip())


cdef void j2k_warning_callback(char *msg, void *client_data) with gil:
    import logging
    logging.warning('J2K warning: %s' % msg.decode('utf8').strip())


cdef void j2k_info_callback(char *msg, void *client_data) with gil:
    import logging
    logging.warning('J2K info: %s' % msg.decode('utf8').strip())


cdef OPJ_COLOR_SPACE opj_colorspace(colorspace):
    """Return OPJ colorspace value from user input."""
    return {
        'GRAY': OPJ_CLRSPC_GRAY,
        'GRAYSCALE': OPJ_CLRSPC_GRAY,
        'MINISWHITE': OPJ_CLRSPC_GRAY,
        'MINISBLACK': OPJ_CLRSPC_GRAY,
        'RGB': OPJ_CLRSPC_SRGB,
        'SRGB': OPJ_CLRSPC_SRGB,
        'RGBA': OPJ_CLRSPC_SRGB,  # ?
        'CMYK': OPJ_CLRSPC_CMYK,
        'SYCC': OPJ_CLRSPC_SYCC,
        'EYCC': OPJ_CLRSPC_EYCC,
        'UNSPECIFIED': OPJ_CLRSPC_UNSPECIFIED,
        'UNKNOWN': OPJ_CLRSPC_UNSPECIFIED,
        None: OPJ_CLRSPC_UNSPECIFIED,
        OPJ_CLRSPC_UNSPECIFIED: OPJ_CLRSPC_UNSPECIFIED,
        OPJ_CLRSPC_SRGB: OPJ_CLRSPC_SRGB,
        OPJ_CLRSPC_GRAY: OPJ_CLRSPC_GRAY,
        OPJ_CLRSPC_SYCC: OPJ_CLRSPC_SYCC,
        OPJ_CLRSPC_EYCC: OPJ_CLRSPC_EYCC,
        OPJ_CLRSPC_CMYK: OPJ_CLRSPC_CMYK,
        }.get(colorspace, OPJ_CLRSPC_UNSPECIFIED)


def jpeg2k_encode(data, level=None, codecformat=None, colorspace=None,
                  tile=None, verbose=0, out=None):
    """Return JPEG 2000 image from numpy array.

    This function is WIP, use at own risk.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        ssize_t byteswritten

        opj_memstream_t memstream
        opj_codec_t *codec = NULL
        opj_image_t *image = NULL
        opj_stream_t *stream = NULL
        opj_cparameters_t parameters
        opj_image_cmptparm_t cmptparms[4]

        OPJ_CODEC_FORMAT codec_format = (
            OPJ_CODEC_JP2 if codecformat == 'JP2' else OPJ_CODEC_J2K)
        OPJ_BOOL ret = OPJ_TRUE
        OPJ_COLOR_SPACE color_space
        OPJ_UINT32 signed, prec, width, height, samples
        ssize_t i, j
        int verbosity = verbose
        int tile_width = 0
        int tile_height = 0

        float rate = 100.0 / _default_value(level, 100, 1, 100)

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        data.dtype in (numpy.int8, numpy.int16, numpy.int32,
                       numpy.uint8, numpy.uint16, numpy.uint32)
        and data.ndim in (2, 3)
        and numpy.PyArray_ISCONTIGUOUS(data)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    signed = 1 if data.dtype in (numpy.int8, numpy.int16, numpy.int32) else 0
    prec = data.itemsize * 8
    width = data.shape[1]
    height = data.shape[0]
    samples = 1 if data.ndim == 2 else data.shape[2]

    if samples > 4 and height <= 4:
        # separate bands
        samples = data.shape[0]
        width = data.shape[1]
        height = data.shape[2]
    elif samples > 1:
        # contig
        # TODO: avoid full copy
        # TODO: doesn't work with e.g. contig (4, 4, 4)
        src = numpy.ascontiguousarray(numpy.moveaxis(data, -1, 0))

    if tile:
        tile_height, tile_width = tile
        # if width % tile_width or height % tile_height:
        #     raise ValueError('invalid tiles')
        raise NotImplementedError('writing tiles not implemented yet')
    else:
        tile_height = height
        tile_width = width

    if colorspace is None:
        if samples <= 2:
            color_space = OPJ_CLRSPC_GRAY
        elif samples <= 4:
            color_space = OPJ_CLRSPC_SRGB
        else:
            color_space = OPJ_CLRSPC_UNSPECIFIED
    else:
        color_space = opj_colorspace(colorspace)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + 2048  # ?
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size * dst.itemsize

    try:
        with nogil:

            # create memory stream
            memstream.data = <OPJ_UINT8 *>&dst[0]
            memstream.size = dstsize
            memstream.offset = 0
            memstream.written = 0

            stream = opj_memstream_create(&memstream, OPJ_FALSE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            # create image
            memset(&cmptparms, 0, sizeof(cmptparms))
            for i in range(samples):
                cmptparms[i].dx = 1  # subsampling
                cmptparms[i].dy = 1
                cmptparms[i].x0 = 0
                cmptparms[i].y0 = 0
                cmptparms[i].h = height
                cmptparms[i].w = width
                cmptparms[i].sgnd = signed
                cmptparms[i].prec = prec
                cmptparms[i].bpp = prec

            if tile_height > 0:
                image = opj_image_tile_create(samples, cmptparms, color_space)
                if image == NULL:
                    raise Jpeg2kError('opj_image_tile_create failed')
            else:
                image = opj_image_create(samples, cmptparms, color_space)
                if image == NULL:
                    raise Jpeg2kError('opj_image_create failed')

            image.x0 = 0
            image.y0 = 0
            image.x1 = width
            image.y1 = height
            image.color_space = color_space
            image.numcomps = samples

            # set parameters
            opj_set_default_encoder_parameters(&parameters)
            # TODO: this crashes
            # parameters.tcp_numlayers = 1  # single quality layer
            parameters.tcp_rates[0] = rate
            parameters.irreversible = 0
            parameters.numresolution = 1

            if tile_height > 0:
                parameters.tile_size_on = OPJ_TRUE
                parameters.cp_tx0 = 0
                parameters.cp_ty0 = 0
                parameters.cp_tdy = tile_height
                parameters.cp_tdx = tile_width
            else:
                parameters.tile_size_on = OPJ_FALSE
                parameters.cp_tx0 = 0
                parameters.cp_ty0 = 0
                parameters.cp_tdy = 0
                parameters.cp_tdx = 0

            # create and setup encoder
            codec = opj_create_compress(codec_format)
            if codec == NULL:
                raise Jpeg2kError('opj_create_compress failed')

            if verbosity > 0:
                opj_set_error_handler(
                    codec,
                    <opj_msg_callback>j2k_error_callback,
                    NULL)
                if verbosity > 1:
                    opj_set_warning_handler(
                        codec,
                        <opj_msg_callback>j2k_warning_callback,
                        NULL)
                    if verbosity > 2:
                        opj_set_info_handler(
                            codec,
                            <opj_msg_callback>j2k_info_callback,
                            NULL)

            ret = opj_setup_encoder(codec, &parameters, image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_encoder failed')

            ret = opj_start_compress(codec, image, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_start_compress failed')

            if tile_height > 0:
                # TODO: loop over tiles
                ret = opj_write_tile(
                    codec,
                    0,
                    <OPJ_BYTE *>src.data,
                    <OPJ_UINT32>srcsize,
                    stream
                )
            else:
                # TODO: copy data to image.comps[band].data[y, x]
                ret = opj_encode(codec, stream)

            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_encode or opj_write_tile failed')

            ret = opj_end_compress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_end_compress failed')

            if memstream.written > memstream.size:
                raise Jpeg2kError('output buffer too small')

            byteswritten = memstream.written

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpeg2k_decode(data, verbose=0, out=None):
    """Decode JPEG 2000 J2K or JP2 image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        int32_t *band
        uint32_t *u4
        uint16_t *u2
        uint8_t *u1
        int32_t *i4
        int16_t *i2
        int8_t *i1
        ssize_t dstsize
        ssize_t itemsize
        opj_memstream_t memstream
        opj_codec_t *codec = NULL
        opj_image_t *image = NULL
        opj_stream_t *stream = NULL
        opj_image_comp_t *comp = NULL
        opj_dparameters_t parameters
        OPJ_BOOL ret = OPJ_FALSE
        OPJ_CODEC_FORMAT codecformat
        OPJ_UINT32 signed, prec, width, height, samples
        ssize_t i, j
        int verbosity = verbose

    if data is out:
        raise ValueError('cannot decode in-place')

    signature = bytearray(src[:12])
    if signature[:4] == b'\xff\x4f\xff\x51':
        codecformat = OPJ_CODEC_J2K
    elif (signature == b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a' or
          signature[:4] == b'\x0d\x0a\x87\x0a'):
        codecformat = OPJ_CODEC_JP2
    else:
        raise Jpeg2kError('not a J2K or JP2 data stream')

    try:
        memstream.data = <OPJ_UINT8 *>&src[0]
        memstream.size = src.size
        memstream.offset = 0
        memstream.written = 0

        with nogil:
            stream = opj_memstream_create(&memstream, OPJ_TRUE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            codec = opj_create_decompress(codecformat)
            if codec == NULL:
                raise Jpeg2kError('opj_create_decompress failed')

            if verbosity > 0:
                opj_set_error_handler(
                    codec, <opj_msg_callback>j2k_error_callback, NULL)
                if verbosity > 1:
                    opj_set_warning_handler(
                        codec, <opj_msg_callback>j2k_warning_callback, NULL)
                    if verbosity > 2:
                        opj_set_info_handler(
                            codec, <opj_msg_callback>j2k_info_callback, NULL)

            opj_set_default_decoder_parameters(&parameters)

            ret = opj_setup_decoder(codec, &parameters)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_decoder failed')

            ret = opj_read_header(stream, codec, &image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_read_header failed')

            ret = opj_set_decode_area(
                codec,
                image,
                <OPJ_INT32>parameters.DA_x0,
                <OPJ_INT32>parameters.DA_y0,
                <OPJ_INT32>parameters.DA_x1,
                <OPJ_INT32>parameters.DA_y1
            )
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_set_decode_area failed')

            # with nogil:
            ret = opj_decode(codec, stream, image)
            if ret != OPJ_FALSE:
                ret = opj_end_decompress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_decode or opj_end_decompress failed')

            # handle subsampling and color profiles
            if (
                image.color_space != OPJ_CLRSPC_SYCC
                and image.numcomps == 3
                and image.comps[0].dx == image.comps[0].dy
                and image.comps[1].dx != 1
            ):
                image.color_space = OPJ_CLRSPC_SYCC
            elif image.numcomps <= 2:
                image.color_space = OPJ_CLRSPC_GRAY

            if image.color_space == OPJ_CLRSPC_SYCC:
                color_sycc_to_rgb(image)
            if image.icc_profile_buf:
                color_apply_icc_profile(image)
                free(image.icc_profile_buf)
                image.icc_profile_buf = NULL
                image.icc_profile_len = 0

            comp = &image.comps[0]
            signed = comp.sgnd
            prec = comp.prec
            height = comp.h * comp.dy
            width = comp.w * comp.dx
            samples = image.numcomps
            itemsize = (prec + 7) // 8

            for i in range(samples):
                comp = &image.comps[i]
                if comp.sgnd != signed or comp.prec != prec:
                    raise Jpeg2kError('components dtype mismatch')
                if comp.w != width or comp.h != height:
                    raise Jpeg2kError('subsampling not supported')
            if itemsize == 3:
                itemsize = 4
            elif itemsize < 1 or itemsize > 4:
                raise Jpeg2kError('unsupported itemsize %i' % int(itemsize))

        dtype = ('i' if signed else 'u') + ('%i' % itemsize)
        if samples > 1:
            shape = int(height), int(width), int(samples)
        else:
            shape = int(height), int(width)
        out = _create_array(out, shape, dtype)
        dst = out
        dstsize = dst.size * itemsize

        with nogil:
            # memset(<void *>dst.data, 0, dstsize)
            # TODO: support separate in addition to contig samples
            if itemsize == 1:
                if signed:
                    for i in range(samples):
                        i1 = <int8_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            i1[j * samples] = <int8_t>band[j]
                else:
                    for i in range(samples):
                        u1 = <uint8_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            u1[j * samples] = <uint8_t>band[j]
            elif itemsize == 2:
                if signed:
                    for i in range(samples):
                        i2 = <int16_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            i2[j * samples] = <int16_t>band[j]
                else:
                    for i in range(samples):
                        u2 = <uint16_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            u2[j * samples] = <uint16_t>band[j]
            elif itemsize == 4:
                if signed:
                    for i in range(samples):
                        i4 = <int32_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            i4[j * samples] = <int32_t>band[j]
                else:
                    for i in range(samples):
                        u4 = <uint32_t *>dst.data + i
                        band = <int32_t *>image.comps[i].data
                        for j in range(height * width):
                            u4[j * samples] = <uint32_t>band[j]

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    return out


def jpeg2k_version():
    """Return OpenJPEG version string."""
    return 'openjpeg ' + opj_version().decode('utf-8')


# JPEG XR #####################################################################

from jxrlib cimport *


class WmpError(RuntimeError):
    """WMP Exceptions."""
    def __init__(self, func, err):
        msg = {
            WMP_errFail: 'WMP_errFail',
            WMP_errNotYetImplemented: 'WMP_errNotYetImplemented',
            WMP_errAbstractMethod: 'WMP_errAbstractMethod',
            WMP_errOutOfMemory: 'WMP_errOutOfMemory',
            WMP_errFileIO: 'WMP_errFileIO',
            WMP_errBufferOverflow: 'WMP_errBufferOverflow',
            WMP_errInvalidParameter: 'WMP_errInvalidParameter',
            WMP_errInvalidArgument: 'WMP_errInvalidArgument',
            WMP_errUnsupportedFormat: 'WMP_errUnsupportedFormat',
            WMP_errIncorrectCodecVersion: 'WMP_errIncorrectCodecVersion',
            WMP_errIndexNotFound: 'WMP_errIndexNotFound',
            WMP_errOutOfSequence: 'WMP_errOutOfSequence',
            WMP_errNotInitialized: 'WMP_errNotInitialized',
            WMP_errAlphaModeCannotBeTranscoded:
                'WMP_errAlphaModeCannotBeTranscoded',
            WMP_errIncorrectCodecSubVersion:
                'WMP_errIncorrectCodecSubVersion',
            WMP_errMustBeMultipleOf16LinesUntilLastCall:
                'WMP_errMustBeMultipleOf16LinesUntilLastCall',
            WMP_errPlanarAlphaBandedEncRequiresTempFile:
                'WMP_errPlanarAlphaBandedEncRequiresTempFile',
            }.get(err, 'unknown error %i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


cdef ERR WriteWS_Memory(WMPStream *pWS, const void *pv, size_t cb) nogil:
    """Relpacement for WriteWS_Memory to keep track of bytes written."""
    if pWS.state.buf.cbCur + cb < pWS.state.buf.cbCur:
        return WMP_errBufferOverflow
    if pWS.state.buf.cbBuf < pWS.state.buf.cbCur + cb:
        return WMP_errBufferOverflow

    memmove(pWS.state.buf.pbBuf + pWS.state.buf.cbCur, pv, cb)
    pWS.state.buf.cbCur += cb

    # keep track of bytes written
    if pWS.state.buf.cbCur > pWS.state.buf.cbBufCount:
        pWS.state.buf.cbBufCount = pWS.state.buf.cbCur

    return WMP_errSuccess


cdef ERR WriteWS_Realloc(WMPStream *pWS, const void *pv, size_t cb) nogil:
    """Relpacement for WriteWS_Memory to realloc buffers on overflow.

    Only use with buffers allocated by malloc.

    """
    cdef:
        size_t newsize = pWS.state.buf.cbCur + cb
    if newsize < pWS.state.buf.cbCur:
        return WMP_errBufferOverflow
    if pWS.state.buf.cbBuf < newsize:
        if newsize <= pWS.state.buf.cbBuf * 1.125:
            # moderate upsize: overallocate
            newsize = newsize + newsize // 8
            newsize = (((newsize-1) // 4096) + 1) * 4096
        else:
            # major upsize: resize to exact size
            newsize = newsize + 1
        pWS.state.buf.pbBuf = <U8 *>realloc(
            <void *>pWS.state.buf.pbBuf,
            newsize)
        if pWS.state.buf.pbBuf == NULL:
            return WMP_errOutOfMemory
        pWS.state.buf.cbBuf = newsize

    memmove(pWS.state.buf.pbBuf + pWS.state.buf.cbCur, pv, cb)
    pWS.state.buf.cbCur += cb

    # keep track of bytes written
    if pWS.state.buf.cbCur > pWS.state.buf.cbBufCount:
        pWS.state.buf.cbBufCount = pWS.state.buf.cbCur

    return WMP_errSuccess


cdef Bool EOSWS_Realloc(WMPStream *pWS) nogil:
    """Relpacement for EOSWS_Memory."""
    # return pWS.state.buf.cbBuf <= pWS.state.buf.cbCur
    return 1


cdef ERR PKCodecFactory_CreateDecoderFromBytes(void *bytes, size_t len,
                                               PKImageDecode **ppDecode) nogil:
    """Create PKImageDecode from byte string."""
    cdef:
        char *pExt = NULL
        const PKIID *pIID = NULL
        WMPStream *stream = NULL
        PKImageDecode *decoder = NULL
        ERR err

    # get decode PKIID
    err = GetImageDecodeIID('.jxr', &pIID)
    if err:
        return err
    # create stream
    err = CreateWS_Memory(&stream, bytes, len)
    if err:
        return err
    # create decoder
    err = PKCodecFactory_CreateCodec(pIID, <void **>ppDecode)
    if err:
        return err
    # attach stream to decoder
    decoder = ppDecode[0]
    err = decoder.Initialize(decoder, stream)
    if err:
        return err
    decoder.fStreamOwner = 1
    return WMP_errSuccess


cdef ERR jxr_decode_guid(PKPixelFormatGUID *pixelformat, int *typenum,
                         ssize_t *samples, U8 *alpha) nogil:
    """Return dtype, samples, alpha from GUID.

    Change pixelformat to output format in-place.

    """
    alpha[0] = 0
    samples[0] = 1

    # bool
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormatBlackWhite):
        pixelformat[0] = GUID_PKPixelFormat8bppGray
        typenum[0] = numpy.NPY_BOOL
        return WMP_errSuccess

    # uint8
    typenum[0] = numpy.NPY_UINT8
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat8bppGray):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bppRGB):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppRGB555):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppRGB565):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bppBGR):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGB):
        pixelformat[0] = GUID_PKPixelFormat24bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppBGRA):
        pixelformat[0] = GUID_PKPixelFormat32bppRGBA
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppPRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGBE):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppCMYK):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bppCMYKAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat24bpp3Channels):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bpp4Channels):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bpp5Channels):
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp6Channels):
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat56bpp7Channels):
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp8Channels):
        samples[0] = 8
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bpp3ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat40bpp4ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp5ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat56bpp6ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp7ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 8
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat72bpp8ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 9
        return WMP_errSuccess

    # uint16
    typenum[0] = numpy.NPY_UINT16
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGray):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppRGB101010):
        pixelformat[0] = GUID_PKPixelFormat48bppRGB
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGB):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppPRGBA):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppCMYK):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bppCMYKAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bpp3Channels):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp4Channels):
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bpp5Channels):
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bpp6Channels):
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat112bpp7Channels):
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bpp8Channels):
        samples[0] = 8
        return WMP_errSuccess

    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bpp3ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat80bpp4ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 5
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bpp5ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 6
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat112bpp6ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 7
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bpp7ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 8
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat144bpp8ChannelsAlpha):
        alpha[0] = 2
        samples[0] = 9
        return WMP_errSuccess

    # float32
    typenum[0] = numpy.NPY_FLOAT32
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat32bppGrayFloat):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat96bppRGBFloat):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBFloat):
        pixelformat[0] = GUID_PKPixelFormat96bppRGBFloat
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppRGBAFloat):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat128bppPRGBAFloat):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess

    # float16
    typenum[0] = numpy.NPY_FLOAT16
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat16bppGrayHalf):
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat48bppRGBHalf):
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBHalf):
        pixelformat[0] = GUID_PKPixelFormat48bppRGBHalf
        samples[0] = 3
        return WMP_errSuccess
    if IsEqualGUID(pixelformat, &GUID_PKPixelFormat64bppRGBAHalf):
        alpha[0] = 2
        samples[0] = 4
        return WMP_errSuccess

    return WMP_errUnsupportedFormat


cdef PKPixelFormatGUID jxr_encode_guid(numpy.dtype dtype, ssize_t samples,
                                       int photometric, int *alpha) nogil:
    """Return pixel format GUID from dtype, samples, and photometric."""
    cdef int typenum = dtype.type_num
    if samples == 1:
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat8bppGray
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat16bppGray
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat32bppGrayFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat16bppGrayHalf
        if typenum == numpy.NPY_BOOL:
            return GUID_PKPixelFormatBlackWhite
    if samples == 3:
        if typenum == numpy.NPY_UINT8:
            if photometric < 0 or photometric == PK_PI_RGB:
                return GUID_PKPixelFormat24bppRGB
            return GUID_PKPixelFormat24bpp3Channels
        if typenum == numpy.NPY_UINT16:
            if photometric < 0 or photometric == PK_PI_RGB:
                return GUID_PKPixelFormat48bppRGB
            return GUID_PKPixelFormat48bpp3Channels
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat96bppRGBFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat48bppRGBHalf
    if samples == 4:
        if typenum == numpy.NPY_UINT8:
            if photometric < 0 or photometric == PK_PI_RGB:
                alpha[0] = 1
                return GUID_PKPixelFormat32bppRGBA
            if photometric == PK_PI_CMYK:
                return GUID_PKPixelFormat32bppCMYK
            if alpha:
                return GUID_PKPixelFormat32bpp3ChannelsAlpha
            return GUID_PKPixelFormat32bpp4Channels
        if typenum == numpy.NPY_UINT16:
            if photometric < 0 or photometric == PK_PI_RGB:
                alpha[0] = 1
                return GUID_PKPixelFormat64bppRGBA
            if photometric == PK_PI_CMYK:
                return GUID_PKPixelFormat64bppCMYK
            if alpha:
                return GUID_PKPixelFormat64bpp3ChannelsAlpha
            return GUID_PKPixelFormat64bpp4Channels
        alpha[0] = 1
        if typenum == numpy.NPY_FLOAT32:
            return GUID_PKPixelFormat128bppRGBAFloat
        if typenum == numpy.NPY_FLOAT16:
            return GUID_PKPixelFormat64bppRGBAHalf
    if samples == 5:
        if photometric == PK_PI_CMYK:
            alpha[0] = 1
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat40bppCMYKAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat80bppCMYKAlpha
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat40bpp4ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat80bpp4ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat40bpp5Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat80bpp5Channels
    if samples == 6:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat48bpp5ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat96bpp5ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat48bpp6Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat96bpp6Channels
    if samples == 7:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat56bpp6ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat112bpp6ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat56bpp7Channels
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat112bpp7Channels
    if samples == 8:
        if alpha[0]:
            if typenum == numpy.NPY_UINT8:
                return GUID_PKPixelFormat64bpp7ChannelsAlpha
            if typenum == numpy.NPY_UINT16:
                return GUID_PKPixelFormat128bpp7ChannelsAlpha
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat64bpp8Channels
        elif typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat128bpp8Channels
    if samples == 9:
        alpha[0] = 1
        if typenum == numpy.NPY_UINT8:
            return GUID_PKPixelFormat72bpp8ChannelsAlpha
        if typenum == numpy.NPY_UINT16:
            return GUID_PKPixelFormat144bpp8ChannelsAlpha
    return GUID_PKPixelFormatDontCare


cdef int jxr_encode_photometric(photometric):
    """Return PK_PI value from photometric argument."""
    if photometric is None:
        return -1
    if isinstance(photometric, int):
        if photometric not in (-1, PK_PI_W0, PK_PI_B0, PK_PI_RGB, PK_PI_CMYK):
            raise ValueError('photometric interpretation not supported')
        return photometric
    photometric = photometric.upper()
    if photometric[:3] == 'RGB':
        return PK_PI_RGB
    if photometric == 'WHITEISZERO' or photometric == 'MINISWHITE':
        return PK_PI_W0
    if photometric in ('BLACKISZERO', 'MINISBLACK', 'GRAY'):
        return PK_PI_B0
    if photometric == 'CMYK' or photometric == 'SEPARATED':
        return PK_PI_CMYK
    # TODO: support more photometric modes
    # if photometric == 'YCBCR':
    #     return PK_PI_YCbCr
    # if photometric == 'CIELAB':
    #     return PK_PI_CIELab
    # if photometric == 'TRANSPARENCYMASK' or photometric == 'MASK':
    #     return PK_PI_TransparencyMask
    # if photometric == 'RGBPALETTE' or photometric == 'PALETTE':
    #     return PK_PI_RGBPalette
    raise ValueError('photometric interpretation not supported')


# Y, U, V, YHP, UHP, VHP
# optimized for PSNR
cdef int *DPK_QPS_420 = [
    66, 65, 70, 72, 72, 77, 59, 58, 63, 64, 63, 68, 52, 51, 57, 56, 56, 61, 48,
    48, 54, 51, 50, 55, 43, 44, 48, 46, 46, 49, 37, 37, 42, 38, 38, 43, 26, 28,
    31, 27, 28, 31, 16, 17, 22, 16, 17, 21, 10, 11, 13, 10, 10, 13, 5, 5, 6, 5,
    5, 6, 2, 2, 3, 2, 2, 2]

cdef int *DPK_QPS_8 = [
    67, 79, 86, 72, 90, 98, 59, 74, 80, 64, 83, 89, 53, 68, 75, 57, 76, 83, 49,
    64, 71, 53, 70, 77, 45, 60, 67, 48, 67, 74, 40, 56, 62, 42, 59, 66, 33, 49,
    55, 35, 51, 58, 27, 44, 49, 28, 45, 50, 20, 36, 42, 20, 38, 44, 13, 27, 34,
    13, 28, 34, 7, 17, 21, 8, 17, 21, 2, 5, 6, 2, 5, 6]

cdef int *DPK_QPS_16 = [
    197, 203, 210, 202, 207, 213, 174, 188, 193, 180, 189, 196, 152, 167, 173,
    156, 169, 174, 135, 152, 157, 137, 153, 158, 119, 137, 141, 119, 138, 142,
    102, 120, 125, 100, 120, 124, 82, 98, 104, 79, 98, 103, 60, 76, 81, 58, 76,
    81, 39, 52, 58, 36, 52, 58, 16, 27, 33, 14, 27, 33, 5, 8, 9, 4, 7, 8]

cdef int *DPK_QPS_16f = [
    148, 177, 171, 165, 187, 191, 133, 155, 153, 147, 172, 181, 114, 133, 138,
    130, 157, 167, 97, 118, 120, 109, 137, 144, 76, 98, 103, 85, 115, 121, 63,
    86, 91, 62, 96, 99, 46, 68, 71, 43, 73, 75, 29, 48, 52, 27, 48, 51, 16, 30,
    35, 14, 29, 34, 8, 14, 17, 7,  13, 17, 3, 5, 7, 3, 5, 6]

cdef int *DPK_QPS_32f = [
    194, 206, 209, 204, 211, 217, 175, 187, 196, 186, 193, 205, 157, 170, 177,
    167, 180, 190, 133, 152, 156, 144, 163, 168, 116, 138, 142, 117, 143, 148,
    98, 120, 123,  96, 123, 126, 80, 99, 102, 78, 99, 102, 65, 79, 84, 63, 79,
    84, 48, 61, 67, 45, 60, 66, 27, 41, 46, 24, 40, 45, 3, 22, 24,  2, 21, 22]


cdef U8 jxr_quantization(int *qps, double quality, ssize_t i) nogil:
    """Return quantization from DPK_QPS table."""
    cdef:
        ssize_t qi = <ssize_t>(10.0 * quality)
        double qf = 10.0 * quality - <double>qi
        int *qps0 = qps + qi * 6
        int *qps1 = qps0 + 6
    return <U8>(<double>qps0[i] * (1.0 - qf) + <double>qps1[i] * qf + 0.5)


cdef ERR jxr_set_encoder(CWMIStrCodecParam *wmiscp, PKPixelInfo *pixelinfo,
                         double quality, int alpha, int pi) nogil:
    """Set encoder compression parameters from level argument and pixel format.

    Code and tables adapted from jxrlib's JxrEncApp.c.

    ImageQuality Q(BD==1) Q(BD==8)    Q(BD==16)   Q(BD==32F)  Subsample Overlap
    [0.0, 0.5)   8-IQ*5   (see table) (see table) (see table) 4:2:0     2
    [0.5, 1.0)   8-IQ*5   (see table) (see table) (see table) 4:4:4     1
    [1.0, 1.0]   1        1           1           1           4:4:4     0

    """
    cdef:
        int *qps

    # default: lossless, no tiles
    wmiscp.uiDefaultQPIndex = 1
    wmiscp.uiDefaultQPIndexAlpha = 1
    wmiscp.olOverlap = OL_NONE
    wmiscp.cfColorFormat = YUV_444
    wmiscp.sbSubband = SB_ALL
    wmiscp.bfBitstreamFormat = SPATIAL
    wmiscp.bProgressiveMode = 0
    wmiscp.cNumOfSliceMinus1H = 0
    wmiscp.cNumOfSliceMinus1V = 0
    wmiscp.uAlphaMode = 2 if alpha else 0
    # wmiscp.bdBitDepth = BD_LONG

    if pi == PK_PI_CMYK:
        wmiscp.cfColorFormat = CMYK

    if quality <= 0.0 or quality == 1.0 or quality >= 100.0:
        return WMP_errSuccess
    if quality > 1.0:
        quality /= 100.0
    if quality >= 1.0:
        return WMP_errSuccess
    if quality < 0.5:
        # overlap
        wmiscp.olOverlap = OL_TWO

    if quality < 0.5 and pixelinfo.uBitsPerSample <= 8 and pi != PK_PI_CMYK:
        # chroma sub-sampling
        wmiscp.cfColorFormat = YUV_420

    # bit depth
    if pixelinfo.bdBitDepth == BD_1:
        wmiscp.uiDefaultQPIndex = <U8>(8 - 5.0 * quality + 0.5)
    else:
        # remap [0.8, 0.866, 0.933, 1.0] to [0.8, 0.9, 1.0, 1.1]
        # to use 8-bit DPK QP table (0.933 == Photoshop JPEG 100)
        if (
            quality > 0.8 and
            pixelinfo.bdBitDepth == BD_8 and
            wmiscp.cfColorFormat != YUV_420 and
            wmiscp.cfColorFormat != YUV_422
        ):
            quality = 0.8 + (quality - 0.8) * 1.5

        if wmiscp.cfColorFormat == YUV_420 or wmiscp.cfColorFormat == YUV_422:
            qps = DPK_QPS_420
        elif pixelinfo.bdBitDepth == BD_8:
            qps = DPK_QPS_8
        elif pixelinfo.bdBitDepth == BD_16:
            qps = DPK_QPS_16
        elif pixelinfo.bdBitDepth == BD_16F:
            qps = DPK_QPS_16f
        else:
            qps = DPK_QPS_32f

        wmiscp.uiDefaultQPIndex = jxr_quantization(qps, quality, 0)
        wmiscp.uiDefaultQPIndexU = jxr_quantization(qps, quality, 1)
        wmiscp.uiDefaultQPIndexV = jxr_quantization(qps, quality, 2)
        wmiscp.uiDefaultQPIndexYHP = jxr_quantization(qps, quality, 3)
        wmiscp.uiDefaultQPIndexUHP = jxr_quantization(qps, quality, 4)
        wmiscp.uiDefaultQPIndexVHP = jxr_quantization(qps, quality, 5)

    return WMP_errSuccess


def jpegxr_encode(data, level=None, photometric=None, hasalpha=None,
                  resolution=None, out=None):
    """Encode numpy array to JPEG XR image."""
    cdef:
        numpy.ndarray src = data
        numpy.dtype dtype = src.dtype
        const uint8_t[::1] dst  # must be const to write to bytes
        U8 *outbuffer = NULL
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        size_t byteswritten = 0
        ssize_t samples
        int pi = jxr_encode_photometric(photometric)
        int alpha = 1 if hasalpha else 0
        float quality = 1.0 if level is None else level

        WMPStream *stream = NULL
        PKImageEncode *encoder = NULL
        PKPixelFormatGUID pixelformat
        PKPixelInfo pixelinfo
        float rx = 96.0
        float ry = 96.0
        I32 width
        I32 height
        U32 stride
        ERR err

    if (
        dtype not in (numpy.uint8, numpy.uint16, numpy.bool,
                      numpy.float16, numpy.float32)
        and data.ndim in (2, 3)
        and numpy.PyArray_ISCONTIGUOUS(data)
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    if resolution:
        rx, ry = resolution

    width = <I32>data.shape[1]
    height = <I32>data.shape[0]
    stride = <U32>data.strides[0]
    samples = 1 if data.ndim == 2 else data.shape[2]

    if width < MB_WIDTH_PIXEL or height < MB_HEIGHT_PIXEL:
        raise ValueError('invalid data shape')

    if dtype == numpy.bool:
        if data.ndim != 2:
            raise ValueError('invalid data shape, strides, or dtype')
        src = numpy.packbits(data, axis=-1)
        stride = <U32>src.strides[0]
        srcsize //= 8

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize <= 0:
            dstsize = srcsize // 2
            dstsize = (((dstsize - 1) // 4096) + 1) * 4096
        elif dstsize < 4096:
            dstsize = 4096
        outbuffer = <U8 *>malloc(dstsize)
        if outbuffer == NULL:
            raise MemoryError('failed to allocate ouput buffer')
    else:
        dst = out
        dstsize = dst.size * dst.itemsize

    try:
        with nogil:
            pixelformat = jxr_encode_guid(dtype, samples, pi, &alpha)
            if IsEqualGUID(&pixelformat, &GUID_PKPixelFormatDontCare):
                raise ValueError('PKPixelFormatGUID not found')
            pixelinfo.pGUIDPixFmt = &pixelformat

            err = PixelFormatLookup(&pixelinfo, LOOKUP_FORWARD)
            if err:
                raise WmpError('PixelFormatLookup', err)

            if outbuffer == NULL:
                err = CreateWS_Memory(&stream, <void *>&dst[0], dstsize)
                if err:
                    raise WmpError('CreateWS_Memory', err)
                stream.Write = WriteWS_Memory
            else:
                err = CreateWS_Memory(&stream, <void *>outbuffer, dstsize)
                if err:
                    raise WmpError('CreateWS_Memory', err)
                stream.Write = WriteWS_Realloc
                stream.EOS = EOSWS_Realloc

            err = PKImageEncode_Create_WMP(&encoder)
            if err:
                raise WmpError('PKImageEncode_Create_WMP', err)

            err = encoder.Initialize(
                encoder,
                stream,
                &encoder.WMP.wmiSCP,
                sizeof(CWMIStrCodecParam)
            )
            if err:
                raise WmpError('PKImageEncode_Initialize', err)

            jxr_set_encoder(
                &encoder.WMP.wmiSCP,
                &pixelinfo,
                quality,
                alpha,
                pi)

            err = encoder.SetPixelFormat(encoder, pixelformat)
            if err:
                raise WmpError('PKImageEncode_SetPixelFormat', err)

            err = encoder.SetSize(encoder, width, height)
            if err:
                raise WmpError('PKImageEncode_SetSize', err)

            err = encoder.SetResolution(encoder, rx, ry)
            if err:
                raise WmpError('PKImageEncode_SetResolution', err)

            err = encoder.WritePixels(encoder, height, <U8 *>src.data, stride)
            if err:
                raise WmpError('PKImageEncode_WritePixels', err)

            byteswritten = stream.state.buf.cbBufCount
            dstsize = stream.state.buf.cbBuf
            if outbuffer != NULL:
                outbuffer = stream.state.buf.pbBuf

    except Exception:
        if outbuffer != NULL:
            if stream != NULL:
                outbuffer = stream.state.buf.pbBuf
            free(outbuffer)
        raise
    finally:
        if encoder != NULL:
            PKImageEncode_Release(&encoder)
        elif stream != NULL:
            stream.Close(&stream)

    if outbuffer != NULL:
        out = _create_output(outtype, <ssize_t>byteswritten, <char *>outbuffer)
        free(outbuffer)
        return out

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpegxr_decode(data, out=None):
    """Decode JPEG XR image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        numpy.dtype dtype
        const uint8_t[::1] src = data
        PKImageDecode *decoder = NULL
        PKFormatConverter *converter = NULL
        PKPixelFormatGUID pixelformat
        PKRect rect
        I32 width
        I32 height
        U32 stride
        ERR err
        U8 alpha
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t samples
        int typenum

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            err = PKCodecFactory_CreateDecoderFromBytes(
                <void *>&src[0],
                srcsize,
                &decoder)
            if err:
                raise WmpError('PKCodecFactory_CreateDecoderFromBytes', err)

            err = PKImageDecode_GetSize(decoder, &width, &height)
            if err:
                raise WmpError('PKImageDecode_GetSize', err)

            err = PKImageDecode_GetPixelFormat(decoder, &pixelformat)
            if err:
                raise WmpError('PKImageDecode_GetPixelFormat', err)

            err = jxr_decode_guid(&pixelformat, &typenum, &samples, &alpha)
            if err:
                raise WmpError('jxr_decode_guid', err)
            decoder.WMP.wmiSCP.uAlphaMode = alpha

            err = PKCodecFactory_CreateFormatConverter(&converter)
            if err:
                raise WmpError('PKCodecFactory_CreateFormatConverter', err)

            err = PKFormatConverter_Initialize(
                converter,
                decoder,
                NULL,
                pixelformat
            )
            if err:
                raise WmpError('PKFormatConverter_Initialize', err)

            with gil:
                shape = height, width
                if samples > 1:
                    shape += samples,
                dtype = PyArray_DescrNewFromType(typenum)
                out = _create_array(out, shape, dtype)
                dst = out
                dstsize = dst.size * dst.itemsize

            rect.X = 0
            rect.Y = 0
            rect.Width = <I32>dst.shape[1]
            rect.Height = <I32>dst.shape[0]
            stride = <U32>dst.strides[0]

            memset(<void *>dst.data, 0, dstsize)  # TODO: still necessary?
            # TODO: check alignment issues
            err = PKFormatConverter_Copy(
                converter,
                &rect,
                <U8 *>dst.data,
                stride)
        if err:
            raise WmpError('PKFormatConverter_Copy', err)

    finally:
        if converter != NULL:
            PKFormatConverter_Release(&converter)
        if decoder != NULL:
            PKImageDecode_Release(&decoder)

    return out


def jpegxr_version():
    """Return jxrlib version string."""
    cdef uint32_t ver = WMP_SDK_VERSION
    return 'jxrlib %i.%i' % (ver >> 8, ver & 255)


# JPEG 12-bit #################################################################

# JPEG 12-bit codecs are implemented in a separate extension module
# due to header and link conflicts with JPEG 8-bit.

try:
    from ._jpeg12 import (
        jpeg12_decode, jpeg12_encode, jpeg12_version, Jpeg12Error
    )
except ImportError:

    Jpeg12Error = RuntimeError

    def jpeg12_decode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpeg12_decode')

    def jpeg12_encode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpeg12_encode')

    def jpeg12_version():
        """Not available."""
        return 'libjpeg12 n/a'


# JPEG LS #################################################################

# JPEG-LS codecs are implemented in a separate extension module
#   because CharLS 2.1 is not commonly available yet.
# TODO: move implementation here once charls2 is available in Debian and
#   Python 2.7 is dropped.

try:
    from ._jpegls import (
        jpegls_decode, jpegls_encode, jpegls_version, JpegLsError
    )
except ImportError:

    JpegLsError = RuntimeError

    def jpegls_decode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpegls_decode')

    def jpegls_encode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpegls_encode')

    def jpegls_version():
        """Not available."""
        return 'charls n/a'


# JPEG XL #################################################################

# JPEG-XL codecs are implemented in a separate extension module
#   because Brunsli is experimental and not commonly available yet.
# TODO: move implementation here once Brunsli is stable, available in Debian
#   and Python 2.7 is dropped.

try:
    from ._jpegxl import (
        jpegxl_decode, jpegxl_encode, jpegxl_version, JpegXlError
    )
except ImportError:

    JpegXlError = RuntimeError

    def jpegxl_decode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpegxl_decode')

    def jpegxl_encode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpegxl_encode')

    def jpegxl_version():
        """Not available."""
        return 'brunsli n/a'


# ZFP #########################################################################

# ZFP codecs are implemented in a separate extension module
#   because ZFP is not commonly available yet and might require OpenMP/CUDA.
# TODO: move implementation here once libzfp is available in Debian

try:
    from ._zfp import (
        zfp_decode, zfp_encode, zfp_version, ZfpError
    )
except ImportError:

    ZfpError = RuntimeError

    def zfp_decode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('zfp_decode')

    def zfp_encode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('zfp_encode')

    def zfp_version():
        """Not available."""
        return 'zfp n/a'


# Imagecodecs Lite ############################################################

# Imagecodecs_lite contains all codecs that do not depend on 3rd party
# external C libraries.

from ._imagecodecs_lite import (
    none_decode, none_encode,
    numpy_decode, numpy_encode, numpy_version,
    delta_decode, delta_encode,
    xor_decode, xor_encode,
    floatpred_decode, floatpred_encode,
    bitorder_decode, bitorder_encode,
    packbits_decode, packbits_encode,
    packints_decode, packints_encode,
    lzw_decode, lzw_encode,
    icd_version, IcdError
)


###############################################################################

# TODO: add options for OpenMP and releasing GIL
# TODO: split into individual extensions?
# TODO: Base64
# TODO: BMP
# TODO: CCITT and JBIG; JBIG-KIT is GPL
# TODO: LZO; http://www.oberhumer.com/opensource/lzo/ is GPL
# TODO: TIFF via libtiff
# TODO: LERC via https://github.com/Esri/lerc; patented but Apache licensed.
# TODO: OpenEXR via ILM's library; C++
