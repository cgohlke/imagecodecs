# imagecodecs.py

# Copyright (c) 2008-2024, Christoph Gohlke
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

r"""Image transformation, compression, and decompression codecs.

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in Tifffile,
Czifile, Zarr, kerchunk, and other scientific image input/output packages.

Decode and/or encode functions are implemented for Zlib (DEFLATE), GZIP, LZMA,
ZStandard (ZSTD), Blosc, Brotli, Snappy, BZ2, LZ4, LZ4F, LZ4HC, LZ4H5, LZW,
LZO, LZF, LZFSE, LZHAM, PGLZ (PostgreSQL LZ), RCOMP (Rice), ZFP, SZ3, Pcodec,
SPERR, AEC, SZIP, LERC, EER, NPY, BCn, DDS, BMP, PNG, APNG, GIF, TIFF, WebP,
JPEG 8 and 12-bit, Lossless JPEG (LJPEG, LJ92, JPEGLL), JPEG 2000 (JP2, J2K),
JPEG LS, JPEG XL, JPEG XS, JPEG XR (WDP, HD Photo), Ultra HDR (JPEG_R),
MOZJPEG, AVIF, HEIF, QOI, RGBE (HDR), Jetraw, DICOMRLE, PackBits,
Packed Integers, Delta, XOR Delta, Floating Point Predictor, Bitorder reversal,
Byteshuffle, Bitshuffle, Float24 (24-bit floating point),
Quantize (Scale, BitGroom, BitRound, GranularBR), and
CMS (color space transformations).
Checksum functions are implemented for crc32, adler32, fletcher32, and
Jenkins lookup3.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.9.22
:DOI: `10.5281/zenodo.6915978 <https://doi.org/10.5281/zenodo.6915978>`_

Quickstart
----------

Install the imagecodecs package and all dependencies from the
`Python Package Index <https://pypi.org/project/imagecodecs/>`_::

    python -m pip install -U "imagecodecs[all]"

Imagecodecs is also available in other package repositories such as
`Anaconda <https://anaconda.org/conda-forge/imagecodecs>`_,
`MSYS2 <https://packages.msys2.org/base/mingw-w64-python-imagecodecs>`_, and
`MacPorts <https://ports.macports.org/port/py-imagecodecs/summary>`_.

See `Requirements`_ and `Notes`_ for building from source.

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/imagecodecs>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.6, 3.13.0rc2 64-bit
- `Numpy <https://pypi.org/project/numpy>`_ 2.1.1
- `numcodecs <https://pypi.org/project/numcodecs/>`_ 0.13.0
  (optional, for Zarr compatible codecs)

Build requirements:

- `Cython <https://github.com/cython/cython>`_ 3.0.11
- `brotli <https://github.com/google/brotli>`_ 1.1.0
- `brunsli <https://github.com/google/brunsli>`_ 0.1
- `bzip2 <https://gitlab.com/bzip2/bzip2>`_ 1.0.8
- `c-blosc <https://github.com/Blosc/c-blosc>`_ 1.21.6
- `c-blosc2 <https://github.com/Blosc/c-blosc2>`_ 2.15.1
- `charls <https://github.com/team-charls/charls>`_ 2.4.2
- `giflib <https://sourceforge.net/projects/giflib/>`_ 5.2.2
- `jetraw <https://github.com/Jetraw/Jetraw>`_ 23.03.16.4
- `jxrlib <https://github.com/cgohlke/jxrlib>`_ 1.2
- `lcms2 <https://github.com/mm2/Little-CMS>`_ 2.16
- `lerc <https://github.com/Esri/lerc>`_ 4.0.4
- `libaec <https://gitlab.dkrz.de/k202009/libaec>`_ 1.1.3
- `libavif <https://github.com/AOMediaCodec/libavif>`_ 1.1.1
  (`aom <https://aomedia.googlesource.com/aom>`_ 3.10.0,
  `dav1d <https://github.com/videolan/dav1d>`_ 1.4.3,
  `rav1e <https://github.com/xiph/rav1e>`_ 0.7.1,
  `svt-av1 <https://gitlab.com/AOMediaCodec/SVT-AV1>`_ 2.2.1
  `libyuv <https://chromium.googlesource.com/libyuv/libyuv>`_ main)
- `libdeflate <https://github.com/ebiggers/libdeflate>`_ 1.21
- `libheif <https://github.com/strukturag/libheif>`_ 1.18.2
  (`libde265 <https://github.com/strukturag/libde265>`_ 1.0.15,
  `x265 <https://bitbucket.org/multicoreware/x265_git/src/master/>`_ 3.6)
- `libjpeg-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_ 3.0.4
- `libjxl <https://github.com/libjxl/libjxl>`_ 0.11.0
- `libjxs <https://jpeg.org/jpegxs/software.html>`_ 2.0.2
- `liblzma <https://github.com/tukaani-project/xz>`_ 5.6.2
- `libpng <https://github.com/glennrp/libpng>`_ 1.6.44
- `libpng-apng <https://sourceforge.net/projects/libpng-apng/>`_ 1.6.44
- `libtiff <https://gitlab.com/libtiff/libtiff>`_ 4.7.0
- `libultrahdr <https://github.com/google/libultrahdr>`_ 1.2.0
- `libwebp <https://github.com/webmproject/libwebp>`_ 1.4.0
- `lz4 <https://github.com/lz4/lz4>`_ 1.10.0
- `lzfse <https://github.com/lzfse/lzfse/>`_ 1.0
- `lzham_codec <https://github.com/richgel999/lzham_codec/>`_ 1.0
- `lzokay <https://github.com/AxioDL/lzokay>`_ db2df1f
- `mozjpeg <https://github.com/mozilla/mozjpeg>`_ 4.1.5
- `openjpeg <https://github.com/uclouvain/openjpeg>`_ 2.5.2
- `pcodec <https://github.com/mwlon/pcodec>`_ 0.3.1
- `snappy <https://github.com/google/snappy>`_ 1.2.1
- `sperr <https://github.com/NCAR/SPERR>`_ 0.8.2
- `sz3 <https://github.com/szcompressor/SZ3>`_ 3.1.8 (3.2.0 crashes)
- `zfp <https://github.com/LLNL/zfp>`_ 1.0.1
- `zlib <https://github.com/madler/zlib>`_ 1.3.1
- `zlib-ng <https://github.com/zlib-ng/zlib-ng>`_ 2.2.2
- `zopfli <https://github.com/google/zopfli>`_ 1.0.3
- `zstd <https://github.com/facebook/zstd>`_ 1.5.6

Vendored requirements:

- `bcdec.h <https://github.com/iOrange/bcdec>`_ 3b29f8f
- `bitshuffle <https://github.com/kiyo-masui/bitshuffle>`_ 0.5.1
- `cfitsio ricecomp.c <https://heasarc.gsfc.nasa.gov/fitsio/>`_ modified
- `h5checksum.c <https://github.com/HDFGroup/hdf5/>`_ modified
- `jpg_0XC3.cpp
  <https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_
  modified
- `liblj92
  <https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/>`_ modified
- `liblzf <http://oldhome.schmorp.de/marc/liblzf.html>`_ 3.6
- `libspng <https://github.com/randy408/libspng>`_ 0.7.4
- `nc4var.c <https://github.com/Unidata/netcdf-c/blob/main/libsrc4/nc4var.c>`_
  modified
- `pg_lzcompress.c <https://github.com/postgres/postgres>`_ modified
- `qoi.h <https://github.com/phoboslab/qoi/>`_ 36190eb
- `rgbe.c <https://www.graphics.cornell.edu/~bjw/rgbe/rgbe.c>`_ modified

Test requirements:

- `tifffile <https://pypi.org/project/tifffile>`_ 2024.9.20
- `czifile <https://pypi.org/project/czifile>`_ 2019.7.2
- `zarr <https://github.com/zarr-developers/zarr-python>`_ 2.18.2
- `python-blosc <https://github.com/Blosc/python-blosc>`_ 1.11.2
- `python-blosc2 <https://github.com/Blosc/python-blosc2>`_ 2.7.1
- `python-brotli <https://github.com/google/brotli/tree/master/python>`_ 1.0.9
- `python-lz4 <https://github.com/python-lz4/python-lz4>`_ 4.3.3
- `python-lzf <https://github.com/teepark/python-lzf>`_ 0.2.6
- `python-snappy <https://github.com/andrix/python-snappy>`_ 0.7.2
- `python-zstd <https://github.com/sergey-dryabzhinsky/python-zstd>`_ 1.5.5.1
- `pyliblzfse <https://github.com/ydkhatri/pyliblzfse>`_ 0.4.1
- `zopflipy <https://github.com/hattya/zopflipy>`_ 1.10

Revisions
---------

2024.9.22

- Pass 7644 tests.
- Use libjpeg-turbo for all Lossless JPEG bit-depths if possible (#105).
- Fix PackBits encoder fails to skip short replication blocks (#107).
- Fix JPEG2K encoder leaving trailing random bytes (#104).
- Fix encoding and decoding JPEG XL with custom bitspersample (#102).
- Improve error handling in lzf_decode (#103).
- Add Ultra HDR (JPEG_R) codec based on libultrahdr library (#108).
- Add JPEGXS codec based on libjxs library (source only).
- Add SZ3 codec based on SZ3 library.
- Deprecate Python 3.9, support Python 3.13.

2024.6.1

- Fix segfault in sperr_decode.
- Fix segfault when strided-decoding into buffers with unexpected shapes (#98).
- Fix jpeg2k_encoder output buffer too small (#101).
- Add PCODEC codec based on pcodec library.
- Support NumPy 2.

2024.1.1

- Add 8/24-bit BMP codec.
- Add SPERR codec based on SPERR library.
- Add LZO decoder based on lzokay library.
- Add DICOMRLE decoder.
- Enable float16 in CMS codec.
- Enable MCT for lossless JPEG2K encoder (#88).
- Ignore pad-byte in PackBits decoder (#86).
- Fix heif_write_callback error message not set.
- Require lcms2 2.16 with issue-420 fixes.
- Require libjxl 0.9, libaec 1.1, Cython 3.

2023.9.18

- Rebuild with updated dependencies fixes CVE-2024-4863.

2023.9.4

- Map avif_encode level parameter to quality (breaking).
- Support monochrome images in avif_encode.
- Add numthreads parameter to avif_decode (fix imread of AVIF).
- Add quantize filter (BitGroom, BitRound, GBR) via nc4var.c.
- Add LZ4H5 codec.
- Support more BCn compressed DDS fourcc types.
- Require libavif 1.0.

2023.8.12

- Add EER (Electron Event Representation) decoder.
- Add option to pass initial value to crc32 and adler32 checksum functions.
- Add fletcher32 and lookup3 checksum functions via HDF5's h5checksum.c.
- Add Checksum codec for numcodecs.

2023.7.10

- Rebuild with optimized compile flags.

2023.7.4

- Add BCn and DDS decoder via bcdec library.
- Add functions to transcode JPEG XL to/from JPEG (#78).
- Add option to decode select frames from animated WebP.
- Use legacy JPEG8 codec when building without libjpeg-turbo 3 (#65).
- Change blosc2_encode defaults to match blosc2-python (breaking).
- Fix segfault writing JPEG2K with more than 4 samples.
- Fix some codecs returning bytearray by default.
- Fully vendor cfitsio's ricecomp.c.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

- …

Refer to the CHANGES file for older revisions.

Objectives
----------

Many scientific image storage formats like TIFF, CZI, DICOM, HDF, and Zarr
are containers that hold large numbers of small data segments (chunks, tiles,
stripes), which are encoded using a variety of compression and pre-filtering
methods. Metadata common to all data segments are typically stored separate
from the segments.

The purpose of the Imagecodecs library is to support Python modules in
encoding and decoding such data segments. The specific aims are:

- Provide functions for encoding and decoding small image data segments
  in-memory (not in-file) from and to bytes or numpy arrays for many
  compression and filtering methods.
- Support image formats and compression methods not available elsewhere in
  the Python ecosystem.
- Reduce the runtime dependency on numerous, large, inapt, or unmaintained
  Python packages. The imagecodecs package only depends on numpy.
- Implement codecs as Cython wrappers of 3rd party libraries with a C API
  and permissive license if exists, else use own C library.
  Provide Cython definition files for the wrapped C libraries.
- Release the Python global interpreter lock (GIL) during extended native/C
  function calls for multi-threaded use.

Accessing parts of large data segments and reading metadata from segments
are out of the scope of this library.

Notes
-----

This library is largely a work in progress.

The API is not stable yet and might change between revisions.

Python <= 3.8 is no longer supported. 32-bit versions are deprecated.

Works on little-endian platforms only.

Supported platforms are ``win_amd64``, ``win_arm64``, ``win32``,
``macosx_x86_64``, ``macosx_arm64``, ``manylinux_x86_64``, and
``manylinux_aarch64``.

Wheels may not be available for all platforms and all releases.

Only the ``win_amd64`` wheels include all features.

The ``tiff``, ``bcn``, ``dds``, ``dicomrle``, ``eer``, ``lzo``, ``packints``,
and ``jpegsof3`` codecs are currently decode-only.

The ``heif``, ``jetraw``, and ``jpegxs`` codecs are distributed as source
code only due to license and possible patent usage issues.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
<https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

This software is based in part on the work of the Independent JPEG Group.

Update pip and setuptools to the latest version before installing imagecodecs::

    python -m pip install -U pip setuptools wheel Cython

Before building imagecodecs from source code, install required tools and
libraries. For example, on latest Ubuntu Linux distributions:

    ``sudo apt-get install build-essential python3-dev cython3 python3-pip
    python3-setuptools python3-wheel python3-numpy libdeflate-dev libjpeg-dev
    libjxr-dev liblcms2-dev liblz4-dev liblerc-dev liblzma-dev libopenjp2-7-dev
    libpng-dev libtiff-dev libwebp-dev libz-dev libzstd-dev``

To build and install imagecodecs from source code, run::

    python -m pip install .

Many extensions are disabled by default when building from source.

To define which extensions are built, or to modify build settings such as
library names and compiler arguments, provide a
``imagecodecs_distributor_setup.customize_build`` function, which is
imported and executed during setup.
See ``setup.py`` for pre-defined ``customize_build`` functions.

Other projects providing imaging or compression codecs:
`Python zlib <https://docs.python.org/3/library/zlib.html>`_,
`Python bz2 <https://docs.python.org/3/library/bz2.html>`_,
`Python lzma <https://docs.python.org/3/library/lzma.html>`_,
`backports.lzma <https://github.com/peterjc/backports.lzma>`_,
`python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_,
`python-lzw <https://github.com/joeatwork/python-lzw>`_,
`python-lerc <https://pypi.org/project/lerc/>`_,
`wavpack-numcodecs
<https://github.com/AllenNeuralDynamics/wavpack-numcodecs>`_,
`packbits <https://github.com/psd-tools/packbits>`_,
`isa-l.igzip <https://github.com/intel/isa-l>`_,
`fpzip <https://github.com/seung-lab/fpzip>`_,
`libmng <https://sourceforge.net/projects/libmng/>`_,
`OpenEXR <https://github.com/AcademySoftwareFoundation/openexr>`_
(EXR, PIZ, PXR24, B44, DWA),
`pyJetraw <https://github.com/Jetraw/pyJetraw>`_,
`tinyexr <https://github.com/syoyo/tinyexr>`_,
`pytinyexr <https://github.com/syoyo/pytinyexr>`_,
`pyroexr <https://github.com/dragly/pyroexr>`_,
`JasPer <https://github.com/jasper-software/jasper>`_,
`libjpeg <https://github.com/thorfdbg/libjpeg>`_ (GPL),
`pylibjpeg <https://github.com/pydicom/pylibjpeg>`_,
`pylibjpeg-libjpeg <https://github.com/pydicom/pylibjpeg-libjpeg>`_ (GPL),
`pylibjpeg-openjpeg <https://github.com/pydicom/pylibjpeg-openjpeg>`_,
`pylibjpeg-rle <https://github.com/pydicom/pylibjpeg-rle>`_,
`glymur <https://github.com/quintusdias/glymur>`_,
`pyheif <https://github.com/carsales/pyheif>`_,
`pyrus-cramjam <https://github.com/milesgranger/pyrus-cramjam>`_,
`PyLZHAM <https://github.com/Galaxy1036/pylzham>`_,
`BriefLZ <https://github.com/jibsen/brieflz>`_,
`QuickLZ <http://www.quicklz.com/>`_ (GPL),
`LZO <http://www.oberhumer.com/opensource/lzo/>`_ (GPL),
`nvJPEG <https://developer.nvidia.com/nvjpeg>`_,
`nvJPEG2K <https://developer.nvidia.com/nvjpeg>`_,
`PyTurboJPEG <https://github.com/lilohuang/PyTurboJPEG>`_,
`CCSDS123 <https://github.com/drowzie/CCSDS123-Issue-2>`_,
`LPC-Rice <https://sourceforge.net/projects/lpcrice/>`_,
`CompressionAlgorithms <https://github.com/glampert/compression-algorithms>`_,
`Compressonator <https://github.com/GPUOpen-Tools/Compressonator>`_,
`Wuffs <https://github.com/google/wuffs>`_,
`TinyDNG <https://github.com/syoyo/tinydng>`_,
`OpenJPH <https://github.com/aous72/OpenJPH>`_,
`Grok <https://github.com/GrokImageCompression/grok>`_ (AGPL),
`MAFISC
<https://wr.informatik.uni-hamburg.de/research/projects/icomex/mafisc>`_,
`B3D <https://github.com/balintbalazs/B3D>`_,
`fo-dicom.Codecs <https://github.com/Efferent-Health/fo-dicom.Codecs>`_,
`jpegli <https://github.com/google/jpegli>`_.

Examples
--------

Import the JPEG2K codec:

>>> from imagecodecs import (
...     jpeg2k_encode,
...     jpeg2k_decode,
...     jpeg2k_check,
...     jpeg2k_version,
...     JPEG2K,
... )

Check that the JPEG2K codec is available in the imagecodecs build:

>>> JPEG2K.available
True

Print the version of the JPEG2K codec's underlying OpenJPEG library:

>>> jpeg2k_version()
'openjpeg 2.5.2'

Encode a numpy array in lossless JP2 format:

>>> array = numpy.random.randint(100, 200, (256, 256, 3), numpy.uint8)
>>> encoded = jpeg2k_encode(array, level=0)
>>> bytes(encoded[:12])
b'\x00\x00\x00\x0cjP  \r\n\x87\n'

Check that the encoded bytes likely contain a JPEG 2000 stream:

>>> jpeg2k_check(encoded)
True

Decode the JP2 encoded bytes to a numpy array:

>>> decoded = jpeg2k_decode(encoded)
>>> numpy.array_equal(decoded, array)
True

Decode the JP2 encoded bytes to an existing numpy array:

>>> out = numpy.empty_like(array)
>>> _ = jpeg2k_decode(encoded, out=out)
>>> numpy.array_equal(out, array)
True

Not all codecs are fully implemented, raising exceptions at runtime:

>>> from imagecodecs import tiff_encode
>>> tiff_encode(array)
Traceback (most recent call last):
 ...
NotImplementedError: tiff_encode

Write the numpy array to a JP2 file:

>>> from imagecodecs import imwrite, imread
>>> imwrite('_test.jp2', array)

Read the image from the JP2 file as numpy array:

>>> image = imread('_test.jp2')
>>> numpy.array_equal(image, array)
True

Create a JPEG 2000 compressed Zarr array:

>>> import zarr
>>> import numcodecs
>>> from imagecodecs.numcodecs import Jpeg2k
>>> numcodecs.register_codec(Jpeg2k)
>>> zarr.zeros(
...     (4, 5, 512, 512, 3),
...     chunks=(1, 1, 256, 256, 3),
...     dtype='u1',
...     compressor=Jpeg2k(),
... )
<zarr.core.Array (4, 5, 512, 512, 3) uint8>

Access image data in a sequence of JP2 files via tifffile.FileSequence and
dask.array:

>>> import tifffile
>>> import dask.array
>>> def jp2_read(filename):
...     with open(filename, 'rb') as fh:
...         data = fh.read()
...     return jpeg2k_decode(data)
...
>>> with tifffile.FileSequence(jp2_read, '*.jp2') as ims:
...     with ims.aszarr() as store:
...         dask.array.from_zarr(store)
...
dask.array<from-zarr, shape=(1, 256, 256, 3)...chunksize=(1, 256, 256, 3)...

Write the Zarr store to a fsspec ReferenceFileSystem in JSON format
and open it as a Zarr array:

>>> store.write_fsspec(
...     'temp.json', url='file://', codec_id='imagecodecs_jpeg2k'
... )
>>> import fsspec
>>> mapper = fsspec.get_mapper(
...     'reference://', fo='temp.json', target_protocol='file'
... )
>>> zarr.open(mapper, mode='r')
<zarr.core.Array (1, 256, 256, 3) uint8 read-only>

View the image in the JP2 file from the command line::

    python -m imagecodecs _test.jp2

"""

from __future__ import annotations

__version__ = '2024.9.22'

import importlib
import io
import os
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mmap
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, IO

    from numpy.typing import ArrayLike, NDArray

import numpy

# map extension module names to attribute names
_MODULES: dict[str, list[str]] = {
    '': [
        '__version__',
        'version',
        'imread',
        'imwrite',
        'imagefileext',
        'DelayedImportError',
        'NONE',
        'none_encode',
        'none_decode',
        'none_check',
        'none_version',
        'NoneError',
        'NUMPY',
        'numpy_encode',
        'numpy_decode',
        'numpy_check',
        'numpy_version',
        'NumpyError',
        'JPEG',
        'jpeg_encode',
        'jpeg_decode',
        'jpeg_check',
        'jpeg_version',
        'JpegError',
    ],
    '_imcd': [
        'imcd_version',
        'numpy_abi_version',
        'cython_version',
        'BITORDER',
        'BitorderError',
        'bitorder_encode',
        'bitorder_decode',
        'bitorder_check',
        'bitorder_version',
        'BYTESHUFFLE',
        'ByteshuffleError',
        'byteshuffle_encode',
        'byteshuffle_decode',
        'byteshuffle_check',
        'byteshuffle_version',
        'DELTA',
        'DeltaError',
        'delta_encode',
        'delta_decode',
        'delta_check',
        'delta_version',
        'DICOMRLE',
        'DicomrleError',
        'dicomrle_encode',
        'dicomrle_decode',
        'dicomrle_check',
        'dicomrle_version',
        'EER',
        'EerError',
        'eer_encode',
        'eer_decode',
        'eer_check',
        'eer_version',
        'FLOAT24',
        'Float24Error',
        'float24_encode',
        'float24_decode',
        'float24_check',
        'float24_version',
        'FLOATPRED',
        'FloatpredError',
        'floatpred_encode',
        'floatpred_decode',
        'floatpred_check',
        'floatpred_version',
        'LZW',
        'LzwError',
        'lzw_encode',
        'lzw_decode',
        'lzw_check',
        'lzw_version',
        # 'MONO12P',
        # 'Mono12pError',
        # 'mono12p_encode',
        # 'mono12p_decode',
        # 'mono12p_check',
        # 'mono12p_version',
        'PACKBITS',
        'PackbitsError',
        'packbits_encode',
        'packbits_decode',
        'packbits_check',
        'packbits_version',
        'PACKINTS',
        'PackintsError',
        'packints_encode',
        'packints_decode',
        'packints_check',
        'packints_version',
        'XOR',
        'XorError',
        'xor_encode',
        'xor_decode',
        'xor_check',
        'xor_version',
    ],
    '_aec': [
        'AEC',
        'AecError',
        'aec_encode',
        'aec_decode',
        'aec_check',
        'aec_version',
    ],
    '_apng': [
        'APNG',
        'ApngError',
        'apng_encode',
        'apng_decode',
        'apng_check',
        'apng_version',
    ],
    '_avif': [
        'AVIF',
        'AvifError',
        'avif_encode',
        'avif_decode',
        'avif_check',
        'avif_version',
    ],
    '_bitshuffle': [
        'BITSHUFFLE',
        'BitshuffleError',
        'bitshuffle_encode',
        'bitshuffle_decode',
        'bitshuffle_check',
        'bitshuffle_version',
    ],
    '_blosc': [
        'BLOSC',
        'BloscError',
        'blosc_encode',
        'blosc_decode',
        'blosc_check',
        'blosc_version',
    ],
    '_blosc2': [
        'BLOSC2',
        'Blosc2Error',
        'blosc2_encode',
        'blosc2_decode',
        'blosc2_check',
        'blosc2_version',
    ],
    '_bmp': [
        'BMP',
        'BmpError',
        'bmp_encode',
        'bmp_decode',
        'bmp_check',
        'bmp_version',
    ],
    '_brotli': [
        'BROTLI',
        'BrotliError',
        'brotli_encode',
        'brotli_decode',
        'brotli_check',
        'brotli_version',
    ],
    '_brunsli': [
        'BRUNSLI',
        'BrunsliError',
        'brunsli_encode',
        'brunsli_decode',
        'brunsli_check',
        'brunsli_version',
    ],
    '_bz2': [
        'BZ2',
        'Bz2Error',
        'bz2_encode',
        'bz2_decode',
        'bz2_check',
        'bz2_version',
    ],
    '_cms': [
        'CMS',
        'CmsError',
        'cms_transform',
        'cms_profile',
        'cms_profile_validate',
        'cms_encode',
        'cms_decode',
        'cms_check',
        'cms_version',
    ],
    '_bcn': [
        'BCN',
        'BcnError',
        'bcn_encode',
        'bcn_decode',
        'bcn_check',
        'bcn_version',
        'DDS',
        'DdsError',
        'dds_encode',
        'dds_decode',
        'dds_check',
        'dds_version',
    ],
    '_deflate': [
        'DEFLATE',
        'DeflateError',
        'deflate_crc32',
        'deflate_adler32',
        'deflate_encode',
        'deflate_decode',
        'deflate_check',
        'deflate_version',
        'GZIP',
        'GzipError',
        'gzip_encode',
        'gzip_decode',
        'gzip_check',
        'gzip_version',
    ],
    '_gif': [
        'GIF',
        'GifError',
        'gif_encode',
        'gif_decode',
        'gif_check',
        'gif_version',
    ],
    '_h5checksum': [
        'H5CHECKSUM',
        'h5checksum_version',
        'h5checksum_fletcher32',
        'h5checksum_lookup3',
        'h5checksum_crc',
        'h5checksum_metadata',
        'h5checksum_hash_string',
    ],
    '_heif': [
        'HEIF',
        'HeifError',
        'heif_encode',
        'heif_decode',
        'heif_check',
        'heif_version',
    ],
    # '_htj2k': [
    #     'HTJ2K',
    #     'Htj2kError',
    #     'htj2k_encode',
    #     'htj2k_decode',
    #     'htj2k_check',
    #     'htj2k_version',
    # ],
    '_jetraw': [
        'JETRAW',
        'JetrawError',
        'jetraw_init',
        'jetraw_encode',
        'jetraw_decode',
        'jetraw_check',
        'jetraw_version',
    ],
    '_jpeg2k': [
        'JPEG2K',
        'Jpeg2kError',
        'jpeg2k_encode',
        'jpeg2k_decode',
        'jpeg2k_check',
        'jpeg2k_version',
    ],
    '_jpeg8': [
        'JPEG8',
        'Jpeg8Error',
        'jpeg8_encode',
        'jpeg8_decode',
        'jpeg8_check',
        'jpeg8_version',
    ],
    # '_jpegli': [
    #     'JPEGLI',
    #     'JpegliError',
    #     'jpegli_encode',
    #     'jpegli_decode',
    #     'jpegli_check',
    #     'jpegli_version',
    # ],
    '_jpegls': [
        'JPEGLS',
        'JpeglsError',
        'jpegls_encode',
        'jpegls_decode',
        'jpegls_check',
        'jpegls_version',
    ],
    '_jpegsof3': [
        'JPEGSOF3',
        'Jpegsof3Error',
        'jpegsof3_encode',
        'jpegsof3_decode',
        'jpegsof3_check',
        'jpegsof3_version',
    ],
    '_jpegxl': [
        'JPEGXL',
        'JpegxlError',
        'jpegxl_encode',
        'jpegxl_decode',
        'jpegxl_encode_jpeg',
        'jpegxl_decode_jpeg',
        'jpegxl_check',
        'jpegxl_version',
    ],
    '_jpegxr': [
        'JPEGXR',
        'JpegxrError',
        'jpegxr_encode',
        'jpegxr_decode',
        'jpegxr_check',
        'jpegxr_version',
    ],
    '_jpegxs': [
        'JPEGXS',
        'JpegxsError',
        'jpegxs_encode',
        'jpegxs_decode',
        'jpegxs_check',
        'jpegxs_version',
    ],
    '_lerc': [
        'LERC',
        'LercError',
        'lerc_encode',
        'lerc_decode',
        'lerc_check',
        'lerc_version',
    ],
    '_ljpeg': [
        'LJPEG',
        'LjpegError',
        'ljpeg_encode',
        'ljpeg_decode',
        'ljpeg_check',
        'ljpeg_version',
    ],
    '_lz4': [
        'LZ4',
        'Lz4Error',
        'lz4_encode',
        'lz4_decode',
        'lz4_check',
        'lz4_version',
        'LZ4H5',
        'Lz4h5Error',
        'lz4h5_encode',
        'lz4h5_decode',
        'lz4h5_check',
        'lz4h5_version',
    ],
    '_lz4f': [
        'LZ4F',
        'Lz4fError',
        'lz4f_encode',
        'lz4f_decode',
        'lz4f_check',
        'lz4f_version',
    ],
    '_lzf': [
        'LZF',
        'LzfError',
        'lzf_encode',
        'lzf_decode',
        'lzf_check',
        'lzf_version',
    ],
    '_lzfse': [
        'LZFSE',
        'LzfseError',
        'lzfse_encode',
        'lzfse_decode',
        'lzfse_check',
        'lzfse_version',
    ],
    '_lzham': [
        'LZHAM',
        'LzhamError',
        'lzham_encode',
        'lzham_decode',
        'lzham_check',
        'lzham_version',
    ],
    '_lzma': [
        'LZMA',
        'LzmaError',
        'lzma_encode',
        'lzma_decode',
        'lzma_check',
        'lzma_version',
    ],
    '_lzo': [
        'LZO',
        'LzoError',
        'lzo_encode',
        'lzo_decode',
        'lzo_check',
        'lzo_version',
    ],
    '_mozjpeg': [
        'MOZJPEG',
        'MozjpegError',
        'mozjpeg_encode',
        'mozjpeg_decode',
        'mozjpeg_check',
        'mozjpeg_version',
    ],
    '_pcodec': [
        'PCODEC',
        'PcodecError',
        'pcodec_encode',
        'pcodec_decode',
        'pcodec_check',
        'pcodec_version',
    ],
    '_pglz': [
        'PGLZ',
        'PglzError',
        'pglz_encode',
        'pglz_decode',
        'pglz_check',
        'pglz_version',
    ],
    '_png': [
        'PNG',
        'PngError',
        'png_encode',
        'png_decode',
        'png_check',
        'png_version',
    ],
    '_qoi': [
        'QOI',
        'QoiError',
        'qoi_encode',
        'qoi_decode',
        'qoi_check',
        'qoi_version',
    ],
    '_quantize': [
        'QUANTIZE',
        'QuantizeError',
        'quantize_encode',
        'quantize_decode',
        'quantize_check',
        'quantize_version',
    ],
    '_rgbe': [
        'RGBE',
        'RgbeError',
        'rgbe_encode',
        'rgbe_decode',
        'rgbe_check',
        'rgbe_version',
    ],
    '_rcomp': [
        'RCOMP',
        'RcompError',
        'rcomp_encode',
        'rcomp_decode',
        'rcomp_check',
        'rcomp_version',
    ],
    '_snappy': [
        'SNAPPY',
        'SnappyError',
        'snappy_encode',
        'snappy_decode',
        'snappy_check',
        'snappy_version',
    ],
    '_sperr': [
        'SPERR',
        'SperrError',
        'sperr_encode',
        'sperr_decode',
        'sperr_check',
        'sperr_version',
    ],
    '_spng': [
        'SPNG',
        'SpngError',
        'spng_encode',
        'spng_decode',
        'spng_check',
        'spng_version',
    ],
    '_sz3': [
        'SZ3',
        'Sz3Error',
        'sz3_encode',
        'sz3_decode',
        'sz3_check',
        'sz3_version',
    ],
    '_szip': [
        'SZIP',
        'SzipError',
        'szip_encode',
        'szip_decode',
        'szip_check',
        'szip_version',
        'szip_params',
    ],
    '_tiff': [
        'TIFF',
        'TiffError',
        'tiff_encode',
        'tiff_decode',
        'tiff_check',
        'tiff_version',
    ],
    '_ultrahdr': [
        'ULTRAHDR',
        'UltrahdrError',
        'ultrahdr_encode',
        'ultrahdr_decode',
        'ultrahdr_check',
        'ultrahdr_version',
    ],
    '_webp': [
        'WEBP',
        'WebpError',
        'webp_encode',
        'webp_decode',
        'webp_check',
        'webp_version',
    ],
    '_zfp': [
        'ZFP',
        'ZfpError',
        'zfp_encode',
        'zfp_decode',
        'zfp_check',
        'zfp_version',
    ],
    '_zlib': [
        'ZLIB',
        'ZlibError',
        'zlib_crc32',
        'zlib_adler32',
        'zlib_encode',
        'zlib_decode',
        'zlib_check',
        'zlib_version',
    ],
    '_zlibng': [
        'ZLIBNG',
        'ZlibngError',
        'zlibng_crc32',
        'zlibng_adler32',
        'zlibng_encode',
        'zlibng_decode',
        'zlibng_check',
        'zlibng_version',
    ],
    '_zopfli': [
        'ZOPFLI',
        'ZopfliError',
        'zopfli_encode',
        'zopfli_decode',
        'zopfli_check',
        'zopfli_version',
    ],
    '_zstd': [
        'ZSTD',
        'ZstdError',
        'zstd_encode',
        'zstd_decode',
        'zstd_check',
        'zstd_version',
    ],
}

# map extra to existing attributes
# for example, keep deprecated names for older versions of tifffile and czifile
_COMPATIBILITY: dict[str, str] = {
    'JPEG': 'JPEG8',
    'JpegError': 'Jpeg8Error',
    'jpeg_check': 'jpeg8_check',
    'jpeg_version': 'jpeg8_version',
    'zopfli_check': 'zlib_check',
    'zopfli_decode': 'zlib_decode',
    # deprecated
    'j2k_encode': 'jpeg2k_encode',
    'j2k_decode': 'jpeg2k_decode',
    'jxr_encode': 'jpegxr_encode',
    'jxr_decode': 'jpegxr_decode',
    # 'JPEG12': 'JPEG8',
    # 'Jpeg12Error': 'Jpeg8Error',
    # 'jpeg12_encode': 'jpeg8_encode',
    # 'jpeg12_decode': 'jpeg8_decode',
    # 'jpeg12_check': 'jpeg8_check',
    # 'jpeg12_version': 'jpeg8_version',
}

# map attribute names to module names
_ATTRIBUTES: dict[str, str] = {
    attribute: module
    for module, attributes in _MODULES.items()
    for attribute in attributes
}

# set of imported modules
_IMPORTED: set[str] = set()

_LOCK = threading.RLock()

__all__ = [
    attribute for attributes in _MODULES.values() for attribute in attributes
]


def _add_codec(
    module: str,
    codec: str | None = None,
    attributes: tuple[str, ...] | None = None,
    /,
) -> None:
    """Register codec in global _MODULES and _ATTRIBUTES."""
    if codec is None:
        codec = module
    if attributes is None:
        attributes = (
            f'{codec}_encode',
            f'{codec}_decode',
            f'{codec}_check',
            f'{codec}_version',
            f'{codec.capitalize()}Error',
            f'{codec.upper()}',
        )
    if module in _MODULES:
        _MODULES[module].extend(attributes)
    else:
        _MODULES[module] = list(attributes)
    _ATTRIBUTES.update({attr: module for attr in attributes})


def _load_all() -> None:
    """Add all registered attributes to package namespace."""
    for name in __dir__():
        __getattr__(name)


def __dir__() -> list[str]:
    """Return list of attribute names accessible on module."""
    return sorted(list(_ATTRIBUTES) + list(_COMPATIBILITY))


def __getattr__(name: str, /) -> Any:
    """Return module attribute after loading it from extension module.

    Load attribute's extension and add its attributes to the package namespace.

    """
    name_ = name
    name = _COMPATIBILITY.get(name, name)

    if name not in _ATTRIBUTES:
        raise AttributeError(f"module 'imagecodecs' has no attribute {name!r}")

    module_name = _ATTRIBUTES[name]
    if not module_name:
        return None

    with _LOCK:
        if module_name in _IMPORTED:
            # extension module was imported in another thread
            # while this thread was waiting for lock
            return getattr(imagecodecs, name)

        try:
            module = importlib.import_module('.' + module_name, 'imagecodecs')
        except ImportError:
            module = None
        except AttributeError:
            # AttributeError: type object 'imagecodecs._module.array' has no
            # attribute '__reduce_cython__'
            # work around Cython raises AttributeError, for example, when
            # the _shared module failed to import due to an incompatible
            # numpy version
            from . import _shared  # noqa

            module = None

        for n in _MODULES[module_name]:
            if n in _COMPATIBILITY:
                continue
            attr = getattr(module, n, None)
            if attr is None:
                attr = _stub(n, module)
            setattr(imagecodecs, n, attr)

        attr = getattr(imagecodecs, name)
        if name != name_:
            setattr(imagecodecs, name_, attr)

        _IMPORTED.add(module_name)
        return attr


class DelayedImportError(ImportError):
    """Delayed ImportError."""

    def __init__(self, name: str, /) -> None:
        """Initialize instance from attribute name."""
        msg = f"could not import name {name!r} from 'imagecodecs'"
        super().__init__(msg)


def _stub(name: str, module: ModuleType | None, /) -> Any:
    """Return stub constant, function, or class."""
    if name.endswith('_version'):
        if module is None:

            def stub_version() -> str:
                """Stub for imagecodecs.codec_version function."""
                return f'{name[:-8]} n/a'

        else:

            def stub_version() -> str:
                """Stub for imagecodecs.codec_version function."""
                return f'{name[:-8]} unknown'

        return stub_version

    if name.endswith('_check'):

        def stub_check(arg: Any, /) -> bool:
            """Stub for imagecodecs.codec_check function."""
            return False

        return stub_check

    if name.endswith('_decode'):

        def stub_decode(*args: Any, **kwargs: Any) -> None:
            """Stub for imagecodecs.codec_decode function."""
            raise DelayedImportError(name)

        return stub_decode

    if name.endswith('_encode'):

        def stub_encode(*args: Any, **kwargs: Any) -> None:
            """Stub for imagecodecs.codec_encode function."""
            raise DelayedImportError(name)

        return stub_encode

    if name.islower():

        def stub_function(*args: Any, **kwargs: Any) -> None:
            """Stub for imagecodecs.codec_function."""
            raise DelayedImportError(name)

        return stub_function

    if name.endswith('Error'):

        class StubError(RuntimeError):
            """Stub for imagecodecs.CodecError class."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise DelayedImportError(name)

        return StubError

    class StubType(type):
        """Stub type metaclass."""

        def __getattr__(cls, arg: str, /) -> Any:
            raise DelayedImportError(name)

        if module is None:

            def __bool__(cls) -> bool:
                return False

    if name.isupper():

        class STUB(metaclass=StubType):
            """Stub for imagecodecs.CODEC constants."""

            available: bool = False

        return STUB

    class Stub(metaclass=StubType):
        """Stub for imagecodecs.Codec class."""

    return Stub


def _extensions() -> tuple[str, ...]:
    """Return sorted names of extension modules."""
    return tuple(sorted(e for e in _MODULES if e))


def _codecs(available: bool | None = None, /) -> tuple[str, ...]:
    """Return sorted names of codecs.

    If `available` is not None, all extension modules are imported into the
    process.

    """
    codecs: tuple[str, ...] = tuple(
        sorted(c.lower() for c in _ATTRIBUTES if c.isupper())
    )
    if available is None:
        return codecs
    if available:
        return tuple(
            c
            for c in codecs
            if getattr(getattr(imagecodecs, c.upper()), 'available')
        )
    return tuple(
        c
        for c in codecs
        if not getattr(getattr(imagecodecs, c.upper()), 'available')
    )


def version(
    astype: type | None = None, /
) -> str | tuple[str, ...] | dict[str, str]:
    """Return version information about all codecs and dependencies.

    All extension modules are imported into the process.

    """
    versions: tuple[str, ...] = (
        f'imagecodecs {__version__}',
        imagecodecs.cython_version(),
        imagecodecs.numpy_version(),
        imagecodecs.numpy_abi_version(),
        imagecodecs.imcd_version(),
        *sorted(
            # use set to filter duplicates
            {
                str(getattr(imagecodecs, v)())
                for v in _ATTRIBUTES
                if v.endswith('_version')
                and v
                not in {
                    'imcd_version',
                    'numpy_abi_version',
                    'numpy_version',
                    'cython_version',
                    'none_version',
                }
            }
        ),
    )
    if astype is None or astype is str:
        return ', '.join(ver.replace(' ', '-') for ver in versions)
    if astype is dict:
        return dict(ver.split(' ') for ver in versions)
    return tuple(versions)


def imread(
    fileobj: str | os.PathLike[Any] | bytes | mmap.mmap,
    /,
    codec: (
        str
        | Callable[..., NDArray[Any]]
        | list[str | Callable[..., NDArray[Any]]]
        | None
    ) = None,
    *,
    memmap: bool = True,
    return_codec: bool = False,
    **kwargs: Any,
) -> NDArray[Any] | tuple[NDArray[Any], Callable[..., NDArray[Any]]]:
    """Return image data from file as numpy array."""
    import mmap

    codecs: list[str | Callable[..., NDArray[Any]]] = []
    if codec is None:
        # find codec based on file extension
        if isinstance(fileobj, (str, os.PathLike)):
            ext = os.path.splitext(os.fspath(fileobj))[-1][1:].lower()
        else:
            ext = None
        if ext in _imcodecs():
            codec = _imcodecs()[ext]
            if codec == 'jpeg':
                codecs.extend(('jpeg8', 'ljpeg'))  # 'jpegsof3'
            else:
                codecs.append(codec)
        # try other imaging codecs
        codecs.extend(
            c
            for c in (
                'tiff',
                'apng',
                'png',
                'gif',
                'webp',
                'jpeg8',
                'ljpeg',
                'jpeg2k',
                'jpegls',
                'jpegxr',
                'jpegxl',
                'jpegxs',
                'avif',
                'heif',
                'bmp',
                # 'jpegli',
                # 'htj2k',
                'ultrahdr',
                # 'brunsli',
                # 'exr',
                'zfp',
                'lerc',
                'rgbe',
                # 'jpegsof3',
                'numpy',
            )
            if c not in codecs
        )
    else:
        # use provided codecs
        if not isinstance(codec, (list, tuple)):  # collections.abc.Iterable
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                c = c.lower()
                c = _imcodecs().get(c, c)
            codecs.append(c)

    data: bytes | mmap.mmap
    offset: int = -1
    close = False
    if isinstance(fileobj, mmap.mmap):
        data = fileobj
        offset = data.tell()
    elif hasattr(fileobj, 'read'):
        # binary stream: open file, BytesIO
        data = fileobj.read()
    elif isinstance(fileobj, (str, os.PathLike)):
        # TODO: support urllib.request.urlopen ?
        # file name
        with open(os.fspath(fileobj), 'rb') as fh:
            if memmap:
                offset = 0
                close = True
                data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                data = fh.read()
    else:
        # binary data
        data = fileobj
    del codec

    func: Callable[..., NDArray[Any]]
    exceptions: list[str] = []
    image: NDArray[Any] | None = None
    for codec in codecs:
        if callable(codec):
            func = codec
        else:
            try:
                func = getattr(imagecodecs, codec + '_decode')
                assert callable(func)
            except Exception as exc:
                exceptions.append(f'{repr(codec).upper()}: {exc}')
                continue

        numthreads = kwargs.pop('numthreads', None)
        if numthreads is not None and func.__name__.split('_')[0] not in {
            'avif',
            'jpeg2k',
            'jpegxl',
        }:
            numthreads = None

        try:
            if numthreads is None:
                image = func(data, **kwargs)
            else:
                image = func(data, numthreads=numthreads, **kwargs)
            assert isinstance(image, numpy.ndarray)
            if image.dtype == 'object':
                image = None
                raise ValueError('failed')
            break
        except DelayedImportError:
            pass
        except Exception as exc:
            # raise
            exceptions.append(f'{func.__name__.upper()}: {exc}')
        if offset >= 0:
            assert isinstance(data, mmap.mmap)
            data.seek(offset)

    if close:
        assert isinstance(data, mmap.mmap)
        data.close()

    if image is None:
        raise ValueError('\n'.join(exceptions))

    if return_codec:
        return image, func
    return image


def imwrite(
    fileobj: str | os.PathLike[Any] | IO[bytes],
    data: ArrayLike,
    /,
    codec: str | Callable[..., bytes | bytearray] | None = None,
    **kwargs: Any,
) -> None:
    """Write numpy array to image file."""
    if codec is None:
        # find codec based on file extension
        if isinstance(fileobj, (str, os.PathLike)):
            ext = os.path.splitext(os.fspath(fileobj))[-1].lower()[1:]
        else:
            raise ValueError('no codec specified')

        codec = _imcodecs().get(ext, ext)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            raise ValueError(f'invalid {codec=!r}') from exc

    elif isinstance(codec, str):
        codec = codec.lower()
        codec = _imcodecs().get(codec, codec)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            raise ValueError(f'invalid {codec=!r}') from exc

    if not callable(codec):
        raise ValueError(f'invalid {codec=!r}')

    image: bytes = codec(data, **kwargs)
    if hasattr(fileobj, 'write'):
        # binary stream: open file, BytesIO
        fileobj.write(image)  # typing: ignore
    else:
        # file name
        with open(fileobj, 'wb') as fh:
            fh.write(image)


def _imcodecs(_codecs: dict[str, str] = {}) -> dict[str, str]:
    """Return map of image file extensions to codec names."""
    with _LOCK:
        if not _codecs:
            codecs = {
                'apng': ('apng',),
                'avif': ('avif', 'avifs'),
                'bmp': ('bmp', 'dip'),  # 'rle'
                'brunsli': ('brn',),
                'dds': ('dds',),
                # 'exr': ('exr',),
                'gif': ('gif',),
                'heif': (
                    'heif',
                    'heic',
                    'heifs',
                    'heics',
                    'hif',  # 'avci', 'avcs'
                ),
                # 'htj2k': ('jph', 'jhc'),  # currently decoded by jpeg2k
                'jpeg2k': (
                    'j2k',
                    'jp2',
                    'j2c',
                    'jpc',
                    'jpx',
                    'jpf',
                    'jpg2',
                    'jph',  # HTJ2K with JP2 boxes
                    'jhc',  # HTJ2K codestream
                ),
                'jpeg8': ('jpg', 'jpeg', 'jpe', 'jfif', 'jfi', 'jif'),
                # 'jpegli': ('jli',),
                'jpegls': ('jls',),
                'jpegxl': ('jxl',),
                'jpegxr': ('jxr', 'hdp', 'wdp'),
                'jpegxs': ('jxs',),
                'lerc': ('lerc1', 'lerc2'),
                'ljpeg': ('ljp', 'ljpg', 'ljpeg'),
                'numpy': ('npy', 'npz'),
                'png': ('png',),
                'qoi': ('qoi',),
                'rgbe': ('hdr', 'rgbe', 'pic'),
                # 'tga': ('tga'),
                'tiff': ('tif', 'tiff', 'ptif', 'ptiff', 'tf8', 'tf2', 'btf'),
                'ultrahdr': ('uhdr', 'jpr'),  # jpg
                'webp': ('webp', 'webm'),
                'zfp': ('zfp',),
            }
            _codecs.update(
                (ext, codec) for codec, exts in codecs.items() for ext in exts
            )
    return _codecs


def imagefileext() -> list[str]:
    """Return list of image file extensions handled by imread and imwrite."""
    return list(_imcodecs().keys())


class NONE:
    """NONE codec constants."""

    available = True
    """NONE codec is available."""


NoneError = RuntimeError


def none_version() -> str:
    """Return empty version string."""
    return ''


def none_check(data: Any, /) -> None:
    """Return None."""


def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""
    return data


def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""
    return data


class NUMPY:
    """NUMPY codec constants."""

    available = True
    """NUMPY codec is available."""


NumpyError = RuntimeError


def numpy_version() -> str:
    """Return Numpy library version string."""
    return f'numpy {numpy.__version__}'


def numpy_check(data: bytes | bytearray, /) -> bool:
    """Return whether data is NPY or NPZ encoded."""
    with io.BytesIO(data) as fh:
        data = fh.read(64)
    magic = b'\x93NUMPY'
    return data.startswith(magic) or (data.startswith(b'PK') and magic in data)


def numpy_decode(
    data: bytes,
    /,
    index: int = 0,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Return decoded NPY or NPZ data."""
    with io.BytesIO(data) as fh:
        try:
            result = numpy.load(fh, **kwargs)
        except ValueError as exc:
            raise ValueError('not a numpy array') from exc
        if hasattr(result, 'files'):
            try:
                index = result.files[index]
            except Exception:
                pass
            result = result[index]
    return result  # type: ignore[no-any-return]


def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes:
    """Return NPY or NPZ encoded data."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data)
        else:
            numpy.save(fh, data)
        fh.seek(0)
        result = fh.read()
    return result


def jpeg_decode(
    data: bytes,
    /,
    *,
    tables: bytes | None = None,
    header: bytes | None = None,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    shape: tuple[int, ...] | None = None,
    bitspersample: int | None = None,  # required for compatibility
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEG image."""
    del bitspersample
    if header is not None:
        data = b''.join((header, data, b'\xff\xd9'))
    try:
        return imagecodecs.jpeg8_decode(  # type: ignore[no-any-return]
            data,
            tables=tables,
            colorspace=colorspace,
            outcolorspace=outcolorspace,
            shape=shape,
            out=out,
        )
    except Exception as exc:
        # try LJPEG codec, which handles more precisions and colorspaces
        msg = str(exc)

        if (
            'Unsupported JPEG data precision' in msg
            or 'Unsupported color conversion' in msg
            or 'Bogus Huffman table definition' in msg
            or 'SOF type' in msg
        ):
            try:
                return imagecodecs.ljpeg_decode(  # type: ignore[no-any-return]
                    data, out=out
                )
            except Exception:
                pass
        # elif 'Empty JPEG image' in msg:
        # for example, Hamamatsu NDPI slides with dimensions > 65500
        # Unsupported marker type
        raise exc


def jpeg_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    lossless: bool | None = None,
    predictor: int | None = None,
    bitspersample: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEG encoded image."""
    if (
        lossless
        and not imagecodecs.JPEG8.all_precisions
        and bitspersample not in {None, 8, 12, 16}
    ):
        return imagecodecs.ljpeg_encode(  # type: ignore[no-any-return]
            data, bitspersample=bitspersample, out=out
        )
    return imagecodecs.jpeg8_encode(  # type: ignore[no-any-return]
        data,
        level=level,
        colorspace=colorspace,
        outcolorspace=outcolorspace,
        subsampling=subsampling,
        optimize=optimize,
        smoothing=smoothing,
        lossless=lossless,
        predictor=predictor,
        bitspersample=bitspersample,
        out=out,
    )


imagecodecs = sys.modules['imagecodecs']
