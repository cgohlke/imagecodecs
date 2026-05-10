# imagecodecs.py

# Copyright (c) 2008-2026, Christoph Gohlke
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
transformation, compression, and decompression functions for use in tifffile,
liffile, czifile, zarr, and other scientific image input/output packages.

Decode and/or encode functions are implemented for the following codecs,
image formats, and data transforms:
Zlib (DEFLATE), GZIP, LZMA, ZStandard (ZSTD), Blosc, Brotli, Snappy, BZ2,
LZ4, LZ4F, LZ4HC, LZ4H5, LZW, LZO, LZF, LZFSE, LZHAM, PGLZ (PostgreSQL LZ),
RCOMP (Rice), HCOMP, PLIO, ZFP, SZ3, Meshopt, Pcodec, SPERR, AEC, SZIP, LERC,
EER, NPY, BCn, DDS, BMP, PNG, APNG, GIF, PCX/DCX, TGA (TARGA), TIFF, WebP,
JPEG (2 to 16-bit), Lossless JPEG (LJPEG, LJ92, JPEGLL), JPEG 2000 (JP2, J2K),
High-throughput JPEG 2000 (HTJ2K, JPH), JPEG LS, JPEG XL, JPEG XS,
JPEG XR (WDP, HD Photo), Ultra HDR (JPEG_R), MOZJPEG, AVIF, HEIF, EXR,
WIC (Windows Imaging Component), WavPack, QOI, RGBE (HDR), PixarLog, Jetraw,
DICOM RLE, CCITT (RLE, T.4 and T.6), PackBits, Packed Integers
(TIFF, MONO p and packed), Delta, XOR Delta, Floating Point Predictor,
Bitorder reversal, Byteshuffle, Bitshuffle, Float24 (24-bit floating point),
Bfloat16 (brain floating point), Quantize (Scale, BitGroom, BitRound,
GranularBR), and CMS (color space transformations).
Checksum functions are implemented for CRC-32, Adler-32, Fletcher-32, and
Jenkins lookup3.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.5.10
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

- `CPython <https://www.python.org>`_ 3.12.10, 3.13.13, 3.14.4 64-bit
- `numpy <https://pypi.org/project/numpy>`_ 2.4.4
- `zarr <https://pypi.org/project/zarr/>`_ 3.2.1
  (optional, for Zarr 3 compatible codecs)
- `numcodecs <https://pypi.org/project/numcodecs/>`_ 0.16.5
  (optional, for Zarr file format 2 compatible codecs)

Build requirements:

- `cython <https://github.com/cython/cython>`_ 3.2.4
- `brotli <https://github.com/google/brotli>`_ 1.2.0
- `bzip2 <https://gitlab.com/bzip2/bzip2>`_ 1.0.8
- `c-blosc <https://github.com/Blosc/c-blosc>`_ 1.21.6
- `c-blosc2 <https://github.com/Blosc/c-blosc2>`_ 3.0.2
- `charls <https://github.com/team-charls/charls>`_ 2.4.3
- `giflib <https://sourceforge.net/projects/giflib/>`_ 6.1.3
- `jxrlib <https://github.com/cgohlke/jxrlib>`_ 1.2
- `lcms2 <https://github.com/mm2/Little-CMS>`_ 2.19.1
- `lerc <https://github.com/Esri/lerc>`_ 4.1.0
- `libaec <https://gitlab.dkrz.de/k202009/libaec>`_ 1.1.6
- `libavif <https://github.com/AOMediaCodec/libavif>`_ 1.4.1
  (`aom <https://aomedia.googlesource.com/aom>`_ 3.13.3,
  `dav1d <https://github.com/videolan/dav1d>`_ 1.5.3,
  `rav1e <https://github.com/xiph/rav1e>`_ 0.8.1,
  `svt-av1 <https://gitlab.com/AOMediaCodec/SVT-AV1>`_ 4.1.0,
  `libyuv <https://chromium.googlesource.com/libyuv/libyuv>`_ main,
  `libxml2 <https://gitlab.gnome.org/GNOME/libxml2>`_ 2.15.3)
- `libdeflate <https://github.com/ebiggers/libdeflate>`_ 1.25
- `libheif <https://github.com/strukturag/libheif>`_ 1.21.2
  (`libde265 <https://github.com/strukturag/libde265>`_ 1.0.18,
  `x265 <https://bitbucket.org/multicoreware/x265_git/src/master/>`_ 4.1)
- `libjpeg-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_ 3.1.4.1
- `libjxl <https://github.com/libjxl/libjxl>`_ 0.11.2
- `libjxs <https://jpeg.org/jpegxs/software.html>`_ 2.0.2
- `liblzma <https://github.com/tukaani-project/xz>`_ 5.8.3
- `libpng <https://github.com/glennrp/libpng>`_ 1.6.58
- `libpng-apng <https://sourceforge.net/projects/libpng-apng/>`_ 1.6.58
- `libtiff <https://gitlab.com/libtiff/libtiff>`_ 4.7.1 (with issue 789 fix)
- `libultrahdr <https://github.com/google/libultrahdr>`_ 1.4.0
- `libwebp <https://github.com/webmproject/libwebp>`_ 1.6.0
- `lz4 <https://github.com/lz4/lz4>`_ 1.10.0
- `meshoptimizer <https://github.com/zeux/meshoptimizer>`_ 1.1
- `openexr <https://github.com/AcademySoftwareFoundation/openexr>`_ 3.4.11
- `openjpeg <https://github.com/uclouvain/openjpeg>`_ 2.5.4
- `openjph <https://github.com/aous72/OpenJPH>`_ 0.27.2
- `pcodec <https://github.com/mwlon/pcodec>`_ 1.0.2
- `snappy <https://github.com/google/snappy>`_ 1.2.2
- `sperr <https://github.com/NCAR/SPERR>`_ 0.8.5
- `sz3 <https://github.com/szcompressor/SZ3>`_ 3.3.2
- `wavpack <https://github.com/dbry/wavpack>`_ 5.9.0
- `zfp <https://github.com/LLNL/zfp>`_ 1.0.1
- `zlib <https://github.com/madler/zlib>`_ 1.3.2
- `zlib-ng <https://github.com/zlib-ng/zlib-ng>`_ 2.3.3
- `zstd <https://github.com/facebook/zstd>`_ 1.5.7

Unmaintained or discontinued build requirements:

- `brunsli <https://github.com/google/brunsli>`_ 0.1
- `jetraw <https://github.com/Jetraw>`_ 23.03.16.4
- `lzfse <https://github.com/lzfse/lzfse/>`_ 1.0
- `lzham_codec <https://github.com/richgel999/lzham_codec/>`_ 1.0
- `lzokay <https://github.com/AxioDL/lzokay>`_ db2df1f
- `mozjpeg <https://github.com/mozilla/mozjpeg>`_ 4.1.5
- `zopfli <https://github.com/google/zopfli>`_ 1.0.3

Bundled source files:

- `bcdec.h <https://github.com/iOrange/bcdec>`_ 93628fe
- `bitshuffle <https://github.com/kiyo-masui/bitshuffle>`_ 0.5.2
- ccitt.c original 0BSD implementation
- `cfitsio ricecomp.c, pliocomp.c, hcompress.c
  <https://heasarc.gsfc.nasa.gov/fitsio/>`_ modified
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
- `libtiff pixarlog.c
  <https://gitlab.com/libtiff/libtiff/-/blob/master/libtiff/tif_pixarlog.c>`_
  v4.7.1 modified
- `qoi.h <https://github.com/phoboslab/qoi/>`_ 4461cc3
- `rgbe.c <https://www.graphics.cornell.edu/~bjw/rgbe/rgbe.c>`_ modified
- wic.cpp original 0BSD implementation

Test requirements:

- `tifffile <https://github.com/cgohlke/tifffile>`_ 2026.5.2
- `czifile <https://github.com/cgohlke/czifile>`_ 2026.4.30
- `liffile <https://github.com/cgohlke/liffile>`_ 2026.4.11
- `kerchunk <https://github.com/fsspec/kerchunk>`_ 0.2.10
- `python-blosc <https://github.com/Blosc/python-blosc>`_ 1.11.4
- `python-blosc2 <https://github.com/Blosc/python-blosc2>`_ 4.2.0
- `python-brotli <https://github.com/google/brotli/tree/master/python>`_ 1.2.0
- `python-lz4 <https://github.com/python-lz4/python-lz4>`_ 4.4.5
- `python-lzf <https://github.com/teepark/python-lzf>`_ 0.2.6
- `python-snappy <https://github.com/andrix/python-snappy>`_ 0.6.1
- `pyliblzfse <https://github.com/ydkhatri/pyliblzfse>`_ 0.4.1
- `backports.zstd <https://github.com/rogdham/backports.zstd>`_ 1.3.0
- `zopflipy <https://github.com/hattya/zopflipy>`_ 1.12

Revisions
---------

2026.5.10

- Add Zarr 3 compatible codecs.
- Add WIC codec based on Windows Imaging Component.
- Add EXR codec based on OpenEXRCore library.
- Add WAVPACK codec based on WavPack library.
- Add HCOMP and PLIO codecs based on modified cfitsio library.
- Add TGA and PCX/DCX legacy codecs.
- Add option to pass SDR image to ultrahdr_encode.
- Add option to specify primaries and transferfunction in jpegxl_encode (#137).
- Add animated WebP encoding and decoding of all frames (breaking).
- Remove cms_encode and cms_decode aliases for cms_transform (breaking).
- Determine colorspace/pixeltype from profiles in cms_transform.
- Allow to pass IntEnum parameters as strings except for levels.
- Support decoding RLE8 and RLE4 compressed BMP.
- Link zopfli_encode level to numiterations parameter.
- Unify image layout handling in encode functions.
- Fix code review issues.
- Drop support for numpy 2.0 (SPEC0), Python 3.11, and macosx_x86_64.

2026.3.6

- Add CCITTRLE, CCITTFAX3 and CCITTFAX4 codecs (decode only).
- Implement packints_encode function.
- Support lerc subcodec in tiff_encode function.
- Support packed integers, ccitt and pixarlog compression in TIFF codec.
- Support bitorder option in PACKINTS codec.
- Support rounding in BFLOAT16 codec.
- Support more BMP types.
- Update PCODEC to new API.
- Fix buffer overflows in third-party code.
- Fix code review issues.

2026.1.14

- Add tiff_encode function.
- Add extra options for HTJ2K (#134).
- Add linear RGB option to cms_profile.
- Change ZSTD default compression level to 3.

2026.1.1

- Enforce positional-only and keyword-only parameters (breaking).
- Base numcodecs.Jpeg on JPEG8 codec (breaking).
- Add HTJ2K codec based on OpenJPH library (#125).
- Add MESHOPT codec based on meshoptimizer library.
- Fix decoding concatenated ZStandard frames.
- Fix potential issues in TIFF and WEBP codecs.
- Fix pyi stub file.
- Change default Brotli compression level to 4.
- Use Brotli streaming API for decoding.
- Enable decoding UltraHDR to uint16.
- Tweak memory allocation and reallocation strategies.
- Use fused types.
- Improve code quality.

2025.11.11

- Fix EER superresolution decoding (breaking; see tifffile #313).
- Add option to eer_decode to add to uint16 array.
- Add option to specify CICP/NCLX parameters in avif_encode (#131).
- Add BFLOAT16 codec.
- Build ABI3 wheels.
- Require Cython >= 3.2.
- Deprecate Python 3.11.

2025.8.2

- Fix szip_encode default output buffer might be too small (#128).
- Fix minor bugs in LZ4H5 codec (#127).
- Avoid grayscale-to-RGB conversions in AVIF codecs.
- Improve AVIF error messages.
- Add flag for free-threading compatibility (#113).
- Do not use zlib uncompress2, which is not available on manylinux.
- Do not build unstable BRUNSLI, PCODEC, SPERR, and SZ3 codecs.
- Require libavif >= 1.3 and Cython >= 3.1.
- Support Python 3.14 and 3.14t.
- Drop support for Python 3.10 and PyPy.

2025.3.30

- …

Refer to the CHANGES file for older revisions.

Objectives
----------

Many scientific image storage formats, such as TIFF, CZI, XLIF, DICOM, HDF,
and Zarr are containers that store numerous small data segments (chunks,
tiles, stripes). These segments are encoded using various compression and
pre-filtering methods. Metadata common to all data segments are typically
stored separately from the segments.

The purpose of the Imagecodecs library is to support Python modules in
encoding and decoding such data segments. The specific aims are:

- Provide functions for encoding and decoding small image data segments
  in-memory (as opposed to in-file) from and to bytes or numpy arrays for
  many compression and filtering methods.
- Support image formats and compression methods that are not available
  elsewhere in the Python ecosystem.
- Reduce the runtime dependency on numerous, large, inapt, or unmaintained
  Python packages. The Imagecodecs package only depends on numpy.
- Implement codecs as Cython wrappers of third-party libraries with a C API
  and permissive license if available; otherwise use own C library.
  Provide Cython definition files for the wrapped C libraries.
- Release the Python global interpreter lock (GIL) during extended native/C
  function calls for multi-threaded use.

Accessing parts of large data segments and reading metadata from segments
are outside the scope of this library.

Notes
-----

This library is largely a work in progress.

The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Supported platforms are ``win_amd64``, ``win_arm64``, ``win32``,
``macosx_arm64``, ``manylinux_x86_64``, and ``manylinux_aarch64``.

Wheels may not be available for all platforms and all releases.

Not all features are available on all platforms.

The ``bcn``, ``ccittfax3``, ``ccittfax4``, ``ccittrle``, ``dds``,
``dicomrle``, ``eer``, ``jpegsof3``, and ``lzo`` codecs are currently
decode-only.

The ``brunsli`` codec is distributed as source code only because the
underlying library is unstable.

The ``heif``, ``jetraw``, and ``jpegxs`` codecs are distributed as source
code only due to license and possible patent usage issues.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2017-2026
<https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

This software is based in part on the work of the Independent JPEG Group.

When building against libjpeg or libjpeg_turbo < 3, set the environment
variable ``IMAGECODECS_JPEG8_LEGACY=1`` to enable legacy API support.

Before building imagecodecs from source code, install required tools and
libraries. For example, on latest Ubuntu Linux distributions::

    sudo apt-get install build-essential python3-dev cython3 python3-pip \
    python3-setuptools python3-wheel python3-numpy libdeflate-dev libjpeg-dev \
    libjxr-dev liblcms2-dev liblz4-dev liblerc-dev liblzma-dev \
    libopenjp2-7-dev libpng-dev libtiff-dev libwebp-dev libz-dev libzstd-dev

To build and install imagecodecs from source code, run::

    python -m pip install .

Many extensions are disabled by default when building from source.

To define which extensions are built, or to modify build settings such as
library names and compiler arguments, provide a
``imagecodecs_distributor_setup.customize_build`` function, which is
imported and executed during setup.
See ``setup.py`` for pre-defined ``customize_build`` functions.

Other projects providing imaging or compression codecs:
`stdlib-zlib <https://docs.python.org/3/library/zlib.html>`_,
`stdlib-bz2 <https://docs.python.org/3/library/bz2.html>`_,
`stdlib-lzma <https://docs.python.org/3/library/lzma.html>`_,
`backports.lzma <https://github.com/peterjc/backports.lzma>`_,
`python-lzo <https://github.com/jd-boyd/python-lzo>`_,
`python-lzw <https://github.com/joeatwork/python-lzw>`_,
`python-lerc <https://pypi.org/project/lerc/>`_,
`wavpack-numcodecs
<https://github.com/AllenNeuralDynamics/wavpack-numcodecs>`_,
`packbits <https://github.com/psd-tools/packbits>`_,
`isa-l.igzip <https://github.com/intel/isa-l>`_,
`fpzip <https://github.com/seung-lab/fpzip>`_,
`libmng <https://sourceforge.net/projects/libmng/>`_,
`openzl <https://github.com/facebook/openzl>`_,
`openhtj2k <https://github.com/osamu620/OpenHTJ2K>`_,
`pyjetraw <https://github.com/Jetraw>`_,
`tinyexr <https://github.com/syoyo/tinyexr>`_,
`pytinyexr <https://github.com/syoyo/pytinyexr>`_,
`pyroexr <https://github.com/dragly/pyroexr>`_,
`jasper <https://github.com/jasper-software/jasper>`_,
`libjpeg <https://github.com/thorfdbg/libjpeg>`_ (gpl),
`pylibjpeg <https://github.com/pydicom/pylibjpeg>`_,
`pylibjpeg-libjpeg <https://github.com/pydicom/pylibjpeg-libjpeg>`_ (gpl),
`pylibjpeg-openjpeg <https://github.com/pydicom/pylibjpeg-openjpeg>`_,
`pylibjpeg-rle <https://github.com/pydicom/pylibjpeg-rle>`_,
`glymur <https://github.com/quintusdias/glymur>`_,
`pyheif <https://github.com/carsales/pyheif>`_,
`pyrus-cramjam <https://github.com/milesgranger/pyrus-cramjam>`_,
`pylzham <https://github.com/Galaxy1036/pylzham>`_,
`brieflz <https://github.com/jibsen/brieflz>`_,
`quicklz <http://www.quicklz.com/>`_ (gpl),
`lzo <http://www.oberhumer.com/opensource/lzo/>`_ (gpl),
`nvjpeg <https://developer.nvidia.com/nvjpeg>`_,
`nvjpeg2k <https://developer.nvidia.com/nvjpeg>`_,
`pyturbojpeg <https://github.com/lilohuang/PyTurboJPEG>`_,
`ccsds123 <https://github.com/drowzie/CCSDS123-Issue-2>`_,
`lpc-rice <https://sourceforge.net/projects/lpcrice/>`_,
`compression-algorithms <https://github.com/glampert/compression-algorithms>`_,
`compressonator <https://github.com/GPUOpen-Tools/Compressonator>`_,
`wuffs <https://github.com/google/wuffs>`_,
`tinydng <https://github.com/syoyo/tinydng>`_,
`grok <https://github.com/GrokImageCompression/grok>`_ (agpl),
`mafisc
<https://wr.informatik.uni-hamburg.de/research/projects/icomex/mafisc>`_,
`b3d <https://github.com/balintbalazs/B3D>`_,
`fo-dicom.codecs <https://github.com/Efferent-Health/fo-dicom.Codecs>`_,
`jpegli <https://github.com/google/jpegli>`_,
`crackle <https://github.com/seung-lab/crackle>`_,
`hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_.

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
'openjpeg 2.5.4'

Encode a numpy array in lossless JP2 format:

>>> import numpy
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

>>> from imagecodecs import dicomrle_encode
>>> dicomrle_encode(array)
Traceback (most recent call last):
 ...
NotImplementedError: dicomrle_encode

Write the numpy array to a JP2 file:

>>> from imagecodecs import imwrite, imread
>>> imwrite('_test.jp2', array)

Read the image from the JP2 file as numpy array:

>>> image = imread('_test.jp2')
>>> numpy.array_equal(image, array)
True

Create a JPEG 2000 compressed Zarr array using numcodecs:

>>> import zarr
>>> from imagecodecs.numcodecs import register_codecs, Jpeg2k
>>> register_codecs()
>>> zarr.zeros(
...     (4, 5, 512, 512, 3),
...     chunks=(1, 1, 256, 256, 3),
...     dtype='u2',
...     compressor=Jpeg2k(bitspersample=10),
...     zarr_format=2,
... )
<Array ... shape=(4, 5, 512, 512, 3) dtype=uint16>

Create a Delta-LZW compressed Zarr array using zarr codecs:

>>> from imagecodecs.zarr import register_codecs, Delta, Lzw
>>> register_codecs()
>>> zarr.zeros(
...     (4, 5, 512, 512, 3),
...     chunks=(1, 1, 256, 256, 3),
...     dtype='u1',
...     codecs=[Delta(), zarr.codecs.BytesCodec(), Lzw()],
... )
<Array ... shape=(4, 5, 512, 512, 3) dtype=uint8>

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
and open it as a Zarr array using kerchunk:

>>> store.write_fsspec(
...     'temp.json', url='file://', codec_id='imagecodecs_jpeg2k'
... )
>>> from kerchunk.utils import refs_as_store
>>> zarr.open(refs_as_store('temp.json'), mode='r')
<Array <FsspecStore(ReferenceFileSystem, /)> shape=(1, 256, 256, 3)...

View the image in the JP2 file from the command line::

    python -m imagecodecs _test.jp2

"""

from __future__ import annotations

__version__ = '2026.5.10'

import contextlib
import functools
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
    from typing import IO, Any

    from numpy.typing import ArrayLike, NDArray

import numpy

# map extension module names to attribute names
# sync with __all__ in __init__.pyi
# uppercase CODEC names are expanded by _expand_modules below
_MODULES: dict[str, list[str]] = {
    '': [
        '__version__',
        'version',
        'imread',
        'imwrite',
        'imagefileext',
        'DelayedImportError',
        'NONE',
        'NUMPY',
        'JPEG',
    ],
    '_imcd': [
        'imcd_version',
        'numpy_abi_version',
        'cython_version',
        'BFLOAT16',
        'BITORDER',
        'BYTESHUFFLE',
        'DELTA',
        'DICOMRLE',
        'EER',
        'FLOAT24',
        'FLOATPRED',
        'LZW',
        'PACKBITS',
        'PACKINTS',
        'XOR',
    ],
    '_aec': ['AEC'],
    '_apng': ['APNG'],
    '_avif': ['AVIF'],
    '_bcn': ['BCN', 'DDS'],
    '_bitshuffle': ['BITSHUFFLE'],
    '_blosc': ['BLOSC'],
    '_blosc2': ['BLOSC2'],
    '_bmp': ['BMP'],
    '_brotli': ['BROTLI'],
    '_brunsli': ['BRUNSLI'],
    '_bz2': ['BZ2'],
    '_ccitt': ['CCITTRLE', 'CCITTFAX3', 'CCITTFAX4'],
    '_cms': [
        'CMS',
        'CmsError',
        'cms_check',
        'cms_version',
        'cms_transform',
        'cms_profile',
        'cms_profile_validate',
    ],
    '_deflate': ['DEFLATE', 'GZIP', 'deflate_adler32', 'deflate_crc32'],
    '_exr': ['EXR'],
    '_gif': ['GIF'],
    '_h5checksum': [
        'H5CHECKSUM',
        'h5checksum_version',
        'h5checksum_crc',
        'h5checksum_fletcher32',
        'h5checksum_hash_string',
        'h5checksum_lookup3',
        'h5checksum_metadata',
    ],
    '_hcomp': ['HCOMP'],
    '_heif': ['HEIF'],
    '_htj2k': ['HTJ2K', 'htj2k_init'],
    # '_isal': ['ISAL', 'isal_adler32', 'isal_crc32'],
    '_jetraw': ['JETRAW', 'jetraw_init'],
    '_jpeg2k': ['JPEG2K'],
    '_jpeg8': ['JPEG8'],
    # '_jpegli': ['JPEGLI'],
    '_jpegls': ['JPEGLS'],
    '_jpegsof3': ['JPEGSOF3'],
    '_jpegxl': ['JPEGXL', 'jpegxl_decode_jpeg', 'jpegxl_encode_jpeg'],
    '_jpegxr': ['JPEGXR'],
    '_jpegxs': ['JPEGXS'],
    '_lerc': ['LERC'],
    '_ljpeg': ['LJPEG'],
    '_lz4': ['LZ4', 'LZ4H5'],
    '_lz4f': ['LZ4F'],
    '_lzf': ['LZF'],
    '_lzfse': ['LZFSE'],
    '_lzham': ['LZHAM'],
    '_lzma': ['LZMA'],
    '_lzo': ['LZO'],
    '_meshopt': ['MESHOPT'],
    '_mozjpeg': ['MOZJPEG'],
    # '_openzl': ['OPENZL'],
    '_pcodec': ['PCODEC'],
    '_pcx': ['PCX'],
    '_pglz': ['PGLZ'],
    '_pixarlog': ['PIXARLOG'],
    '_plio': ['PLIO'],
    '_png': ['PNG'],
    '_qoi': ['QOI'],
    '_quantize': ['QUANTIZE'],
    '_rcomp': ['RCOMP'],
    '_rgbe': ['RGBE'],
    '_snappy': ['SNAPPY'],
    '_sperr': ['SPERR'],
    '_spng': ['SPNG'],
    '_sz3': ['SZ3'],
    '_szip': ['SZIP', 'szip_params'],
    '_tga': ['TGA'],
    '_tiff': ['TIFF'],
    '_ultrahdr': ['ULTRAHDR'],
    '_wavpack': ['WAVPACK', 'wavpack_info'],
    '_webp': ['WEBP'],
    '_wic': ['WIC'],
    '_zfp': ['ZFP'],
    '_zlib': ['ZLIB', 'zlib_adler32', 'zlib_crc32'],
    '_zlibng': ['ZLIBNG', 'zlibng_adler32', 'zlibng_crc32'],
    '_zopfli': ['ZOPFLI'],
    '_zstd': ['ZSTD'],
}

# map extra to existing attributes
_ALIASES: dict[str, str] = {
    'JPEG': 'JPEG8',
    'JpegError': 'Jpeg8Error',
    'jpeg_check': 'jpeg8_check',
    'jpeg_version': 'jpeg8_version',
    'zopfli_check': 'zlib_check',
    'zopfli_decode': 'zlib_decode',
    # 'zopfli_encode' is a different algorithm
}


def _add_codec(
    module: str,
    codec: str | None = None,
    attributes: tuple[str, ...] | None = None,
    /,
) -> None:
    """Register codec in global _MODULES and _ATTRIBUTES (for testing)."""
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
    _ATTRIBUTES.update(dict.fromkeys(attributes, module))


def _load_all() -> None:
    """Add all registered attributes to package namespace (for testing)."""
    for name in __dir__():
        __getattr__(name)


def _set_module() -> None:
    """Set __module__ attribute on public objects."""
    globs = globals()
    for item in _MODULES['']:
        if item in globs:
            obj = globs[item]
            if hasattr(obj, '__module__'):
                obj.__module__ = 'imagecodecs'


def _expand_modules(modules: dict[str, list[str]], /) -> dict[str, list[str]]:
    """Expand uppercase codec names to standard codec attributes.

    An uppercase item like 'AEC' expands to:
        AEC, AecError, aec_encode, aec_decode, aec_check, aec_version.

    Expansion is skipped when any standard name (e.g. 'aec_version') is
    already listed explicitly alongside the uppercase token, as is the case
    for non-standard codec-like constants such as H5CHECKSUM.

    """
    result: dict[str, list[str]] = {}
    for module, items in modules.items():
        attrs: list[str] = []
        items_set = set(items)
        for item in items:
            if item.isupper():
                codec = item.lower()
                standard = {
                    f'{item.capitalize()}Error',
                    f'{codec}_encode',
                    f'{codec}_decode',
                    f'{codec}_check',
                    f'{codec}_version',
                }
                if standard.isdisjoint(items_set):
                    attrs += [
                        item,
                        f'{item.capitalize()}Error',
                        f'{codec}_encode',
                        f'{codec}_decode',
                        f'{codec}_check',
                        f'{codec}_version',
                    ]
                else:
                    attrs.append(item)
            else:
                attrs.append(item)
        result[module] = attrs
    return result


_MODULES = _expand_modules(_MODULES)

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


def __dir__() -> list[str]:
    """Return list of attribute names accessible on module."""
    return sorted(list(_ATTRIBUTES) + list(_ALIASES))


def __getattr__(name: str, /) -> Any:
    """Return module attribute after loading it from extension module.

    Load attribute's extension and add its attributes to the package namespace.

    """
    name_ = name
    name = _ALIASES.get(name, name)

    if name not in _ATTRIBUTES:
        msg = f"module 'imagecodecs' has no attribute {name!r}"
        raise AttributeError(msg)

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
            from . import _shared  # noqa: F401

            module = None

        for n in _MODULES[module_name]:
            if n in _ALIASES:
                continue
            attr = getattr(module, n, None)
            if attr is None:
                attr = _stub(n, module)
                # TODO: do not set __module__?
                # it might interfere with introspection tools
            attr.__module__ = 'imagecodecs'
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


def _codecs(*, available: bool | None = None) -> tuple[str, ...]:
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
            c for c in codecs if getattr(imagecodecs, c.upper()).available
        )
    return tuple(
        c for c in codecs if not getattr(imagecodecs, c.upper()).available
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

    # determine list of codecs based on file extension or provided codec
    codecs: list[str | Callable[..., NDArray[Any]]] = []
    if codec is None:
        # find codec based on file extension
        ext_to_codec, codec_list = _imcodecs()
        if isinstance(fileobj, (str, os.PathLike)):
            ext = os.path.splitext(os.fspath(fileobj))[-1][1:].lower()
        else:
            ext = None
        if ext in ext_to_codec:
            codec = ext_to_codec[ext]
            if codec == 'jpeg8':
                codecs.extend(('jpeg8', 'ljpeg'))  # 'jpegsof3'
            else:
                codecs.append(codec)
        # also try other imaging codecs
        codecs.extend(c for c in codec_list if c not in codecs)
    else:
        # use provided codecs
        if not isinstance(codec, (list, tuple)):  # collections.abc.Iterable
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                cl = c.lower()
                codecs.append(_imcodecs()[0].get(cl, cl))
            else:
                codecs.append(c)
    del codec

    # read data from file object
    func: Callable[..., NDArray[Any]] | None = None
    exceptions: list[str] = []
    image: NDArray[Any] | None = None
    data: bytes | bytearray | mmap.mmap
    offset: int = -1
    close = False

    try:
        # TODO: support urllib.request.urlopen ?
        if isinstance(fileobj, (str, os.PathLike)):
            # file name
            with open(os.fspath(fileobj), 'rb') as fh:
                if memmap:
                    offset = 0
                    close = True
                    data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                else:
                    data = fh.read()
        elif isinstance(fileobj, mmap.mmap):
            data = fileobj
            offset = data.tell()
        elif hasattr(fileobj, 'read'):
            # binary stream: open file, BytesIO
            data = fileobj.read()
            assert isinstance(data, (bytes, bytearray))
        else:
            # binary data
            data = fileobj
            assert isinstance(data, (bytes, bytearray, mmap.mmap))

        # try decode data using codecs
        numthreads = kwargs.pop('numthreads', None)
        for codec in codecs:

            # get decoder function or class
            if callable(codec):
                func = codec
            else:
                try:
                    func = getattr(imagecodecs, codec + '_decode')
                    assert callable(func)
                except Exception as exc:
                    exceptions.append(f'{repr(codec).upper()}: {exc}')
                    continue

            _numthreads = numthreads
            if _numthreads is not None and not func.__name__.startswith(
                ('avif', 'jpeg2k', 'jpegxl', 'zfp')
            ):
                _numthreads = None

            # decode data
            try:
                if _numthreads is None:
                    image = func(data, **kwargs)
                else:
                    image = func(data, numthreads=_numthreads, **kwargs)
                assert isinstance(image, numpy.ndarray)
                if image.dtype.kind == 'O':
                    msg = 'failed'
                    raise ValueError(msg)  # noqa: TRY301
                break
            except DelayedImportError:
                pass
            except Exception as exc:
                # raise  # uncomment for debugging
                exceptions.append(f'{func.__name__.upper()}: {exc}')
            if offset >= 0:
                assert isinstance(data, mmap.mmap)
                data.seek(offset)

    finally:
        if close:
            assert isinstance(data, mmap.mmap)
            data.close()

    if image is None:
        raise ValueError('\n'.join(exceptions))

    if return_codec:
        if func is None:
            msg = 'no valid decoder found'
            raise ValueError(msg)
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
            msg = 'no codec specified'
            raise ValueError(msg)

        codec = _imcodecs()[0].get(ext, ext)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            msg = f'invalid {codec=!r}'
            raise ValueError(msg) from exc

    elif isinstance(codec, str):
        codec = codec.lower()
        codec = _imcodecs()[0].get(codec, codec)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            msg = f'invalid {codec=!r}'
            raise ValueError(msg) from exc

    if not callable(codec):
        msg = f'invalid {codec=!r}'
        raise TypeError(msg)

    image: bytes | bytearray = codec(data, **kwargs)
    if hasattr(fileobj, 'write'):
        # binary stream: open file, BytesIO
        fileobj.write(image)
    else:
        # file name
        with open(fileobj, 'wb') as fh:
            fh.write(image)


@functools.cache
def _imcodecs() -> tuple[dict[str, str], list[str]]:
    """Return map of image file extensions to codec names and codec names."""
    codecs = {
        'apng': ('apng',),
        'avif': ('avif', 'avifs'),
        'bmp': ('bmp', 'dip'),  # 'rle'
        'brunsli': ('brn',),
        'dds': ('dds',),
        'exr': ('exr',),
        'gif': ('gif',),
        'heif': (
            'heif',
            'heic',
            'heifs',
            'heics',
            'hif',  # 'avci', 'avcs'
        ),
        'htj2k': (
            'htj2k',
            'jph',  # HTJ2K with JP2 boxes
            'jhc',  # HTJ2K codestream
            # 'j2c',
        ),
        'jpeg2k': (
            'j2k',
            'jp2',
            'j2c',
            'jpc',
            'jpx',
            'jpf',
            'jpg2',
            # 'jph',  # HTJ2K with JP2 boxes
            # 'jhc',  # HTJ2K codestream
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
        'pcx': ('pcx', 'dcx'),
        'png': ('png',),
        'qoi': ('qoi',),
        'rgbe': ('hdr', 'rgbe', 'pic'),
        'tga': ('tga',),
        'tiff': ('tif', 'tiff', 'ptif', 'ptiff', 'tf8', 'tf2', 'btf'),
        'ultrahdr': ('uhdr', 'jpr'),  # jpg
        'wavpack': ('wv',),
        'webp': ('webp', 'webm'),
        'wic': ('ico',),  # bmp, heif, jpg, png, tiff, webp
        'zfp': ('zfp',),
    }
    return (
        {ext: codec for codec, exts in codecs.items() for ext in exts},
        list(codecs.keys()),
    )


def imagefileext() -> list[str]:
    """Return list of image file extensions handled by imread and imwrite."""
    return list(_imcodecs()[0].keys())


class NONE:
    """NONE codec constants."""

    available = True
    """NONE codec is available."""


class NoneError(RuntimeError):
    """NONE codec error."""


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


class NumpyError(RuntimeError):
    """NumPy codec error."""


def numpy_version() -> str:
    """Return Numpy library version string."""
    return f'numpy {numpy.__version__}'


def numpy_check(data: bytes | bytearray, /) -> bool:
    """Return whether data is NPY or NPZ encoded or None if unknown."""
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
            msg = 'not a numpy array'
            raise ValueError(msg) from exc
        if hasattr(result, 'files'):
            with contextlib.suppress(Exception):
                index = result.files[index]
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
        return fh.read()


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
            with contextlib.suppress(Exception):
                return imagecodecs.ljpeg_decode(  # type: ignore[no-any-return]
                    data, out=out
                )
        # elif 'Empty JPEG image' in msg:
        # for example, Hamamatsu NDPI slides with dimensions > 65500
        # Unsupported marker type
        raise


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
        and not getattr(imagecodecs.JPEG8, 'all_precisions', False)
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

_set_module()
