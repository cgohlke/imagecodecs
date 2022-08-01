Image transformation, compression, and decompression codecs
===========================================================

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in Tifffile,
Czifile, Zarr, and other scientific image input/output packages.

Decode and/or encode functions are implemented for Zlib (DEFLATE), GZIP,
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZ4F, LZ4HC, LZW,
LZF, PGLZ (PostgreSQL LZ), RCOMP (Rice), ZFP, AEC, LERC, NPY, PNG, APNG,
GIF, TIFF, WebP, QOI, JPEG 8-bit, JPEG 12-bit, Lossless JPEG (LJPEG, SOF3),
JPEG 2000 (JP2, J2K), JPEG LS, JPEG XR (WDP, HD Photo), JPEG XL, MOZJPEG,
AVIF, HEIF, Jetraw, PackBits, Packed Integers, Delta, XOR Delta,
Floating Point Predictor, Bitorder reversal, Byteshuffle, Bitshuffle,
CMS (color space transformations), and Float24 (24-bit floating point).

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.7.31
:DOI: 10.5281/zenodo.6915978

Installation
------------

Install the imagecodecs package from the Python Package Index::

    python -m pip install -U imagecodecs

Wheels for macOS may not be available for the latest releases.

Imagecodecs is also available in other package repositories such as
`Anaconda <https://anaconda.org/conda-forge/imagecodecs>`_,
`MSYS2 <https://packages.msys2.org/base/mingw-w64-python-imagecodecs>`_, and
`MacPorts <https://ports.macports.org/port/py-imagecodecs/summary>`_.

See `Requirements`_ and `Notes`_ for building from source.

Requirements
------------

This release has been tested with the following requirements and dependencies
(other versions may work):

- `CPython 3.8.10, 3.9.13, 3.10.5, 3.11.0b5 <https://www.python.org>`_
  (AMD64 platforms only, 32-bit versions are deprecated)
- `Numpy 1.21.5 <https://pypi.org/project/numpy>`_

Build requirements:

- `Cython 0.29.32 <https://github.com/cython/cython>`_
- `bitshuffle 0.4.2 <https://github.com/kiyo-masui/bitshuffle>`_
- `brotli 1.0.9 <https://github.com/google/brotli>`_
- `brunsli 0.1 <https://github.com/google/brunsli>`_
- `bzip2 1.0.8 <https://gitlab.com/bzip2/bzip2>`_
- `c-blosc 1.21.1 <https://github.com/Blosc/c-blosc>`_
- `c-blosc2 2.2.0 <https://github.com/Blosc/c-blosc2>`_
- `cfitsio 3.49 <https://heasarc.gsfc.nasa.gov/fitsio/>`_
- `charls 2.3.4 <https://github.com/team-charls/charls>`_
- `giflib 5.2.1 <http://giflib.sourceforge.net/>`_
- `jetraw 22.02.16.1 <https://github.com/Jetraw/Jetraw>`_
- `jxrlib 1.1 <https://packages.debian.org/source/sid/jxrlib>`_
- `lcms 2.13.1 <https://github.com/mm2/Little-CMS>`_
- `lerc 4.0.0 <https://github.com/Esri/lerc>`_
- `libaec 1.0.6 <https://gitlab.dkrz.de/k202009/libaec>`_
- `libavif 0.10.1 <https://github.com/AOMediaCodec/libavif>`_
  (`aom 3.4.0 <https://aomedia.googlesource.com/aom>`_,
  `dav1d 1.0.0 <https://github.com/videolan/dav1d>`_,
  `rav1e 0.5.1 <https://github.com/xiph/rav1e>`_)
- `libdeflate 1.12 <https://github.com/ebiggers/libdeflate>`_
- `libheif 1.12.0 <https://github.com/strukturag/libheif>`_
  (`libde265 1.0.8 <https://github.com/strukturag/libde265>`_)
- `libjpeg 9d <http://libjpeg.sourceforge.net/>`_
- `libjpeg-turbo 2.1.3 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
- `libjxl 0.6.1 <https://github.com/libjxl/libjxl>`_
- `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
- `liblzma 5.2.5 <https://github.com/xz-mirror/xz>`_
- `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
- `libpng-apng 1.6.37 <https://sourceforge.net/projects/libpng-apng/>`_
- `libspng 0.7.2 <https://github.com/randy408/libspng>`_
- `libtiff 4.4.0 <https://gitlab.com/libtiff/libtiff>`_
- `libwebp 1.2.3 <https://github.com/webmproject/libwebp>`_
- `lz4 1.9.3 <https://github.com/lz4/lz4>`_
- `mozjpeg 4.0.3 <https://github.com/mozilla/mozjpeg>`_
- `openjpeg 2.5.0 <https://github.com/uclouvain/openjpeg>`_
- `qoi 75e7f30 <https://github.com/phoboslab/qoi>`_
- `snappy 1.1.9 <https://github.com/google/snappy>`_
- `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
- `zlib 1.2.12 <https://github.com/madler/zlib>`_
- `zlib-ng 2.0.6 <https://github.com/zlib-ng/zlib-ng>`_
- `zopfli-1.0.3 <https://github.com/google/zopfli>`_
- `zstd 1.5.2 <https://github.com/facebook/zstd>`_

Test requirements:

- `tifffile 2022.7.31 <https://pypi.org/project/tifffile>`_
- `czifile 2019.7.2 <https://pypi.org/project/czifile>`_
- `python-blosc 1.10.6 <https://github.com/Blosc/python-blosc>`_
- `python-blosc2-0.2.0 <https://github.com/Blosc/python-blosc2>`_
- `python-brotli 1.0.9 <https://github.com/google/brotli/tree/master/python>`_
- `python-lz4 4.0.2 <https://github.com/python-lz4/python-lz4>`_
- `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
- `python-snappy 0.6.1 <https://github.com/andrix/python-snappy>`_
- `python-zstd 1.5.2 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
- `bitshuffle 0.4.2 <https://github.com/kiyo-masui/bitshuffle>`_
- `numcodecs 0.10.1 <https://github.com/zarr-developers/numcodecs>`_
- `zarr 2.12.0 <https://github.com/zarr-developers/zarr-python>`_
- `zopflipy 1.7 <https://github.com/hattya/zopflipy>`_

Revisions
---------

2022.7.31

- Pass 6348 tests.
- Add option to decode WebP as RGBA.
- Add option to specify WebP compression method.
- Use exact lossless WebP encoding.

2022.7.27

- Add LZW encoder.
- Add QOI codec via qoi.h (#37).
- Add HEIF codec via libheif (source only; #33).
- Add JETRAW codec via Jetraw demo (source only).
- Add ByteShuffle codec, a generic version of FloatPred.
- Replace imcd_floatpred by imcd_byteshuffle (breaking).
- Use bool type in imcd (breaking).

2022.2.22

- Fix jpeg numcodecs with tables (#28).
- Add APNG codec via libpng-apng patch.
- Add lossless and decodingspeed parameters to jpegxl_encode (#30).
- Add option to read JPEG XL animations.
- Add dummy numthreads parameter to codec functions.
- Set default numthreads to 1 (disable multi-threading).
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.11.20

- Fix testing on pypy and Python 3.10.

2021.11.11

- Require libjxl 0.6.x.
- Add CMS codec via Little CMS library for color space transformations (WIP).
- Add MOZJPEG codec via mozjpeg library (Windows only).
- Add SPNG codec via libspng library.
- Rename avif_encode maxthreads parameter to numthreads (breaking).
- Accept n-dimensional output in non-image numcodecs decoders.
- Support masks in LERC codec.
- Support multi-threading and planar format in JPEG2K codec.
- Support multi-resolution, MCT, bitspersample, and 32-bit in jpeg2k encoder.
- Change jpeg2k_encode level parameter to fixed quality psnr (breaking).
- Change jpegxl_encode effort parameter default to minimum 3.
- Change JPEG encoders to use YCbCr for RGB images by default.
- Replace lerc_encode planarconfig with planar parameter (breaking).
- Add option to specify omp numthreads and chunksize in ZFP codec.
- Set default numthreads to 0.
- Fix Blosc default typesize.
- Fix segfault in jpegxl_encode.
- Replace many constants with enums (breaking).

2021.8.26

- Add BLOSC2 codec via c-blosc2 library.
- Require LERC 3 and libjxl 0.5.
- Do not exceed literal-only size in PackBits encoder.
- Raise ImcdError if output is insufficient in PackBits codecs (breaking).
- Raise ImcdError if input is corrupt in PackBits decoder (breaking).
- Fix delta codec for non-native byteorder.

2021.7.30

- ...

Refer to the CHANGES file for older revisions.

Notes
-----

This library is largely work in progress.

The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

The ``tiff``, ``packints``, and ``jpegsof3`` codecs are currently decode-only.

The ``heif`` and ``jetraw`` codecs are distributed as source code only due to
license and possible patent usage issues.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
<https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

This software is based in part on the work of the Independent JPEG Group.

This software includes a modified version of `dcm2niix's jpg_0XC3.cpp
<https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_.

This software includes a modified version of `PostgreSQL's pg_lzcompress.c
<https://github.com/postgres/postgres/blob/REL_13_STABLE/src/common/
pg_lzcompress.c>`_.

This software includes a modified version of `bitshuffle
<https://github.com/kiyo-masui/bitshuffle>`_.

This software includes a modified version of `liblj92
<https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/>`_.

This software includes `qoi.h <https://github.com/phoboslab/qoi/>`_.

Build instructions and wheels for manylinux and macOS courtesy of
`Grzegorz Bokota <https://github.com/Czaki/imagecodecs_build>`_.

Update pip and setuptools to the latest version before installing imagecodecs::

    python -m pip install -U pip setuptools wheel Cython

Install the requirements for building imagecodecs from source code on
latest Ubuntu Linux distributions:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy python3-zarr
    python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-dev libjxr-dev
    liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev
    libzopfli-dev libgif-dev libtiff-dev libdeflate-dev libavif-dev
    libheif-dev libcfitsio-dev``

Use the ``--lite`` build option to only build extensions without 3rd-party
dependencies. Use the ``--skip-extension`` build options to skip building
specific extensions, e.g.:

    ``python -m pip install imagecodecs --global-option="build_ext"
    --global-option="--skip-bitshuffle"``

The ``apng``, ``avif``, ``jetraw``, ``jpeg12``, ``jpegls``, ``jpegxl``,
``lerc``, ``lz4f``, ``mozjpeg``, ``zfp``, and ``zlibng`` extensions are
disabled by default when building from source.

To modify other build settings such as library names and compiler arguments,
provide a ``imagecodecs_distributor_setup.customize_build`` function, which
is imported and executed during setup. See ``setup.py`` for examples.

Other Python packages and C libraries providing imaging or compression codecs:
`Python zlib <https://docs.python.org/3/library/zlib.html>`_,
`Python bz2 <https://docs.python.org/3/library/bz2.html>`_,
`Python lzma <https://docs.python.org/3/library/lzma.html>`_,
`backports.lzma <https://github.com/peterjc/backports.lzma>`_,
`python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_,
`python-lzw <https://github.com/joeatwork/python-lzw>`_,
`python-lerc <https://pypi.org/project/lerc/>`_,
`packbits <https://github.com/psd-tools/packbits>`_,
`isa-l.igzip <https://github.com/intel/isa-l>`_,
`fpzip <https://github.com/seung-lab/fpzip>`_,
`libmng <https://sourceforge.net/projects/libmng/>`_,
`OpenEXR <https://github.com/AcademySoftwareFoundation/openexr>`_
(EXR, PIZ, PXR24, B44, DWA),
`pyJetraw <https://github.com/Jetraw/pyJetraw>`_,
`tinyexr <https://github.com/syoyo/tinyexr>`_,
`pytinyexr <https://github.com/syoyo/pytinyexr>`_,
`libjpeg <https://github.com/thorfdbg/libjpeg>`_ (GPL),
`pylibjpeg <https://github.com/pydicom/pylibjpeg>`_,
`pylibjpeg-libjpeg <https://github.com/pydicom/pylibjpeg-libjpeg>`_ (GPL),
`pylibjpeg-openjpeg <https://github.com/pydicom/pylibjpeg-openjpeg>`_,
`pylibjpeg-rle <https://github.com/pydicom/pylibjpeg-rle>`_,
`glymur <https://github.com/quintusdias/glymur>`_,
`pyheif <https://github.com/carsales/pyheif>`_,
`pyrus-cramjam <https://github.com/milesgranger/pyrus-cramjam>`_,
`QuickLZ <http://www.quicklz.com/>`_ (GPL),
`LZO <http://www.oberhumer.com/opensource/lzo/>`_ (GPL),
`nvJPEG <https://developer.nvidia.com/nvjpeg>`_,
`nvJPEG2K <https://developer.nvidia.com/nvjpeg>`_,
`PyTurboJPEG <https://github.com/lilohuang/PyTurboJPEG>`_,
`CCSDS123 <https://github.com/drowzie/CCSDS123-Issue-2>`_,
`LPC-Rice <https://sourceforge.net/projects/lpcrice/>`_,
`MAFISC
<https://wr.informatik.uni-hamburg.de/research/projects/icomex/mafisc>`_.

Examples
--------

Import the JPEG2K codec:

>>> from imagecodecs import (
...     jpeg2k_encode, jpeg2k_decode, jpeg2k_check, jpeg2k_version, JPEG2K
... )

Check that the JPEG2K codec is available in the imagecodecs build:

>>> bool(JPEG2K)
True

Print the version of the JPEG2K codec's underlying OpenJPEG library:

>>> jpeg2k_version()
'openjpeg 2.5.0'

Encode a numpy array in lossless JP2 format:

>>> array = numpy.random.randint(100, 200, (256, 256, 3), numpy.uint8)
>>> encoded = jpeg2k_encode(array, level=0)
>>> encoded[:12]
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

View the image in the JP2 file from the command line::

    $ python -m imagecodecs _test.jp2
