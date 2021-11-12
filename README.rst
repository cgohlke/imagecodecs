Image transformation, compression, and decompression codecs
===========================================================

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in the
tifffile, czifile, zarr, and other scientific image input/output modules.

Decode and/or encode functions are implemented for Zlib (DEFLATE), GZIP,
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZ4F, LZ4HC,
LZW, LZF, PGLZ (PostgreSQL LZ), RCOMP (Rice), ZFP, AEC, LERC, NPY,
PNG, GIF, TIFF, WebP, JPEG 8-bit, JPEG 12-bit, Lossless JPEG (LJPEG, SOF3),
MOZJPEG, JPEG 2000, JPEG LS, JPEG XR (WDP, HD Photo), JPEG XL, AVIF,
PackBits, Packed Integers, Delta, XOR Delta, Floating Point Predictor,
Bitorder reversal, Bitshuffle, CMS (color space transformations), and
Float24 (24-bit floating point).

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.11.11

:Status: Alpha

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.7.9, 3.8.10, 3.9.8, 3.10.0, 64-bit <https://www.python.org>`_
* `Cython 0.29.24 <https://cython.org>`_
* `Numpy 1.21.4 <https://pypi.org/project/numpy>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `brotli 1.0.9 <https://github.com/google/brotli>`_
* `brunsli 0.1 <https://github.com/google/brunsli>`_
* `bzip2 1.0.8 <https://gitlab.com/bzip2/bzip2>`_
* `c-blosc 1.21.1 <https://github.com/Blosc/c-blosc>`_
* `c-blosc2 2.0.4 <https://github.com/Blosc/c-blosc2>`_
* `cfitsio 3.49 <https://heasarc.gsfc.nasa.gov/fitsio/>`_
* `charls 2.2.0 <https://github.com/team-charls/charls>`_
* `giflib 5.2.1 <http://giflib.sourceforge.net/>`_
* `jxrlib 1.1 <https://packages.debian.org/source/sid/jxrlib>`_
* `lcms 2.12 <https://github.com/mm2/Little-CMS>`_
* `lerc 3.0 <https://github.com/Esri/lerc>`_
* `libaec 1.0.6 <https://gitlab.dkrz.de/k202009/libaec>`_
* `libavif 0.9.2 <https://github.com/AOMediaCodec/libavif>`_
  (`aom 3.1.3 <https://aomedia.googlesource.com/aom>`_,
  `dav1d 0.9.2 <https://github.com/videolan/dav1d>`_,
  `rav1e 0.4.1 <https://github.com/xiph/rav1e>`_)
* `libdeflate 1.8 <https://github.com/ebiggers/libdeflate>`_
* `libjpeg 9d <http://libjpeg.sourceforge.net/>`_
* `libjpeg-turbo 2.1.1 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
* `libjxl 0.6.1 <https://github.com/libjxl/libjxl>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `liblzma 5.2.5 <https://github.com/xz-mirror/xz>`_
* `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
* `libspng 0.7.0 <https://github.com/randy408/libspng>`_
* `libtiff 4.3.0 <https://gitlab.com/libtiff/libtiff>`_
* `libwebp 1.2.1 <https://github.com/webmproject/libwebp>`_
* `lz4 1.9.3 <https://github.com/lz4/lz4>`_
* `mozjpeg 4.0.3 <https://github.com/mozilla/mozjpeg>`_
* `openjpeg 2.4.0 <https://github.com/uclouvain/openjpeg>`_
* `snappy 1.1.9 <https://github.com/google/snappy>`_
* `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
* `zlib 1.2.11 <https://github.com/madler/zlib>`_
* `zlib-ng 2.0.5 <https://github.com/zlib-ng/zlib-ng>`_
* `zopfli-1.0.3 <https://github.com/google/zopfli>`_
* `zstd 1.5.0 <https://github.com/facebook/zstd>`_

Required Python packages for testing (other versions may work):

* `tifffile 2021.11.2  <https://pypi.org/project/tifffile>`_
* `czifile 2019.7.2 <https://pypi.org/project/czifile>`_
* `python-blosc 1.10.6 <https://github.com/Blosc/python-blosc>`_
* `python-blosc2-0.2.0 <https://github.com/Blosc/python-blosc2>`_
* `python-brotli 1.0.9 <https://github.com/google/brotli/tree/master/python>`_
* `python-lz4 3.1.3 <https://github.com/python-lz4/python-lz4>`_
* `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
* `python-snappy 0.6.0 <https://github.com/andrix/python-snappy>`_
* `python-zstd 1.5.0.2 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `numcodecs 0.9.1 <https://github.com/zarr-developers/numcodecs>`_
* `zarr 2.10.2 <https://github.com/zarr-developers/zarr-python>`_
* `zopflipy 1.7 <https://github.com/hattya/zopflipy>`_

Notes
-----
This library is largely work in progress.

The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 32-bit versions are deprecated. Python <= 3.6 are no longer supported.

MacOS wheels may not be available for latest releases. Consider using
`previous <https://pypi.org/project/imagecodecs/#history>`_,
`conda-forge <https://github.com/conda-forge/imagecodecs-feedstock>`_ or
`macports <https://ports.macports.org/port/py-imagecodecs/summary>`_ releases.

Some codecs are currently decode-only: ``tiff``, ``lzw``, ``packints``, and
``jpegsof3``.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017
and 2019 <https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

This software is based in part on the work of the Independent JPEG Group.

This software includes a modified version of `dcm2niix's jpg_0XC3.cpp
<https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_.

This software includes a modified version of `PostgreSQL's pg_lzcompress.c
<https://github.com/postgres/postgres/blob/REL_13_STABLE/src/common/
pg_lzcompress.c>`_.

This software includes a modified version of `liblj92
<https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/>`_.

Build instructions and wheels for manylinux and macOS courtesy of
`Grzegorz Bokota <https://github.com/Czaki/imagecodecs_build>`_.

Update pip and setuptools to the latest version before installing imagecodecs:

    ``python -m pip install --upgrade pip setuptools``

Install imagecodecs using precompiled wheels:

    ``python -m pip install --upgrade imagecodecs``

Install the requirements for building imagecodecs from source code on
latest Ubuntu Linux distributions:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy python3-zarr
    python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-dev libjxr-dev
    liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev
    libzopfli-dev libgif-dev libtiff-dev libdeflate-dev libavif-dev
    libcfitsio-dev``

Use the ``--lite`` build option to only build extensions without 3rd-party
dependencies. Use the ``--skip-extension`` build options to skip building
specific extensions, e.g.:

    ``python -m pip install imagecodecs --global-option="build_ext"
    --global-option="--skip-bitshuffle"``

The ``jpeg12``, ``jpegls``, ``jpegxl``, ``zfp``, ``avif``, ``lz4f``, ``lerc``,
``mozjpeg``, and ``zlibng`` extensions are disabled by default when building
from source.

To modify other build settings such as library names and compiler arguments,
provide a ``imagecodecs_distributor_setup.customize_build`` function, which
is imported and executed during setup. See ``setup.py`` for examples.

Other Python packages and C libraries providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_
* `backports.lzma <https://github.com/peterjc/backports.lzma>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `python-lerc <https://pypi.org/project/lerc/>`_
* `packbits <https://github.com/psd-tools/packbits>`_
* `isa-l.igzip <https://github.com/intel/isa-l>`_
* `fpzip <https://github.com/seung-lab/fpzip>`_
* `libmng <https://sourceforge.net/projects/libmng/>`_
* `APNG patch for libpng <https://sourceforge.net/projects/libpng-apng/>`_
* `OpenEXR <https://github.com/AcademySoftwareFoundation/openexr>`_
* `tinyexr <https://github.com/syoyo/tinyexr>`_
* `pytinyexr <https://github.com/syoyo/pytinyexr>`_
* `libjpeg <https://github.com/thorfdbg/libjpeg>`_ (GPL)
* `pylibjpeg <https://github.com/pydicom/pylibjpeg>`_
* `pylibjpeg-libjpeg <https://github.com/pydicom/pylibjpeg-libjpeg>`_ (GPL)
* `pylibjpeg-openjpeg <https://github.com/pydicom/pylibjpeg-openjpeg>`_
* `pylibjpeg-rle <https://github.com/pydicom/pylibjpeg-rle>`_
* `glymur <https://github.com/quintusdias/glymur>`_
* `pyheif <https://github.com/carsales/pyheif>`_
* `libheif <https://github.com/strukturag/libheif>`_ (LGPL)
* `pyrus-cramjam <https://github.com/milesgranger/pyrus-cramjam>`_
* `QuickLZ <http://www.quicklz.com/>`_ (GPL)
* `LZO <http://www.oberhumer.com/opensource/lzo/>`_ (GPL)

Revisions
---------
2021.11.11
    Pass 5947 tests.
    Require libjxl 0.6.x.
    Add CMS codec via Little CMS library for color space transformations (WIP).
    Add MOZJPEG codec via mozjpeg library (Windows only).
    Add SPNG codec via libspng library.
    Rename avif_encode maxthreads parameter to numthreads (breaking).
    Accept n-dimensional output in non-image numcodecs decoders.
    Support masks in LERC codec.
    Support multi-threading and planar format in JPEG2K codec.
    Support multi-resolution, MCT, bitspersample, and 32-bit in jpeg2k encoder.
    Change jpeg2k_encode level parameter to fixed quality psnr (breaking).
    Change jpegxl_encode effort parameter default to minimum 3.
    Change JPEG encoders to use YCbCr for RGB images by default.
    Replace lerc_encode planarconfig with planar parameter (breaking).
    Add option to specify omp numthreads and chunksize in ZFP codec.
    Set default numthreads to 0.
    Fix Blosc default typesize.
    Fix segfault in jpegxl_encode.
    Replace many constants with enums (breaking).
2021.8.26
    Add BLOSC2 codec via c-blosc2 library.
    Require LERC 3 and libjxl 0.5.
    Do not exceed literal-only size in PackBits encoder.
    Raise ImcdError if output is insufficient in PackBits codecs (breaking).
    Raise ImcdError if input is corrupt in PackBits decoder (breaking).
    Fix delta codec for non-native byteorder.
2021.7.30
    Support more dtypes and axes argument in PackBits encoder.
    Fix worst case output size in PackBits encoder.
    Fix decoding AVIF created with older libavif.
    Fix decoding GIF with disposal to previous for first frame.
    Add lossless option in jpeg_encode.
2021.6.8
    Fix building with Cython 0.3a7.
    Decode TIFF with JPEG compression, YCBCR or CMYK colorspace as RGB24.
    Vendor cfitsio/ricecomp.c for shared library builds on Windows (#18).
2021.5.20
    Add ZLIBNG codec via zlib-ng library.
    Add RCOMP (Rice) codec via cfitsio library.
    Fix decoding of 16-bit JPEG with jpeg_decode.
    Relax user provided output array shape requirement.
2021.4.28
    Change WebP default compression level to lossless.
    Rename jpegxl codec to brunsli (breaking).
    Add new JPEG XL codec via jpeg-xl library.
    Add PGLZ codec via PostgreSQL's pg_lzcompress.c.
    Update to libtiff 4.3 and libjpeg-turbo 2.1.
    Enable JPEG 12-bit codec in manylinux wheels.
    Drop manylinux2010 wheels.
2021.3.31
    Add numcodecs compatible codecs for use by Zarr (experimental).
    Support separate JPEG header in jpeg_decode.
    Do not decode JPEG LS and XL in jpeg_decode (breaking).
    Fix ZFP with partial header.
    Fix JPEG LS tests (#15).
    Fix LZ4F contentchecksum.
    Remove blosc Snappy tests.
    Fix docstrings.
2021.2.26
    Support X2 and X4 floating point predictors (found in DNG).
2021.1.28
    Add option to return JPEG XR fixed point pixel types as integers.
    Add LJPEG codec via liblj92 (alternative to JPEGSOF3 codec).
    Change zopfli header location.
2021.1.11
    Fix build issues (#7, #8).
    Return bytearray instead of bytes on PyPy.
    Raise TypeError if output provided is bytes (breaking).
2021.1.8
    Add float24 codec.
    Update copyrights.
2020.12.24
    Update dependencies and build scripts.
2020.12.22
    Add AVIF codec via libavif.
    Add DEFLATE/Zlib and GZIP codecs via libdeflate.
    Add LZ4F codec.
    Add high compression mode option to lz4_encode.
    Convert JPEG XR 16 and 32-bit fixed point pixel types to float32.
    Fix JPEG 2000 lossy encoding.
    Fix GIF disposal handling.
    Remove support for Python 3.6 (NEP 29).
2020.5.30
    Add LERC codec via ESRI's lerc library.
    Enable building JPEG extensions with libjpeg >= 8.
    Enable distributors to modify build settings.
2020.2.18
    Fix segfault when decoding corrupted LZW segments.
    Work around Cython raises AttributeError when using incompatible numpy.
    Raise ValueError if in-place decoding is not possible (except floatpred).
2020.1.31
    Add GIF codec via giflib.
    Add TIFF decoder via libtiff.
    Add codec_check functions.
    Fix formatting libjpeg error messages.
    Use xfail in tests.
    Load extensions on demand on Python >= 3.7.
    Add build options to skip building specific extensions.
    Split imagecodecs extension into individual extensions.
    Move shared code into shared extension.
    Rename imagecodecs_lite extension and imagecodecs C library to 'imcd'.
    Remove support for Python 2.7 and 3.5.
2019.12.31
    Fix decoding of indexed PNG with transparency.
    Last version to support Python 2.7 and 3.5.
2019.12.16
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
    Add Brotli codec.
    Add optional JPEG XL codec via Brunsli repacker.
2019.12.3
    Sync with imagecodecs-lite.
2019.11.28
    Add AEC codec via libaec.
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
    Add optional ZFP codec via zfp library.
    Add numpy NPY and NPZ codecs.
    Fix some static codechecker errors.
2019.1.1
    ...

Refer to the CHANGES file for older revisions.
