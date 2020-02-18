Image transformation, compression, and decompression codecs
===========================================================

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in the
tifffile, czifile, and other scientific imaging modules.

Decode and/or encode functions are implemented for Zlib (DEFLATE),
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZW, LZF, ZFP, AEC,
NPY, PNG, GIF, TIFF, WebP, JPEG 8-bit, JPEG 12-bit, JPEG SOF3, JPEG 2000,
JPEG LS, JPEG XR, JPEG XL, PackBits, Packed Integers, Delta, XOR Delta,
Floating Point Predictor, Bitorder reversal, and Bitshuffle.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.2.18

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.6.8, 3.7.6, 3.8.1 64-bit <https://www.python.org>`_
* `Numpy 1.16.6 <https://www.numpy.org>`_
* `Cython 0.29.15 <https://cython.org>`_
* `zlib 1.2.11 <https://github.com/madler/zlib>`_
* `lz4 1.9.2 <https://github.com/lz4/lz4>`_
* `zstd 1.4.4 <https://github.com/facebook/zstd>`_
* `blosc 1.17.1 <https://github.com/Blosc/c-blosc>`_
* `bzip2 1.0.8 <https://sourceware.org/bzip2>`_
* `liblzma 5.2.4 <https://github.com/xz-mirror/xz>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
* `libwebp 1.0.3 <https://github.com/webmproject/libwebp>`_
* `libtiff 4.1.0 <https://gitlab.com/libtiff/libtiff>`_
* `libjpeg-turbo 2.0.4 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
  (8 and 12-bit)
* `charls 2.1.0 <https://github.com/team-charls/charls>`_
* `openjpeg 2.3.1 <https://github.com/uclouvain/openjpeg>`_
* `jxrlib 1.1 <https://packages.debian.org/source/sid/jxrlib>`_
* `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `libaec 1.0.4 <https://gitlab.dkrz.de/k202009/libaec>`_
* `snappy 1.1.8 <https://github.com/google/snappy>`_
* `zopfli-1.0.3 <https://github.com/google/zopfli>`_
* `brotli 1.0.7 <https://github.com/google/brotli>`_
* `brunsli 0.1 <https://github.com/google/brunsli>`_
* `giflib 5.2.1 <http://giflib.sourceforge.net/>`_
* `lcms 2.9 <https://github.com/mm2/Little-CMS>`_

Required Python packages for testing (other versions may work):

* `tifffile 2020.2.16 <https://pypi.org/project/tifffile/>`_
* `czifile 2019.7.2 <https://pypi.org/project/czifile/>`_
* `python-blosc 1.8.3 <https://github.com/Blosc/python-blosc>`_
* `python-lz4 3.0.2 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd 1.4.4 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
* `python-brotli 1.0.7 <https://github.com/google/brotli/tree/master/python>`_
* `python-snappy 0.5.4 <https://github.com/andrix/python-snappy>`_
* `zopflipy 1.3 <https://github.com/hattya/zopflipy>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_

Notes
-----
The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 32-bit versions are deprecated. Python 2.7 and 3.5 are no longer
supported.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017
and 2019 <https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd party library licenses.

This software is based in part on the work of the Independent JPEG Group.

This software includes modified versions of `dcm2niix's jpg_0XC3.cpp
<https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_
and `OpenJPEG's color.c
<https://github.com/uclouvain/openjpeg/blob/master/src/bin/common/color.c>`_.

Build instructions and wheels for manylinux and macOS courtesy of
`Grzegorz Bokota <https://github.com/Czaki/imagecodecs>`_.

To install the requirements for building imagecodecs from source code on
latest Ubuntu Linux distributions, run:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy
    python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-turbo8-dev libjxr-dev
    liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev
    libzopfli-dev libgif-dev libtiff-dev``

Use the ``--skip-extension`` build options to skip building specific
extensions. Use the ``--lite`` build option to only build extensions without
3rd-party dependencies. Edit ``setup.py`` to modify other build options.

Other Python packages and C libraries providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_
* `backports.lzma <https://github.com/peterjc/backports.lzma>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `packbits <https://github.com/psd-tools/packbits>`_
* `fpzip <https://github.com/seung-lab/fpzip>`_
* `libmng <https://sourceforge.net/projects/libmng/>`_
* `APNG patch for libpng <https://sourceforge.net/projects/libpng-apng/>`_
* `OpenEXR <https://github.com/AcademySoftwareFoundation/openexr>`_
* `LERC <https://github.com/Esri/lerc>`_

Revisions
---------
2020.2.18
    Pass 3469 tests.
    Fix segfault when decoding corrupted LZW segments.
    Work around Cython raises AttributeError when using incompatible numpy.
    Raise ValueError if in-place decoding is not possible (except floatpred).
2020.1.31
    Add GIF codec via giflib.
    Add TIFF decoder via libtiff (WIP).
    Add codec_check functions (WIP).
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
    ...

Refer to the CHANGES file for older revisions.
