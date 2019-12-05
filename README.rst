Image transformation, compression, and decompression codecs
===========================================================

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in the
tifffile, czifile, and other scientific imaging modules.

Decode and/or encode functions are currently implemented for Zlib DEFLATE,
ZStandard (ZSTD), Blosc, LZMA, BZ2, LZ4, LZW, LZF, ZFP, AEC, NPY,
PNG, WebP, JPEG 8-bit, JPEG 12-bit, JPEG SOF3, JPEG LS, JPEG 2000, JPEG XR,
PackBits, Packed Integers, Delta, XOR Delta, Floating Point Predictor,
Bitorder reversal, and Bitshuffle.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.3

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 2.7.16, 3.5.4, 3.6.8, 3.7.5, 3.8.0 64-bit <https://www.python.org>`_
* `Numpy 1.16.5 <https://www.numpy.org>`_
* `Cython 0.29.14 <https://cython.org>`_
* `zlib 1.2.11 <https://github.com/madler/zlib>`_
* `lz4 1.9.2 <https://github.com/lz4/lz4>`_
* `zstd 1.4.4 <https://github.com/facebook/zstd>`_
* `blosc 1.17.0 <https://github.com/Blosc/c-blosc>`_
* `bzip2 1.0.8 <https://sourceware.org/bzip2>`_
* `xz liblzma 5.2.4 <https://github.com/xz-mirror/xz>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
* `libwebp 1.0.3 <https://github.com/webmproject/libwebp>`_
* `libjpeg-turbo 2.0.3 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
  (8 and 12-bit)
* `charls 2.1.0 <https://github.com/team-charls/charls>`_
* `openjpeg 2.3.1 <https://github.com/uclouvain/openjpeg>`_
* `jxrlib 0.2.1 <https://github.com/glencoesoftware/jxrlib>`_
* `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `libaec 1.0.4 <https://gitlab.dkrz.de/k202009/libaec>`_
* `lcms 2.9 <https://github.com/mm2/Little-CMS>`_

Required for testing (other versions may work):

* `tifffile 2019.7.26 <https://pypi.org/project/tifffile/>`_
* `czifile 2019.7.2 <https://pypi.org/project/czifile/>`_
* `python-blosc 1.8.1 <https://github.com/Blosc/python-blosc>`_
* `python-lz4 2.2.1 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd 1.4.4 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
* `backports.lzma 0.0.14 <https://github.com/peterjc/backports.lzma>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_

Notes
-----
Imagecodecs is currently developed, built, and tested on Windows only.

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
current Ubuntu Linux distributions, run:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy python3-pytest
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg62-turbo-dev
    libjpeg-turbo8-dev libjxr-dev liblcms2-dev libcharls-dev libaec-dev
    libtiff-dev python3-blosc``

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
* `python-snappy <https://github.com/andrix/python-snappy>`_
* `python-brotli <https://github.com/google/brotli/tree/master/python>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `packbits <https://github.com/psd-tools/packbits>`_
* `fpzip <https://github.com/seung-lab/fpzip>`_

Revisions
---------
2019.12.3
    Pass 2795 tests.
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
    Add ZFP codec via zfp library (WIP).
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
    Add JPEG LS codec via CharLS.
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
