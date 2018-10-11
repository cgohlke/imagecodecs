Image transformation, compression, and decompression codecs
===========================================================

The imagecodecs package provides various block-oriented, in-memory
buffer transformation, compression, and decompression functions
for use in the tiffile, czifile, and other Python imaging modules.

Decode and/or encode functions are currently implemented for Zlib DEFLATE,
ZStandard, LZMA, BZ2, LZ4, LZW, LZF, PNG, WebP, JPEG, JPEG 12-bit, JPEG 2000,
JPEG XR, PackBits, Packed Integers, Delta, XOR Delta, Floating Point Predictor,
and Bitorder reversal.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2018.10.10

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Cython 0.28 <http://cython.org/>`_
* `zlib 1.2.11 <https://github.com/madler/zlib/>`_
* `lz4 1.8.2 <https://github.com/lz4/lz4/>`_
* `zstd 1.3.5 <https://github.com/facebook/zstd/>`_
* `bzip2 1.0.6 <http://www.bzip.org/>`_
* `xz lzma 5.2.4 <https://github.com/xz-mirror/xz/>`_
* `libpng 1.6.35 <https://github.com/glennrp/libpng/>`_
* `libwebp 1.0 <https://github.com/webmproject/libwebp/>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libjpeg-turbo 2.0 <https://libjpeg-turbo.org/>`_
* `openjpeg 2.3 <http://www.openjpeg.org/>`_
* `jxrlib 0.2.0 <https://github.com/glencoesoftware/jxrlib/>`_
  with `patch <https://www.lfd.uci.edu/~gohlke/code/
  jxrlib_CreateDecoderFromBytes.diff.html>`_
* A Python distutils compatible C compiler

Notes
-----
Imagecodecs is currently developed, built, and tested on Windows only.

The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 2.7 and 3.4 are deprecated.

Other Python packages providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `python-blosc <https://github.com/Blosc/python-blosc>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_ and
  `backports.lzma <https://github.com/peterjc/backports.lzma>`_
* `python-snappy <https://github.com/andrix/python-snappy>`_
* `python-brotli <https://github.com/google/brotli/tree/master/python>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lz4 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `python-lzf <https://github.com/teepark/python-lzf>`_
* `packbits <https://github.com/psd-tools/packbits>`_

Revisions
---------
2018.10.10
    Add PNG codecs.
    Add option to specify output colorspace in JPEG decoder.
    Fix Delta codec for floating point numbers.
    Fix XOR Delta codecs.
2018.9.30
    Add LZF codecs.
2018.9.22
    Add WebP codecs.
2018.8.29
    Pass 396 tests.
    Add PackBits encoder.
2018.8.22
    Add link library version information.
    Add option to specify size of LZW buffer.
    Add JPEG 2000 decoder.
    Add XOR Delta codec.
2018.8.16
    Link to libjpeg-turbo.
    Support Python 2.7 and Visual Studio 2008.
2018.8.10
    Initial alpha release.
    Add LZW, PackBits, PackInts and FloatPred decoders from tifffile.c module.
    Add JPEG and JXR decoders from czifile.pyx module.
