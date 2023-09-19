Image transformation, compression, and decompression codecs
===========================================================

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in Tifffile,
Czifile, Zarr, kerchunk, and other scientific image input/output packages.

Decode and/or encode functions are implemented for Zlib (DEFLATE), GZIP,
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZ4F, LZ4HC, LZ4H5,
LZW, LZF, LZFSE, LZHAM, PGLZ (PostgreSQL LZ), RCOMP (Rice), ZFP, AEC, SZIP,
LERC, EER, NPY, BCn, DDS, PNG, APNG, GIF, TIFF, WebP, QOI, JPEG 8 and 12-bit,
Lossless JPEG (LJPEG, LJ92, JPEGLL), JPEG 2000 (JP2, J2K), JPEG LS, JPEG XL,
JPEG XR (WDP, HD Photo), MOZJPEG, AVIF, HEIF, RGBE (HDR), Jetraw, PackBits,
Packed Integers, Delta, XOR Delta, Floating Point Predictor, Bitorder reversal,
Byteshuffle, Bitshuffle, Quantize (Scale, BitGroom, BitRound, GranularBR),
Float24 (24-bit floating point), and CMS (color space transformations).
Checksum functions are implemented for crc32, adler32, fletcher32, and
Jenkins lookup3.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2023.9.18
:DOI: `10.5281/zenodo.6915978 <https://doi.org/10.5281/zenodo.6915978>`_

Quickstart
----------

Install the imagecodecs package and all dependencies from the
`Python Package Index <https://pypi.org/project/imagecodecs/>`_::

    python -m pip install -U imagecodecs[all]

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

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.4, 3.12.0rc, 64-bit
- `Numpy <https://pypi.org/project/numpy>`_ 1.25.2
- `numcodecs <https://pypi.org/project/numcodecs/>`_ 0.11.0
  (optional, for Zarr compatible codecs)

Build requirements:

- `Cython <https://github.com/cython/cython>`_ 0.29.36
- `brotli <https://github.com/google/brotli>`_ 1.1.0
- `brunsli <https://github.com/google/brunsli>`_ 0.1
- `bzip2 <https://gitlab.com/bzip2/bzip2>`_ 1.0.8
- `c-blosc <https://github.com/Blosc/c-blosc>`_ 1.21.5
- `c-blosc2 <https://github.com/Blosc/c-blosc2>`_ 2.10.3
- `charls <https://github.com/team-charls/charls>`_ 2.4.2
- `giflib <https://sourceforge.net/projects/giflib/>`_ 5.2.1
- `jetraw <https://github.com/Jetraw/Jetraw>`_ 22.02.16.1
- `jxrlib <https://github.com/cgohlke/jxrlib>`_ 1.2
- `lcms <https://github.com/mm2/Little-CMS>`_ 2.15
- `lerc <https://github.com/Esri/lerc>`_ 4.0.0
- `libaec <https://gitlab.dkrz.de/k202009/libaec>`_ 1.0.6
- `libavif <https://github.com/AOMediaCodec/libavif>`_ 1.0.1
  (`aom <https://aomedia.googlesource.com/aom>`_ 3.7.0,
  `dav1d <https://github.com/videolan/dav1d>`_ 1.2.1,
  `rav1e <https://github.com/xiph/rav1e>`_ 0.6.6,
  `svt-av1 <https://gitlab.com/AOMediaCodec/SVT-AV1>`_ 1.7.0)
- `libdeflate <https://github.com/ebiggers/libdeflate>`_ 1.19
- `libheif <https://github.com/strukturag/libheif>`_ 1.16.2
  (`libde265 <https://github.com/strukturag/libde265>`_ 1.0.12,
  `x265 <https://bitbucket.org/multicoreware/x265_git/src/master/>`_ 3.5)
- `libjpeg-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_ 3.0.0
- `libjxl <https://github.com/libjxl/libjxl>`_ 0.8.2
- `liblzma <https://git.tukaani.org/?p=xz.git>`_ 5.4.4
- `libpng <https://github.com/glennrp/libpng>`_ 1.6.40
- `libpng-apng <https://sourceforge.net/projects/libpng-apng/>`_ 1.6.40
- `libtiff <https://gitlab.com/libtiff/libtiff>`_ 4.6.0
- `libwebp <https://github.com/webmproject/libwebp>`_ 1.3.2
- `lz4 <https://github.com/lz4/lz4>`_ 1.9.4
- `lzfse <https://github.com/lzfse/lzfse/>`_ 1.0
- `lzham_codec <https://github.com/richgel999/lzham_codec/>`_ 1.0
- `mozjpeg <https://github.com/mozilla/mozjpeg>`_ 4.1.1
- `openjpeg <https://github.com/uclouvain/openjpeg>`_ 2.5.0
- `snappy <https://github.com/google/snappy>`_ 1.1.10
- `zfp <https://github.com/LLNL/zfp>`_ 1.0.0
- `zlib <https://github.com/madler/zlib>`_ 1.3
- `zlib-ng <https://github.com/zlib-ng/zlib-ng>`_ 2.1.3
- `zopfli <https://github.com/google/zopfli>`_ 1.0.3
- `zstd <https://github.com/facebook/zstd>`_ 1.5.5

Vendored requirements:

- `bcdec.h <https://github.com/iOrange/bcdec>`_ 026acf9
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

- `tifffile <https://pypi.org/project/tifffile>`_ 2023.9.18
- `czifile <https://pypi.org/project/czifile>`_ 2019.7.2
- `zarr <https://github.com/zarr-developers/zarr-python>`_ 2.16.1
- `python-blosc <https://github.com/Blosc/python-blosc>`_ 1.11.1
- `python-blosc2 <https://github.com/Blosc/python-blosc2>`_ 2.2.7
- `python-brotli <https://github.com/google/brotli/tree/master/python>`_ 1.0.9
- `python-lz4 <https://github.com/python-lz4/python-lz4>`_ 4.3.2
- `python-lzf <https://github.com/teepark/python-lzf>`_ 0.2.4
- `python-snappy <https://github.com/andrix/python-snappy>`_ 0.6.1
- `python-zstd <https://github.com/sergey-dryabzhinsky/python-zstd>`_ 1.5.5.1
- `pyliblzfse <https://github.com/ydkhatri/pyliblzfse>`_ 0.4.1
- `zopflipy <https://github.com/hattya/zopflipy>`_ 1.8

Revisions
---------

2023.9.18

- Pass 7110 tests.
- Rebuild with updated dependencies fixes CVE-2023-4863.

2023.9.4

- Map avif_encode level parameter to quality (breaking).
- Support monochrome images in avif_encode.
- Add numthreads parameter to avif_decode (fix imread of AVIF).
- Add experimental quantize filter (BitGroom, BitRound, GBR) via nc4var.c.
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

2023.3.16

- Require libjpeg-turbo 2.1.91 (3.0 beta) and c-blosc2 2.7.1.
- Add experimental type hints.
- Add SZIP codec via libaec library.
- Use Zstd streaming API to decode blocks with unknown decompressed size.
- Remove unused level, index, and numthreads parameters (breaking).
- Make AEC and BLOSC constants enums (breaking).
- Capitalize numcodecs class names (breaking).
- Remove JPEG12 codec (breaking; use JPEG8 instead).
- Encode and decode lossless and 12-bit JPEG with JPEG8 codec by default.
- Remove JPEGSOF3 fallback in JPEG codec.
- Fix slow IFD seeking with libtiff 4.5.
- Fixes for Cython 3.0.

2023.1.23

- Require libjxl 0.8.
- Change mapping of level to distance parameter in jpegxl_encode.
- Add option to specify bitspersample in jpegxl_encode.
- Add option to pass de/linearize tables to LJPEG codec.
- Fix lj92 decoder for SSSS=16 (#59).
- Prefer ljpeg over jpegsof3 codec.
- Add option to specify AVIF encoder codec.
- Support LERC with Zstd or Deflate compression.
- Squeeze chunk arrays by default in numcodecs image compression codecs.

2022.12.24

- Fix PNG codec error handling.
- Fix truncated transferfunctions in cms_profile (#57).
- Fix exceptions not raised in cdef functions not returning Python object.

2022.12.22

- Require libtiff 4.5.
- Require libavif 0.11.
- Change jpegxl_encode level parameter to resemble libjpeg quality (breaking).
- Add LZFSE codec via lzfse library.
- Add LZHAM codec via lzham library.
- Fix AttributeError in cms_profile (#52).
- Support gamma argument in cms_profile (#53).
- Raise limit of TIFF pages to 1048576.
- Use libtiff thread-safe error/warning handlers.
- Add option to specify filters and strategy in png_encode.
- Add option to specify integrity check type in lzma_encode.
- Fix DeprecationWarning with NumPy 1.24.
- Support Python 3.11 and win-arm64.

2022.9.26

- Support JPEG XL multi-channel (planar grayscale only) and multi-frame.
- Require libjxl 0.7.
- Switch to Blosc2 API and require c-blosc 2.4 (breaking).
- Return LogLuv encoded TIFF as float32.
- Add RGBE codec via rgbe.c.

2022.8.8

- Drop support for libjpeg.
- Fix encoding JPEG in RGB color space.
- Require ZFP 1.0.

2022.7.31

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
``macosx_x86_64``, ``macosx_arm64``, and ``manylinux_x86_64``.

Wheels may not be available for all platforms and all releases.

Only the ``win_amd64`` wheels include all features.

The ``tiff``, ``bcn``, ``dds``, ``eer``, ``packints``, and ``jpegsof3`` codecs
are currently decode-only.

The ``heif`` and ``jetraw`` codecs are distributed as source code only due to
license and possible patent usage issues.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
<https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

This software is based in part on the work of the Independent JPEG Group.

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
    libheif-dev``

Use the ``--lite`` build option to only build extensions without 3rd-party
dependencies. Use the ``--skip-extension`` build options to skip building
specific extensions, for example:

    ``python -m pip install imagecodecs --global-option="build_ext"
    --global-option="--skip-bitshuffle"``

The ``apng``, ``avif``, ``jetraw``, ``jpegls``, ``jpegxl``, ``lerc``,
``lz4f``, ``lzfse``, ``lzham``, ``mozjpeg``, ``zfp``, and ``zlibng``
extensions are disabled by default when building from source.

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
`SPERR <https://github.com/NCAR/SPERR>`_ (GPL),
`MAFISC
<https://wr.informatik.uni-hamburg.de/research/projects/icomex/mafisc>`_,
`B3D <https://github.com/balintbalazs/B3D>`_.

Examples
--------

Import the JPEG2K codec:

>>> from imagecodecs import (
...     jpeg2k_encode, jpeg2k_decode, jpeg2k_check, jpeg2k_version, JPEG2K
... )

Check that the JPEG2K codec is available in the imagecodecs build:

>>> JPEG2K.available
True

Print the version of the JPEG2K codec's underlying OpenJPEG library:

>>> jpeg2k_version()
'openjpeg 2.5.0'

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
...     compressor=Jpeg2k()
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
>>> with tifffile.FileSequence(jp2_read, '*.jp2') as ims:
...     with ims.aszarr() as store:
...         dask.array.from_zarr(store)
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

    $ python -m imagecodecs _test.jp2
