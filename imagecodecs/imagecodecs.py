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
#
# Bundled source files in the 3rdparty folder and third-party libraries
# listed in the imagecodecs/licenses folder are distributed under their own
# licenses, as stated in the respective files.

r"""Image transformation, compression, and decompression codecs.

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in tifffile,
liffile, czifile, zarr, and other scientific image input/output packages.

Decode and/or encode functions are implemented for
**Byte compressors**: AEC, Blosc, Blosc2, Brotli, BZ2, CZI ZSTD1 and
CHUNKED (with hilo filter), DEFLATE (zlib, isa-l, libdeflate, zlib-ng),
GZIP, LZ4, LZ4F, LZ4H5, LZF, LZFSE, LZHAM, LZMA, LZO, LZW, OpenZL, PackBits,
PGLZ (PostgreSQL LZ), Snappy, Szip, Zopfli, and ZStandard (ZSTD);
**Image formats**: APNG, AVIF, BCn, BMP, DDS, EXR, GIF, HEIF, High-throughput
JPEG 2000 (HTJ2K, JPH), JPEG (2 to 16-bit), JPEG 2000 (JP2, J2K), JPEG LS,
JPEG XR, JPEG XS, Lossless JPEG (LJPEG, LJ92, JPEGLL), JPEGLI, MOZJPEG,
PCX, PNG, QOI, RGBE (HDR), TGA (TARGA), TIFF, Ultra HDR (JPEG_R), WebP,
and WIC (Windows Imaging Component);
**Image compressors**: CCITT (RLE, T.4 and T.6), DICOM RLE, and PixarLog;
**Scientific compressors**: B2ND, EER, HCOMP, Jetraw, LERC, NPZ, Pcodec, PLIO,
RCOMP (Rice), SPERR, SZ3, and ZFP;
**Filters**: Bitorder reversal, CMS (color space transformations), Delta,
Quantize (Scale, BitGroom, BitRound, GranularBR), and XOR Delta;
**Shufflers**: Bitshuffle, Byteshuffle, Floating Point Predictor, Meshopt,
Packed Integers (TIFF, MONO p and packed), and WavPack,
**Type conversions**: Bfloat16 (brain floating point), and Float24 (24-bit
floating point); and
**Checksums**: CRC-32, Adler-32, Fletcher-32, and Jenkins lookup3.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.11.11
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

See `Programming interface`_ and `Examples`_ for using the library.

Source code and support are available on
`GitHub <https://github.com/cgohlke/imagecodecs>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.12.10, 3.13.17, 3.14.10, 3.15.2 64-bit
- `numpy <https://pypi.org/project/numpy>`_ 2.5.4
- `zarr <https://pypi.org/project/zarr/>`_ 3.4.2
  (optional, for Zarr 3 compatible codecs)
- `numcodecs <https://pypi.org/project/numcodecs/>`_ 0.17.1
  (optional, for Zarr file format 2 compatible codecs)

Build requirements:

- `Cython <https://github.com/cython/cython>`_ 3.3.1
- `brotli <https://github.com/google/brotli>`_ 1.2.1
- `bzip2 <https://gitlab.com/bzip2/bzip2>`_ 1.0.8
- `c-blosc <https://github.com/Blosc/c-blosc>`_ 1.21.7
- `c-blosc2 <https://github.com/Blosc/c-blosc2>`_ 3.6.0
- `charls <https://github.com/team-charls/charls>`_ 2.4.5
- `giflib <https://sourceforge.net/projects/giflib/>`_ 6.1.3
- `isa-l <https://github.com/intel/isa-l>`_ 2.32.1
- `jpegli <https://github.com/google/jpegli>`_ main
- `jxrlib <https://github.com/cgohlke/jxrlib>`_ 1.2
- `lcms2 <https://github.com/mm2/Little-CMS>`_ 2.20.0
- `lerc <https://github.com/Esri/lerc>`_ 4.3.0
- `libaec <https://github.com/Deutsches-Klimarechenzentrum/libaec>`_ 1.1.8
- `libavif <https://github.com/AOMediaCodec/libavif>`_ 1.4.2
  (`aom <https://aomedia.googlesource.com/aom>`_ 3.15.3,
  `dav1d <https://github.com/videolan/dav1d>`_ 1.5.4,
  `rav1e <https://github.com/xiph/rav1e>`_ 0.8.1,
  `svt-av1 <https://gitlab.com/AOMediaCodec/SVT-AV1>`_ 4.3.0,
  `libyuv <https://chromium.googlesource.com/libyuv/libyuv>`_ main,
  `libxml2 <https://gitlab.gnome.org/GNOME/libxml2>`_ 2.15.4)
- `libdeflate <https://github.com/ebiggers/libdeflate>`_ 1.26
- `libheif <https://github.com/strukturag/libheif>`_ 1.23.7
  (`libde265 <https://github.com/strukturag/libde265>`_ 1.1.3,
  `x265 <https://bitbucket.org/multicoreware/x265_git/src/master/>`_ 4.2)
- `libjpeg-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_ 3.2.1
- `libjxl <https://github.com/libjxl/libjxl>`_ 0.12.0
- `libjxs <https://jpeg.org/jpegxs/software.html>`_ 3.0.2-ED3
- `liblzma <https://github.com/tukaani-project/xz>`_ 5.8.4
- `libpng <https://github.com/glennrp/libpng>`_ 1.6.59
- `libpng-apng <https://sourceforge.net/projects/libpng-apng/>`_ 1.6.59
- `libtiff <https://gitlab.com/libtiff/libtiff>`_ 4.7.2 (with issue 864 patch)
- `libultrahdr <https://github.com/google/libultrahdr>`_ 2.0.2
- `libwebp <https://github.com/webmproject/libwebp>`_ 1.6.0
- `lz4 <https://github.com/lz4/lz4>`_ 1.11.1
- `meshoptimizer <https://github.com/zeux/meshoptimizer>`_ 1.3
- `openexr <https://github.com/AcademySoftwareFoundation/openexr>`_ 3.5.4
  (`imath <https://github.com/AcademySoftwareFoundation/Imath>`_ 3.2.3)
- `openjpeg <https://github.com/uclouvain/openjpeg>`_ 2.5.4
- `openjph <https://github.com/aous72/OpenJPH>`_ 0.33.0
- `openzl <https://github.com/facebook/openzl>`_ 0.3.0
- `pcodec <https://github.com/mwlon/pcodec>`_ 1.0.4
- `snappy <https://github.com/google/snappy>`_ 1.3.1
- `sperr <https://github.com/NCAR/SPERR>`_ 0.8.5
- `sz3 <https://github.com/szcompressor/SZ3>`_ 3.4.0
- `wavpack <https://github.com/dbry/wavpack>`_ 5.9.0
- `zfp <https://github.com/LLNL/zfp>`_ 1.0.1
- `zlib <https://github.com/madler/zlib>`_ 1.4
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

- `bcdec.h <https://github.com/iOrange/bcdec>`_ 80859ed
- `bitshuffle <https://github.com/kiyo-masui/bitshuffle>`_ 0.5.2
- bmp.c original 0BSD implementation
- ccitt.c original 0BSD implementation
- `cfitsio ricecomp.c, pliocomp.c, hcompress.c
  <https://heasarc.gsfc.nasa.gov/fitsio/>`_ modified
- `h5checksum.c <https://github.com/HDFGroup/hdf5/>`_ modified
- jpeg_tools.c original 0BSD implementation
- `jpg_0XC3.cpp
  <https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_
  modified
- `liblj92
  <https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/>`_ modified
- `liblzf <http://oldhome.schmorp.de/marc/liblzf.html>`_ 3.6
- `libspng <https://github.com/randy408/libspng>`_ 0d37ad9
- `nc4var.c <https://github.com/Unidata/netcdf-c/blob/main/libsrc4/nc4var.c>`_
  modified
- pcx.c original 0BSD implementation
- `pg_lzcompress.c <https://github.com/postgres/postgres>`_ modified
- `libtiff pixarlog.c
  <https://gitlab.com/libtiff/libtiff/-/blob/master/libtiff/tif_pixarlog.c>`_
  v4.7.1 modified
- `qoi.h <https://github.com/phoboslab/qoi/>`_ 4461cc3
- `rgbe.c <https://www.graphics.cornell.edu/~bjw/rgbe/rgbe.c>`_ modified
- tga.c original 0BSD implementation
- wic.cpp original 0BSD implementation

Test requirements:

- `tifffile <https://github.com/cgohlke/tifffile>`_ 2026.11.8
- `czifile <https://github.com/cgohlke/czifile>`_ 2026.11.8
- `liffile <https://github.com/cgohlke/liffile>`_ 2026.11.8
- `sdtfile <https://github.com/cgohlke/sdtfile>`_ 2026.11.6
- `roifile <https://github.com/cgohlke/roifile>`_ 2026.11.6
- `kerchunk <https://github.com/fsspec/kerchunk>`_ 0.2.10
- `python-blosc <https://github.com/Blosc/python-blosc>`_ 1.11.4
- `python-blosc2 <https://github.com/Blosc/python-blosc2>`_ 4.14.1
- `python-brotli <https://github.com/google/brotli/tree/master/python>`_ 1.2.0
- `python-lz4 <https://github.com/python-lz4/python-lz4>`_ 4.4.5
- `python-lzf <https://github.com/teepark/python-lzf>`_ 0.2.6
- `python-snappy <https://github.com/andrix/python-snappy>`_ 0.6.1
- `pyliblzfse <https://github.com/ydkhatri/pyliblzfse>`_ 0.4.1
- `backports.zstd <https://github.com/rogdham/backports.zstd>`_ 1.7.0
- `zopflipy <https://github.com/hattya/zopflipy>`_ 1.13

Revisions
---------

2026.11.11

- Rename JPEG8 codec to JPEG and deprecate JPEG8 alias (breaking).
- Rename codec parameter of imread and imwrite to imagecodec (breaking).
- Rename ultrahdr_encode_jpeg parameters and metadata keys (breaking).
- Move Brunsli JPEG transcoding to brunsli_encode_jpeg/decode_jpeg (breaking).
- Remove LJPEG fallback from jpeg_decode and jpeg_encode (breaking).
- Remove Python bz2 module fallback from bz2_encode and bz2_decode.
- Remove RGB565 color spaces from JPEG and MOZJPEG codecs (breaking).
- Remove planar auto-detection in jpeg2k_encode and htj2k_encode (breaking).
- Reject read-only output arrays in decoders (breaking).
- Return 1D uint8 view or byte memoryview of user-provided out (breaking).
- Reject bitspersample > 26 in jpeg2k_encode (breaking).
- Reject invalid colorspace and subsampling in JPEG encoders (breaking).
- Reject non-identity cICP matrix coefficients in PNG/APNG encoders (breaking).
- Reject non-contiguous output arrays in cms_transform (breaking).
- Reject unused whitepoint and primaries arguments in cms_profile (breaking).
- Reject unknown codec names in register_codecs.
- Reject object and zero itemsize dtypes in array codecs.
- Change imread return_codec parameter to info (breaking).
- Change integer out parameter to exact output size in all encoders (breaking).
- Change mozjpeg_decode to alias of jpeg_decode (breaking).
- Change jpegxl_encode/decode_jpeg options to keyword-only (breaking).
- Change BLOSC2/B2ND parameters and enums to match python-blosc2 (breaking).
- Change packints to pack 16, 32, and 64-bit integers big-endian (breaking).
- Change webp_encode default to lossy if level is specified (breaking).
- Change htj2k_encode resolutions to number of resolution levels (breaking).
- Change ISAL.COMPRESSION.SPEED to level 0 (breaking).
- Change numcodecs Jpegxl lossless default to None (breaking).
- Change meshopt_encode default level to 2 and use vertex codec v1.
- Change ultrahdr_encode default gainmap scale to library default.
- Change b2nd_decode to disable threading.
- Change BMP, PCX, and TGA codecs to thread-safe C libraries.
- Change imread to try APNG for PNG and WIC for camera raw files.
- Change AVIF creation times and cms_profile creation date to constant.
- Derive jpeg2k_encode default colorspace from image layout (breaking).
- Derive jpeg2k_encode color samples from colorspace parameter.
- Derive JPEG default color space from CMYK and extended RGB input.
- Convert CIELab to RGB in jpeg2k_decode (breaking).
- Add JPEGLI codec based on jpegli library (Windows only).
- Add iminfo function and info functions to many codecs.
- Add imagecodecs.typing module with type aliases.
- Add JpegCodec implementing the zarr-extensions jpeg spec.
- Add more filters and codecs to BLOSC2/B2ND (#163).
- Add jpeg_transcode and jpeg_merge functions (#160).
- Add scale parameter to jpeg_decode (#159).
- Add iccprofile parameter to JPEG, MOZJPEG, AVIF, and HEIF encoders.
- Add hasalpha parameter to HEIF and JPEGXL encoders.
- Add whitepoint and gamma parameters to jpegxl_encode.
- Add primaries, transfer, whitepoint, and gamma parameters to jpegxl_decode.
- Add planar parameter to jpegls_encode and jpegls_decode functions.
- Add pixelformat parameter to jpegxr_decode.
- Add extrasample parameter to jpeg2k_encode and write JP2 alpha channels.
- Add option to create wavpack correction data.
- Add hc parameter to lz4h5_encode.
- Add byteorder parameter to PixarLog codecs.
- Add displayp3, rec2020, sgray profiles and named curves to cms_profile.
- Add new encoder parameters to numcodecs and zarr codecs.
- Support typed contiguous output buffers in byte decompressors.
- Support negative index in AVIF, HEIF, APNG, and EXR decoders (breaking).
- Support non-native byte order input in most array codecs.
- Support non-native byte order output in PCODEC, SPERR, and SZ3 decoders.
- Support bytes-like ICC profiles in CMS functions.
- Support short and prefixed codec names in register_codecs.
- Support LERC blobs that use noData values (via lerc_decode_4D).
- Support interrupting zopfli_encode.
- Support zero delay and combined filters in APNG/PNG.
- Support more DDS DX10 formats.
- Support multi-member and padded GZIP streams and raw DEFLATE without size.
- Support uniformly subsampled components in jpeg2k_decode.
- Support 16-bit gray, 16-bit RGB, and gray-alpha images in spng_decode.
- Support 15/16-bit colormaps and right-to-left images in tga_decode.
- Support negative minexp, non-native byte order, and dtype strings in ZFP.
- Improve parameter validation, error messages, and check functions.
- Improve documentation.
- Improve type hints.
- Fix image layout of samples misinterpreted as frames.
- Fix zarr/numcodecs codecs shape handling, serialization, and Cms metadata.
- Fix photometric enum handling in HEIF, JPEGXL, APNG, and TIFF encoders.
- Fix RGBE, ULTRAHDR, QUANTIZE, and TIFF colormap with non-native byte order.
- Fix silent truncation and wrong checksums with data larger than 4 GB.
- Fix silent data truncation in LZMA, LZ4F, LZ4H5, BZ2, SZIP, and AEC codecs.
- Fix JPEGXR, JPEGXS, LJPEG encoders silently truncating output to small out.
- Fix decoding concatenated streams in LZMA and BZ2 codecs.
- Fix decoding highly compressible data in LZ4, Brotli, and AEC codecs.
- Fix crashes, out-of-bounds access, and writes to read-only buffers in IMCD.
- Fix aec_encode signedness of int8 and byte-order prefixed buffers.
- Fix apng_decode alpha compositing and apng_encode delays > 65535 ms.
- Fix (a)png_decode accepting truncated input or missing IEND chunk.
- Fix interlace handling in png_decode and apng_decode.
- Fix bitshuffle codecs ignoring itemsize of numpy arrays (breaking).
- Fix bmp_decode of bitfields not 8 bits and segfaults with zero channel mask.
- Fix CCITT decoders with 1D, ND, or buffer output.
- Fix ccittfax4_decode infinite loop on crafted input (#158).
- Fix cms_transform of gray volumes and big-endian floats, cms_info of v4.
- Fix thread-safety of error logging in cms_transform and cms_profile_validate.
- Fix decoding DDS cubemaps, texture arrays, and odd-sized textures.
- Fix eer_decode dropping last code and with invalid parameters (#162).
- Fix exr_encode default DWA and LJ2K compression levels.
- Fix float24_encode rounding overflow and directed rounding of large values.
- Fix gif_encode color resolution and gif_decode disposal and frame index.
- Fix h5checksum functions ignoring value and racing CRC table (breaking).
- Fix heif_decode of monochrome images.
- Fix isal_decode ignoring checksum errors and isal_crc32c non-standard values.
- Fix jpegsof3_decode OOB reads, uint8 wraparound, zero dimensions (#157).
- Fix ljpeg_decode and jpegsof3_decode ignoring point transform.
- Fix ljpeg_decode returning wrong samples with restart intervals.
- Fix jpegxl_decode returning uninitialized frames with invalid index.
- Fix jpegxs_decode hanging on truncated codestreams.
- Fix jpegxs_encode silently corrupting data deeper than 14 bits (breaking).
- Fix lzham_encode, lzfse_decode, and lzf_decode failing on specific data.
- Fix lzw_decode output truncation beyond 4 GB.
- Fix meshopt_decode ignoring shape and dtype if out is specified.
- Fix quanttable and progressive parameters of mozjpeg_encode.
- Fix packbits_encode quadratic runtime and silently truncated output.
- Fix pcodec_encode failing with small pagesize.
- Fix pglz_decode truncating output and reading past input (breaking).
- Fix pixarlog_decode returning uninitialized rows and failing on long strips.
- Fix plio_encode silently truncating floats.
- Fix quantize_encode corrupting non-finite values and with too large nsd.
- Fix rcomp_encode crash with nblock <= 0 and too small output estimate.
- Fix rgbe_encode/decode output size and dtype, read-only out, invalid values.
- Fix snappy_encode requiring level argument.
- Fix sperr_decode of single-slice 3D data, short headers, and ignored dtype.
- Fix sz3_decode heap overflow and uninitialized output with mismatched shape.
- Fix tiff_encode writing non-deterministic gap bytes.
- Fix tiff_decode silently accepting incomplete strip data.
- Fix wavpack_encode of unsigned and non-native byte order data (breaking).
- Fix webp_encode out-of-bounds reads with leading frame dimensions.
- Fix WIC pixel formats, frame shapes, and conversion crashes (breaking).
- Fix zfp_decode of data with full header when shape and dtype are specified.
- Fix ZSTD1 and CHUNKED codecs with empty, truncated, or plain Zstd data.
- Fix memory safety issues (buffer overflow, memory leaks, etc).
- Fix many other issues.

2026.10.10

- Reject 0-D input to array codecs (#151, breaking).
- Reject bitspersample outside dtype range (breaking).
- Reject dtype too wide for bitspersample (#153, breaking).
- Reject big-endian arrays in many codecs.
- Add skipres parameter to jpeg2k_decode (#155).
- Add bitspersample parameter to htj2k_encode (#153).
- Add bitspersample and color transformation parameter to jpegls_encode (#150).
- Add option to write SPIFF header in jpegls_encode (breaking).
- Add perceptualheuristics parameter to jpegxl_encode (#149).
- Add numthreads parameter to zstd_encode and zstd1_encode.
- Add gainmap_gamma, gainmap_level, and multichannel to ultrahdr_encode.
- Add ultrahdr_encode_jpeg and ultrahdr_decode_jpeg functions.
- Add colorinteropid parameter to exr_encode.
- Add compression level parameter to snappy_encode (require Snappy 1.3).
- Add contentsize parameter to lz4f_encode.
- Add chunked decoder to lz4f_decode, used when content size not in header.
- Add ZSTD and LJ2K compression to EXR codec (require OpenEXR 3.5).
- Add colorspace parameter to qoi_encode.
- Add codec parameter to avif_decode.
- Add numthreads parameter to heif_encode and heif_decode.
- Add rgbe_info function, and orientation field to rgbe_header_info C struct.
- Set GIF background alpha to 0 when alpha channel is present (breaking).
- Set JPEG8 lossless data precision to 16 for uint16 by default.
- Set AVIF encoder default speed to match avifenc (breaking).
- Enable writing lossy monochrome AVIF images.
- Update OPENZL codec to FrameInfo API (require OpenZL 0.3.0).
- Support more pixelformats in jpegxr_decode.
- Avoid copy of non-planar input in jpeg2k_encode.
- Avoid extra copy in CZI hilo unpacking.
- Avoid memset on CLEAR in lzw_encode.
- Improve lzw_decode using literal table.
- Improve inner loops in bmp_encode.
- Improve HTJ2K error handling.
- Use uint64 accumulators in imcd_delta, imcd_xor, and imcd_packints_decode.
- Fix issues in static Pyodide builds.

2026.8.16

- …

Refer to the CHANGES file for older revisions.

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
``dicomrle``, ``eer``, ``jpegsof3``, ``lzo``, and ``openzl`` codecs are
currently decode-only.

The ``brunsli`` codec is distributed as source code only because the
underlying library is unstable.

The ``heif``, ``jetraw``, and ``jpegxs`` codecs are distributed as source
code only due to license and possible patent usage issues.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2017-2026
<https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist>`_
is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd-party library licenses.

Bundled source files in the 3rdparty folder are distributed under their
own licenses.

This software is based in part on the work of the Independent JPEG Group.

When building against libjpeg or libjpeg_turbo < 3, set the environment
variable ``IMAGECODECS_JPEG_LEGACY=1`` to enable legacy API support.

Before building imagecodecs from source code, install required tools and
libraries. For example, on latest Ubuntu Linux distributions::

    sudo apt-get install build-essential python3-dev cython3 python3-pip \
    python3-setuptools python3-wheel python3-numpy \
    libbrotli-dev libcharls-dev libdeflate-dev liblerc-dev libjpeg-dev \
    libjxr-dev liblcms2-dev liblz4-dev liblzma-dev libopenjp2-7-dev \
    libpng-dev libtiff-dev libwebp-dev libz-dev libzstd-dev

To build and install imagecodecs from source code, run::

    python -m pip install .

Many extensions are disabled by default when building from source.

To define which extensions are built, or to modify build settings such as
library names and compiler arguments, provide a
``imagecodecs_distributor_setup.customize_build`` function, which is
imported and executed during setup.
See ``setup.py`` for pre-defined ``customize_build`` functions.

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

Random access to parts of large segments is generally not supported.

Programming interface
---------------------

Each codec is exposed through a set of module attributes, where ``CODEC``
and ``codec`` stand for the upper and lower case codec name, for example,
``JPEG2K`` and ``jpeg2k``:

- ``CODEC``: class with an ``available`` attribute indicating whether
  the codec is available in the build, and enums of codec constants.
- ``CodecError``: exception raised by the codec functions.
- ``codec_version()``: return version of the underlying library.
- ``codec_check(data)``: return whether data likely are encoded by the
  codec, or ``None`` if unknown.
- ``codec_info(data)``: return information about encoded data, such as
  shape and dtype, without decoding (not supported by all codecs).
- ``codec_encode(data, ..., out)``: return encoded data.
- ``codec_decode(data, ..., out)``: return decoded data.

The encode, decode, and info functions of codecs not available in the
build raise ``DelayedImportError``. Functions not implemented by a codec
raise ``NotImplementedError``.

Encode functions take numpy arrays or bytes-like objects and return
bytes-like encoded data. Decode functions take bytes-like encoded data
and return numpy arrays (image and array codecs) or bytes-like objects
(byte compressors and filters).

The optional ``out`` parameter of encode and decode functions controls
the output:
``None`` lets the codec allocate the output;
``bytes`` or ``bytearray`` select the type of the allocated output;
an integer specifies the output size;
a writable, contiguous buffer receives the output directly.
Functions returning numpy arrays return a new array, or ``out`` or a view
of it with the decoded shape and dtype.
Functions returning bytes-like objects return a new ``bytes`` or
``bytearray`` object, or a view of the provided ``out`` buffer:
a 1D uint8 view of a numpy array, a byte memoryview of other buffers,
or the bytearray itself if completely filled.
The length of the returned object is the number of bytes written.

Some codecs deviate from these rules, for example, filters and shufflers
may return numpy arrays from encode functions, and some functions return
tuples. Refer to the docstrings and type hints of individual functions.

Additional functions, such as checksums (``zlib_crc32``),
color management (``cms_transform``, ``cms_profile``),
JPEG transcoding (``jpeg_transcode``, ``jpeg_merge``, ``jpegli_transcode``,
``jpegxl_encode_jpeg``), codec initialization (``htj2k_init``),
and the high level ``imread``, ``imwrite``, and ``iminfo`` functions
do not follow the codec schema.

Examples
--------
Import the JPEG2K codec used in these examples:

>>> from imagecodecs import (
...     JPEG2K,
...     jpeg2k_check,
...     jpeg2k_decode,
...     jpeg2k_encode,
...     jpeg2k_info,
...     jpeg2k_version,
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

Check that the encoded bytes contain a JPEG 2000 stream:

>>> jpeg2k_check(encoded)
True

Obtain information about the encoded image:

>>> jpeg2k_info(encoded)
{..., 'shape': (256, 256, 3), 'dtype': dtype('uint8'), ...}

Decode the JP2 encoded bytes to a numpy array:

>>> decoded = jpeg2k_decode(encoded)
>>> numpy.array_equal(decoded, array)
True

Decode the JP2 encoded bytes to an existing numpy array:

>>> out = numpy.empty_like(array)
>>> _ = jpeg2k_decode(encoded, out=out)
>>> numpy.array_equal(out, array)
True

Transform the decoded image from sRGB to linear RGB:

>>> from imagecodecs import cms_transform, cms_profile
>>> transformed = cms_transform(
...     decoded, cms_profile('srgb'), cms_profile('linearrgb')
... )
>>> transformed.shape
(256, 256, 3)

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

Create a JPEG 2000 compressed Zarr format 2 (legacy) array using numcodecs:

>>> import zarr
>>> from imagecodecs.numcodecs import register_codecs, Jpeg2k
>>> register_codecs(['jpeg2k'])
>>> zarr.zeros(
...     (4, 5, 512, 512, 3),
...     chunks=(1, 1, 256, 256, 3),
...     dtype='u2',
...     compressor=Jpeg2k(bitspersample=10),
...     zarr_format=2,
... )
<Array ... shape=(4, 5, 512, 512, 3) dtype=uint16>

Create a Delta-LZW compressed Zarr format 3 array using native zarr codecs:

>>> from imagecodecs.zarr import register_codecs, Delta, Lzw
>>> register_codecs(['delta', 'lzw'])
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
...     'temp.json',
...     url='file://',
...     codec_id='imagecodecs_jpeg2k',
...     zarr_format=3,
... )
>>> register_codecs(['jpeg2k'])
>>> from kerchunk.utils import refs_as_store
>>> zarr.open(refs_as_store('temp.json'), mode='r')
<Array <FsspecStore(ReferenceFileSystem, /)> shape=(1, 256, 256, 3)...

View the image in the JP2 file from the command line::

    python -m imagecodecs _test.jp2

"""

from __future__ import annotations

__version__ = '2026.11.11'

import functools
import importlib
import os
import sys
import threading
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import mmap
    from collections.abc import Mapping, Sequence
    from types import ModuleType
    from typing import IO, Any, Literal

    from numpy.typing import ArrayLike, NDArray

    from .typing import ArrayDecoder, ArrayEncoder, CodecInfo

import numpy

# map extension module names to attribute names
# sync with __all__ in __init__.pyi
# uppercase CODEC names are expanded by _expand_modules below
_MODULES: dict[str, list[str]] = {
    '': [
        '__version__',
        'version',
        'imread',
        'iminfo',
        'imwrite',
        'imagefileext',
        'DelayedImportError',
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
    '_blosc2': ['BLOSC2', 'B2ND'],
    '_bmp': ['BMP'],
    '_brotli': ['BROTLI'],
    '_brunsli': ['BRUNSLI', 'brunsli_decode_jpeg', 'brunsli_encode_jpeg'],
    '_bz2': ['BZ2'],
    '_ccitt': ['CCITTRLE', 'CCITTFAX3', 'CCITTFAX4'],
    '_cfitsio': ['HCOMP', 'PLIO', 'RCOMP'],
    '_cms': [
        'CMS',
        'CmsError',
        'cms_check',
        'cms_info',
        'cms_version',
        'cms_transform',
        'cms_profile',
        'cms_profile_validate',
    ],
    '_czi': ['CHUNKED', 'ZSTD1'],
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
    '_heif': ['HEIF'],
    '_htj2k': ['HTJ2K', 'htj2k_init'],
    '_isal': ['ISAL', 'isal_adler32', 'isal_crc32', 'isal_crc32c'],
    '_jetraw': ['JETRAW', 'jetraw_init'],
    '_jpeg2k': ['JPEG2K'],
    '_jpeg': ['JPEG', 'jpeg_merge', 'jpeg_transcode'],
    '_jpegli': ['JPEGLI', 'jpegli_transcode'],
    '_jpegls': ['JPEGLS'],
    '_jpegsof3': ['JPEGSOF3'],
    '_jpegxl': ['JPEGXL', 'jpegxl_decode_jpeg', 'jpegxl_encode_jpeg'],
    '_jpegxr': ['JPEGXR'],
    '_jpegxs': ['JPEGXS'],
    '_lerc': ['LERC'],
    '_ljpeg': ['LJPEG'],
    '_lz4': ['LZ4', 'LZ4F', 'LZ4H5'],
    '_lzf': ['LZF'],
    '_lzfse': ['LZFSE'],
    '_lzham': ['LZHAM'],
    '_lzma': ['LZMA'],
    '_lzo': ['LZO'],
    '_meshopt': ['MESHOPT'],
    '_mozjpeg': ['MOZJPEG'],
    '_numpy': ['NONE', 'NUMPY'],
    '_openzl': ['OPENZL'],
    '_pcodec': ['PCODEC'],
    '_pcx': ['PCX'],
    '_pglz': ['PGLZ'],
    '_pixarlog': ['PIXARLOG'],
    '_png': ['PNG'],
    '_qoi': ['QOI'],
    '_quantize': ['QUANTIZE'],
    '_rgbe': ['RGBE'],
    '_snappy': ['SNAPPY'],
    '_sperr': ['SPERR'],
    '_spng': ['SPNG'],
    '_sz3': ['SZ3'],
    '_szip': ['SZIP', 'szip_params'],
    '_tga': ['TGA'],
    '_tiff': ['TIFF'],
    '_ultrahdr': ['ULTRAHDR', 'ultrahdr_decode_jpeg', 'ultrahdr_encode_jpeg'],
    '_wavpack': ['WAVPACK'],
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
    'JPEG8': 'JPEG',
    'Jpeg8Error': 'JpegError',
    'jpeg8_check': 'jpeg_check',
    'jpeg8_decode': 'jpeg_decode',
    'jpeg8_encode': 'jpeg_encode',
    'jpeg8_version': 'jpeg_version',
    'jpegli_info': 'jpeg_info',
    'mozjpeg_decode': 'jpeg_decode',
    'mozjpeg_info': 'jpeg_info',
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
                    f'{codec}_check',
                    f'{codec}_decode',
                    f'{codec}_encode',
                    f'{codec}_info',
                    f'{codec}_version',
                }
                if standard.isdisjoint(items_set):
                    attrs += [
                        item,
                        f'{item.capitalize()}Error',
                        f'{codec}_check',
                        f'{codec}_decode',
                        f'{codec}_encode',
                        f'{codec}_info',
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

# derive __all__ from _MODULES at import time,
# see __init__.pyi for a static listing used by type checkers, IDEs, and agents
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
    if name.endswith('_info'):

        def stub_info(*args: Any, **kwargs: Any) -> None:
            """Stub for imagecodecs.codec_info function."""
            raise DelayedImportError(name)

        return stub_info

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


@overload
def imread(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    /,
    imagecodec: (
        str | ArrayDecoder | Sequence[str | ArrayDecoder] | None
    ) = None,
    *,
    memmap: bool = True,
    info: Literal[False] = False,
    **kwargs: Any,
) -> NDArray[Any]: ...


@overload
def imread(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    /,
    imagecodec: (
        str | ArrayDecoder | Sequence[str | ArrayDecoder] | None
    ) = None,
    *,
    memmap: bool = True,
    info: Literal[True],
    **kwargs: Any,
) -> tuple[NDArray[Any], dict[str, Any]]: ...


@overload
def imread(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    /,
    imagecodec: (
        str | ArrayDecoder | Sequence[str | ArrayDecoder] | None
    ) = None,
    *,
    memmap: bool = True,
    info: bool = False,
    **kwargs: Any,
) -> NDArray[Any] | tuple[NDArray[Any], dict[str, Any]]: ...


def imread(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    /,
    imagecodec: (
        str | ArrayDecoder | Sequence[str | ArrayDecoder] | None
    ) = None,
    *,
    memmap: bool = True,
    info: bool = False,
    **kwargs: Any,
) -> NDArray[Any] | tuple[NDArray[Any], dict[str, Any]]:
    """Return image data from file as numpy array.

    Parameters:
        fileobj:
            File name, binary stream, memory-mapped file, or encoded bytes.
        imagecodec:
            Name of codec, decode function, or list thereof to try in order.
            By default, the codec is determined from the file name
            extension, falling back to trying all image codecs.
        memmap:
            Memory-map file instead of reading it into memory.
        info:
            Also return information about the image.
        **kwargs:
            Additional arguments passed to the decode function,
            for example, `codec` of `avif_decode`.
            `numthreads` is only passed to codecs supporting it.

    Returns:
        Decoded image, and if `info` is True, dictionary with information
        about image.

    Raises:
        ValueError: Image could not be decoded by any codec.

    """
    image, info_dict = _imread(
        fileobj, imagecodec, memmap=memmap, decode=True, info=info, **kwargs
    )
    assert image is not None
    if info:
        return image, info_dict
    return image


def iminfo(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    /,
    imagecodec: (
        str | ArrayDecoder | Sequence[str | ArrayDecoder] | None
    ) = None,
    *,
    memmap: bool = True,
) -> dict[str, Any]:
    """Return information about image in file."""
    return _imread(
        fileobj, imagecodec, memmap=memmap, decode=False, info=True
    )[1]


def _imread(
    fileobj: (
        str | os.PathLike[Any] | IO[bytes] | bytes | bytearray | mmap.mmap
    ),
    codec: str | ArrayDecoder | Sequence[str | ArrayDecoder] | None,
    /,
    *,
    memmap: bool,
    decode: bool,
    info: bool,
    **kwargs: Any,
) -> tuple[NDArray[Any] | None, dict[str, Any]]:
    """Return decoded image and/or image information from file."""
    import mmap

    # determine list of codecs based on file extension or provided codec
    codecs: list[str | ArrayDecoder] = []
    if codec is None:
        # find codec based on file extension
        ext_to_codec, codec_list = _imcodecs()
        if isinstance(fileobj, str | os.PathLike):
            ext = os.path.splitext(os.fspath(fileobj))[-1][1:].lower()
        else:
            ext = ''
        if ext in ext_to_codec:
            codec = ext_to_codec[ext]
            if codec == 'jpeg':
                codecs.extend(('jpeg', 'ljpeg'))  # 'jpegsof3'
            elif codec == 'png' and imagecodecs.APNG.available:
                codecs.append('apng')
            else:
                codecs.append(codec)
        # also try other imaging codecs
        codecs.extend(c for c in codec_list if c not in codecs)
    else:
        # use provided codecs
        if isinstance(codec, str) or callable(codec):
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                cl = c.lower()
                codecs.append(_imcodecs()[0].get(cl, cl))
            else:
                codecs.append(c)
    del codec

    # read data from file object
    func: ArrayDecoder | None = None
    info_func: CodecInfo | None = None
    info_dict: dict[str, Any] = {}
    exceptions: list[str] = []
    image: NDArray[Any] | None = None
    data: bytes | bytearray | mmap.mmap
    offset: int = -1
    close = False

    try:
        # TODO: support urllib.request.urlopen ?
        if isinstance(fileobj, str | os.PathLike):
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
        elif isinstance(fileobj, bytes | bytearray):
            data = fileobj
        else:
            # binary stream: open file, BytesIO
            data = fileobj.read()
            assert isinstance(data, bytes | bytearray)

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
                if decode:
                    if _numthreads is None:
                        image = func(data, **kwargs)
                    else:
                        image = func(data, numthreads=_numthreads, **kwargs)
                    assert isinstance(image, numpy.ndarray)
                    if image.dtype.kind == 'O':
                        msg = 'failed'
                        raise ValueError(msg)  # noqa: TRY301

                info_dict = {}
                if info:
                    name = func.__name__
                    info_func = None
                    if name.endswith('_decode'):
                        info_func = getattr(
                            imagecodecs, name[:-7] + '_info', None
                        )
                    try:
                        if info_func is None:
                            msg = 'no info function'
                            raise ValueError(msg)  # noqa: TRY301
                        info_dict = info_func(data)
                    except Exception:
                        # info-only mode must fail over to next codec
                        if not decode:
                            raise
                        assert image is not None
                        info_dict = {
                            '_codec': name[:-7],
                            'shape': image.shape,
                            'dtype': image.dtype,
                        }
                break
            except DelayedImportError:
                pass
            except Exception as exc:
                # raise  # uncomment for debugging
                exceptions.append(f'{func.__name__.upper()}: {exc}')
            if offset >= 0:
                assert isinstance(data, mmap.mmap)
                data.seek(offset)
        else:
            raise ValueError('\n'.join(exceptions))

    finally:
        if close:
            assert isinstance(data, mmap.mmap)
            data.close()

    return image, info_dict


def imwrite(
    fileobj: str | os.PathLike[Any] | IO[bytes],
    data: ArrayLike,
    /,
    imagecodec: str | ArrayEncoder | None = None,
    **kwargs: Any,
) -> None:
    """Write numpy array to image file.

    Parameters:
        fileobj:
            File name or writable binary stream.
        data:
            Image data to encode.
        imagecodec:
            Name of codec or encode function.
            By default, the codec is determined from the file name extension.
            Required if `fileobj` is a binary stream.
        **kwargs:
            Additional arguments passed to the encode function,
            for example, `codec` of `avif_encode`.

    Raises:
        ValueError: Codec is not specified or invalid.
        TypeError: Codec is not callable.

    """
    if imagecodec is None:
        # find codec based on file extension
        if isinstance(fileobj, str | os.PathLike):
            ext = os.path.splitext(os.fspath(fileobj))[-1].lower()[1:]
        else:
            msg = 'no imagecodec specified'
            raise ValueError(msg)

        imagecodec = _imcodecs()[0].get(ext, ext)
        try:
            imagecodec = getattr(imagecodecs, imagecodec + '_encode')
        except AttributeError as exc:
            msg = f'invalid {imagecodec=!r}'
            raise ValueError(msg) from exc

    elif isinstance(imagecodec, str):
        imagecodec = imagecodec.lower()
        imagecodec = _imcodecs()[0].get(imagecodec, imagecodec)
        try:
            imagecodec = getattr(imagecodecs, imagecodec + '_encode')
        except AttributeError as exc:
            msg = f'invalid {imagecodec=!r}'
            raise ValueError(msg) from exc

    if not callable(imagecodec):
        msg = f'invalid {imagecodec=!r}'
        raise TypeError(msg)

    image: bytes | bytearray | memoryview = imagecodec(data, **kwargs)
    if hasattr(fileobj, 'write'):
        # binary stream: open file, BytesIO
        fileobj.write(image)
    else:
        # file name
        with open(fileobj, 'wb') as fh:
            fh.write(image)


@functools.cache
def _imcodecs() -> tuple[Mapping[str, str], tuple[str, ...]]:
    """Return map of image file extensions to codec names and codec names."""
    codecs = {
        'apng': ('apng',),
        'avif': ('avif', 'avifs'),
        'b2nd': ('b2nd',),
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
        'jpeg': ('jpg', 'jpeg', 'jpe', 'jfif', 'jfi', 'jif'),
        'jpeg2k': (
            'j2k',
            'jp2',
            'j2c',
            'jpc',
            'jpf',
            'jpg2',
            # 'jpx',
            # 'jpm',
            # 'mj2',
            # 'jph',  # HTJ2K with JP2 boxes
            # 'jhc',  # HTJ2K codestream
        ),
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
        'wic': (
            # bmp, heif, jpg, jxr, png, tiff, webp, wpd
            'ico',
            # digital camera raw formats
            '3fr',
            'arw',
            'cr2',
            'cr3',
            'crw',
            'dcr',
            'dng',
            'nef',
            'orf',
            'pef',
            'raf',
            'rwl',
            'rw2',
            'sr2',
            'srf',
            'srw',
        ),
        'zfp': ('zfp',),
    }
    return (
        {ext: codec for codec, exts in codecs.items() for ext in exts},
        tuple(codecs.keys()),
    )


def imagefileext() -> tuple[str, ...]:
    """Return image file extensions handled by imread and imwrite."""
    return tuple(_imcodecs()[0].keys())


imagecodecs = sys.modules['imagecodecs']

_set_module()
