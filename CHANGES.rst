Revisions
---------

2024.6.1

- Pass 7486 tests.
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

- Support more dtypes and axes argument in PackBits encoder.
- Fix worst case output size in PackBits encoder.
- Fix decoding AVIF created with older libavif.
- Fix decoding GIF with disposal to previous for first frame.
- Add lossless option in jpeg_encode.

2021.6.8

- Fix building with Cython 0.3a7.
- Decode TIFF with JPEG compression, YCBCR or CMYK colorspace as RGB24.
- Vendor cfitsio/ricecomp.c for shared library builds on Windows (#18).

2021.5.20

- Add ZLIBNG codec via zlib-ng library.
- Add RCOMP (Rice) codec via cfitsio library.
- Fix decoding of 16-bit JPEG with jpeg_decode.
- Relax user provided output array shape requirement.

2021.4.28

- Change WebP default compression level to lossless.
- Rename jpegxl codec to brunsli (breaking).
- Add new JPEG XL codec via jpeg-xl library.
- Add PGLZ codec via PostgreSQL's pg_lzcompress.c.
- Update to libtiff 4.3 and libjpeg-turbo 2.1.
- Enable JPEG 12-bit codec in manylinux wheels.
- Drop manylinux2010 wheels.

2021.3.31

- Add numcodecs compatible codecs for use by Zarr (experimental).
- Support separate JPEG header in jpeg_decode.
- Do not decode JPEG LS and XL in jpeg_decode (breaking).
- Fix ZFP with partial header.
- Fix JPEG LS tests (#15).
- Fix LZ4F contentchecksum.
- Remove blosc Snappy tests.
- Fix docstrings.

2021.2.26

- Support X2 and X4 floating point predictors (found in DNG).

2021.1.28

- Add option to return JPEG XR fixed point pixel types as integers.
- Add LJPEG codec via liblj92 (alternative to JPEGSOF3 codec).
- Change zopfli header location.

2021.1.11

- Fix build issues (#7, #8).
- Return bytearray instead of bytes on PyPy.
- Raise TypeError if output provided is bytes (breaking).

2021.1.8

- Add float24 codec.
- Update copyrights.

2020.12.24

- Update dependencies and build scripts.

2020.12.22

- Add AVIF codec via libavif.
- Add DEFLATE/Zlib and GZIP codecs via libdeflate.
- Add LZ4F codec.
- Add high compression mode option to lz4_encode.
- Convert JPEG XR 16 and 32-bit fixed point pixel types to float32.
- Fix JPEG 2000 lossy encoding.
- Fix GIF disposal handling.
- Remove support for Python 3.6 (NEP 29).

2020.5.30

- Add LERC codec via ESRI's lerc library.
- Enable building JPEG extensions with libjpeg >= 8.
- Enable distributors to modify build settings.

2020.2.18

- Fix segfault when decoding corrupted LZW segments.
- Work around Cython raises AttributeError when using incompatible numpy.
- Raise ValueError if in-place decoding is not possible (except floatpred).

2020.1.31

- Add GIF codec via giflib.
- Add TIFF decoder via libtiff.
- Add codec_check functions.
- Fix formatting libjpeg error messages.
- Use xfail in tests.
- Load extensions on demand on Python >= 3.7.
- Add build options to skip building specific extensions.
- Split imagecodecs extension into individual extensions.
- Move shared code into shared extension.
- Rename imagecodecs_lite extension and imagecodecs C library to 'imcd'.
- Remove support for Python 2.7 and 3.5.

2019.12.31

- Fix decoding of indexed PNG with transparency.
- Last version to support Python 2.7 and 3.5.

2019.12.16

- Add Zopfli codec.
- Add Snappy codec.
- Rename j2k codec to jpeg2k.
- Rename jxr codec to jpegxr.
- Use Debian's jxrlib.
- Support pathlib and binary streams in imread and imwrite.
- Move external C declarations to pxd files.
- Move shared code to pxi file.
- Update copyright notices.

2019.12.10

- Add version functions.
- Add Brotli codec.
- Add optional JPEG XL codec via Brunsli repacker.

2019.12.3

- Sync with imagecodecs-lite.

2019.11.28

- Add AEC codec via libaec.
- Do not require scikit-image for testing.
- Require CharLS 2.1.

2019.11.18

- Add bitshuffle codec.
- Fix formatting of unknown error numbers.
- Fix test failures with official python-lzf.

2019.11.5

- Rebuild with updated dependencies.

2019.5.22

- Add optional YCbCr chroma subsampling to JPEG encoder.
- Add default reversible mode to ZFP encoder.
- Add imread and imwrite helper functions.

2019.4.20

- Fix setup requirements.

2019.2.22

- Move codecs without 3rd-party C library dependencies to imagecodecs_lite.

2019.2.20

- Rebuild with updated dependencies.

2019.1.20

- Add more pixel formats to JPEG XR codec.
- Add JPEG XR encoder.

2019.1.14

- Add optional ZFP codec via zfp library.
- Add numpy NPY and NPZ codecs.
- Fix some static codechecker errors.

2019.1.1

- Update copyright year.
- Do not install package if Cython extension fails to build.
- Fix compiler warnings.

2018.12.16

- Reallocate LZW buffer on demand.
- Ignore integer type output arguments for codecs returning images.

2018.12.12

- Enable decoding of subsampled J2K images via conversion to RGB.
- Enable decoding of large JPEG using patched libjpeg-turbo.
- Switch to Cython 0.29, language_level=3.

2018.12.1

- Add J2K encoder (WIP).
- Use ZStd content size 1 MB if it cannot be determined.
- Use logging.warning instead of warnings.warn or print.

2018.11.8

- Decode LSB style LZW.
- Fix last byte not written by LZW decoder (bug fix).
- Permit unknown colorspaces in JPEG codecs (e.g. CFA used in TIFF).

2018.10.30

- Add JPEG 8-bit and 12-bit encoders.
- Improve color space handling in JPEG codecs.

2018.10.28

- Rename jpeg0xc3 to jpegsof3.
- Add optional JPEG LS codec via CharLS.
- Fix missing alpha values in jxr_decode.
- Fix decoding JPEG SOF3 with multiple DHTs.

2018.10.22

- Add Blosc codec via libblosc.

2018.10.21

- Builds on Ubuntu 18.04 WSL.
- Include liblzf in srcdist.
- Do not require CreateDecoderFromBytes patch to jxrlib.

2018.10.18

- Improve jpeg_decode wrapper.

2018.10.17

- Add JPEG SOF3 decoder based on jpg_0XC3.cpp.

2018.10.10

- Add PNG codec via libpng.
- Add option to specify output colorspace in JPEG decoder.
- Fix Delta codec for floating point numbers.
- Fix XOR Delta codec.

2018.9.30

- Add LZF codec via liblzf.

2018.9.22

- Add WebP codec via libwebp.

2018.8.29

- Add PackBits encoder.

2018.8.22

- Add link library version information.
- Add option to specify size of LZW buffer.
- Add JPEG 2000 decoder via OpenJPEG.
- Add XOR Delta codec.

2018.8.16

- Link to libjpeg-turbo.
- Support Python 2.7 and Visual Studio 2008.

2018.8.10

- Initial alpha release.
- Add LZW, PackBits, PackInts and FloatPred decoders from tifffile.c module.
- Add JPEG and JPEG XR decoders from czifile.pyx module.
- …
