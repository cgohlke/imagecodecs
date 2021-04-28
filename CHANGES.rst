Revisions
---------
2021.4.28
    Pass 5119 tests.
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
    Add AVIF codec via libavif (WIP).
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
    Add optional JPEG LS codec via CharLS.
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
