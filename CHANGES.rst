Revisions
---------

2025.8.2

- Pass 7339 tests.
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

- Fix some codecs for use with Zarr 3, zarr_format=2 (#123).
- Fix LZ4H5 codec when block size is less than compressed size (#126).
- Fix pglz_compress is not thread-safe.
- Set __module__ attribute on public objects.
- Drop support for Python 3.9, deprecate Python 3.10.

2024.12.30

- Fix out parameter array not zeroed in some cases.
- Fix ultrahdr_encode with linear rgbaf16 input (#108).
- Fix jpegls_encode with level greater than 9 (#119).
- Fix jpeg8_encode with bitspersample and lossless=False (#116).
- Fix excessive buffer allocation in lz4h5_encode (#112).
- Fix build error with libjpeg (#111).

2024.9.22

- Use libjpeg-turbo for all Lossless JPEG bit-depths if possible (#105).
- Fix PackBits encoder fails to skip short replication blocks (#107).
- Fix JPEG2K encoder leaving trailing random bytes (#104).
- Fix encoding and decoding JPEG XL with custom bitspersample (#102).
- Improve error handling in lzf_decode (#103).
- Add Ultra HDR (JPEG_R) codec based on libultrahdr library (#108).
- Add JPEGXS codec based on libjxs library (source only).
- Add SZ3 codec based on SZ3 library.
- Deprecate Python 3.9, support Python 3.13.

2024.6.1

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