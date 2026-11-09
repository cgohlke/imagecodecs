# imagecodecs/__init__.pyi

# Copyright (c) 2023-2026, Christoph Gohlke
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

# Public interface for the imagecodecs package.

import enum
from collections.abc import Buffer, Sequence
from typing import Any, Literal, NoReturn, overload

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .imagecodecs import DelayedImportError as DelayedImportError
from .imagecodecs import imagefileext as imagefileext
from .imagecodecs import iminfo as iminfo
from .imagecodecs import imread as imread
from .imagecodecs import imwrite as imwrite
from .typing import ByteOrder, BytesOut, BytesResult, EnumLike

__all__ = [
    'AEC',
    'APNG',
    'AVIF',
    'B2ND',
    'BCN',
    'BFLOAT16',
    'BITORDER',
    'BITSHUFFLE',
    'BLOSC',
    'BLOSC2',
    'BMP',
    'BROTLI',
    'BRUNSLI',
    'BYTESHUFFLE',
    'BZ2',
    'CCITTFAX3',
    'CCITTFAX4',
    'CCITTRLE',
    'CHUNKED',
    'CMS',
    'DDS',
    'DEFLATE',
    'DELTA',
    'DICOMRLE',
    'EER',
    'EXR',
    'FLOAT24',
    'FLOATPRED',
    'GIF',
    'GZIP',
    'H5CHECKSUM',
    'HCOMP',
    'HEIF',
    'HTJ2K',
    'ISAL',
    'JETRAW',
    'JPEG',
    'JPEG2K',
    'JPEGLI',
    'JPEGLS',
    'JPEGSOF3',
    'JPEGXL',
    'JPEGXR',
    'JPEGXS',
    'LERC',
    'LJPEG',
    'LZ4',
    'LZ4F',
    'LZ4H5',
    'LZF',
    'LZFSE',
    'LZHAM',
    'LZMA',
    'LZO',
    'LZW',
    'MESHOPT',
    'MOZJPEG',
    'NONE',
    'NUMPY',
    'OPENZL',
    'PACKBITS',
    'PACKINTS',
    'PCODEC',
    'PCX',
    'PGLZ',
    'PIXARLOG',
    'PLIO',
    'PNG',
    'QOI',
    'QUANTIZE',
    'RCOMP',
    'RGBE',
    'SNAPPY',
    'SPERR',
    'SPNG',
    'SZ3',
    'SZIP',
    'TGA',
    'TIFF',
    'ULTRAHDR',
    'WAVPACK',
    'WEBP',
    'WIC',
    'XOR',
    'ZFP',
    'ZLIB',
    'ZLIBNG',
    'ZOPFLI',
    'ZSTD',
    'ZSTD1',
    'AecError',
    'ApngError',
    'AvifError',
    'B2ndError',
    'BcnError',
    'Bfloat16Error',
    'BitorderError',
    'BitshuffleError',
    'Blosc2Error',
    'BloscError',
    'BmpError',
    'BrotliError',
    'BrunsliError',
    'ByteshuffleError',
    'Bz2Error',
    'Ccittfax3Error',
    'Ccittfax4Error',
    'CcittrleError',
    'ChunkedError',
    'CmsError',
    'DdsError',
    'DeflateError',
    'DelayedImportError',
    'DeltaError',
    'DicomrleError',
    'EerError',
    'ExrError',
    'Float24Error',
    'FloatpredError',
    'GifError',
    'GzipError',
    'HcompError',
    'HeifError',
    'Htj2kError',
    'IsalError',
    'JetrawError',
    'Jpeg2kError',
    'JpegError',
    'JpegliError',
    'JpeglsError',
    'Jpegsof3Error',
    'JpegxlError',
    'JpegxrError',
    'JpegxsError',
    'LercError',
    'LjpegError',
    'Lz4Error',
    'Lz4fError',
    'Lz4h5Error',
    'LzfError',
    'LzfseError',
    'LzhamError',
    'LzmaError',
    'LzoError',
    'LzwError',
    'MeshoptError',
    'MozjpegError',
    'NoneError',
    'NumpyError',
    'OpenzlError',
    'PackbitsError',
    'PackintsError',
    'PcodecError',
    'PcxError',
    'PglzError',
    'PixarlogError',
    'PlioError',
    'PngError',
    'QoiError',
    'QuantizeError',
    'RcompError',
    'RgbeError',
    'SnappyError',
    'SperrError',
    'SpngError',
    'Sz3Error',
    'SzipError',
    'TgaError',
    'TiffError',
    'UltrahdrError',
    'WavpackError',
    'WebpError',
    'WicError',
    'XorError',
    'ZfpError',
    'ZlibError',
    'ZlibngError',
    'ZopfliError',
    'Zstd1Error',
    'ZstdError',
    '__version__',
    'aec_check',
    'aec_decode',
    'aec_encode',
    'aec_info',
    'aec_version',
    'apng_check',
    'apng_decode',
    'apng_encode',
    'apng_info',
    'apng_version',
    'avif_check',
    'avif_decode',
    'avif_encode',
    'avif_info',
    'avif_version',
    'b2nd_check',
    'b2nd_decode',
    'b2nd_encode',
    'b2nd_info',
    'b2nd_version',
    'bcn_check',
    'bcn_decode',
    'bcn_encode',
    'bcn_info',
    'bcn_version',
    'bfloat16_check',
    'bfloat16_decode',
    'bfloat16_encode',
    'bfloat16_info',
    'bfloat16_version',
    'bitorder_check',
    'bitorder_decode',
    'bitorder_encode',
    'bitorder_info',
    'bitorder_version',
    'bitshuffle_check',
    'bitshuffle_decode',
    'bitshuffle_encode',
    'bitshuffle_info',
    'bitshuffle_version',
    'blosc2_check',
    'blosc2_decode',
    'blosc2_encode',
    'blosc2_info',
    'blosc2_version',
    'blosc_check',
    'blosc_decode',
    'blosc_encode',
    'blosc_info',
    'blosc_version',
    'bmp_check',
    'bmp_decode',
    'bmp_encode',
    'bmp_info',
    'bmp_version',
    'brotli_check',
    'brotli_decode',
    'brotli_encode',
    'brotli_info',
    'brotli_version',
    'brunsli_check',
    'brunsli_decode',
    'brunsli_decode_jpeg',
    'brunsli_encode',
    'brunsli_encode_jpeg',
    'brunsli_info',
    'brunsli_version',
    'byteshuffle_check',
    'byteshuffle_decode',
    'byteshuffle_encode',
    'byteshuffle_info',
    'byteshuffle_version',
    'bz2_check',
    'bz2_decode',
    'bz2_encode',
    'bz2_info',
    'bz2_version',
    'ccittfax3_check',
    'ccittfax3_decode',
    'ccittfax3_encode',
    'ccittfax3_info',
    'ccittfax3_version',
    'ccittfax4_check',
    'ccittfax4_decode',
    'ccittfax4_encode',
    'ccittfax4_info',
    'ccittfax4_version',
    'ccittrle_check',
    'ccittrle_decode',
    'ccittrle_encode',
    'ccittrle_info',
    'ccittrle_version',
    'chunked_check',
    'chunked_decode',
    'chunked_encode',
    'chunked_info',
    'chunked_version',
    'cms_check',
    'cms_info',
    'cms_profile',
    'cms_profile_validate',
    'cms_transform',
    'cms_version',
    'cython_version',
    'dds_check',
    'dds_decode',
    'dds_encode',
    'dds_info',
    'dds_version',
    'deflate_adler32',
    'deflate_check',
    'deflate_crc32',
    'deflate_decode',
    'deflate_encode',
    'deflate_info',
    'deflate_version',
    'delta_check',
    'delta_decode',
    'delta_encode',
    'delta_info',
    'delta_version',
    'dicomrle_check',
    'dicomrle_decode',
    'dicomrle_encode',
    'dicomrle_info',
    'dicomrle_version',
    'eer_check',
    'eer_decode',
    'eer_encode',
    'eer_info',
    'eer_version',
    'exr_check',
    'exr_decode',
    'exr_encode',
    'exr_info',
    'exr_version',
    'float24_check',
    'float24_decode',
    'float24_encode',
    'float24_info',
    'float24_version',
    'floatpred_check',
    'floatpred_decode',
    'floatpred_encode',
    'floatpred_info',
    'floatpred_version',
    'gif_check',
    'gif_decode',
    'gif_encode',
    'gif_info',
    'gif_version',
    'gzip_check',
    'gzip_decode',
    'gzip_encode',
    'gzip_info',
    'gzip_version',
    'h5checksum_crc',
    'h5checksum_fletcher32',
    'h5checksum_hash_string',
    'h5checksum_lookup3',
    'h5checksum_metadata',
    'h5checksum_version',
    'hcomp_check',
    'hcomp_decode',
    'hcomp_encode',
    'hcomp_info',
    'hcomp_version',
    'heif_check',
    'heif_decode',
    'heif_encode',
    'heif_info',
    'heif_version',
    'htj2k_check',
    'htj2k_decode',
    'htj2k_encode',
    'htj2k_info',
    'htj2k_init',
    'htj2k_version',
    'imagefileext',
    'imcd_version',
    'iminfo',
    'imread',
    'imwrite',
    'isal_adler32',
    'isal_check',
    'isal_crc32',
    'isal_crc32c',
    'isal_decode',
    'isal_encode',
    'isal_info',
    'isal_version',
    'jetraw_check',
    'jetraw_decode',
    'jetraw_encode',
    'jetraw_info',
    'jetraw_init',
    'jetraw_version',
    'jpeg2k_check',
    'jpeg2k_decode',
    'jpeg2k_encode',
    'jpeg2k_info',
    'jpeg2k_version',
    'jpeg_check',
    'jpeg_decode',
    'jpeg_encode',
    'jpeg_info',
    'jpeg_merge',
    'jpeg_transcode',
    'jpeg_version',
    'jpegli_check',
    'jpegli_decode',
    'jpegli_encode',
    'jpegli_info',
    'jpegli_transcode',
    'jpegli_version',
    'jpegls_check',
    'jpegls_decode',
    'jpegls_encode',
    'jpegls_info',
    'jpegls_version',
    'jpegsof3_check',
    'jpegsof3_decode',
    'jpegsof3_encode',
    'jpegsof3_info',
    'jpegsof3_version',
    'jpegxl_check',
    'jpegxl_decode',
    'jpegxl_decode_jpeg',
    'jpegxl_encode',
    'jpegxl_encode_jpeg',
    'jpegxl_info',
    'jpegxl_version',
    'jpegxr_check',
    'jpegxr_decode',
    'jpegxr_encode',
    'jpegxr_info',
    'jpegxr_version',
    'jpegxs_check',
    'jpegxs_decode',
    'jpegxs_encode',
    'jpegxs_info',
    'jpegxs_version',
    'lerc_check',
    'lerc_decode',
    'lerc_encode',
    'lerc_info',
    'lerc_version',
    'ljpeg_check',
    'ljpeg_decode',
    'ljpeg_encode',
    'ljpeg_info',
    'ljpeg_version',
    'lz4_check',
    'lz4_decode',
    'lz4_encode',
    'lz4_info',
    'lz4_version',
    'lz4f_check',
    'lz4f_decode',
    'lz4f_encode',
    'lz4f_info',
    'lz4f_version',
    'lz4h5_check',
    'lz4h5_decode',
    'lz4h5_encode',
    'lz4h5_info',
    'lz4h5_version',
    'lzf_check',
    'lzf_decode',
    'lzf_encode',
    'lzf_info',
    'lzf_version',
    'lzfse_check',
    'lzfse_decode',
    'lzfse_encode',
    'lzfse_info',
    'lzfse_version',
    'lzham_check',
    'lzham_decode',
    'lzham_encode',
    'lzham_info',
    'lzham_version',
    'lzma_check',
    'lzma_decode',
    'lzma_encode',
    'lzma_info',
    'lzma_version',
    'lzo_check',
    'lzo_decode',
    'lzo_encode',
    'lzo_info',
    'lzo_version',
    'lzw_check',
    'lzw_decode',
    'lzw_encode',
    'lzw_info',
    'lzw_version',
    'meshopt_check',
    'meshopt_decode',
    'meshopt_encode',
    'meshopt_info',
    'meshopt_version',
    'mozjpeg_check',
    'mozjpeg_decode',
    'mozjpeg_encode',
    'mozjpeg_info',
    'mozjpeg_version',
    'none_check',
    'none_decode',
    'none_encode',
    'none_info',
    'none_version',
    'numpy_abi_version',
    'numpy_check',
    'numpy_decode',
    'numpy_encode',
    'numpy_info',
    'numpy_version',
    'openzl_check',
    'openzl_decode',
    'openzl_encode',
    'openzl_info',
    'openzl_version',
    'packbits_check',
    'packbits_decode',
    'packbits_encode',
    'packbits_info',
    'packbits_version',
    'packints_check',
    'packints_decode',
    'packints_encode',
    'packints_info',
    'packints_version',
    'pcodec_check',
    'pcodec_decode',
    'pcodec_encode',
    'pcodec_info',
    'pcodec_version',
    'pcx_check',
    'pcx_decode',
    'pcx_encode',
    'pcx_info',
    'pcx_version',
    'pglz_check',
    'pglz_decode',
    'pglz_encode',
    'pglz_info',
    'pglz_version',
    'pixarlog_check',
    'pixarlog_decode',
    'pixarlog_encode',
    'pixarlog_info',
    'pixarlog_version',
    'plio_check',
    'plio_decode',
    'plio_encode',
    'plio_info',
    'plio_version',
    'png_check',
    'png_decode',
    'png_encode',
    'png_info',
    'png_version',
    'qoi_check',
    'qoi_decode',
    'qoi_encode',
    'qoi_info',
    'qoi_version',
    'quantize_check',
    'quantize_decode',
    'quantize_encode',
    'quantize_info',
    'quantize_version',
    'rcomp_check',
    'rcomp_decode',
    'rcomp_encode',
    'rcomp_info',
    'rcomp_version',
    'rgbe_check',
    'rgbe_decode',
    'rgbe_encode',
    'rgbe_info',
    'rgbe_version',
    'snappy_check',
    'snappy_decode',
    'snappy_encode',
    'snappy_info',
    'snappy_version',
    'sperr_check',
    'sperr_decode',
    'sperr_encode',
    'sperr_info',
    'sperr_version',
    'spng_check',
    'spng_decode',
    'spng_encode',
    'spng_info',
    'spng_version',
    'sz3_check',
    'sz3_decode',
    'sz3_encode',
    'sz3_info',
    'sz3_version',
    'szip_check',
    'szip_decode',
    'szip_encode',
    'szip_info',
    'szip_params',
    'szip_version',
    'tga_check',
    'tga_decode',
    'tga_encode',
    'tga_info',
    'tga_version',
    'tiff_check',
    'tiff_decode',
    'tiff_encode',
    'tiff_info',
    'tiff_version',
    'ultrahdr_check',
    'ultrahdr_decode',
    'ultrahdr_decode_jpeg',
    'ultrahdr_encode',
    'ultrahdr_encode_jpeg',
    'ultrahdr_info',
    'ultrahdr_version',
    'version',
    'wavpack_check',
    'wavpack_decode',
    'wavpack_encode',
    'wavpack_info',
    'wavpack_version',
    'webp_check',
    'webp_decode',
    'webp_encode',
    'webp_info',
    'webp_version',
    'wic_check',
    'wic_decode',
    'wic_encode',
    'wic_info',
    'wic_version',
    'xor_check',
    'xor_decode',
    'xor_encode',
    'xor_info',
    'xor_version',
    'zfp_check',
    'zfp_decode',
    'zfp_encode',
    'zfp_info',
    'zfp_version',
    'zlib_adler32',
    'zlib_check',
    'zlib_crc32',
    'zlib_decode',
    'zlib_encode',
    'zlib_info',
    'zlib_version',
    'zlibng_adler32',
    'zlibng_check',
    'zlibng_crc32',
    'zlibng_decode',
    'zlibng_encode',
    'zlibng_info',
    'zlibng_version',
    'zopfli_check',
    'zopfli_decode',
    'zopfli_encode',
    'zopfli_info',
    'zopfli_version',
    'zstd1_check',
    'zstd1_decode',
    'zstd1_encode',
    'zstd1_info',
    'zstd1_version',
    'zstd_check',
    'zstd_decode',
    'zstd_encode',
    'zstd_info',
    'zstd_version',
]
__version__: str

def __dir__() -> list[str]: ...
def __getattr__(name: str, /) -> Any: ...
@overload
def version(astype: None = None) -> str: ...
@overload
def version(astype: type[str]) -> str: ...
@overload
def version(
    astype: type[tuple],  # type: ignore[type-arg]
) -> tuple[str, ...]: ...
@overload
def version(
    astype: type[dict],  # type: ignore[type-arg]
) -> dict[str, str]: ...
def cython_version() -> str: ...
def numpy_abi_version() -> str: ...
def imcd_version() -> str: ...
def ccitt_version() -> str: ...

class ImcdError(RuntimeError): ...
class CcittError(RuntimeError): ...

class AEC:
    available: bool

    class FLAG(enum.IntEnum):

        DATA_SIGNED = ...
        DATA_3BYTE = ...
        DATA_PREPROCESS = ...
        RESTRICTED = ...
        PAD_RSI = ...
        NOT_ENFORCE = ...

class AecError(RuntimeError): ...

def aec_version() -> str: ...
def aec_check(data: Buffer, /) -> bool | None: ...
def aec_encode(
    data: Buffer | ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def aec_decode(
    data: Buffer,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def aec_decode(
    data: Buffer,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def aec_info(data: Buffer, /) -> NoReturn: ...

class APNG:

    available: bool

    class COLOR_TYPE(enum.IntEnum):

        GRAY = ...
        GRAY_ALPHA = ...
        RGB = ...
        RGB_ALPHA = ...
        PALETTE = ...

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...

    class STRATEGY(enum.IntEnum):

        DEFAULT = ...
        FILTERED = ...
        HUFFMAN_ONLY = ...
        RLE = ...
        FIXED = ...

    class FILTER(enum.IntFlag):

        NO = ...
        NONE = ...
        SUB = ...
        UP = ...
        AVG = ...
        PAETH = ...
        FAST = ...
        ALL = ...

    class COLOR_PRIMARIES(enum.IntEnum):

        BT709 = ...
        SRGB = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        GENERIC_FILM = ...
        BT2020 = ...
        BT2100 = ...
        XYZ = ...
        SMPTE431 = ...
        SMPTE432 = ...
        DCI_P3 = ...
        EBU3213 = ...

    class TRANSFER_CHARACTERISTICS(enum.IntEnum):

        BT709 = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        LINEAR = ...
        LOG100 = ...
        LOG100_SQRT10 = ...
        IEC61966 = ...
        BT1361 = ...
        SRGB = ...
        BT2020_10BIT = ...
        BT2020_12BIT = ...
        PQ = ...
        SMPTE428 = ...
        HLG = ...

    class MATRIX_COEFFICIENTS(enum.IntEnum):

        IDENTITY = ...
        BT709 = ...
        UNSPECIFIED = ...
        FCC = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        YCGCO = ...
        BT2020_NCL = ...
        BT2020_CL = ...
        SMPTE2085 = ...
        CHROMA_DERIVED_NCL = ...
        CHROMA_DERIVED_CL = ...
        ICTCP = ...
        YCGCO_RE = ...
        YCGCO_RO = ...

class ApngError(RuntimeError): ...

def apng_version() -> str: ...
def apng_check(data: Buffer, /) -> bool | None: ...
def apng_encode(
    data: ArrayLike,
    /,
    level: APNG.COMPRESSION | int | None = None,
    *,
    strategy: EnumLike[APNG.STRATEGY] | None = None,
    filter: EnumLike[APNG.FILTER] | None = None,
    photometric: EnumLike[APNG.COLOR_TYPE] | None = None,
    delay: int | None = None,
    primaries: EnumLike[APNG.COLOR_PRIMARIES] | None = None,
    transfer: EnumLike[APNG.TRANSFER_CHARACTERISTICS] | None = None,
    matrix: EnumLike[APNG.MATRIX_COEFFICIENTS] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def apng_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def apng_info(data: Buffer, /, index: int | None = 0) -> dict[str, Any]: ...

class AVIF:

    available: bool

    class PIXEL_FORMAT(enum.IntEnum):

        NONE = ...
        YUV444 = ...
        YUV422 = ...
        YUV420 = ...
        YUV400 = ...
        COUNT = ...

    class QUALITY(enum.IntEnum):

        DEFAULT = ...
        LOSSLESS = ...
        WORST = ...
        BEST = ...

    class SPEED(enum.IntEnum):

        DEFAULT = ...
        SLOWEST = ...
        FASTEST = ...

    class CHROMA_UPSAMPLING(enum.IntEnum):

        AUTOMATIC = ...
        FASTEST = ...
        BEST_QUALITY = ...
        NEAREST = ...
        BILINEAR = ...

    class CODEC_CHOICE(enum.IntEnum):

        AUTO = ...
        AOM = ...
        DAV1D = ...
        LIBGAV1 = ...
        RAV1E = ...
        SVT = ...
        AVM = ...

    class COLOR_PRIMARIES(enum.IntEnum):

        UNKNOWN = ...
        BT709 = ...
        SRGB = ...
        IEC61966_2_4 = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        GENERIC_FILM = ...
        BT2020 = ...
        BT2100 = ...
        XYZ = ...
        SMPTE431 = ...
        SMPTE432 = ...
        DCI_P3 = ...
        EBU3213 = ...

    class TRANSFER_CHARACTERISTICS(enum.IntEnum):

        UNKNOWN = ...
        BT709 = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        LINEAR = ...
        LOG100 = ...
        LOG100_SQRT10 = ...
        IEC61966 = ...
        BT1361 = ...
        SRGB = ...
        BT2020_10BIT = ...
        BT2020_12BIT = ...
        PQ = ...
        SMPTE2084 = ...
        SMPTE428 = ...
        HLG = ...

    class MATRIX_COEFFICIENTS(enum.IntEnum):

        IDENTITY = ...
        BT709 = ...
        UNSPECIFIED = ...
        FCC = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        YCGCO = ...
        BT2020_NCL = ...
        BT2020_CL = ...
        SMPTE2085 = ...
        CHROMA_DERIVED_NCL = ...
        CHROMA_DERIVED_CL = ...
        ICTCP = ...
        YCGCO_RE = ...
        YCGCO_RO = ...

class AvifError(RuntimeError): ...

def avif_version() -> str: ...
def avif_check(data: Buffer, /) -> bool | None: ...
def avif_encode(
    data: ArrayLike,
    /,
    level: AVIF.QUALITY | int | None = None,
    *,
    speed: AVIF.SPEED | int | None = None,
    tilelog2: tuple[int, int] | None = None,
    bitspersample: int | None = None,
    pixelformat: EnumLike[AVIF.PIXEL_FORMAT] | None = None,
    codec: EnumLike[AVIF.CODEC_CHOICE] | None = None,
    primaries: EnumLike[AVIF.COLOR_PRIMARIES] | None = None,
    transfer: EnumLike[AVIF.TRANSFER_CHARACTERISTICS] | None = None,
    matrix: EnumLike[AVIF.MATRIX_COEFFICIENTS] | None = None,
    iccprofile: Buffer | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def avif_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    codec: EnumLike[AVIF.CODEC_CHOICE] | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def avif_info(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    codec: EnumLike[AVIF.CODEC_CHOICE] | None = None,
) -> dict[str, Any]: ...

class B2ND:

    available: bool

    class FILTER(enum.IntEnum):

        NOFILTER = ...
        NOSHUFFLE = ...
        SHUFFLE = ...  # default
        BITSHUFFLE = ...
        DELTA = ...
        TRUNC_PREC = ...
        NDCELL = ...
        NDMEAN = ...
        BYTEDELTA = ...
        INT_TRUNC = ...

    class CODEC(enum.IntEnum):

        BLOSCLZ = ...
        LZ4 = ...
        LZ4HC = ...
        ZLIB = ...
        ZSTD = ...  # default
        NDLZ = ...
        ZFP_ACC = ...
        ZFP_PREC = ...
        ZFP_RATE = ...

class B2ndError(RuntimeError): ...

def b2nd_version() -> str: ...
def b2nd_check(data: Buffer, /) -> bool | None: ...
def b2nd_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    codec: EnumLike[B2ND.CODEC] | None = None,
    codec_meta: int | None = None,
    filters: Sequence[EnumLike[B2ND.FILTER]] | None = None,
    filters_meta: Sequence[int] | None = None,
    chunkshape: Sequence[int] | None = None,
    blockshape: Sequence[int] | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def b2nd_decode(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def b2nd_info(data: Buffer, /) -> dict[str, Any]: ...

class BCN:

    available: bool

    class FORMAT(enum.IntEnum):

        BC1 = ...  # DXT1
        BC2 = ...  # DXT3
        BC3 = ...  # DXT5
        BC4 = ...  # BC4_UNORM
        BC5 = ...  # BC5_UNORM
        BC6HU = ...  # BC6H_UF16
        BC6HS = ...  # BC6H_SF16
        BC7 = ...  # BC7_UNORM

class BcnError(RuntimeError): ...

def bcn_version() -> str: ...
def bcn_check(data: Buffer, /) -> bool | None: ...
def bcn_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def bcn_decode(
    data: Buffer,
    /,
    format: BCN.FORMAT | int,
    *,
    shape: Sequence[int] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def bcn_info(
    data: Buffer,
    /,
    format: BCN.FORMAT | int,
    *,
    shape: Sequence[int] | None = None,
) -> dict[str, Any]: ...

class BFLOAT16:

    available: bool

    class ROUND(enum.IntEnum):

        TONEAREST = ...
        UPWARD = ...
        DOWNWARD = ...
        TOWARDZERO = ...

Bfloat16Error = ImcdError
bfloat16_version = imcd_version

def bfloat16_check(data: Buffer, /) -> bool | None: ...
def bfloat16_encode(
    data: ArrayLike,
    /,
    *,
    byteorder: ByteOrder | None = None,
    rounding: EnumLike[BFLOAT16.ROUND] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def bfloat16_decode(
    data: Buffer,
    /,
    *,
    byteorder: ByteOrder | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def bfloat16_info(data: Buffer, /) -> NoReturn: ...

class BITORDER:

    available: bool

BitorderError = ImcdError
bitorder_version = imcd_version

def bitorder_check(data: Buffer, /) -> bool | None: ...
@overload
def bitorder_encode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def bitorder_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...

bitorder_decode = bitorder_encode

def bitorder_info(data: Buffer, /) -> NoReturn: ...

class BITSHUFFLE:

    available: bool

class BitshuffleError(RuntimeError): ...

def bitshuffle_version() -> str: ...
def bitshuffle_check(data: Buffer, /) -> bool | None: ...
@overload
def bitshuffle_encode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    itemsize: int | None = None,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def bitshuffle_encode(
    data: Buffer,
    /,
    *,
    itemsize: int | None = None,
    blocksize: int = 0,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def bitshuffle_decode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    itemsize: int | None = None,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def bitshuffle_decode(
    data: Buffer,
    /,
    *,
    itemsize: int | None = None,
    blocksize: int = 0,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def bitshuffle_decode(
    data: Buffer,
    /,
    *,
    itemsize: int | None = None,
    blocksize: int = 0,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def bitshuffle_info(data: Buffer, /) -> NoReturn: ...

class BLOSC:

    available: bool

    class SHUFFLE(enum.IntEnum):

        NOSHUFFLE = ...
        SHUFFLE = ...
        BITSHUFFLE = ...

    class COMPRESSOR(enum.IntEnum):

        BLOSCLZ = ...
        LZ4 = ...
        LZ4HC = ...
        SNAPPY = ...
        ZLIB = ...
        ZSTD = ...

class BloscError(RuntimeError): ...

def blosc_version() -> str: ...
def blosc_check(data: Buffer, /) -> bool | None: ...
def blosc_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    compressor: EnumLike[BLOSC.COMPRESSOR] | None = None,
    shuffle: EnumLike[BLOSC.SHUFFLE] | None = None,
    typesize: int | None = None,
    blocksize: int | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def blosc_decode(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def blosc_decode(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def blosc_info(data: Buffer, /) -> dict[str, Any]: ...

class BLOSC2:

    available: bool

    class FILTER(enum.IntEnum):

        NOFILTER = ...
        NOSHUFFLE = ...
        SHUFFLE = ...  # default
        BITSHUFFLE = ...
        DELTA = ...
        TRUNC_PREC = ...
        BYTEDELTA = ...
        INT_TRUNC = ...

    class CODEC(enum.IntEnum):

        BLOSCLZ = ...
        LZ4 = ...
        LZ4HC = ...
        ZLIB = ...
        ZSTD = ...  # default

    class SPLIT(enum.IntEnum):

        ALWAYS = ...  # default
        NEVER = ...
        AUTO = ...
        FORWARD_COMPAT = ...
        FORWARD = ...  # alias

class Blosc2Error(RuntimeError): ...

def blosc2_version() -> str: ...
def blosc2_check(data: Buffer, /) -> bool | None: ...
def blosc2_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    codec: EnumLike[BLOSC2.CODEC] | None = None,
    filters: Sequence[EnumLike[BLOSC2.FILTER]] | None = None,
    filters_meta: Sequence[int] | None = None,
    splitmode: EnumLike[BLOSC2.SPLIT] | None = None,
    typesize: int | None = None,
    blocksize: int | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def blosc2_decode(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def blosc2_decode(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def blosc2_info(data: Buffer, /) -> dict[str, Any]: ...

class BMP:

    available: bool

class BmpError(RuntimeError): ...

def bmp_version() -> str: ...
def bmp_check(data: Buffer, /) -> bool | None: ...
def bmp_encode(
    data: ArrayLike,
    /,
    *,
    ppm: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def bmp_decode(
    data: Buffer,
    /,
    *,
    asrgb: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def bmp_info(data: Buffer, /, asrgb: bool | None = None) -> dict[str, Any]: ...

class BROTLI:

    available: bool

    class MODE(enum.IntEnum):

        GENERIC = ...
        TEXT = ...
        FONT = ...

class BrotliError(RuntimeError): ...

def brotli_version() -> str: ...
def brotli_check(data: Buffer, /) -> bool | None: ...
def brotli_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    mode: EnumLike[BROTLI.MODE] | None = None,
    lgwin: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def brotli_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def brotli_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def brotli_info(data: Buffer, /) -> NoReturn: ...

class BRUNSLI:

    available: bool

class BrunsliError(RuntimeError): ...

def brunsli_version() -> str: ...
def brunsli_check(data: Buffer, /) -> bool | None: ...
def brunsli_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: str | int | None = None,
    outcolorspace: int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    predictor: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def brunsli_decode(
    data: Buffer,
    /,
    *,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def brunsli_encode_jpeg(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def brunsli_decode_jpeg(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def brunsli_info(
    data: Buffer,
    /,
    *,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
) -> dict[str, Any]: ...

class BYTESHUFFLE:

    available: bool

ByteshuffleError = ImcdError
byteshuffle_version = imcd_version

def byteshuffle_check(data: Buffer, /) -> bool | None: ...
def byteshuffle_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    delta: bool = False,
    reorder: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def byteshuffle_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    delta: bool = False,
    reorder: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def byteshuffle_info(data: Buffer, /) -> NoReturn: ...

class BZ2:

    available: bool

class Bz2Error(RuntimeError): ...

def bz2_version() -> str: ...
def bz2_check(data: Buffer, /) -> bool | None: ...
def bz2_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def bz2_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def bz2_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def bz2_info(data: Buffer, /) -> NoReturn: ...

class CCITTFAX3:

    available: bool

Ccittfax3Error = CcittError
ccittfax3_version = ccitt_version

def ccittfax3_check(data: Buffer, /) -> bool | None: ...
def ccittfax3_encode(
    data: Buffer | ArrayLike,
    /,
    level: int | None = None,
    *,
    axis: int | None = None,
    out: BytesOut | None = None,
) -> NoReturn: ...
def ccittfax3_decode(
    data: Buffer,
    /,
    height: int = 0,
    width: int = 0,
    *,
    t4options: int = 0,
    out: NDArray[Any] | bytearray | memoryview | None = None,
) -> NDArray[Any]: ...
def ccittfax3_info(data: Buffer, /) -> NoReturn: ...

class CCITTFAX4:

    available: bool

Ccittfax4Error = CcittError
ccittfax4_version = ccitt_version

def ccittfax4_check(data: Buffer, /) -> bool | None: ...
def ccittfax4_encode(
    data: Buffer | ArrayLike,
    /,
    level: int | None = None,
    *,
    axis: int | None = None,
    out: BytesOut | None = None,
) -> NoReturn: ...
def ccittfax4_decode(
    data: Buffer,
    /,
    height: int = 0,
    width: int = 0,
    *,
    out: NDArray[Any] | bytearray | memoryview | None = None,
) -> NDArray[Any]: ...
def ccittfax4_info(data: Buffer, /) -> NoReturn: ...

class CCITTRLE:

    available: bool

CcittrleError = CcittError
ccittrle_version = ccitt_version

def ccittrle_check(data: Buffer, /) -> bool | None: ...
def ccittrle_encode(
    data: Buffer | ArrayLike,
    /,
    level: int | None = None,
    *,
    axis: int | None = None,
    out: BytesOut | None = None,
) -> NoReturn: ...
def ccittrle_decode(
    data: Buffer,
    /,
    height: int = 0,
    width: int = 0,
    *,
    out: NDArray[Any] | bytearray | memoryview | None = None,
) -> NDArray[Any]: ...
def ccittrle_info(data: Buffer, /) -> NoReturn: ...

class CHUNKED:
    available: bool

    class CODEC(enum.IntEnum):
        ZSTD = ...
        LZ4 = ...

class ChunkedError(RuntimeError): ...

def chunked_version() -> str: ...
def chunked_check(data: Buffer, /) -> bool | None: ...
def chunked_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    codec: EnumLike[CHUNKED.CODEC] | None = None,
    chunksize: int | None = None,
    itemsize: int = 1,
    hilo: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def chunked_decode(
    data: Buffer,
    /,
    *,
    itemsize: int = 1,
    samples: int = 1,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def chunked_decode(
    data: Buffer,
    /,
    *,
    itemsize: int = 1,
    samples: int = 1,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def chunked_info(data: Buffer, /) -> NoReturn: ...

class CMS:

    available: bool

    class INTENT(enum.IntEnum):

        PERCEPTUAL = ...
        RELATIVE_COLORIMETRIC = ...
        SATURATION = ...
        ABSOLUTE_COLORIMETRIC = ...

    class FLAGS(enum.IntEnum):

        NOCACHE = ...
        NOOPTIMIZE = ...
        NULLTRANSFORM = ...
        GAMUTCHECK = ...
        SOFTPROOFING = ...
        BLACKPOINTCOMPENSATION = ...
        NOWHITEONWHITEFIXUP = ...
        HIGHRESPRECALC = ...
        LOWRESPRECALC = ...
        EIGHTBITS_DEVICELINK = ...
        GUESSDEVICECLASS = ...
        KEEP_SEQUENCE = ...
        FORCE_CLUT = ...
        CLUT_POST_LINEARIZATION = ...
        CLUT_PRE_LINEARIZATION = ...
        NONEGATIVES = ...
        COPY_ALPHA = ...
        NODEFAULTRESOURCEDEF = ...

    class PT(enum.IntEnum):

        GRAY = ...
        RGB = ...
        CMY = ...
        CMYK = ...
        YCBCR = ...
        YUV = ...
        XYZ = ...
        LAB = ...
        YUVK = ...
        HSV = ...
        HLS = ...
        YXY = ...
        MCH1 = ...
        MCH2 = ...
        MCH3 = ...
        MCH4 = ...
        MCH5 = ...
        MCH6 = ...
        MCH7 = ...
        MCH8 = ...
        MCH9 = ...
        MCH10 = ...
        MCH11 = ...
        MCH12 = ...
        MCH13 = ...
        MCH14 = ...
        MCH15 = ...

class CmsError(RuntimeError): ...

def cms_version() -> str: ...
def cms_check(data: Buffer, /) -> bool | None: ...
def cms_info(profile: Buffer, /) -> dict[str, Any]: ...
def cms_transform(
    data: ArrayLike,
    /,
    profile: Buffer | None,
    outprofile: Buffer | None,
    *,
    colorspace: str | None = None,
    planar: bool | None = None,
    outcolorspace: str | None = None,
    outplanar: bool | None = None,
    outdtype: DTypeLike | None = None,
    intent: EnumLike[CMS.INTENT] | None = None,
    flags: CMS.FLAGS | int | None = None,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def cms_profile(
    profile: (
        Literal[
            'null',
            'gray',
            'rgb',
            'srgb',
            'xyz',
            'lab2',
            'lab4',
            'adobergb',
            'linearrgb',
            'displayp3',
            'rec2020',
            'sgray',
        ]
        | None
    ),
    /,
    *,
    whitepoint: Sequence[float] | None = None,
    primaries: Sequence[float] | None = None,
    transferfunction: ArrayLike | Literal['srgb', 'rec709'] | None = None,
    gamma: float | None = None,
) -> bytes: ...
def cms_profile_validate(
    profile: Buffer,
    /,
    *,
    verbose: int | None = None,
) -> None: ...

class DDS:

    available: bool

    class FOURCC(enum.IntEnum):

        DDS = ...
        DXT1 = ...
        DXT3 = ...
        DXT5 = ...
        DX10 = ...
        BC4_UNORM = ...
        BC5_UNORM = ...
        BC6H_UF16 = ...
        BC6H_SF16 = ...
        BC7_UNORM = ...

class DdsError(RuntimeError): ...

def dds_version() -> str: ...
def dds_check(data: Buffer, /) -> bool | None: ...
def dds_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def dds_decode(
    data: Buffer,
    /,
    *,
    mipmap: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def dds_info(data: Buffer, /, *, mipmap: int = 0) -> dict[str, Any]: ...

class DEFLATE:

    available: bool

class DeflateError(RuntimeError): ...

def deflate_version() -> str: ...
def deflate_check(data: Buffer, /) -> bool | None: ...
def deflate_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    raw: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def deflate_decode(
    data: Buffer,
    /,
    *,
    raw: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def deflate_decode(
    data: Buffer,
    /,
    *,
    raw: bool = False,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def deflate_crc32(data: Buffer, /, value: int | None = None) -> int: ...
def deflate_adler32(data: Buffer, /, value: int | None = None) -> int: ...
def deflate_info(data: Buffer, /) -> NoReturn: ...

class DELTA:

    available: bool

DeltaError = ImcdError
delta_version = imcd_version

def delta_check(data: Buffer, /) -> bool | None: ...
@overload
def delta_encode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def delta_encode(
    data: Buffer,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def delta_decode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def delta_decode(
    data: Buffer,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: BytesOut | None = None,
) -> BytesResult: ...
def delta_info(data: Buffer, /) -> NoReturn: ...

class DICOMRLE:

    available: bool

DicomrleError = ImcdError
dicomrle_version = imcd_version

def dicomrle_check(data: Buffer, /) -> bool | None: ...
def dicomrle_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def dicomrle_decode(
    data: Buffer,
    /,
    dtype: DTypeLike,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def dicomrle_info(data: Buffer, /, dtype: DTypeLike) -> dict[str, Any]: ...

class EER:

    available: bool

EerError = ImcdError
eer_version = imcd_version

def eer_check(data: Buffer, /) -> bool | None: ...
def eer_encode(
    data: ArrayLike,
    /,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def eer_decode(
    data: Buffer,
    /,
    shape: tuple[int, int],
    skipbits: int,
    horzbits: int,
    vertbits: int,
    *,
    superres: int = 0,
    out: NDArray[Any] | int | None = None,
) -> NDArray[Any]: ...
def eer_info(data: Buffer, /) -> NoReturn: ...

class EXR:

    available: bool

    class COMPRESSION(enum.IntEnum):

        NONE = ...
        RLE = ...
        ZIPS = ...
        ZIP = ...
        PIZ = ...
        PXR24 = ...
        B44 = ...
        B44A = ...
        DWAA = ...
        DWAB = ...
        HTJ2K256 = ...
        HTJ2K32 = ...
        ZSTD = ...
        LJ2K = ...

class ExrError(RuntimeError): ...

def exr_version() -> str: ...
def exr_check(data: Buffer, /) -> bool | None: ...
def exr_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    compression: EnumLike[EXR.COMPRESSION] | None = None,
    colorinteropid: str | None = None,
    planar: bool | None = None,
    frames: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def exr_decode(
    data: Buffer,
    /,
    *,
    index: int | None = 0,
    planar: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def exr_info(
    data: Buffer, /, *, index: int | None = 0, planar: bool | None = None
) -> dict[str, Any]: ...

class FLOAT24:

    available: bool

    class ROUND(enum.IntEnum):

        TONEAREST = ...
        UPWARD = ...
        DOWNWARD = ...
        TOWARDZERO = ...

Float24Error = ImcdError
float24_version = imcd_version

def float24_check(data: Buffer, /) -> bool | None: ...
def float24_encode(
    data: ArrayLike,
    /,
    *,
    byteorder: ByteOrder | None = None,
    rounding: EnumLike[FLOAT24.ROUND] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def float24_decode(
    data: Buffer,
    /,
    *,
    byteorder: ByteOrder | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def float24_info(data: Buffer, /) -> NoReturn: ...

class FLOATPRED:

    available: bool

FloatpredError = ImcdError
floatpred_version = imcd_version

def floatpred_check(data: Buffer, /) -> bool | None: ...
def floatpred_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def floatpred_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def floatpred_info(data: Buffer, /) -> NoReturn: ...

class GIF:

    available: bool

class GifError(RuntimeError): ...

def gif_version() -> str: ...
def gif_check(data: Buffer, /) -> bool | None: ...
def gif_encode(
    data: ArrayLike,
    /,
    *,
    colormap: ArrayLike | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def gif_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    asrgb: bool = True,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def gif_info(
    data: Buffer, /, index: int | None = None, *, asrgb: bool = True
) -> dict[str, Any]: ...

class GZIP:

    available: bool

GzipError = DeflateError
gzip_version = deflate_version

def gzip_check(data: Buffer, /) -> bool | None: ...
def gzip_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def gzip_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def gzip_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def gzip_info(data: Buffer, /) -> NoReturn: ...

class H5CHECKSUM:

    available: bool

def h5checksum_version() -> str: ...
def h5checksum_fletcher32(
    data: Buffer, /, value: int | None = None
) -> int: ...
def h5checksum_lookup3(data: Buffer, /, value: int | None = None) -> int: ...
def h5checksum_crc(data: Buffer, /, value: int | None = None) -> int: ...
def h5checksum_metadata(data: Buffer, /, value: int | None = None) -> int: ...
def h5checksum_hash_string(
    data: Buffer, /, value: int | None = None
) -> int: ...

class HCOMP:

    available: bool

class HcompError(RuntimeError): ...

def hcomp_version() -> str: ...
def hcomp_check(data: Buffer, /) -> bool | None: ...
def hcomp_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def hcomp_decode(
    data: Buffer,
    /,
    *,
    smooth: int | None = None,
    safe32: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def hcomp_info(data: Buffer, /) -> NoReturn: ...

class HEIF:

    available: bool

    class COMPRESSION(enum.IntEnum):

        UNDEFINED = ...
        HEVC = ...
        AVC = ...
        JPEG = ...
        AV1 = ...
        VVC = ...
        EVC = ...
        JPEG2000 = ...
        UNCOMPRESSED = ...
        MASK = ...
        HTJ2K = ...

    class COLORSPACE(enum.IntEnum):

        UNDEFINED = ...
        YCBCR = ...
        RGB = ...
        MONOCHROME = ...
        GRAY = ...  # alias

class HeifError(RuntimeError): ...

def heif_version() -> str: ...
def heif_check(data: Buffer, /) -> bool | None: ...
def heif_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bitspersample: int | None = None,
    photometric: EnumLike[HEIF.COLORSPACE] | None = None,
    hasalpha: bool | None = None,
    compression: EnumLike[HEIF.COMPRESSION] | None = None,
    iccprofile: Buffer | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def heif_decode(
    data: Buffer,
    /,
    index: int | None = 0,
    *,
    photometric: EnumLike[HEIF.COLORSPACE] | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def heif_info(
    data: Buffer,
    /,
    index: int | None = 0,
    *,
    photometric: EnumLike[HEIF.COLORSPACE] | None = None,
) -> dict[str, Any]: ...

class HTJ2K:

    available: bool

    class TILEPART(enum.IntFlag):

        RESOLUTIONS = ...
        COMPONENTS = ...

class Htj2kError(RuntimeError): ...

def htj2k_init(
    *,
    verbose: int | None = None,
) -> None: ...
def htj2k_version() -> str: ...
def htj2k_check(data: Buffer, /) -> bool | None: ...
def htj2k_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    rgb: bool | None = None,
    planar: bool | None = None,
    bitspersample: int | None = None,
    tile: tuple[int, int] | None = None,
    resolutions: int | None = None,
    reversible: bool | None = None,
    tlm: bool | None = None,
    tilepart: EnumLike[HTJ2K.TILEPART] | None = None,
    block_size: tuple[int, int] | None = None,
    prog_order: str | None = None,
    profile: str | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def htj2k_decode(
    data: Buffer,
    /,
    *,
    planar: bool | None = None,
    skipres: int | tuple[int, int] | None = None,
    resilient: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def htj2k_info(
    data: Buffer,
    /,
    *,
    planar: bool | None = None,
    skipres: int | tuple[int, int] | None = None,
    resilient: bool = False,
) -> dict[str, Any]: ...

class ISAL:

    available: bool

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...

class IsalError(RuntimeError): ...

def isal_version() -> str: ...
def isal_check(data: Buffer, /) -> bool | None: ...
def isal_encode(
    data: Buffer,
    /,
    level: ISAL.COMPRESSION | int | None = None,
    *,
    raw: bool = False,
    gzip: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def isal_decode(
    data: Buffer,
    /,
    *,
    raw: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def isal_decode(
    data: Buffer,
    /,
    *,
    raw: bool = False,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def isal_crc32(data: Buffer, /, value: int | None = None) -> int: ...
def isal_crc32c(data: Buffer, /, value: int | None = None) -> int: ...
def isal_adler32(data: Buffer, /, value: int | None = None) -> int: ...
def isal_info(data: Buffer, /) -> NoReturn: ...

class JETRAW:

    available: bool

class JetrawError(RuntimeError): ...

def jetraw_init(
    parameters: str | None = None,
    *,
    verbose: int | None = None,
) -> None: ...
def jetraw_version() -> str: ...
def jetraw_check(data: Buffer, /) -> bool | None: ...
def jetraw_encode(
    data: ArrayLike,
    /,
    identifier: str,
    *,
    errorbound: float | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jetraw_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jetraw_info(data: Buffer, /) -> NoReturn: ...

class JPEG2K:

    available: bool

    class CODEC(enum.IntEnum):

        JP2 = ...
        J2K = ...
        # JPT = ...
        # JPP = ...
        # JPX = ...

    class CLRSPC(enum.IntEnum):

        UNSPECIFIED = ...
        SRGB = ...
        GRAY = ...
        SYCC = ...
        EYCC = ...
        CMYK = ...

    class EXTRA(enum.IntEnum):

        UNSPECIFIED = ...
        ASSOCALPHA = ...
        UNASSALPHA = ...

class Jpeg2kError(RuntimeError): ...

def jpeg2k_version() -> str: ...
def jpeg2k_check(data: Buffer, /) -> bool | None: ...
def jpeg2k_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    codecformat: EnumLike[JPEG2K.CODEC] | None = None,
    colorspace: EnumLike[JPEG2K.CLRSPC] | None = None,
    extrasample: EnumLike[JPEG2K.EXTRA] | None = None,
    planar: bool | None = None,
    tile: tuple[int, int] | None = None,
    bitspersample: int | None = None,
    resolutions: int | None = None,
    reversible: bool | None = None,
    mct: bool = True,
    verbose: int | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpeg2k_decode(
    data: Buffer,
    /,
    *,
    planar: bool | None = None,
    skipres: int | None = None,
    verbose: int | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpeg2k_info(
    data: Buffer,
    /,
    *,
    planar: bool | None = None,
    skipres: int | None = None,
    verbose: int | None = None,
) -> dict[str, Any]: ...

class JPEG:

    available: bool

    legacy: bool

    all_precisions: bool

    class CS(enum.IntEnum):

        UNKNOWN = ...
        GRAYSCALE = ...
        RGB = ...
        YCbCr = ...
        CMYK = ...
        YCCK = ...
        EXT_RGB = ...
        EXT_RGBX = ...
        EXT_BGR = ...
        EXT_BGRX = ...
        EXT_XBGR = ...
        EXT_XRGB = ...
        EXT_RGBA = ...
        EXT_BGRA = ...
        EXT_ABGR = ...
        EXT_ARGB = ...

class JpegError(RuntimeError): ...

def jpeg_version() -> str: ...
def jpeg_check(data: Buffer, /) -> bool | None: ...
def jpeg_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: EnumLike[JPEG.CS] | None = None,
    outcolorspace: EnumLike[JPEG.CS] | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    lossless: bool | None = None,
    predictor: int | None = None,
    bitspersample: int | None = None,
    iccprofile: Buffer | None = None,
    validate: bool | None = None,  # for compatibility
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpeg_decode(
    data: Buffer,
    /,
    *,
    tables: bytes | None = None,
    header: bytes | None = None,
    colorspace: EnumLike[JPEG.CS] | None = None,
    outcolorspace: EnumLike[JPEG.CS] | None = None,
    fancyupsampling: bool | None = None,
    shape: tuple[int, int] | None = None,
    scale: tuple[int, int] | int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpeg_info(
    data: Buffer,
    /,
    *,
    tables: bytes | None = None,
    header: bytes | None = None,
    colorspace: EnumLike[JPEG.CS] | None = None,
    outcolorspace: EnumLike[JPEG.CS] | None = None,
    shape: tuple[int, int] | None = None,
    scale: tuple[int, int] | int | None = None,
) -> dict[str, Any]: ...
def jpeg_transcode(
    data: Buffer | Sequence[Buffer | None],
    /,
    *,
    tables: Buffer | None = None,
    header: Buffer | None = None,
    colorspace: EnumLike[JPEG.CS] | None = None,
    tiles: tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    optimize: bool | None = None,
    progressive: bool | None = None,
    copy: str | None = None,
    restart: int | None = None,
    grayscale: bool | None = None,
    arithmetic: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpeg_merge(
    data: Buffer | Sequence[Buffer],
    /,
    *,
    tables: Buffer | None = None,
    header: Buffer | None = None,
    colorspace: EnumLike[JPEG.CS] | None = None,
    tiles: tuple[int, int] | None = None,
) -> bytes: ...

class JPEGLI:

    available: bool

    class CS(enum.IntEnum):

        UNKNOWN = ...
        GRAYSCALE = ...
        RGB = ...
        YCbCr = ...
        CMYK = ...
        YCCK = ...
        EXT_RGB = ...
        EXT_RGBX = ...
        EXT_BGR = ...
        EXT_BGRX = ...
        EXT_XBGR = ...
        EXT_XRGB = ...
        EXT_RGBA = ...
        EXT_BGRA = ...
        EXT_ABGR = ...
        EXT_ARGB = ...

class JpegliError(RuntimeError): ...

def jpegli_version() -> str: ...
def jpegli_check(data: Buffer, /) -> bool | None: ...
def jpegli_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    distance: float | None = None,
    colorspace: EnumLike[JPEGLI.CS] | None = None,
    outcolorspace: EnumLike[JPEGLI.CS] | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: int | None = None,
    progressive: bool | int | None = None,
    adaptive: bool | None = None,
    standardtables: bool | None = None,
    iccprofile: Buffer | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegli_decode(
    data: Buffer,
    /,
    *,
    tables: bytes | None = None,
    header: bytes | None = None,
    colorspace: EnumLike[JPEGLI.CS] | None = None,
    outcolorspace: EnumLike[JPEGLI.CS] | None = None,
    fancyupsampling: bool | None = None,
    scale: tuple[int, int] | int | None = None,
    dtype: DTypeLike | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegli_transcode(
    data: Buffer,
    /,
    *,
    optimize: bool | None = None,
    progressive: bool | int | None = None,
    copy: str | None = None,
    restart: int | None = None,
    grayscale: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...

jpegli_info = jpeg_info

class JPEGLS:

    available: bool

    class COLOR_TRANSFORMATION(enum.IntEnum):

        NONE = ...
        HP1 = ...
        HP2 = ...
        HP3 = ...

    class INTERLEAVE(enum.IntEnum):

        NONE = ...
        LINE = ...
        SAMPLE = ...

class JpeglsError(RuntimeError): ...

def jpegls_version() -> str: ...
def jpegls_check(data: Buffer, /) -> bool | None: ...
def jpegls_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bitspersample: int | None = None,
    planar: bool | None = None,
    transformation: EnumLike[JPEGLS.COLOR_TRANSFORMATION] | None = None,
    spiff: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegls_decode(
    data: Buffer,
    /,
    *,
    planar: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegls_info(
    data: Buffer, /, *, planar: bool | None = None
) -> dict[str, Any]: ...

class JPEGSOF3:

    available: bool

class Jpegsof3Error(RuntimeError): ...

def jpegsof3_version() -> str: ...
def jpegsof3_check(data: Buffer, /) -> bool | None: ...
def jpegsof3_encode(
    data: ArrayLike,
    /,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def jpegsof3_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegsof3_info(data: Buffer, /) -> NoReturn: ...

class JPEGXL:

    available: bool

    class COLOR_SPACE(enum.IntEnum):

        UNKNOWN = ...
        RGB = ...
        GRAY = ...
        XYB = ...

    class CHANNEL(enum.IntEnum):

        UNKNOWN = ...
        ALPHA = ...
        DEPTH = ...
        SPOT_COLOR = ...
        SELECTION_MASK = ...
        BLACK = ...
        CFA = ...
        THERMAL = ...
        OPTIONAL = ...

    class WHITE_POINT(enum.IntEnum):

        D65 = ...
        CUSTOM = ...
        E = ...
        DCI = ...

    class PRIMARIES(enum.IntEnum):

        SRGB = ...
        CUSTOM = ...
        BT2100 = ...
        P3 = ...

    class TRANSFER_FUNCTION(enum.IntEnum):

        BT709 = ...
        UNKNOWN = ...
        LINEAR = ...
        SRGB = ...
        PQ = ...
        DCI = ...
        HLG = ...
        GAMMA = ...

class JpegxlError(RuntimeError): ...

def jpegxl_version() -> str: ...
def jpegxl_check(data: Buffer, /) -> bool | None: ...
def jpegxl_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    effort: int | None = None,
    distance: float | None = None,
    lossless: bool | None = None,
    decodingspeed: int | None = None,
    photometric: EnumLike[JPEGXL.COLOR_SPACE] | None = None,
    hasalpha: bool | None = None,
    bitspersample: int | None = None,
    # extrasamples: Sequence[JPEGXL.CHANNEL] | None = None,
    planar: bool | None = None,
    primaries: EnumLike[JPEGXL.PRIMARIES] | None = None,
    transfer: EnumLike[JPEGXL.TRANSFER_FUNCTION] | None = None,
    whitepoint: EnumLike[JPEGXL.WHITE_POINT] | None = None,
    gamma: float | None = None,
    perceptualheuristics: bool | None = None,
    usecontainer: bool | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegxl_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    keeporientation: bool | None = None,
    primaries: EnumLike[JPEGXL.PRIMARIES] | None = None,
    transfer: EnumLike[JPEGXL.TRANSFER_FUNCTION] | None = None,
    whitepoint: EnumLike[JPEGXL.WHITE_POINT] | None = None,
    gamma: float | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegxl_encode_jpeg(
    data: Buffer,
    /,
    *,
    usecontainer: bool | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegxl_decode_jpeg(
    data: Buffer,
    /,
    *,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegxl_info(
    data: Buffer,
    /,
    *,
    keeporientation: bool | None = None,
    primaries: EnumLike[JPEGXL.PRIMARIES] | None = None,
    transfer: EnumLike[JPEGXL.TRANSFER_FUNCTION] | None = None,
    whitepoint: EnumLike[JPEGXL.WHITE_POINT] | None = None,
    gamma: float | None = None,
) -> dict[str, Any]: ...

class JPEGXR:

    available: bool

    class PI(enum.IntEnum):

        W0 = ...
        B0 = ...
        RGB = ...
        RGBPalette = ...
        TransparencyMask = ...
        CMYK = ...
        YCbCr = ...
        CIELab = ...
        NCH = ...
        RGBE = ...
        DEFAULT = ...
        GRAY = ...  # alias

    class PF(enum.IntEnum):

        Format8bppGray = ...
        Format24bppRGB = ...
        Format32bppRGBA = ...
        Format32bppPRGBA = ...
        Format32bppRGBE = ...
        Format32bppCMYK = ...
        Format40bppCMYKAlpha = ...
        Format24bpp3Channels = ...
        Format32bpp4Channels = ...
        Format40bpp5Channels = ...
        Format48bpp6Channels = ...
        Format56bpp7Channels = ...
        Format64bpp8Channels = ...
        Format32bpp3ChannelsAlpha = ...
        Format40bpp4ChannelsAlpha = ...
        Format48bpp5ChannelsAlpha = ...
        Format56bpp6ChannelsAlpha = ...
        Format64bpp7ChannelsAlpha = ...
        Format72bpp8ChannelsAlpha = ...
        Format16bppGray = ...
        Format48bppRGB = ...
        Format64bppRGBA = ...
        Format64bppPRGBA = ...
        Format64bppCMYK = ...
        Format80bppCMYKAlpha = ...
        Format48bpp3Channels = ...
        Format64bpp4Channels = ...
        Format80bpp5Channels = ...
        Format96bpp6Channels = ...
        Format112bpp7Channels = ...
        Format128bpp8Channels = ...
        Format64bpp3ChannelsAlpha = ...
        Format80bpp4ChannelsAlpha = ...
        Format96bpp5ChannelsAlpha = ...
        Format112bpp6ChannelsAlpha = ...
        Format128bpp7ChannelsAlpha = ...
        Format144bpp8ChannelsAlpha = ...
        Format16bppGrayHalf = ...
        Format48bppRGBHalf = ...
        Format64bppRGBAHalf = ...
        Format32bppGrayFloat = ...
        Format96bppRGBFloat = ...
        Format128bppRGBAFloat = ...
        Format128bppPRGBAFloat = ...

class JpegxrError(RuntimeError): ...

def jpegxr_version() -> str: ...
def jpegxr_check(data: Buffer, /) -> bool | None: ...
def jpegxr_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    photometric: EnumLike[JPEGXR.PI] | None = None,
    hasalpha: bool | None = None,
    resolution: tuple[float, float] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegxr_decode(
    data: Buffer,
    /,
    *,
    pixelformat: EnumLike[JPEGXR.PF] | None = None,
    fp2int: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegxr_info(
    data: Buffer,
    /,
    *,
    pixelformat: EnumLike[JPEGXR.PF] | None = None,
    fp2int: bool = False,
) -> dict[str, Any]: ...

class JPEGXS:

    available: bool

class JpegxsError(RuntimeError): ...

def jpegxs_version() -> str: ...
def jpegxs_check(data: Buffer, /) -> bool | None: ...
def jpegxs_encode(
    data: ArrayLike,
    /,
    config: str | bytes | None = None,
    *,
    bitspersample: int | None = None,
    verbose: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def jpegxs_decode(
    data: Buffer,
    /,
    *,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegxs_info(data: Buffer, /) -> dict[str, Any]: ...

class LERC:

    available: bool

class LercError(RuntimeError): ...

def lerc_version() -> str: ...
def lerc_check(data: Buffer, /) -> bool | None: ...
def lerc_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    masks: ArrayLike | None = None,
    version: int | None = None,
    planar: bool | None = None,
    compression: Literal['zstd', 'deflate'] | None = None,
    compressionargs: dict[str, Any] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lerc_decode(
    data: Buffer,
    /,
    *,
    masks: Literal[False] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def lerc_decode(
    data: Buffer,
    /,
    *,
    masks: Literal[True] | NDArray[Any],
    out: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], NDArray[Any] | None]: ...
def lerc_info(data: Buffer, /) -> dict[str, Any]: ...

class LJPEG:

    available: bool

class LjpegError(RuntimeError): ...

def ljpeg_version() -> str: ...
def ljpeg_check(data: Buffer, /) -> bool | None: ...
def ljpeg_encode(
    data: ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    delinearize: ArrayLike | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def ljpeg_decode(
    data: Buffer,
    /,
    *,
    linearize: ArrayLike | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def ljpeg_info(data: Buffer, /) -> NoReturn: ...

class LZ4:

    available: bool

    class CLEVEL(enum.IntEnum):

        DEFAULT = ...
        MIN = ...
        MAX = ...
        OPT_MIN = ...

class Lz4Error(RuntimeError): ...

def lz4_version() -> str: ...
def lz4_check(data: Buffer, /) -> bool | None: ...
def lz4_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    hc: bool = False,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lz4_info(data: Buffer, /) -> NoReturn: ...

class LZ4F:

    available: bool

    VERSION: int

class Lz4fError(RuntimeError): ...

def lz4f_version() -> str: ...
def lz4f_check(data: Buffer, /) -> bool | None: ...
def lz4f_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    blocksizeid: int | None = None,
    blockchecksum: bool | None = None,
    contentchecksum: bool | None = None,
    contentsize: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4f_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4f_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lz4f_info(data: Buffer, /) -> dict[str, Any]: ...

class LZ4H5:

    available: bool

    CLEVEL: LZ4.CLEVEL

class Lz4h5Error(RuntimeError): ...

def lz4h5_version() -> str: ...
def lz4h5_check(data: Buffer, /) -> bool | None: ...
def lz4h5_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    hc: bool = False,
    blocksize: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4h5_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lz4h5_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lz4h5_info(data: Buffer, /) -> NoReturn: ...

class LZF:

    available: bool

class LzfError(RuntimeError): ...

def lzf_version() -> str: ...
def lzf_check(data: Buffer, /) -> bool | None: ...
def lzf_encode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzf_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzf_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lzf_info(data: Buffer, /) -> NoReturn: ...

class LZFSE:

    available: bool

class LzfseError(RuntimeError): ...

def lzfse_version() -> str: ...
def lzfse_check(data: Buffer, /) -> bool | None: ...
def lzfse_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzfse_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzfse_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lzfse_info(data: Buffer, /) -> NoReturn: ...

class LZHAM:

    available: bool

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...
        UBER = ...

    class STRATEGY(enum.IntEnum):

        DEFAULT = ...
        FILTERED = ...
        HUFFMAN_ONLY = ...
        RLE = ...
        FIXED = ...

class LzhamError(RuntimeError): ...

def lzham_version() -> str: ...
def lzham_check(data: Buffer, /) -> bool | None: ...
def lzham_encode(
    data: Buffer,
    /,
    level: EnumLike[LZHAM.COMPRESSION] | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzham_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzham_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lzham_info(data: Buffer, /) -> NoReturn: ...

class LZMA:

    available: bool

    class CHECK(enum.IntEnum):

        NONE = ...
        CRC32 = ...
        CRC64 = ...
        SHA256 = ...

class LzmaError(RuntimeError): ...

def lzma_version() -> str: ...
def lzma_check(data: Buffer, /) -> bool | None: ...
def lzma_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    check: EnumLike[LZMA.CHECK] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzma_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzma_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lzma_info(data: Buffer, /) -> dict[str, Any]: ...

class LZO:

    available: bool

class LzoError(RuntimeError): ...

def lzo_version() -> str: ...
def lzo_check(data: Buffer, /) -> bool | None: ...
def lzo_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> NoReturn: ...
def lzo_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
def lzo_info(data: Buffer, /) -> NoReturn: ...

class LZW:

    available: bool

LzwError = ImcdError

lzw_version = imcd_version

def lzw_check(data: Buffer, /) -> bool | None: ...
def lzw_encode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzw_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def lzw_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def lzw_info(data: Buffer, /) -> dict[str, Any]: ...

class MESHOPT:

    available: bool

class MeshoptError(RuntimeError): ...

def meshopt_version() -> str: ...
def meshopt_check(data: Buffer, /) -> bool | None: ...
def meshopt_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    items: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def meshopt_decode(
    data: Buffer,
    /,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    *,
    items: int | None = None,
    out: NDArray[Any] | bytearray | memoryview | None = None,
) -> NDArray[Any]: ...
def meshopt_info(data: Buffer, /) -> NoReturn: ...

class MOZJPEG:

    available: bool

    class CS(enum.IntEnum):

        UNKNOWN = ...
        GRAYSCALE = ...
        RGB = ...
        YCbCr = ...
        CMYK = ...
        YCCK = ...
        EXT_RGB = ...
        EXT_RGBX = ...
        EXT_BGR = ...
        EXT_BGRX = ...
        EXT_XBGR = ...
        EXT_XRGB = ...
        EXT_RGBA = ...
        EXT_BGRA = ...
        EXT_ABGR = ...
        EXT_ARGB = ...

class MozjpegError(RuntimeError): ...

def mozjpeg_version() -> str: ...
def mozjpeg_check(data: Buffer, /) -> bool | None: ...
def mozjpeg_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: EnumLike[MOZJPEG.CS] | None = None,
    outcolorspace: EnumLike[MOZJPEG.CS] | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    notrellis: bool | None = None,
    quanttable: int | None = None,
    progressive: bool | None = None,
    iccprofile: Buffer | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...

mozjpeg_decode = jpeg_decode
mozjpeg_info = jpeg_info

class NONE:

    available: bool

NoneError = RuntimeError

def none_version() -> str: ...
def none_check(data: Any, /) -> bool | None: ...
def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any: ...
def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any: ...
def none_info(data: Buffer, /) -> dict[str, Any]: ...

class NUMPY:

    available: bool

NumpyError = RuntimeError

def numpy_version() -> str: ...
def numpy_check(data: Buffer, /) -> bool | None: ...
def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> bytes: ...
def numpy_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]: ...
def numpy_info(data: Buffer, /, index: int | str = 0) -> dict[str, Any]: ...

class OPENZL:

    available: bool

class OpenzlError(RuntimeError): ...

def openzl_version() -> str: ...
def openzl_check(data: Buffer, /) -> bool | None: ...
def openzl_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> NoReturn: ...
def openzl_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def openzl_info(data: Buffer, /) -> NoReturn: ...

class PACKBITS:

    available: bool

PackbitsError = ImcdError

packbits_version = imcd_version

def packbits_check(data: Buffer, /) -> bool | None: ...
def packbits_encode(
    data: Buffer | ArrayLike,
    /,
    *,
    axis: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def packbits_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def packbits_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def packbits_info(data: Buffer, /) -> dict[str, Any]: ...

class PACKINTS:

    available: bool

PackintsError = ImcdError

packints_version = imcd_version

def packints_check(data: Buffer, /) -> bool | None: ...
def packints_encode(
    data: ArrayLike,
    bitspersample: int,
    /,
    *,
    bitorder: Literal['>', '<'] | None = None,
    runlen: int = 0,
    out: BytesOut | None = None,
) -> BytesResult: ...
def packints_decode(
    data: Buffer,
    dtype: DTypeLike,
    bitspersample: int,
    /,
    *,
    bitorder: Literal['>', '<'] | None = None,
    runlen: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def packints_info(
    data: Buffer,
    dtype: DTypeLike,
    bitspersample: int,
    /,
    *,
    bitorder: Literal['>', '<'] | None = None,
    runlen: int = 0,
) -> dict[str, Any]: ...

class PCX:

    available: bool

class PcxError(RuntimeError): ...

def pcx_version() -> str: ...
def pcx_check(data: Buffer, /) -> bool | None: ...
def pcx_encode(
    data: ArrayLike,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def pcx_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def pcx_info(data: Buffer, /, index: int | None = None) -> dict[str, Any]: ...

class PCODEC:

    available: bool

class PcodecError(RuntimeError): ...

def pcodec_version() -> str: ...
def pcodec_check(data: Buffer, /) -> bool | None: ...
def pcodec_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    pagesize: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def pcodec_decode(
    data: Buffer,
    /,
    shape: Sequence[int] | int | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def pcodec_info(data: Buffer, /) -> NoReturn: ...

class PGLZ:

    available: bool

class PglzError(RuntimeError): ...

def pglz_version() -> str: ...
def pglz_check(data: Buffer, /) -> bool | None: ...
def pglz_encode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    strategy: str | tuple[int, int, int, int, int, int] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def pglz_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    checkcomplete: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def pglz_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    checkcomplete: bool | None = None,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def pglz_info(data: Buffer, /) -> NoReturn: ...

class PLIO:

    available: bool

class PlioError(RuntimeError): ...

def plio_version() -> str: ...
def plio_check(data: Buffer, /) -> bool | None: ...
def plio_encode(
    data: ArrayLike,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def plio_decode(
    data: Buffer,
    /,
    npix: int | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def plio_info(data: Buffer, /) -> NoReturn: ...

class PIXARLOG:

    available: bool

class PixarlogError(RuntimeError): ...

def pixarlog_version() -> str: ...
def pixarlog_check(data: Buffer, /) -> bool | None: ...
def pixarlog_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    deflate: bool = True,
    byteorder: ByteOrder | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def pixarlog_decode(
    data: Buffer,
    /,
    *,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    deflate: bool = True,
    byteorder: ByteOrder | None = None,
    out: NDArray[Any] | bytearray | memoryview | None = None,
) -> NDArray[Any]: ...
def pixarlog_info(data: Buffer, /) -> NoReturn: ...

class PNG:

    available: bool

    class COLOR_TYPE(enum.IntEnum):

        GRAY = ...
        GRAY_ALPHA = ...
        RGB = ...
        RGB_ALPHA = ...
        PALETTE = ...

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...

    class STRATEGY(enum.IntEnum):

        DEFAULT = ...
        FILTERED = ...
        HUFFMAN_ONLY = ...
        RLE = ...
        FIXED = ...

    class FILTER(enum.IntFlag):

        NO = ...
        NONE = ...
        SUB = ...
        UP = ...
        AVG = ...
        PAETH = ...
        FAST = ...
        ALL = ...

    class COLOR_PRIMARIES(enum.IntEnum):

        BT709 = ...
        SRGB = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        GENERIC_FILM = ...
        BT2020 = ...
        BT2100 = ...
        XYZ = ...
        SMPTE431 = ...
        SMPTE432 = ...
        DCI_P3 = ...
        EBU3213 = ...

    class TRANSFER_CHARACTERISTICS(enum.IntEnum):

        BT709 = ...
        UNSPECIFIED = ...
        BT470M = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        LINEAR = ...
        LOG100 = ...
        LOG100_SQRT10 = ...
        IEC61966 = ...
        BT1361 = ...
        SRGB = ...
        BT2020_10BIT = ...
        BT2020_12BIT = ...
        PQ = ...
        SMPTE428 = ...
        HLG = ...

    class MATRIX_COEFFICIENTS(enum.IntEnum):

        IDENTITY = ...
        BT709 = ...
        UNSPECIFIED = ...
        FCC = ...
        BT470BG = ...
        BT601 = ...
        SMPTE240 = ...
        YCGCO = ...
        BT2020_NCL = ...
        BT2020_CL = ...
        SMPTE2085 = ...
        CHROMA_DERIVED_NCL = ...
        CHROMA_DERIVED_CL = ...
        ICTCP = ...
        YCGCO_RE = ...
        YCGCO_RO = ...

class PngError(RuntimeError): ...

def png_version() -> str: ...
def png_check(data: Buffer, /) -> bool | None: ...
def png_encode(
    data: ArrayLike,
    /,
    level: PNG.COMPRESSION | int | None = None,
    *,
    strategy: EnumLike[PNG.STRATEGY] | None = None,
    filter: EnumLike[PNG.FILTER] | None = None,
    primaries: EnumLike[PNG.COLOR_PRIMARIES] | None = None,
    transfer: EnumLike[PNG.TRANSFER_CHARACTERISTICS] | None = None,
    matrix: EnumLike[PNG.MATRIX_COEFFICIENTS] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def png_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def png_info(data: Buffer, /) -> dict[str, Any]: ...

class QOI:

    available: bool

    class COLORSPACE(enum.IntEnum):

        SRGB = ...
        LINEAR = ...

class QoiError(RuntimeError): ...

def qoi_version() -> str: ...
def qoi_check(data: Buffer, /) -> bool | None: ...
def qoi_encode(
    data: ArrayLike,
    /,
    *,
    colorspace: EnumLike[QOI.COLORSPACE] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def qoi_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def qoi_info(data: Buffer, /) -> dict[str, Any]: ...

class QUANTIZE:

    available: bool

    class MODE(enum.IntEnum):

        NOQUANTIZE = ...
        BITGROOM = ...
        GRANULARBR = ...
        GBR = ...  # alias
        BITROUND = ...
        SCALE = ...

class QuantizeError(RuntimeError): ...

def quantize_version() -> str: ...
def quantize_check(data: Buffer, /) -> bool | None: ...
def quantize_encode(
    data: NDArray[Any],
    /,
    mode: (
        QUANTIZE.MODE
        | Literal[
            'noquantize', 'bitgroom', 'granularbr', 'gbr', 'bitround', 'scale'
        ]
    ),
    nsd: int,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def quantize_decode(
    data: NDArray[Any],
    /,
    mode: (
        QUANTIZE.MODE
        | Literal[
            'noquantize', 'bitgroom', 'granularbr', 'gbr', 'bitround', 'scale'
        ]
    ),
    nsd: int,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def quantize_info(data: Buffer, /) -> NoReturn: ...

class RCOMP:

    available: bool

class RcompError(RuntimeError): ...

def rcomp_version() -> str: ...
def rcomp_check(data: Buffer, /) -> bool | None: ...
def rcomp_encode(
    data: ArrayLike,
    /,
    *,
    nblock: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def rcomp_decode(
    data: Buffer,
    /,
    shape: Sequence[int] | int | None = None,
    dtype: DTypeLike | None = None,
    *,
    nblock: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def rcomp_info(data: Buffer, /) -> NoReturn: ...

class RGBE:

    available: bool

    class ORIENT(enum.IntFlag):
        IDENTITY = ...
        FLIP_X = ...
        FLIP_Y = ...
        TRANSPOSE = ...

class RgbeError(RuntimeError): ...

def rgbe_version() -> str: ...
def rgbe_check(data: Buffer, /) -> bool | None: ...
def rgbe_info(
    data: Buffer,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
) -> dict[str, Any]: ...
@overload
def rgbe_encode(
    data: ArrayLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def rgbe_encode(
    data: ArrayLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: NDArray[Any],
) -> NDArray[Any]: ...
def rgbe_decode(
    data: Buffer,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class SNAPPY:

    available: bool

    class COMPRESSION_LEVEL(enum.IntEnum):
        DEFAULT = ...
        MIN = ...
        MAX = ...

class SnappyError(RuntimeError): ...

def snappy_version() -> str: ...
def snappy_check(data: Buffer, /) -> bool | None: ...
def snappy_encode(
    data: Buffer,
    /,
    level: SNAPPY.COMPRESSION_LEVEL | int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def snappy_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def snappy_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def snappy_info(data: Buffer, /) -> dict[str, Any]: ...

class SPERR:

    available: bool

    class MODE(enum.IntEnum):

        BPP = ...
        PSNR = ...
        PWE = ...

class SperrError(RuntimeError): ...

def sperr_version() -> str: ...
def sperr_check(data: Buffer, /) -> bool | None: ...
def sperr_encode(
    data: ArrayLike,
    /,
    level: float,
    mode: SPERR.MODE | Literal['bpp', 'psnr', 'pwe'],
    *,
    chunks: tuple[int, int, int] | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def sperr_decode(
    data: Buffer,
    /,
    *,
    shape: Sequence[int] | int | None = None,
    dtype: DTypeLike | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def sperr_info(data: Buffer, /) -> dict[str, Any]: ...

class SPNG:

    available: bool

    class COLOR_TYPE(enum.IntEnum):

        GRAY = ...
        GRAY_ALPHA = ...
        RGB = ...
        RGB_ALPHA = ...
        PALETTE = ...

    class FMT(enum.IntEnum):

        RGBA8 = ...
        RGBA16 = ...
        RGB8 = ...
        GA8 = ...
        GA16 = ...
        G8 = ...

    class FILTER(enum.IntEnum):

        NO = ...
        NONE = ...
        SUB = ...
        UP = ...
        AVG = ...
        PAETH = ...
        FAST = ...
        ALL = ...

class SpngError(RuntimeError): ...

def spng_version() -> str: ...
def spng_check(data: Buffer, /) -> bool | None: ...
def spng_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    filter: EnumLike[SPNG.FILTER] | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def spng_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def spng_info(data: Buffer, /) -> dict[str, Any]: ...

class SZ3:

    available: bool

    class MODE(enum.IntEnum):

        ABS = ...
        REL = ...
        ABS_AND_REL = ...
        ABS_OR_REL = ...
        # PSNR = ...
        # NORM = ...
        # PW_REL = ...
        # ABS_AND_PW_REL = ...
        # ABS_OR_PW_REL = ...
        # REL_AND_PW_REL = ...
        # REL_OR_PW_REL = ...

class Sz3Error(RuntimeError): ...

def sz3_version() -> str: ...
def sz3_check(data: Buffer, /) -> bool | None: ...
def sz3_encode(
    data: ArrayLike,
    /,
    mode: EnumLike[SZ3.MODE] | None = None,
    abs: float | None = None,
    rel: float | None = None,
    # pwr: float | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
def sz3_decode(
    data: Buffer,
    /,
    shape: Sequence[int] | int,
    dtype: DTypeLike,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def sz3_info(data: Buffer, /) -> NoReturn: ...

class SZIP:

    available: bool

    class OPTION_MASK(enum.IntEnum):

        ALLOW_K13 = ...
        CHIP = ...
        EC = ...
        LSB = ...
        MSB = ...
        NN = ...
        RAW = ...

class SzipError(RuntimeError): ...

def szip_version() -> str: ...
def szip_check(data: Buffer, /) -> bool | None: ...
def szip_encode(
    data: Buffer,
    /,
    options_mask: SZIP.OPTION_MASK | int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def szip_decode(
    data: Buffer,
    /,
    options_mask: int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def szip_decode(
    data: Buffer,
    /,
    options_mask: int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def szip_params(
    data: NDArray[Any], /, options_mask: int = 4, pixels_per_block: int = 32
) -> dict[str, int]: ...
def szip_info(data: Buffer, /) -> NoReturn: ...

class TGA:

    available: bool

class TgaError(RuntimeError): ...

def tga_version() -> str: ...
def tga_check(data: Buffer, /) -> bool | None: ...
def tga_encode(
    data: ArrayLike,
    /,
    *,
    rle: bool = False,
    out: BytesOut | None = None,
) -> BytesResult: ...
def tga_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def tga_info(data: Buffer, /) -> dict[str, Any]: ...

class TIFF:

    available: bool

    class VERSION(enum.IntEnum):

        CLASSIC = ...
        BIG = ...

    class ENDIAN(enum.IntEnum):

        BIG = ...
        LITTLE = ...

    class COMPRESSION(enum.IntEnum):

        NONE = ...
        CCITTRLE = ...
        CCITTFAX3 = ...
        CCITTFAX4 = ...
        LZW = ...
        JPEG = ...
        PACKBITS = ...
        DEFLATE = ...
        ADOBE_DEFLATE = ...
        LZMA = ...
        ZSTD = ...
        WEBP = ...
        LERC = ...
        PIXARLOG = ...
        # JXL = ...

    class PHOTOMETRIC(enum.IntEnum):

        MINISWHITE = ...
        MINISBLACK = ...
        RGB = ...
        PALETTE = ...
        MASK = ...
        SEPARATED = ...
        YCBCR = ...

    class PLANARCONFIG(enum.IntEnum):

        CONTIG = ...
        SEPARATE = ...

    class PREDICTOR(enum.IntEnum):

        NONE = ...
        HORIZONTAL = ...
        FLOATINGPOINT = ...

    class EXTRASAMPLE(enum.IntEnum):

        UNSPECIFIED = ...
        ASSOCALPHA = ...
        UNASSALPHA = ...

    class RESUNIT(enum.IntEnum):

        NONE = ...
        INCH = ...
        CENTIMETER = ...

    class ORIENTATION(enum.IntEnum):

        TOPLEFT = ...
        TOPRIGHT = ...
        BOTRIGHT = ...
        BOTLEFT = ...
        LEFTTOP = ...
        RIGHTTOP = ...
        RIGHTBOT = ...
        LEFTBOT = ...

    class FILETYPE(enum.IntFlag):

        REDUCEDIMAGE = ...
        PAGE = ...
        MASK = ...

    class SUBCODEC(enum.IntEnum):

        NONE = ...
        DEFLATE = ...
        ZSTD = ...

class TiffError(RuntimeError): ...

def tiff_version() -> str: ...
def tiff_check(data: Buffer, /) -> bool | None: ...
def tiff_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bigtiff: bool | None = None,
    byteorder: EnumLike[TIFF.ENDIAN] | None = None,
    subfiletype: EnumLike[TIFF.FILETYPE] | None = None,
    photometric: EnumLike[TIFF.PHOTOMETRIC] | None = None,
    planarconfig: EnumLike[TIFF.PLANARCONFIG] | None = None,
    extrasample: EnumLike[TIFF.EXTRASAMPLE] | None = None,
    # volumetric: bool = False,
    tile: tuple[int, int] | None = None,
    rowsperstrip: int | None = None,
    bitspersample: int | None = None,
    compression: EnumLike[TIFF.COMPRESSION] | None = None,
    subcodec: EnumLike[TIFF.SUBCODEC] | None = None,
    predictor: bool | EnumLike[TIFF.PREDICTOR] | None = None,
    colormap: ArrayLike | None = None,
    iccprofile: Buffer | None = None,
    resolution: tuple[float, float] | None = None,
    resolutionunit: EnumLike[TIFF.RESUNIT] | None = None,
    description: str | None = None,
    datetime: str | None = None,
    software: str | None = None,
    verbose: int | None = None,
    appendto: bytes | bytearray | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def tiff_decode(
    data: Buffer,
    /,
    index: int | Sequence[int] | slice | None = 0,
    *,
    asrgb: bool = False,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def tiff_info(
    data: Buffer,
    /,
    index: int | Sequence[int] | slice | None = 0,
    *,
    asrgb: bool = False,
    verbose: int | None = None,
) -> dict[str, Any]: ...

class ULTRAHDR:

    available: bool

    class CG(enum.IntEnum):

        UNSPECIFIED = ...
        BT_709 = ...
        DISPLAY_P3 = ...
        BT_2100 = ...

    class CT(enum.IntEnum):

        UNSPECIFIED = ...
        LINEAR = ...
        HLG = ...
        PQ = ...
        SRGB = ...

    class CR(enum.IntEnum):

        UNSPECIFIED = ...
        LIMITED_RANGE = ...
        FULL_RANGE = ...

    class CODEC(enum.IntEnum):

        JPEG = ...
        HEIF = ...
        AVIF = ...

    class USAGE(enum.IntEnum):

        REALTIME = ...
        QUALITY = ...

class UltrahdrError(RuntimeError): ...

def ultrahdr_version() -> str: ...
def ultrahdr_check(data: Buffer, /) -> bool | None: ...
def ultrahdr_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    sdr: ArrayLike | None = None,
    scale: int | None = None,
    gamut: EnumLike[ULTRAHDR.CG] | None = None,
    transfer: EnumLike[ULTRAHDR.CT] | None = None,
    nits: float | None = None,
    boostmin: float | None = None,
    boostmax: float | None = None,
    crange: EnumLike[ULTRAHDR.CR] | None = None,
    usage: EnumLike[ULTRAHDR.USAGE] | None = None,
    codec: EnumLike[ULTRAHDR.CODEC] | None = None,
    gainmap_gamma: float | None = None,
    gainmap_level: int | None = None,
    multichannel: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def ultrahdr_decode(
    data: Buffer,
    /,
    *,
    dtype: DTypeLike | None = None,
    transfer: EnumLike[ULTRAHDR.CT] | None = None,
    boost: float | None = None,
    gpu: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def ultrahdr_decode_jpeg(
    data: Buffer,
    /,
) -> tuple[bytes, bytes, dict[str, Any]]: ...
def ultrahdr_encode_jpeg(
    base_jpeg: Buffer,
    gainmap_jpeg: Buffer,
    /,
    *,
    gamut: EnumLike[ULTRAHDR.CG] | None = None,
    gainmap_gamma: tuple[float, float, float] | float | None = None,
    boostmin: tuple[float, float, float] | float | None = None,
    boostmax: tuple[float, float, float] | float | None = None,
    offset_sdr: tuple[float, float, float] | float | None = None,
    offset_hdr: tuple[float, float, float] | float | None = None,
    hdr_capacity_min: float | None = None,
    hdr_capacity_max: float | None = None,
    use_base_cg: bool | None = None,
) -> bytes: ...
def ultrahdr_info(
    data: Buffer,
    /,
    *,
    dtype: DTypeLike | None = None,
    transfer: EnumLike[ULTRAHDR.CT] | None = None,
    boost: float | None = None,
) -> dict[str, Any]: ...

class WAVPACK:

    available: bool

    class LEVEL(enum.IntEnum):

        FAST = ...
        DEFAULT = ...
        HIGH = ...
        VERY_HIGH = ...

class WavpackError(RuntimeError): ...

def wavpack_version() -> str: ...
def wavpack_check(data: Buffer, /) -> bool | None: ...
def wavpack_info(
    data: Buffer, /, *, correction: Buffer | None = None
) -> dict[str, Any]: ...
@overload
def wavpack_encode(
    data: ArrayLike,
    /,
    level: WAVPACK.LEVEL | int | None = None,
    *,
    bitrate: float | None = None,
    correction: Literal[False] = False,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def wavpack_encode(
    data: ArrayLike,
    /,
    level: WAVPACK.LEVEL | int | None = None,
    *,
    bitrate: float | None = None,
    correction: Literal[True],
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> tuple[BytesResult, bytes]: ...
def wavpack_decode(
    data: Buffer,
    /,
    *,
    correction: Buffer | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class WEBP:

    available: bool

class WebpError(RuntimeError): ...

def webp_version() -> str: ...
def webp_check(data: Buffer, /) -> bool | None: ...
def webp_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    lossless: bool | None = None,
    method: int | None = None,
    delay: int | None = None,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def webp_decode(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    hasalpha: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def webp_info(
    data: Buffer,
    /,
    index: int | None = None,
    *,
    hasalpha: bool | None = None,
) -> dict[str, Any]: ...

class WIC:

    available: bool

    class FORMAT(enum.IntEnum):

        BMP = ...
        PNG = ...
        JPEG = ...
        TIFF = ...
        GIF = ...
        WMP = ...
        HEIF = ...
        WEBP = ...

class WicError(RuntimeError): ...

def wic_version() -> str: ...
def wic_check(data: Buffer, /) -> bool | None: ...
def wic_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    format: WIC.FORMAT | str | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def wic_decode(
    data: Buffer,
    /,
    index: int = 0,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def wic_info(data: Buffer, /, index: int = 0) -> dict[str, Any]: ...

class XOR:

    available: bool

XorError = ImcdError

xor_version = imcd_version

def xor_check(data: Any, /) -> bool | None: ...
@overload
def xor_encode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def xor_encode(
    data: Buffer,
    /,
    *,
    axis: int = -1,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def xor_decode(  # type: ignore[overload-overlap]
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def xor_decode(
    data: Buffer,
    /,
    *,
    axis: int = -1,
    out: BytesOut | None = None,
) -> BytesResult: ...
def xor_info(data: Buffer, /) -> NoReturn: ...

class ZFP:

    available: bool

    class EXEC(enum.IntEnum):

        SERIAL = ...
        OMP = ...
        CUDA = ...

    class MODE(enum.IntEnum):

        NONE = ...
        EXPERT = ...
        FIXED_RATE = ...
        FIXED_PRECISION = ...
        FIXED_ACCURACY = ...
        REVERSIBLE = ...
        RATE = ...
        PRECISION = ...
        ACCURACY = ...

    class HEADER(enum.IntEnum):

        MAGIC = ...
        META = ...
        MODE = ...
        FULL = ...

class ZfpError(RuntimeError): ...

def zfp_version() -> str: ...
def zfp_check(data: Buffer, /) -> bool | None: ...
def zfp_encode(
    data: ArrayLike,
    /,
    level: float | tuple[int, int, int, int] | None = None,
    *,
    mode: EnumLike[ZFP.MODE] | None = None,
    execution: EnumLike[ZFP.EXEC] | None = None,
    chunksize: int | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
def zfp_decode(
    data: Buffer,
    /,
    *,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    strides: Sequence[int] | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def zfp_info(
    data: Buffer,
    /,
    *,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    strides: Sequence[int] | None = None,
) -> dict[str, Any]: ...

class ZLIB:

    available: bool

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...

    class STRATEGY(enum.IntEnum):

        DEFAULT = ...
        FILTERED = ...
        HUFFMAN_ONLY = ...
        RLE = ...
        FIXED = ...

class ZlibError(RuntimeError): ...

def zlib_version() -> str: ...
def zlib_check(data: Buffer, /) -> bool | None: ...
def zlib_encode(
    data: Buffer,
    /,
    level: ZLIB.COMPRESSION | int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zlib_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zlib_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def zlib_crc32(data: Buffer, /, value: int | None = None) -> int: ...
def zlib_adler32(data: Buffer, /, value: int | None = None) -> int: ...
def zlib_info(data: Buffer, /) -> NoReturn: ...

class ZLIBNG:

    available: bool

    class COMPRESSION(enum.IntEnum):

        DEFAULT = ...
        NO = ...
        BEST = ...
        SPEED = ...

    class STRATEGY(enum.IntEnum):

        DEFAULT = ...
        FILTERED = ...
        HUFFMAN_ONLY = ...
        RLE = ...
        FIXED = ...

class ZlibngError(RuntimeError): ...

def zlibng_version() -> str: ...
def zlibng_check(data: Buffer, /) -> bool | None: ...
def zlibng_encode(
    data: Buffer,
    /,
    level: ZLIBNG.COMPRESSION | int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zlibng_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zlibng_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def zlibng_crc32(data: Buffer, /, value: int | None = None) -> int: ...
def zlibng_adler32(data: Buffer, /, value: int | None = None) -> int: ...
def zlibng_info(data: Buffer, /) -> NoReturn: ...

class ZOPFLI:

    available: bool

    class FORMAT(enum.IntEnum):

        GZIP = ...
        ZLIB = ...
        DEFLATE = ...

class ZopfliError(RuntimeError): ...

def zopfli_version() -> str: ...

zopfli_check = zlib_check

def zopfli_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    format: EnumLike[ZOPFLI.FORMAT] | None = None,
    blocksplitting: bool | None = None,
    blocksplittingmax: int | None = None,
    verbose: bool | None = None,
    verbose_more: bool | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...

zopfli_decode = zlib_decode

def zopfli_info(data: Buffer, /) -> NoReturn: ...

class ZSTD:

    available: bool

class ZstdError(RuntimeError): ...

def zstd_version() -> str: ...
def zstd_check(data: Buffer, /) -> bool | None: ...
def zstd_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zstd_decode(
    data: Buffer,
    /,
    *,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zstd_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def zstd_info(data: Buffer, /) -> dict[str, Any]: ...

class ZSTD1:
    available: bool

class Zstd1Error(RuntimeError): ...

def zstd1_version() -> str: ...
def zstd1_check(data: Buffer, /) -> bool | None: ...
def zstd1_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    itemsize: int = 1,
    hilo: bool = False,
    numthreads: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zstd1_decode(
    data: Buffer,
    /,
    *,
    itemsize: int = 1,
    samples: int = 1,
    out: BytesOut | None = None,
) -> BytesResult: ...
@overload
def zstd1_decode(
    data: Buffer,
    /,
    *,
    itemsize: int = 1,
    samples: int = 1,
    out: NDArray[Any],
) -> NDArray[numpy.uint8]: ...
def zstd1_info(data: Buffer, /) -> NoReturn: ...
