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
# This interface document is updated manually and considered experimental.

import enum
import mmap
import os
from collections.abc import Callable, Sequence
from typing import IO, Any, Literal, NoReturn, TypeAlias, overload

from numpy.typing import ArrayLike, DTypeLike, NDArray

BytesLike: TypeAlias = bytes | bytearray | mmap.mmap

__all__ = [
    'AEC',
    'APNG',
    'AVIF',
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
    'CMS',
    'DDS',
    'DEFLATE',
    'DELTA',
    'DICOMRLE',
    'EER',
    'FLOAT24',
    'FLOATPRED',
    'GIF',
    'GZIP',
    'H5CHECKSUM',
    'HEIF',
    'HTJ2K',
    'JETRAW',
    'JPEG',
    'JPEG2K',
    'JPEG8',
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
    'PACKBITS',
    'PACKINTS',
    'PCODEC',
    'PGLZ',
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
    'TIFF',
    'ULTRAHDR',
    'WEBP',
    'XOR',
    'ZFP',
    'ZLIB',
    'ZLIBNG',
    'ZOPFLI',
    'ZSTD',
    'AecError',
    'ApngError',
    'AvifError',
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
    'CmsError',
    'DdsError',
    'DeflateError',
    'DelayedImportError',
    'DeltaError',
    'DicomrleError',
    'EerError',
    'Float24Error',
    'FloatpredError',
    'GifError',
    'GzipError',
    'HeifError',
    'Htj2kError',
    'JetrawError',
    'Jpeg2kError',
    'Jpeg8Error',
    'JpegError',
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
    'PackbitsError',
    'PackintsError',
    'PcodecError',
    'PglzError',
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
    'TiffError',
    'UltrahdrError',
    'WebpError',
    'XorError',
    'ZfpError',
    'ZlibError',
    'ZlibngError',
    'ZopfliError',
    'ZstdError',
    '__version__',
    'aec_check',
    'aec_decode',
    'aec_encode',
    'aec_version',
    'apng_check',
    'apng_decode',
    'apng_encode',
    'apng_version',
    'avif_check',
    'avif_decode',
    'avif_encode',
    'avif_version',
    'bcn_check',
    'bcn_decode',
    'bcn_encode',
    'bcn_version',
    'bfloat16_check',
    'bfloat16_decode',
    'bfloat16_encode',
    'bfloat16_version',
    'bitorder_check',
    'bitorder_decode',
    'bitorder_encode',
    'bitorder_version',
    'bitshuffle_check',
    'bitshuffle_decode',
    'bitshuffle_encode',
    'bitshuffle_version',
    'blosc2_check',
    'blosc2_decode',
    'blosc2_encode',
    'blosc2_version',
    'blosc_check',
    'blosc_decode',
    'blosc_encode',
    'blosc_version',
    'bmp_check',
    'bmp_decode',
    'bmp_encode',
    'bmp_version',
    'brotli_check',
    'brotli_decode',
    'brotli_encode',
    'brotli_version',
    'brunsli_check',
    'brunsli_decode',
    'brunsli_encode',
    'brunsli_version',
    'byteshuffle_check',
    'byteshuffle_decode',
    'byteshuffle_encode',
    'byteshuffle_version',
    'bz2_check',
    'bz2_decode',
    'bz2_encode',
    'bz2_version',
    'cms_check',
    'cms_decode',
    'cms_encode',
    'cms_profile',
    'cms_profile_validate',
    'cms_transform',
    'cms_version',
    'cython_version',
    'dds_check',
    'dds_decode',
    'dds_encode',
    'dds_version',
    'deflate_adler32',
    'deflate_check',
    'deflate_crc32',
    'deflate_decode',
    'deflate_encode',
    'deflate_version',
    'delta_check',
    'delta_decode',
    'delta_encode',
    'delta_version',
    'dicomrle_check',
    'dicomrle_decode',
    'dicomrle_encode',
    'dicomrle_version',
    'eer_check',
    'eer_decode',
    'eer_encode',
    'eer_version',
    'float24_check',
    'float24_decode',
    'float24_encode',
    'float24_version',
    'floatpred_check',
    'floatpred_decode',
    'floatpred_encode',
    'floatpred_version',
    'gif_check',
    'gif_decode',
    'gif_encode',
    'gif_version',
    'gzip_check',
    'gzip_decode',
    'gzip_encode',
    'gzip_version',
    'h5checksum_crc',
    'h5checksum_fletcher32',
    'h5checksum_hash_string',
    'h5checksum_lookup3',
    'h5checksum_metadata',
    'h5checksum_version',
    'heif_check',
    'heif_decode',
    'heif_encode',
    'heif_version',
    'htj2k_check',
    'htj2k_decode',
    'htj2k_encode',
    'htj2k_init',
    'htj2k_version',
    'imagefileext',
    'imcd_version',
    'imread',
    'imwrite',
    'jetraw_check',
    'jetraw_decode',
    'jetraw_encode',
    'jetraw_init',
    'jetraw_version',
    'jpeg2k_check',
    'jpeg2k_decode',
    'jpeg2k_encode',
    'jpeg2k_version',
    'jpeg8_check',
    'jpeg8_decode',
    'jpeg8_encode',
    'jpeg8_version',
    'jpeg_check',
    'jpeg_decode',
    'jpeg_encode',
    'jpeg_version',
    'jpegls_check',
    'jpegls_decode',
    'jpegls_encode',
    'jpegls_version',
    'jpegsof3_check',
    'jpegsof3_decode',
    'jpegsof3_encode',
    'jpegsof3_version',
    'jpegxl_check',
    'jpegxl_decode',
    'jpegxl_decode_jpeg',
    'jpegxl_encode',
    'jpegxl_encode_jpeg',
    'jpegxl_version',
    'jpegxr_check',
    'jpegxr_decode',
    'jpegxr_encode',
    'jpegxr_version',
    'jpegxs_check',
    'jpegxs_decode',
    'jpegxs_encode',
    'jpegxs_version',
    'lerc_check',
    'lerc_decode',
    'lerc_encode',
    'lerc_version',
    'ljpeg_check',
    'ljpeg_decode',
    'ljpeg_encode',
    'ljpeg_version',
    'lz4_check',
    'lz4_decode',
    'lz4_encode',
    'lz4_version',
    'lz4f_check',
    'lz4f_decode',
    'lz4f_encode',
    'lz4f_version',
    'lz4h5_check',
    'lz4h5_decode',
    'lz4h5_encode',
    'lz4h5_version',
    'lzf_check',
    'lzf_decode',
    'lzf_encode',
    'lzf_version',
    'lzfse_check',
    'lzfse_decode',
    'lzfse_encode',
    'lzfse_version',
    'lzham_check',
    'lzham_decode',
    'lzham_encode',
    'lzham_version',
    'lzma_check',
    'lzma_decode',
    'lzma_encode',
    'lzma_version',
    'lzo_check',
    'lzo_decode',
    'lzo_encode',
    'lzo_version',
    'lzw_check',
    'lzw_decode',
    'lzw_encode',
    'lzw_version',
    'meshopt_check',
    'meshopt_decode',
    'meshopt_encode',
    'meshopt_version',
    'mozjpeg_check',
    'mozjpeg_decode',
    'mozjpeg_encode',
    'mozjpeg_version',
    'none_check',
    'none_decode',
    'none_encode',
    'none_version',
    'numpy_abi_version',
    'numpy_check',
    'numpy_decode',
    'numpy_encode',
    'numpy_version',
    'packbits_check',
    'packbits_decode',
    'packbits_encode',
    'packbits_version',
    'packints_check',
    'packints_decode',
    'packints_encode',
    'packints_version',
    'pcodec_check',
    'pcodec_decode',
    'pcodec_encode',
    'pcodec_version',
    'pglz_check',
    'pglz_decode',
    'pglz_encode',
    'pglz_version',
    'png_check',
    'png_decode',
    'png_encode',
    'png_version',
    'qoi_check',
    'qoi_decode',
    'qoi_encode',
    'qoi_version',
    'quantize_check',
    'quantize_decode',
    'quantize_encode',
    'quantize_version',
    'rcomp_check',
    'rcomp_decode',
    'rcomp_encode',
    'rcomp_version',
    'rgbe_check',
    'rgbe_decode',
    'rgbe_encode',
    'rgbe_version',
    'snappy_check',
    'snappy_decode',
    'snappy_encode',
    'snappy_version',
    'sperr_check',
    'sperr_decode',
    'sperr_encode',
    'sperr_version',
    'spng_check',
    'spng_decode',
    'spng_encode',
    'spng_version',
    'sz3_check',
    'sz3_decode',
    'sz3_encode',
    'sz3_version',
    'szip_check',
    'szip_decode',
    'szip_encode',
    'szip_params',
    'szip_version',
    'tiff_check',
    'tiff_decode',
    'tiff_encode',
    'tiff_version',
    'ultrahdr_check',
    'ultrahdr_decode',
    'ultrahdr_encode',
    'ultrahdr_version',
    'version',
    'webp_check',
    'webp_decode',
    'webp_encode',
    'webp_version',
    'xor_check',
    'xor_decode',
    'xor_encode',
    'xor_version',
    'zfp_check',
    'zfp_decode',
    'zfp_encode',
    'zfp_version',
    'zlib_adler32',
    'zlib_check',
    'zlib_crc32',
    'zlib_decode',
    'zlib_encode',
    'zlib_version',
    'zlibng_adler32',
    'zlibng_check',
    'zlibng_crc32',
    'zlibng_decode',
    'zlibng_encode',
    'zlibng_version',
    'zopfli_check',
    'zopfli_decode',
    'zopfli_encode',
    'zopfli_version',
    'zstd_check',
    'zstd_decode',
    'zstd_encode',
    'zstd_version',
]
__version__: str

def __dir__() -> list[str]: ...
def __getattr__(name: str, /) -> Any: ...

class DelayedImportError(ImportError):
    def __init__(self, name: str, /) -> None: ...

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
@overload
def imread(
    fileobj: str | os.PathLike[Any] | bytes | mmap.mmap,
    /,
    codec: (
        str
        | Callable[..., NDArray[Any]]
        | list[str | Callable[..., NDArray[Any]]]
        | None
    ) = None,
    *,
    memmap: bool = False,
    return_codec: Literal[False] = ...,
    **kwargs: Any,
) -> NDArray[Any]: ...
@overload
def imread(
    fileobj: str | os.PathLike[Any] | bytes | mmap.mmap,
    /,
    codec: (
        str
        | Callable[..., NDArray[Any]]
        | list[str | Callable[..., NDArray[Any]]]
        | None
    ) = None,
    *,
    memmap: bool = False,
    return_codec: Literal[True],
    **kwargs: Any,
) -> tuple[NDArray[Any], Callable[..., NDArray[Any]]]: ...
@overload
def imread(
    fileobj: str | os.PathLike[Any] | bytes | mmap.mmap,
    /,
    codec: (
        str
        | Callable[..., NDArray[Any]]
        | list[str | Callable[..., NDArray[Any]]]
        | None
    ) = None,
    *,
    memmap: bool = False,
    return_codec: bool,
    **kwargs: Any,
) -> NDArray[Any] | tuple[NDArray[Any], Callable[..., NDArray[Any]]]: ...
def imwrite(
    fileobj: str | os.PathLike[Any] | IO[bytes],
    data: ArrayLike,
    /,
    codec: str | Callable[..., bytes | bytearray] | None = None,
    **kwargs: Any,
) -> None: ...
def imagefileext() -> tuple[str, ...]: ...
def cython_version() -> str: ...
def numpy_abi_version() -> str: ...
def imcd_version() -> str: ...

class ImcdError(RuntimeError): ...

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
def aec_check(data: BytesLike, /) -> bool | None: ...
def aec_encode(
    data: BytesLike | ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def aec_decode(
    data: BytesLike,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def aec_decode(
    data: BytesLike,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: NDArray[Any],
) -> NDArray[Any]: ...

class APNG:

    available: bool

    class COLOR_TYPE(enum.IntEnum):

        GRAY = ...
        GRAY_ALPHA = ...
        RGB = ...
        RGB_ALPHA = ...

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

    class FILTER(enum.IntEnum):  # IntFlag

        NO = ...
        NONE = ...
        SUB = ...
        UP = ...
        AVG = ...
        PAETH = ...
        FAST = ...
        ALL = ...

class ApngError(RuntimeError): ...

def apng_version() -> str: ...
def apng_check(data: BytesLike, /) -> bool | None: ...
def apng_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    strategy: APNG.STRATEGY | int | None = None,
    filter: APNG.FILTER | int | None = None,
    photometric: APNG.COLOR_TYPE | int | None = None,
    delay: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def apng_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def avif_check(data: BytesLike, /) -> bool | None: ...
def avif_encode(
    data: ArrayLike,
    /,
    level: AVIF.QUALITY | int | None = None,
    *,
    speed: AVIF.SPEED | int | None = None,
    tilelog2: tuple[int, int] | None = None,
    bitspersample: int | None = None,
    pixelformat: AVIF.PIXEL_FORMAT | int | str | None = None,
    codec: AVIF.CODEC_CHOICE | int | str | None = None,
    primaries: AVIF.COLOR_PRIMARIES | int | None = None,
    transfer: AVIF.TRANSFER_CHARACTERISTICS | int | None = None,
    matrix: AVIF.MATRIX_COEFFICIENTS | int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
def avif_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def bcn_check(data: BytesLike, /) -> bool | None: ...
def bcn_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def bcn_decode(
    data: BytesLike,
    /,
    format: BCN.FORMAT | int,
    *,
    shape: tuple[int, ...] | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class BFLOAT16:

    available: bool

    class ROUND(enum.IntEnum):

        TONEAREST = ...
        UPWARD = ...
        DOWNWARD = ...
        TOWARDZERO = ...

Bfloat16Error = ImcdError
bfloat16_version = imcd_version

def bfloat16_check(data: BytesLike, /) -> bool | None: ...
def bfloat16_encode(
    data: ArrayLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    rounding: BFLOAT16.ROUND | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def bfloat16_decode(
    data: BytesLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class BITORDER:

    available: bool

BitorderError = ImcdError
bitorder_version = imcd_version

def bitorder_check(data: BytesLike, /) -> bool | None: ...
@overload
def bitorder_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def bitorder_encode(
    data: NDArray[Any],
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

bitorder_decode = bitorder_encode

class BITSHUFFLE:

    available: bool

class BitshuffleError(RuntimeError): ...

def bitshuffle_version() -> str: ...
def bitshuffle_check(data: BytesLike, /) -> bool | None: ...
@overload
def bitshuffle_encode(
    data: BytesLike,
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
@overload
def bitshuffle_encode(
    data: NDArray[Any],
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def bitshuffle_decode(
    data: BytesLike,
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def bitshuffle_decode(
    data: NDArray[Any],
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def blosc_check(data: BytesLike, /) -> bool | None: ...
def blosc_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    compressor: BLOSC.COMPRESSOR | int | str | None = None,
    shuffle: BLOSC.SHUFFLE | int | str | None = None,
    typesize: int | None = None,
    blocksize: int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def blosc_decode(
    data: BytesLike,
    /,
    *,
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class BLOSC2:

    available: bool

    class FILTER(enum.IntEnum):

        NOFILTER = ...
        NOSHUFFLE = ...
        SHUFFLE = ...  # default
        BITSHUFFLE = ...
        DELTA = ...
        TRUNC_PREC = ...

    class COMPRESSOR(enum.IntEnum):

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

class Blosc2Error(RuntimeError): ...

def blosc2_version() -> str: ...
def blosc2_check(data: BytesLike, /) -> bool | None: ...
def blosc2_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    compressor: BLOSC2.COMPRESSOR | int | str | None = None,
    shuffle: BLOSC2.FILTER | int | str | None = None,
    splitmode: BLOSC2.SPLIT | int | str | None = None,
    typesize: int | None = None,
    blocksize: int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def blosc2_decode(
    data: BytesLike,
    /,
    *,
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class BMP:

    available: bool

class BmpError(RuntimeError): ...

def bmp_version() -> str: ...
def bmp_check(data: BytesLike, /) -> bool | None: ...
def bmp_encode(
    data: ArrayLike,
    /,
    *,
    ppm: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def bmp_decode(
    data: BytesLike,
    /,
    *,
    asrgb: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class BROTLI:

    available: bool

    class MODE(enum.IntEnum):

        GENERIC = ...
        TEXT = ...
        FONT = ...

class BrotliError(RuntimeError): ...

def brotli_version() -> str: ...
def brotli_check(data: BytesLike, /) -> bool | None: ...
def brotli_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    mode: BROTLI.MODE | int | None = None,
    lgwin: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def brotli_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class BRUNSLI:

    available: bool

class BrunsliError(RuntimeError): ...

def brunsli_version() -> str: ...
def brunsli_check(data: BytesLike, /) -> bool | None: ...
def brunsli_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    colorspace: str | int | None = None,
    outcolorspace: int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    predictor: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def brunsli_decode(
    data: BytesLike,
    /,
    *,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    asjpeg: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class BYTESHUFFLE:

    available: bool

ByteshuffleError = ImcdError
byteshuffle_version = imcd_version

def byteshuffle_check(data: BytesLike, /) -> bool | None: ...
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

class BZ2:

    available: bool

class Bz2Error(RuntimeError): ...

def bz2_version() -> str: ...
def bz2_check(data: BytesLike, /) -> bool | None: ...
def bz2_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def bz2_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

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
def cms_check(data: BytesLike, /) -> bool | None: ...
def cms_transform(
    data: ArrayLike,
    /,
    profile: bytes,
    outprofile: bytes,
    *,
    colorspace: str | None = None,
    planar: bool | None = None,
    outcolorspace: str | None = None,
    outplanar: bool | None = None,
    outdtype: DTypeLike | None = None,
    intent: CMS.INTENT | int | None = None,
    flags: CMS.FLAGS | int | None = None,
    verbose: bool | None = None,
    out: int | bytearray | None = None,
) -> NDArray[Any]: ...

cms_encode = cms_transform
cms_decode = cms_transform

def cms_profile(
    profile: str,
    /,
    *,
    whitepoint: Sequence[float] | None = None,
    primaries: Sequence[float] | None = None,
    transferfunction: ArrayLike | None = None,
    gamma: float | None = None,
) -> bytes: ...
def cms_profile_validate(
    profile: bytes,
    /,
    *,
    verbose: bool = False,
) -> None: ...

class DDS:

    available: bool

class DdsError(RuntimeError): ...

def dds_version() -> str: ...
def dds_check(data: BytesLike, /) -> bool | None: ...
def dds_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def dds_decode(
    data: BytesLike,
    /,
    *,
    mipmap: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class DEFLATE:

    available: bool

class DeflateError(RuntimeError): ...

def deflate_version() -> str: ...
def deflate_check(data: BytesLike, /) -> bool | None: ...
def deflate_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    raw: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def deflate_decode(
    data: BytesLike,
    /,
    *,
    raw: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
def deflate_crc32(data: BytesLike, /, value: int | None = None) -> int: ...
def deflate_adler32(data: BytesLike, /, value: int | None = None) -> int: ...

class DELTA:

    available: bool

DeltaError = ImcdError
delta_version = imcd_version

def delta_check(data: BytesLike, /) -> bool | None: ...
@overload
def delta_encode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
@overload
def delta_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def delta_decode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def delta_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class EER:

    available: bool

EerError = ImcdError
eer_version = imcd_version

def eer_check(data: BytesLike, /) -> bool | None: ...
def eer_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def eer_decode(
    data: BytesLike,
    /,
    shape: tuple[int, int],
    skipbits: int,
    horzbits: int,
    vertbits: int,
    *,
    superres: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class DICOMRLE:

    available: bool

DicomrleError = ImcdError
dicomrle_version = imcd_version

def dicomrle_check(data: BytesLike, /) -> bool | None: ...
def dicomrle_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def dicomrle_decode(
    data: BytesLike,
    /,
    dtype: DTypeLike,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class FLOAT24:

    available: bool

    class ROUND(enum.IntEnum):

        TONEAREST = ...
        UPWARD = ...
        DOWNWARD = ...
        TOWARDZERO = ...

Float24Error = ImcdError
float24_version = imcd_version

def float24_check(data: BytesLike, /) -> bool | None: ...
def float24_encode(
    data: ArrayLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    rounding: FLOAT24.ROUND | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def float24_decode(
    data: BytesLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class FLOATPRED:

    available: bool

FloatpredError = ImcdError
floatpred_version = imcd_version

def floatpred_check(data: BytesLike, /) -> bool | None: ...
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

class GIF:

    available: bool

class GifError(RuntimeError): ...

def gif_version() -> str: ...
def gif_check(data: BytesLike, /) -> bool | None: ...
def gif_encode(
    data: ArrayLike,
    /,
    *,
    colormap: ArrayLike | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def gif_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    asrgb: bool = True,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class GZIP:

    available: bool

GzipError = DeflateError
gzip_version = deflate_version

def gzip_check(data: BytesLike, /) -> bool | None: ...
def gzip_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def gzip_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class H5CHECKSUM:

    available: bool

def h5checksum_version() -> str: ...
def h5checksum_fletcher32(
    data: BytesLike, /, value: int | None = None
) -> int: ...
def h5checksum_lookup3(
    data: BytesLike, /, value: int | None = None
) -> int: ...
def h5checksum_crc(data: BytesLike, /, value: int | None = None) -> int: ...
def h5checksum_metadata(
    data: BytesLike, /, value: int | None = None
) -> int: ...
def h5checksum_hash_string(
    data: BytesLike, /, value: int | None = None
) -> int: ...

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

    class COLORSPACE(enum.IntEnum):

        UNDEFINED = ...
        YCBCR = ...
        RGB = ...
        MONOCHROME = ...

class HeifError(RuntimeError): ...

def heif_version() -> str: ...
def heif_check(data: BytesLike, /) -> bool | None: ...
def heif_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bitspersample: int | None = None,
    photometric: HEIF.COLORSPACE | int | str | None = None,
    compression: HEIF.COMPRESSION | int | str | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def heif_decode(
    data: BytesLike,
    /,
    index: int | None = 0,
    *,
    photometric: HEIF.COLORSPACE | int | str | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class HTJ2K:

    available: bool

class Htj2kError(RuntimeError): ...

def htj2k_init(
    *,
    verbose: int | None = None,
) -> None: ...
def htj2k_version() -> str: ...
def htj2k_check(data: BytesLike, /) -> bool | None: ...
def htj2k_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    rgb: bool | None = None,
    planar: bool | None = None,
    tile: tuple[int, int] | None = None,
    resolutions: int | None = None,
    reversible: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def htj2k_decode(
    data: BytesLike,
    /,
    *,
    planar: bool | None = None,
    skipres: int | tuple[int, int] | None = None,
    resilient: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class JETRAW:

    available: bool

class JetrawError(RuntimeError): ...

def jetraw_init(
    parameters: str | None = None,
    *,
    verbose: int | None = None,
) -> None: ...
def jetraw_version() -> str: ...
def jetraw_check(data: BytesLike, /) -> bool | None: ...
def jetraw_encode(
    data: ArrayLike,
    /,
    identifier: str,
    *,
    errorbound: float | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jetraw_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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

class Jpeg2kError(RuntimeError): ...

def jpeg2k_version() -> str: ...
def jpeg2k_check(data: BytesLike, /) -> bool | None: ...
def jpeg2k_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    codecformat: JPEG2K.CODEC | int | str | None = None,
    colorspace: JPEG2K.CLRSPC | int | str | None = None,
    planar: bool | None = None,
    tile: tuple[int, int] | None = None,
    bitspersample: int | None = None,
    resolutions: int | None = None,
    reversible: bool | None = None,
    mct: bool = True,
    verbose: int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpeg2k_decode(
    data: BytesLike,
    /,
    *,
    planar: bool | None = None,
    verbose: int | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class JPEG8:

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
        RGB565 = ...

class Jpeg8Error(RuntimeError): ...

def jpeg8_version() -> str: ...
def jpeg8_check(data: BytesLike, /) -> bool | None: ...
def jpeg8_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: JPEG8.CS | int | str | None = None,
    outcolorspace: JPEG8.CS | int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    lossless: bool | None = None,
    predictor: int | None = None,
    bitspersample: int | None = None,
    validate: bool | None = None,  # for compatibility
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpeg8_decode(
    data: BytesLike,
    /,
    *,
    tables: bytes | None = None,
    colorspace: JPEG8.CS | int | str | None = None,
    outcolorspace: JPEG8.CS | int | str | None = None,
    shape: tuple[int, int] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

JPEG = JPEG8

JpegError = Jpeg8Error

jpeg_version = jpeg8_version

jpeg_check = jpeg8_check

def jpeg_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: JPEG.CS | int | str | None = None,
    outcolorspace: JPEG.CS | int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    lossless: bool | None = None,
    predictor: int | None = None,
    bitspersample: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpeg_decode(
    data: BytesLike,
    /,
    *,
    tables: bytes | None = None,
    header: bytes | None = None,
    colorspace: JPEG.CS | int | str | None = None,
    outcolorspace: JPEG.CS | int | str | None = None,
    shape: tuple[int, int] | None = None,
    bitspersample: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class JPEGLS:

    available: bool

class JpeglsError(RuntimeError): ...

def jpegls_version() -> str: ...
def jpegls_check(data: BytesLike, /) -> bool | None: ...
def jpegls_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpegls_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class JPEGSOF3:

    available: bool

class Jpegsof3Error(RuntimeError): ...

def jpegsof3_version() -> str: ...
def jpegsof3_check(data: BytesLike, /) -> bool | None: ...
def jpegsof3_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def jpegsof3_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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

class JpegxlError(RuntimeError): ...

def jpegxl_version() -> str: ...
def jpegxl_check(data: BytesLike, /) -> bool | None: ...
def jpegxl_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    effort: int | None = None,
    distance: float | None = None,
    lossless: bool | None = None,
    decodingspeed: int | None = None,
    photometric: JPEGXL.COLOR_SPACE | int | str | None = None,
    bitspersample: int | None = None,
    # extrasamples: Sequence[JPEGXL.CHANNEL] | None = None,
    planar: bool | None = None,
    usecontainer: bool | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpegxl_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    keeporientation: bool | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
def jpegxl_encode_jpeg(
    data: BytesLike,
    /,
    usecontainer: bool | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpegxl_decode_jpeg(
    data: BytesLike,
    /,
    numthreads: int,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...

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

class JpegxrError(RuntimeError): ...

def jpegxr_version() -> str: ...
def jpegxr_check(data: BytesLike, /) -> bool | None: ...
def jpegxr_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    photometric: JPEGXR.PI | int | str | None = None,
    hasalpha: bool | None = None,
    resolution: tuple[float, float] | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpegxr_decode(
    data: BytesLike,
    /,
    *,
    fp2int: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class JPEGXS:

    available: bool

class JpegxsError(RuntimeError): ...

def jpegxs_version() -> str: ...
def jpegxs_check(data: BytesLike, /) -> bool | None: ...
def jpegxs_encode(
    data: ArrayLike,
    /,
    config: str | None = None,
    *,
    bitspersample: int | None = None,
    verbose: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def jpegxs_decode(
    data: BytesLike,
    /,
    *,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class LERC:

    available: bool

class LercError(RuntimeError): ...

def lerc_version() -> str: ...
def lerc_check(data: BytesLike, /) -> bool | None: ...
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
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
@overload
def lerc_decode(
    data: BytesLike,
    /,
    *,
    masks: Literal[False] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def lerc_decode(
    data: BytesLike,
    /,
    *,
    masks: Literal[True] | NDArray[Any],
    out: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]: ...

class LJPEG:

    available: bool

class LjpegError(RuntimeError): ...

def ljpeg_version() -> str: ...
def ljpeg_check(data: BytesLike, /) -> bool | None: ...
def ljpeg_encode(
    data: ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    delinearize: ArrayLike | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def ljpeg_decode(
    data: BytesLike,
    /,
    *,
    linearize: ArrayLike | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class LZ4:

    available: bool

    class CLEVEL(enum.IntEnum):

        DEFAULT = ...
        MIN = ...
        MAX = ...
        OPT_MIN = ...

class Lz4Error(RuntimeError): ...

def lz4_version() -> str: ...
def lz4_check(data: BytesLike, /) -> bool | None: ...
def lz4_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    hc: bool = False,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lz4_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZ4F:

    available: bool

    VERSION: int

class Lz4fError(RuntimeError): ...

def lz4f_version() -> str: ...
def lz4f_check(data: BytesLike, /) -> bool | None: ...
def lz4f_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    blocksizeid: int | None = None,
    contentchecksum: bool | None = None,
    blockchecksum: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lz4f_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZ4H5:

    available: bool

    CLEVEL: LZ4.CLEVEL

class Lz4h5Error(RuntimeError): ...

def lz4h5_version() -> str: ...
def lz4h5_check(data: BytesLike, /) -> bool | None: ...
def lz4h5_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    blocksize: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lz4h5_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZF:

    available: bool

class LzfError(RuntimeError): ...

def lzf_version() -> str: ...
def lzf_check(data: BytesLike, /) -> bool | None: ...
def lzf_encode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lzf_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZFSE:

    available: bool

class LzfseError(RuntimeError): ...

def lzfse_version() -> str: ...
def lzfse_check(data: BytesLike, /) -> bool | None: ...
def lzfse_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lzfse_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

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
def lzham_check(data: BytesLike, /) -> bool | None: ...
def lzham_encode(
    data: BytesLike,
    /,
    level: LZHAM.COMPRESSION | int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lzham_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZMA:

    available: bool

    class CHECK(enum.IntEnum):

        NONE = ...
        CRC32 = ...
        CRC64 = ...
        SHA256 = ...

class LzmaError(RuntimeError): ...

def lzma_version() -> str: ...
def lzma_check(data: BytesLike, /) -> bool | None: ...
def lzma_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    check: LZMA.CHECK | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lzma_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZO:

    available: bool

class LzoError(RuntimeError): ...

def lzo_version() -> str: ...
def lzo_check(data: BytesLike, /) -> bool | None: ...
def lzo_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def lzo_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class LZW:

    available: bool

LzwError = ImcdError

lzw_version = imcd_version

def lzw_check(data: BytesLike, /) -> bool | None: ...
def lzw_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def lzw_decode(
    data: BytesLike,
    /,
    *,
    buffersize: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class MESHOPT:

    available: bool

class MeshoptError(RuntimeError): ...

def meshopt_version() -> str: ...
def meshopt_check(data: BytesLike, /) -> bool | None: ...
def meshopt_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    items: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def meshopt_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    items: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

# TODO: MONO12P codec implementation pending
# class MONO12P:
#
#     available: bool
#
# Mono12pError = ImcdError
# mono12p_version = imcd_version
#
# def mono12p_check(data: BytesLike, /) -> bool | None: ...
#
# def mono12p_encode(
#     data: ArrayLike,
#     /,
#     msfirst: bool = False,
#     *,
#     axis: int = -1,
#     out: int | bytearray | None = None,
# ) -> bytes | bytearray: ...
#
#
# def mono12p_decode(
#     data: BytesLike,
#     /,
#     msfirst: bool = False,
#     *,
#     runlen: int = 0,
#     out: NDArray[Any] | None = None,
# ) -> NDArray[Any]: ...

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
        RGB565 = ...

class MozjpegError(RuntimeError): ...

def mozjpeg_version() -> str: ...
def mozjpeg_check(data: BytesLike, /) -> bool | None: ...
def mozjpeg_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    colorspace: MOZJPEG.CS | int | str | None = None,
    outcolorspace: MOZJPEG.CS | int | str | None = None,
    subsampling: str | tuple[int, int] | None = None,
    optimize: bool | None = None,
    smoothing: bool | None = None,
    notrellis: bool | None = None,
    quanttable: int | None = None,
    progressive: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def mozjpeg_decode(
    data: BytesLike,
    /,
    *,
    tables: bytes | None = None,
    colorspace: MOZJPEG.CS | int | str | None = None,
    outcolorspace: MOZJPEG.CS | int | str | None = None,
    shape: tuple[int, int] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class NONE:

    available: bool

NoneError = RuntimeError

def none_version() -> str: ...
def none_check(data: Any, /) -> bool | None: ...
def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any: ...
def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any: ...

class NUMPY:

    available: bool

NumpyError = RuntimeError

def numpy_version() -> str: ...
def numpy_check(data: BytesLike, /) -> bool | None: ...
def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes: ...
def numpy_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]: ...

class PACKBITS:

    available: bool

PackbitsError = ImcdError

packbits_version = imcd_version

def packbits_check(data: BytesLike, /) -> bool | None: ...
def packbits_encode(
    data: BytesLike | ArrayLike,
    /,
    *,
    axis: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def packbits_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class PACKINTS:

    available: bool

PackintsError = ImcdError

packints_version = imcd_version

def packints_check(data: BytesLike, /) -> bool | None: ...
def packints_encode(
    data: ArrayLike,
    bitspersample: int,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def packints_decode(
    data: BytesLike,
    dtype: DTypeLike,
    bitspersample: int,
    /,
    *,
    runlen: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class PCODEC:

    available: bool

class PcodecError(RuntimeError): ...

def pcodec_version() -> str: ...
def pcodec_check(data: BytesLike, /) -> bool | None: ...
def pcodec_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def pcodec_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class PGLZ:

    available: bool

class PglzError(RuntimeError): ...

def pglz_version() -> str: ...
def pglz_check(data: BytesLike, /) -> bool | None: ...
def pglz_encode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    strategy: str | tuple[int, int, int, int, int, int] | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def pglz_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    checkcomplete: bool | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class PNG:

    available: bool

    class COLOR_TYPE(enum.IntEnum):

        GRAY = ...
        GRAY_ALPHA = ...
        RGB = ...
        RGB_ALPHA = ...

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

    class FILTER(enum.IntEnum):  # IntFlag

        NO = ...
        NONE = ...
        SUB = ...
        UP = ...
        AVG = ...
        PAETH = ...
        FAST = ...
        ALL = ...

class PngError(RuntimeError): ...

def png_version() -> str: ...
def png_check(data: BytesLike, /) -> bool | None: ...
def png_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    strategy: PNG.STRATEGY | int | None = None,
    filter: PNG.FILTER | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def png_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class QOI:

    available: bool

    class COLORSPACE(enum.IntEnum):

        SRGB = ...
        LINEAR = ...

class QoiError(RuntimeError): ...

def qoi_version() -> str: ...
def qoi_check(data: BytesLike, /) -> bool | None: ...
def qoi_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def qoi_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class QUANTIZE:

    available: bool

    class MODE(enum.IntEnum):

        NOQUANTIZE = ...
        BITGROOM = ...
        GRANULARBR = ...
        BITROUND = ...
        SCALE = ...

class QuantizeError(RuntimeError): ...

def quantize_version() -> str: ...
def quantize_check(data: BytesLike, /) -> bool | None: ...
def quantize_encode(
    data: NDArray[Any],
    /,
    mode: (
        QUANTIZE.MODE
        | Literal['bitgroom', 'granularbr', 'gbr', 'bitround', 'scale']
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
        | Literal['bitgroom', 'granularbr', 'gbr', 'bitround', 'scale']
    ),
    nsd: int,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class RCOMP:

    available: bool

class RcompError(RuntimeError): ...

def rcomp_version() -> str: ...
def rcomp_check(data: BytesLike, /) -> bool | None: ...
def rcomp_encode(
    data: ArrayLike,
    /,
    *,
    nblock: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def rcomp_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    nblock: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class RGBE:

    available: bool

class RgbeError(RuntimeError): ...

def rgbe_version() -> str: ...
def rgbe_check(data: BytesLike, /) -> bool | None: ...
def rgbe_encode(
    data: ArrayLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def rgbe_decode(
    data: BytesLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class SNAPPY:

    available: bool

class SnappyError(RuntimeError): ...

def snappy_version() -> str: ...
def snappy_check(data: BytesLike, /) -> bool | None: ...
def snappy_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def snappy_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...

class SPERR:

    available: bool

    class MODE(enum.IntEnum):

        BPP = ...
        PSNR = ...
        PWE = ...

class SperrError(RuntimeError): ...

def sperr_version() -> str: ...
def sperr_check(data: BytesLike, /) -> bool | None: ...
def sperr_encode(
    data: ArrayLike,
    /,
    level: float,
    mode: SPERR.MODE | Literal['bpp', 'psnr', 'pwe'],
    *,
    chunks: tuple[int, int, int] | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def sperr_decode(
    data: BytesLike,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class SPNG:

    available: bool

    class FMT(enum.IntEnum):

        RGBA8 = ...
        RGBA16 = ...
        RGB8 = ...
        GA8 = ...
        GA16 = ...
        G8 = ...

class SpngError(RuntimeError): ...

def spng_version() -> str: ...
def spng_check(data: BytesLike, /) -> bool | None: ...
def spng_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def spng_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def sz3_check(data: BytesLike, /) -> bool | None: ...
def sz3_encode(
    data: ArrayLike,
    /,
    mode: SZ3.MODE | int | str | None = None,
    abs: float | None = None,
    rel: float | None = None,
    # pwr: float | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def sz3_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def szip_check(data: BytesLike, /) -> bool | None: ...
def szip_encode(
    data: BytesLike,
    /,
    options_mask: SZIP.OPTION_MASK | int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def szip_decode(
    data: BytesLike,
    /,
    options_mask: int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
def szip_params(
    data: NDArray[Any], /, options_mask: int = 4, pixels_per_block: int = 32
) -> dict[str, int]: ...

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
        LZW = ...
        JPEG = ...
        PACKBITS = ...
        DEFLATE = ...
        ADOBE_DEFLATE = ...
        LZMA = ...
        ZSTD = ...
        WEBP = ...
        # LERC = ...
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

class TiffError(RuntimeError): ...

def tiff_version() -> str: ...
def tiff_check(data: BytesLike, /) -> bool | None: ...
def tiff_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bigtiff: bool | None = None,
    append: bool | None = None,
    photometric: TIFF.PHOTOMETRIC | int | None = None,
    planarconfig: TIFF.PLANARCONFIG | int | None = None,
    extrasamples: tuple[TIFF.EXTRASAMPLE | int, ...] | None = None,
    # volumetric: bool = False,
    tile: tuple[int, int] | None = None,
    rowsperstrip: int | None = None,
    bitspersample: int | None = None,
    compression: TIFF.COMPRESSION | int | None = None,
    predictor: TIFF.PREDICTOR | int | None = None,
    # colormap: NDArray[Any] | None = None,
    description: str | None = None,
    datetime: str | None = None,
    resolution: tuple[float, float] | None = None,
    subfiletype: int = 0,
    software: str | None = None,
    verbose: bool | None = None,
    out: int | bytearray | None = None,
) -> NoReturn: ...
def tiff_decode(
    data: BytesLike,
    /,
    index: int | Sequence[int] | slice | None = 0,
    *,
    asrgb: bool = False,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def ultrahdr_check(data: BytesLike, /) -> bool | None: ...
def ultrahdr_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    scale: int | None = None,
    gamut: ULTRAHDR.CG | int | None = None,
    transfer: ULTRAHDR.CT | int | None = None,
    nits: float | None = None,
    crange: ULTRAHDR.CR | int | None = None,
    usage: ULTRAHDR.USAGE | int | None = None,
    codec: ULTRAHDR.CODEC | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def ultrahdr_decode(
    data: BytesLike,
    /,
    *,
    dtype: DTypeLike | None = None,
    transfer: ULTRAHDR.CT | int | None = None,
    boost: float | None = None,
    gpu: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class WEBP:

    available: bool

class WebpError(RuntimeError): ...

def webp_version() -> str: ...
def webp_check(data: BytesLike, /) -> bool | None: ...
def webp_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    lossless: bool | None = None,
    method: int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def webp_decode(
    data: BytesLike,
    /,
    index: int | None = 0,
    *,
    hasalpha: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

class XOR:

    available: bool

XorError = ImcdError

xor_version = imcd_version

def xor_check(data: Any, /) -> bool | None: ...
@overload
def xor_encode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
@overload
def xor_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...
@overload
def xor_decode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
@overload
def xor_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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

    class HEADER(enum.IntEnum):

        MAGIC = ...
        META = ...
        MODE = ...
        FULL = ...

class ZfpError(RuntimeError): ...

def zfp_version() -> str: ...
def zfp_check(data: BytesLike, /) -> bool | None: ...
def zfp_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    mode: ZFP.MODE | int | str | None = None,
    execution: ZFP.EXEC | int | str | None = None,
    chunksize: int | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def zfp_decode(
    data: BytesLike,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    strides: tuple[int, ...] | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]: ...

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
def zlib_check(data: BytesLike, /) -> bool | None: ...
def zlib_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def zlib_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
def zlib_crc32(data: BytesLike, /, value: int | None = None) -> int: ...
def zlib_adler32(data: BytesLike, /, value: int | None = None) -> int: ...

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
def zlibng_check(data: BytesLike, /) -> bool | None: ...
def zlibng_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def zlibng_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
def zlibng_crc32(data: BytesLike, /, value: int | None = None) -> int: ...
def zlibng_adler32(data: BytesLike, /, value: int | None = None) -> int: ...

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
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
    **kwargs: Any,
) -> bytes | bytearray: ...

zopfli_decode = zlib_decode

class ZSTD:

    available: bool

class ZstdError(RuntimeError): ...

def zstd_version() -> str: ...
def zstd_check(data: BytesLike, /) -> bool | None: ...
def zstd_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray: ...
def zstd_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray: ...
