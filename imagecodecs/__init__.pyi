# imagecodecs/__init__.pyi

# Copyright (c) 2023-2024, Christoph Gohlke
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
# Requires Python 3.10 and numpy 1.20.

"""Image transformation, compression, and decompression codecs."""

import enum
import mmap
import os
from collections.abc import Sequence
from typing import IO, Any, Callable, Literal, overload

from numpy.typing import ArrayLike, DTypeLike, NDArray

BytesLike = bytes | bytearray | mmap.mmap

__version__: str


def __dir__() -> list[str]: ...


def __getattr__(name: str, /) -> Any: ...


class DelayedImportError(ImportError):
    """Delayed ImportError."""

    def __init__(self, name: str, /) -> None: ...


def version(
    astype: type | None = None,
) -> str:  # | tuple[str, ...] | dict[str, str]: ...
    """Return version information about all codecs and dependencies.

    All extension modules are imported into the process.
    """


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
) -> NDArray[Any]:
    """Return image array from file."""


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
) -> tuple[NDArray[Any], Callable[..., NDArray[Any]]]:
    """Return image array and decode function from file."""


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
) -> NDArray[Any] | tuple[NDArray[Any], Callable[..., NDArray[Any]]]:
    """Return image array and decode function from file."""


def imwrite(
    fileobj: str | os.PathLike[Any] | IO[bytes],
    data: ArrayLike,
    /,
    codec: str | Callable[..., bytes | bytearray] | None = None,
    **kwargs: Any,
) -> None:
    """Write image array to file."""


def imagefileext() -> list[str]:
    """Return list of image file extensions handled by imread and imwrite."""


def cython_version() -> str:
    """Return Cython version string."""


def numpy_abi_version() -> str:
    """Return Numpy ABI version string."""


def imcd_version() -> str:
    """Return imcd library version string."""


class ImcdError(RuntimeError):
    """IMCD codec exceptions."""


class AEC:
    """AEC codec constants."""

    available: bool
    """AEC codec is available."""

    class FLAG(enum.IntEnum):
        """AEC codec flags."""

        DATA_SIGNED: int
        DATA_3BYTE: int
        DATA_PREPROCESS: int
        RESTRICTED: int
        PAD_RSI: int
        NOT_ENFORCE: int


class AecError(RuntimeError):
    """AEC codec exceptions."""


def aec_version() -> str:
    """Return libaec library version string."""


def aec_check(data: BytesLike, /) -> None:
    """Return whether data is AEC encoded."""


def aec_encode(
    data: BytesLike | ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    flags: int | None = None,
    blocksize: int | None = None,
    rsi: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return AEC encoded data."""


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
) -> bytes | bytearray:
    """Return decoded AEC data."""


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
) -> NDArray[Any]:
    """Return decoded AEC data."""


class APNG:
    """APNG codec constants."""

    available: bool
    """APNG codec is available."""

    class COLOR_TYPE(enum.IntEnum):
        """APNG codec color types."""

        GRAY: int
        GRAY_ALPHA: int
        RGB: int
        RGB_ALPHA: int

    class COMPRESSION(enum.IntEnum):
        """APNG codec compression levels."""

        DEFAULT: int
        NO: int
        BEST: int
        SPEED: int

    class STRATEGY(enum.IntEnum):
        """APNG codec strategies."""

        DEFAULT: int
        FILTERED: int
        HUFFMAN_ONLY: int
        RLE: int
        FIXED: int

    class FILTER(enum.IntEnum):  # IntFlag
        """APNG codec filters."""

        NO: int
        NONE: int
        SUB: int
        UP: int
        AVG: int
        PAETH: int
        FAST: int
        ALL: int


class ApngError(RuntimeError):
    """APNG codec exceptions."""


def apng_version() -> str:
    """Return libpng-apng library version string."""


def apng_check(data: BytesLike, /) -> bool:
    """Return whether data is APNG encoded image."""


def apng_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    strategy: int | None = None,
    filter: int | None = None,
    photometric: int | None = None,
    delay: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return APNG encoded image."""


def apng_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded APNG image."""


class AVIF:
    """AVIF codec constants."""

    available: bool
    """AVIF codec is available."""

    class PIXEL_FORMAT(enum.IntEnum):
        """AVIF codec pixel formats."""

        NONE: int
        YUV444: int
        YUV422: int
        YUV420: int
        YUV400: int

    class QUALITY(enum.IntEnum):
        """AVIF codec quality."""

        DEFAULT: int
        LOSSLESS: int
        WORST: int
        BEST: int

    class SPEED(enum.IntEnum):
        """AVIF codec speeds."""

        DEFAULT: int
        SLOWEST: int
        FASTEST: int

    class CHROMA_UPSAMPLING(enum.IntEnum):
        """AVIF codec chroma upsampling types."""

        AUTOMATIC: int
        FASTEST: int
        BEST_QUALITY: int
        NEAREST: int
        BILINEAR: int

    class CODEC_CHOICE(enum.IntEnum):
        """AVIF codec choices."""

        AUTO: int
        AOM: int
        DAV1D: int
        LIBGAV1: int
        RAV1E: int
        SVT: int
        AVM: int


class AvifError(RuntimeError):
    """AVIF codec exceptions."""


def avif_version() -> str:
    """Return libavif library version string."""


def avif_check(data: BytesLike, /) -> bool | None:
    """Return whether data is AVIF encoded image."""


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
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return AVIF encoded image."""


def avif_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded AVIF image."""


class BCN:
    """BCn codec constants."""

    available: bool
    """BCn codec is available."""

    class FORMAT(enum.IntEnum):
        """BCn compression format."""

        BC1 = 1  # DXT1
        BC2 = 2  # DXT3
        BC3 = 3  # DXT5
        BC4 = 4  # BC4_UNORM
        BC5 = 5  # BC5_UNORM
        BC6HU = 6  # BC6H_UF16
        BC6HS = -6  # BC6H_SF16
        BC7 = 7  # BC7_UNORM


class BcnError(RuntimeError):
    """BCn codec exceptions."""


def bcn_version() -> str:
    """Return bcdec library version string."""


def bcn_check(data: BytesLike, /) -> None:
    """Return whether data is BCn encoded."""


def bcn_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return BCn encoded data (not implemented)."""


def bcn_decode(
    data: BytesLike,
    format: BCN.FORMAT | int,
    /,
    shape: tuple[int, ...] | None = None,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BCn data."""


class BITORDER:
    """BITORDER codec constants."""

    available: bool
    """BITORDER codec is available."""


BitorderError = ImcdError
bitorder_version = imcd_version


def bitorder_check(data: BytesLike, /) -> None:
    """Return whether data is BITORDER encoded."""


@overload
def bitorder_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return data with reversed bit-order in each byte."""


@overload
def bitorder_encode(
    data: NDArray[Any],
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return data with reversed bit-order in each byte."""


bitorder_decode = bitorder_encode


class BITSHUFFLE:
    """BITSHUFFLE codec constants."""

    available: bool
    """BITSHUFFLE codec is available."""


class BitshuffleError(RuntimeError):
    """BITSHUFFLE codec exceptions."""


def bitshuffle_version() -> str:
    """Return Bitshuffle library version string."""


def bitshuffle_check(data: BytesLike, /) -> bool | None:
    """Return whether data is BITSHUFFLE encoded."""


@overload
def bitshuffle_encode(
    data: BytesLike,
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return BITSHUFFLE encoded data."""


@overload
def bitshuffle_encode(
    data: NDArray[Any],
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return BITSHUFFLE encoded data."""


@overload
def bitshuffle_decode(
    data: BytesLike,
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BITSHUFFLE data."""


@overload
def bitshuffle_decode(
    data: NDArray[Any],
    /,
    *,
    itemsize: int = 1,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> bytes | bytearray:
    """Return decoded BITSHUFFLE data."""


class BLOSC:
    """BLOSC codec constants."""

    available: bool
    """BLOSC codec is available."""

    class SHUFFLE(enum.IntEnum):
        """BLOSC codec shuffle types."""

        NOSHUFFLE: int
        SHUFFLE: int
        BITSHUFFLE: int

    class COMPRESSOR(enum.IntEnum):
        """BLOSC codec compressors."""

        BLOSCLZ: int
        LZ4: int
        LZ4HC: int
        SNAPPY: int
        ZLIB: int
        ZSTD: int


class BloscError(RuntimeError):
    """BLOSC coec exceptions."""


def blosc_version() -> str:
    """Return C-Blosc library version string."""


def blosc_check(data: BytesLike, /) -> None:
    """Return whether data is BLOSC encoded."""


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
) -> bytes | bytearray:
    """Return BLOSC encoded data."""


def blosc_decode(
    data: BytesLike,
    /,
    *,
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BLOSC data."""


class BLOSC2:
    """BLOSC2 codec constants."""

    available: bool
    """BLOSC2 codec is available."""

    class FILTER(enum.IntEnum):
        """BLOSC2 codec filters."""

        NOFILTER: int
        NOSHUFFLE: int
        SHUFFLE: int  # default
        BITSHUFFLE: int
        DELTA: int
        TRUNC_PREC: int

    class COMPRESSOR(enum.IntEnum):
        """BLOSC2 codec compressors."""

        BLOSCLZ: int
        LZ4: int
        LZ4HC: int
        ZLIB: int
        ZSTD: int  # default

    class SPLIT(enum.IntEnum):
        """BLOSC2 split modes."""

        ALWAYS: int  # default
        NEVER: int
        AUTO: int
        FORWARD_COMPAT: int


class Blosc2Error(RuntimeError):
    """BLOSC2 codec exceptions."""


def blosc2_version() -> str:
    """Return C-Blosc2 library version string."""


def blosc2_check(data: BytesLike, /) -> None:
    """Return whether data is BLOSC2 encoded."""


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
) -> bytes | bytearray:
    """Return decoded BLOSC2 data."""


def blosc2_decode(
    data: BytesLike,
    /,
    *,
    numthreads: int | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return BLOSC2 encoded data."""


class BMP:
    """BMP codec constants."""

    available: bool
    """BMP codec is available."""


class BmpError(RuntimeError):
    """BMP codec exceptions."""


def bmp_version() -> str:
    """Return EasyBMP library version string."""


def bmp_check(data: BytesLike, /) -> bool:
    """Return whether data is BMP encoded image."""


def bmp_encode(
    data: ArrayLike,
    /,
    *,
    ppm: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return BMP encoded image."""


def bmp_decode(
    data: BytesLike,
    /,
    *,
    asrgb: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded BMP image."""


class BROTLI:
    """BROTLI codec constants."""

    available: bool
    """BROTLI codec is available."""

    class MODE(enum.IntEnum):
        """BROTLI codec modes."""

        GENERIC: int
        TEXT: int
        FONT: int


class BrotliError(RuntimeError):
    """BROTLI codec exceptions."""


def brotli_version() -> str:
    """Return Brotli library version string."""


def brotli_check(data: BytesLike, /) -> None:
    """Return whether data is BROTLI encoded."""


def brotli_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    mode: BROTLI.MODE | int | None = None,
    lgwin: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return BROTLI encoded data."""


def brotli_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BROTLI data."""


class BRUNSLI:
    """BRUNSLI codec constants."""

    available: bool
    """BRUNSLI codec is available."""


class BrunsliError(RuntimeError):
    """BRUNSLI codec exceptions."""


def brunsli_version() -> str:
    """Return Brunsli library version string."""


def brunsli_check(data: BytesLike, /) -> bool | None:
    """Return whether data is BRUNSLI/JPEG encoded."""


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
) -> bytes | bytearray:
    """Return BRUNSLI/JPEG encoded image."""


def brunsli_decode(
    data: BytesLike,
    /,
    *,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    asjpeg: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BRUNSLI/JPEG image."""


class BYTESHUFFLE:
    """BYTESHUFFLE codec constants."""

    available: bool
    """BYTESHUFFLE codec is available."""


ByteshuffleError = ImcdError
byteshuffle_version = imcd_version


def byteshuffle_check(data: BytesLike, /) -> None:
    """Return whether data is BYTESHUFFLE encoded."""


def byteshuffle_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    delta: bool = False,
    reorder: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return byte-shuffled data."""


def byteshuffle_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    delta: bool = False,
    reorder: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return un-byte-shuffled data."""


class BZ2:
    """BZ2 codec constants."""

    available: bool
    """BZ2 codec is available."""


class Bz2Error(RuntimeError):
    """BZ2 codec exceptions."""


def bz2_version() -> str:
    """Return libbzip2 library version string."""


def bz2_check(data: BytesLike, /) -> None:
    """Return whether data is BZ2 encoded."""


def bz2_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return BZ2 encoded data."""


def bz2_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded BZ2 data."""


class CMS:
    """CMS codec constants."""

    available: bool
    """CMS codec is available."""

    class INTENT(enum.IntEnum):
        """CMS codec intent types."""

        PERCEPTUAL: int
        RELATIVE_COLORIMETRIC: int
        SATURATION: int
        ABSOLUTE_COLORIMETRIC: int

    class FLAGS(enum.IntEnum):
        """CMS codec flags."""

        NOCACHE: int
        NOOPTIMIZE: int
        NULLTRANSFORM: int
        GAMUTCHECK: int
        SOFTPROOFING: int
        BLACKPOINTCOMPENSATION: int
        NOWHITEONWHITEFIXUP: int
        HIGHRESPRECALC: int
        LOWRESPRECALC: int
        EIGHTBITS_DEVICELINK: int
        GUESSDEVICECLASS: int
        KEEP_SEQUENCE: int
        FORCE_CLUT: int
        CLUT_POST_LINEARIZATION: int
        CLUT_PRE_LINEARIZATION: int
        NONEGATIVES: int
        COPY_ALPHA: int
        NODEFAULTRESOURCEDEF: int

    class PT(enum.IntEnum):
        """CMS codec pixel types."""

        GRAY: int
        RGB: int
        CMY: int
        CMYK: int
        YCBCR: int
        YUV: int
        XYZ: int
        LAB: int
        YUVK: int
        HSV: int
        HLS: int
        YXY: int
        MCH1: int
        MCH2: int
        MCH3: int
        MCH4: int
        MCH5: int
        MCH6: int
        MCH7: int
        MCH8: int
        MCH9: int
        MCH10: int
        MCH11: int
        MCH12: int
        MCH13: int
        MCH14: int
        MCH15: int


class CmsError(RuntimeError):
    """CMS codec exceptions."""


def cms_version() -> str:
    """Return Little-CMS library version string."""


def cms_check(data: BytesLike, /) -> bool:
    """Return whether data is ICC profile."""


def cms_transform(
    data: ArrayLike,
    profile: bytes,
    outprofile: bytes,
    /,
    *,
    colorspace: str | None = None,
    planar: bool | None = None,
    outcolorspace: str | None = None,
    outplanar: bool | None = None,
    outdtype: DTypeLike | None = None,
    intent: int | None = None,
    flags: int | None = None,
    verbose: bool | None = None,
    out: int | bytearray | None = None,
) -> NDArray[Any]:
    """Return color-transformed array (experimental)."""


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
) -> bytes:
    """Return ICC profile."""


def cms_profile_validate(
    profile: bytes,
    /,
    *,
    verbose: bool = False,
) -> None:
    """Raise CmsError if ICC profile is invalid."""


class DDS:
    """DDS codec constants."""

    available: bool
    """DDS codec is available."""


class DdsError(RuntimeError):
    """DDS codec exceptions."""


def dds_version() -> str:
    """Return bcdec library version string."""


def dds_check(data: BytesLike, /) -> bool | None:
    """Return whether data is DDS encoded."""


def dds_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DDS encoded data (not implemented)."""


def dds_decode(
    data: BytesLike,
    /,
    *,
    mipmap: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DDS data."""


class DEFLATE:
    """DEFLATE codec constants."""

    available: bool
    """DEFLATE codec is available."""


class DeflateError(RuntimeError):
    """DEFLATE codec exceptions."""


def deflate_version() -> str:
    """Return libdeflate library version string."""


def deflate_check(data: BytesLike, /) -> bool | None:
    """Return whether data is Zlib/Deflate encoded."""


def deflate_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    raw: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DEFLATE encoded data."""


def deflate_decode(
    data: BytesLike,
    /,
    *,
    raw: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DEFLATE data."""


def deflate_crc32(data: BytesLike, /, value: int | None = None) -> int:
    """Return CRC32 checksum of data."""


def deflate_adler32(data: BytesLike, /, value: int | None = None) -> int:
    """Return Adler-32 checksum of data."""


class DELTA:
    """DELTA codec constants."""

    available: bool
    """DELTA codec is available."""


DeltaError = ImcdError
delta_version = imcd_version


def delta_check(data: BytesLike, /) -> None:
    """Return whether data is DELTA encoded."""


@overload
def delta_encode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DELTA encoded data."""


@overload
def delta_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return DELTA encoded data."""


@overload
def delta_decode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DELTA data."""


@overload
def delta_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded DELTA data."""


class EER:
    """EER codec constants."""

    available: bool
    """EER codec is available."""


EerError = ImcdError
eer_version = imcd_version


def eer_check(data: BytesLike, /) -> None:
    """Return whether data is EER encoded."""


def eer_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> None:
    """Return EER encoded image (not implemented)."""


def eer_decode(
    data: BytesLike,
    /,
    shape: tuple[int, int],
    rlebits: int,
    horzbits: int,
    vertbits: int,
    *,
    superres: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded EER image."""


class DICOMRLE:
    """DICOMRLE codec constants."""

    available: bool
    """DICOMRLE codec is available."""


DicomrleError = ImcdError
dicomrle_version = imcd_version


def dicomrle_check(data: BytesLike, /) -> bool:
    """Return whether data is DICOMRLE encoded."""


def dicomrle_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DICOMRLE encoded data (not implemented)."""
    raise NotImplementedError('dicomrle_encode')


def dicomrle_decode(
    data: BytesLike,
    /,
    dtype: DTypeLike,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DICOMRLE data."""


class FLOAT24:
    """FLOAT24 codec constants."""

    available: bool
    """FLOAT24 codec is available."""

    class ROUND(enum.IntEnum):
        """FLOAT24 codec rounding types."""

        TONEAREST: int
        UPWARD: int
        DOWNWARD: int
        TOWARDZERO: int


Float24Error = ImcdError
float24_version = imcd_version


def float24_check(data: BytesLike, /) -> bool | None:
    """Return whether data is FLOAT24 encoded."""


def float24_encode(
    data: ArrayLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    rounding: FLOAT24.ROUND | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return FLOAT24 encoded array."""


def float24_decode(
    data: BytesLike,
    /,
    *,
    byteorder: Literal['>', '<', '='] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded FLOAT24 array."""


class FLOATPRED:
    """FLOATPRED codec constants."""

    available: bool
    """FLOATPRED codec is available."""


FloatpredError = ImcdError
floatpred_version = imcd_version


def floatpred_check(data: BytesLike, /) -> bool | None:
    """Return whether data is FLOATPRED encoded."""


def floatpred_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return floating-point predicted array."""


def floatpred_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return un-predicted floating-point array."""


class GIF:
    """GIF codec constants."""

    available: bool
    """GIF codec is available."""


class GifError(RuntimeError):
    """GIF codec exceptions."""


def gif_version() -> str:
    """Return giflib library version string."""


def gif_check(data: BytesLike, /) -> bool | None:
    """Return whether data is GIF encoded image."""


def gif_encode(
    data: ArrayLike,
    /,
    *,
    colormap: ArrayLike | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return GIF encoded image."""


def gif_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    asrgb: bool = True,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded GIF image."""


class GZIP:
    """GZIP codec constants."""

    available: bool
    """GZIP codec is available."""


GzipError = DeflateError
gzip_version = deflate_version


def gzip_check(data: BytesLike, /) -> bool:
    """Return whether data is GZIP encoded."""


def gzip_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return GZIP encoded data."""


def gzip_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded GZIP data."""


class H5CHECKSUM:
    """H5checksum codec constants."""

    available: bool
    """H5checksum codec is available."""


def h5checksum_version() -> str:
    """Return h5checksum library version string."""


def h5checksum_fletcher32(data: BytesLike, /, value: int | None = None) -> int:
    """Return fletcher32 checksum of data (value is ignored)."""


def h5checksum_lookup3(data: BytesLike, /, value: int | None = None) -> int:
    """Return Jenkins lookup3 checksum of data."""


def h5checksum_crc(data: BytesLike, /, value: int | None = None) -> int:
    """Return crc checksum of data (value is ignored)."""


def h5checksum_metadata(data: BytesLike, /, value: int | None = None) -> int:
    """Return checksum of metadata."""


def h5checksum_hash_string(
    data: BytesLike, /, value: int | None = None
) -> int:
    """Return hash of bytes string (value is ignored)."""


class HEIF:
    """HEIF codec constants."""

    available: bool
    """HEIF codec is available."""

    class COMPRESSION(enum.IntEnum):
        """HEIF codec compression levels."""

        UNDEFINED: int
        HEVC: int
        AVC: int
        JPEG: int
        AV1: int
        # VVC
        # EVC
        # JPEG2000
        # UNCOMPRESSED

    class COLORSPACE(enum.IntEnum):
        """HEIF codec color spaces."""

        UNDEFINED: int
        YCBCR: int
        RGB: int
        MONOCHROME: int


class HeifError(RuntimeError):
    """HEIF codec exceptions."""


def heif_version() -> str:
    """Return libheif library version string."""


def heif_check(data: BytesLike, /) -> bool | None:
    """Return whether data is HEIF encoded image."""


def heif_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    bitspersample: int | None = None,
    photometric: HEIF.COLORSPACE | int | str | None = None,
    compression: HEIF.COMPRESSION | int | str | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return HEIF encoded image."""


def heif_decode(
    data: BytesLike,
    /,
    index: int | None = 0,
    *,
    photometric: HEIF.COLORSPACE | int | str | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded HEIF image."""


class JETRAW:
    """JETRAW codec constants."""

    available: bool
    """JETRAW codec is available."""


class JetrawError(RuntimeError):
    """JETRAW codec exceptions."""


def jetraw_version() -> str:
    """Return Jetraw library version string."""


def jetraw_check(data: BytesLike, /) -> None:
    """Return whether data is JETRAW encoded image."""


def jetraw_encode(
    data: ArrayLike,
    /,
    identifier: str,
    *,
    errorbound: float | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JETRAW encoded image."""


def jetraw_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JETRAW image."""


def jetraw_init(
    parameters: str | None = None,
    *,
    verbose: int | None = None,
) -> None:
    """Initialize JETRAW codec."""


class JPEG2K:
    """JPEG2K codec constants."""

    available: bool
    """JPEG2K codec is available."""

    class CODEC(enum.IntEnum):
        """JPEG2K codec file formats."""

        JP2: int
        J2K: int
        # JPT: int
        # JPP: int
        # JPX: int

    class CLRSPC(enum.IntEnum):
        """JPEG2K codec color spaces."""

        UNSPECIFIED: int
        SRGB: int
        GRAY: int
        SYCC: int
        EYCC: int
        CMYK: int


class Jpeg2kError(RuntimeError):
    """JPEG2K codec exceptions."""


def jpeg2k_version() -> str:
    """Return OpenJPEG library version string."""


def jpeg2k_check(data: BytesLike, /) -> bool | None:
    """Return whether data is JPEG 2000 encoded image."""


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
) -> bytes | bytearray:
    """Return JPEG 2000 encoded image."""


def jpeg2k_decode(
    data: BytesLike,
    /,
    *,
    planar: bool | None = None,
    verbose: int | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEG 2000 image."""


class JPEG8:
    """JPEG8 codec constants."""

    available: bool
    """JPEG8 codec is available."""

    legacy: bool
    """JPEG8 codec is not linked to libjpeg-turbo 3."""

    all_precisions: bool
    """JPEG8 codec supports all precisions from 2 to 16-bit."""

    class CS(enum.IntEnum):
        """JPEG8 codec color spaces."""

        UNKNOWN: int
        GRAYSCALE: int
        RGB: int
        YCbCr: int
        CMYK: int
        YCCK: int
        EXT_RGB: int
        EXT_RGBX: int
        EXT_BGR: int
        EXT_BGRX: int
        EXT_XBGR: int
        EXT_XRGB: int
        EXT_RGBA: int
        EXT_BGRA: int
        EXT_ABGR: int
        EXT_ARGB: int
        RGB565: int


class Jpeg8Error(RuntimeError):
    """JPEG8 codec exceptions."""


def jpeg8_version() -> str:
    """Return libjpeg-turbo library version string."""


def jpeg8_check(data: BytesLike, /) -> bool:
    """Return whether data is JPEG encoded image."""


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
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEG encoded image."""


def jpeg8_decode(
    data: BytesLike,
    /,
    *,
    tables: bytes | None = None,
    colorspace: JPEG8.CS | int | str | None = None,
    outcolorspace: JPEG8.CS | int | str | None = None,
    shape: tuple[int, int] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEG image."""


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
) -> bytes | bytearray:
    """Return JPEG encoded image."""


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
) -> NDArray[Any]:
    """Return decoded JPEG image."""


class JPEGLS:
    """JPEGLS codec constants."""

    available: bool
    """JPEGLS codec is available."""


class JpeglsError(RuntimeError):
    """JPEGLS codec exceptions."""


def jpegls_version() -> str:
    """Return CharLS library version string."""


def jpegls_check(data: BytesLike, /) -> None:
    """Return whether data is JPEGLS encoded image."""


def jpegls_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEGLS encoded image."""


def jpegls_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEGLS image."""


class JPEGSOF3:
    """JPEGSOF3 codec constants."""

    available: bool
    """JPEGSOF3 codec is available."""


class Jpegsof3Error(RuntimeError):
    """JPEGSOF3 codec exceptions."""


def jpegsof3_version() -> str:
    """Return jpegsof3 library version string."""


def jpegsof3_check(data: BytesLike, /) -> None:
    """Return whether data is Lossless JPEG encoded image."""


def jpegsof3_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> None:
    """Return Lossless JPEG encoded image (not implemented)."""


def jpegsof3_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded Lossless JPEG image."""


class JPEGXL:
    """JPEGXL codec constants."""

    available: bool
    """JPEGXL codec is available."""

    class COLOR_SPACE(enum.IntEnum):
        """JPEGXL codec color spaces."""

        UNKNOWN: int
        RGB: int
        GRAY: int
        XYB: int

    class CHANNEL(enum.IntEnum):
        """JPEGXL codec channel types."""

        UNKNOWN: int
        ALPHA: int
        DEPTH: int
        SPOT_COLOR: int
        SELECTION_MASK: int
        BLACK: int
        CFA: int
        THERMAL: int
        OPTIONAL: int


class JpegxlError(RuntimeError):
    """JPEGXL codec exceptions."""


def jpegxl_version() -> str:
    """Return libjxl library version string."""


def jpegxl_check(data: BytesLike, /) -> bool:
    """Return whether data is JPEGXL encoded image."""


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
) -> bytes | bytearray:
    """Return JPEGXL encoded image."""


def jpegxl_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    keeporientation: bool | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEGXL image."""


def jpegxl_encode_jpeg(
    data: BytesLike,
    /,
    usecontainer: bool | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEGXL encoded image from JPEG stream."""


def jpegxl_decode_jpeg(
    data: BytesLike,
    /,
    numthreads: int,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEG encoded image from JPEG XL stream."""


class JPEGXR:
    """JPEGXR codec constants."""

    available: bool
    """JPEGXR codec is available."""

    class PI(enum.IntEnum):
        """JPEGXR codec photometric interpretations."""

        W0: int
        B0: int
        RGB: int
        RGBPalette: int
        TransparencyMask: int
        CMYK: int
        YCbCr: int
        CIELab: int
        NCH: int
        RGBE: int


class JpegxrError(RuntimeError):
    """JPEGXR codec exceptions."""


def jpegxr_version() -> str:
    """Return jxrlib library version string."""


def jpegxr_check(data: BytesLike, /) -> bool | None:
    """Return whether data is JPEGXR encoded image."""


def jpegxr_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    photometric: JPEGXR.PI | int | str | None = None,
    hasalpha: bool | None = None,
    resolution: tuple[float, float] | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEGXR encoded image."""


def jpegxr_decode(
    data: BytesLike,
    /,
    *,
    fp2int: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEGXR image."""


class JPEGXS:
    """JPEGXS codec constants."""

    available: bool
    """JPEGXS codec is available."""


class JpegxsError(RuntimeError):
    """JPEGXS codec exceptions."""


def jpegxs_version() -> str:
    """Return libjxs library version string."""


def jpegxs_check(data: BytesLike, /) -> bool:
    """Return whether data is JPEGXS encoded image."""


def jpegxs_encode(
    data: ArrayLike,
    /,
    config: str | None = None,
    *,
    bitspersample: int | None = None,
    verbose: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return JPEGXS encoded image."""


def jpegxs_decode(
    data: BytesLike,
    /,
    *,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEGXS image."""


class LERC:
    """LERC codec constants."""

    available: bool
    """LERC codec is available."""


class LercError(RuntimeError):
    """LERC codec exceptions."""


def lerc_version() -> str:
    """Return LERC library version string."""


def lerc_check(data: BytesLike, /) -> bool:
    """Return whether data is LERC encoded."""


def lerc_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    masks: ArrayLike | None = None,
    version: int | None = None,
    planar: bool | None = None,
    compression: Literal['zstd'] | Literal['deflate'] | None = None,
    compressionargs: dict[str, Any] | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LERC encoded image."""


@overload
def lerc_decode(
    data: BytesLike,
    /,
    *,
    masks: Literal[False] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded LERC image."""


@overload
def lerc_decode(
    data: BytesLike,
    /,
    *,
    masks: Literal[True] | NDArray[Any],
    out: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return LERC encoded image."""


class LJPEG:
    """LJPEG codec constants."""

    available: bool
    """LJPEG codec is available."""


class LjpegError(RuntimeError):
    """LJPEG codec exceptions."""


def ljpeg_version() -> str:
    """Return liblj92 library version string."""


def ljpeg_check(data: BytesLike, /) -> None:
    """Return whether data is Lossless JPEG encoded image."""


def ljpeg_encode(
    data: ArrayLike,
    /,
    *,
    bitspersample: int | None = None,
    delinearize: ArrayLike | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return Lossless JPEG encoded image."""


def ljpeg_decode(
    data: BytesLike,
    /,
    *,
    linearize: ArrayLike | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded Lossless JPEG image."""


class LZ4:
    """LZ4 codec constants."""

    available: bool
    """LZ4 codec is available."""

    class CLEVEL(enum.IntEnum):
        """LZ4 codec compression levels."""

        DEFAULT: int
        MIN: int
        MAX: int
        OPT_MIN: int


class Lz4Error(RuntimeError):
    """LZ4 codec exceptions."""


def lz4_version() -> str:
    """Return LZ4 library version string."""


def lz4_check(data: BytesLike, /) -> None:
    """Return whether data is LZ4 encoded."""


def lz4_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    hc: bool = False,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZ4 encoded data."""


def lz4_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZ4 data."""


class LZ4F:
    """LZ4F codec constants."""

    available: bool
    """LZ4F codec is available."""

    VERSION: int
    """LZ4F file version."""


class Lz4fError(RuntimeError):
    """LZ4F codec exceptions."""


def lz4f_version() -> str:
    """Return LZ4 library version string."""


def lz4f_check(data: BytesLike, /) -> bool:
    """Return whether data is LZ4F encoded."""


def lz4f_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    blocksizeid: int | None = None,
    contentchecksum: bool | None = None,
    blockchecksum: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZ4F encoded data."""


def lz4f_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZ4F data."""


class LZ4H5:
    """LZ4H5 codec constants."""

    available: bool
    """LZ4H5 codec is available."""

    CLEVEL: LZ4.CLEVEL
    """LZ4 codec compression levels."""


class Lz4h5Error(RuntimeError):
    """LZ4H5 codec exceptions."""


def lz4h5_version() -> str:
    """Return LZ4 library version string."""


def lz4h5_check(data: BytesLike, /) -> bool | None:
    """Return whether data is LZ4H5 encoded."""


def lz4h5_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    blocksize: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZ4H5 encoded data."""


def lz4h5_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZ4H5 data."""


class LZF:
    """LZF codec constants."""

    available: bool
    """LZF codec is available."""


class LzfError(RuntimeError):
    """LZF codec exceptions."""


def lzf_version() -> str:
    """Return LibLZF library version string."""


def lzf_check(data: BytesLike, /) -> None:
    """Return whether data is LZF encoded."""


def lzf_encode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZF encoded data."""


def lzf_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZF data."""


class LZFSE:
    """LZFSE codec constants."""

    available: bool
    """LZFSE codec is available."""


class LzfseError(RuntimeError):
    """LZFSE codec exceptions."""


def lzfse_version() -> str:
    """Return LZFSE library version string."""


def lzfse_check(data: BytesLike, /) -> bool:
    """Return whether data is LZFSE encoded."""


def lzfse_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZFSE encoded data."""


def lzfse_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZFSE data."""


class LZHAM:
    """LZHAM codec constants."""

    available: bool
    """LZHAM codec is available."""

    class COMPRESSION(enum.IntEnum):
        """LZHAM codec compression levels."""

        DEFAULT: int
        NO: int
        BEST: int
        SPEED: int
        UBER: int

    class STRATEGY(enum.IntEnum):
        """LZHAM codec compression strategies."""

        DEFAULT: int
        FILTERED: int
        HUFFMAN_ONLY: int
        RLE: int
        FIXED: int


class LzhamError(RuntimeError):
    """LZHAM codec exceptions."""


def lzham_version() -> str:
    """Return LZHAM library version string."""


def lzham_check(data: BytesLike, /) -> None:
    """Return whether data is LZHAM encoded."""


def lzham_encode(
    data: BytesLike,
    /,
    level: LZHAM.COMPRESSION | int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZHAM encoded data."""


def lzham_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZHAM data."""


class LZMA:
    """LZMA codec constants."""

    available: bool
    """LZMA codec is available."""

    class CHECK(enum.IntEnum):
        """LZMA codec checksums."""

        NONE: int
        CRC32: int
        CRC64: int
        SHA256: int


class LzmaError(RuntimeError):
    """LZMA codec exceptions."""


def lzma_version() -> str:
    """Return liblzma library version string."""


def lzma_check(data: BytesLike, /) -> None:
    """Return whether data is LZMA encoded."""


def lzma_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    check: LZMA.CHECK | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZMA encoded data."""


def lzma_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZMA data."""


class LZO:
    """LZO codec constants."""

    available: bool
    """LZO codec is available."""


class LzoError(RuntimeError):
    """LZO codec exceptions."""


def lzo_version() -> str:
    """Return lzokay library version string."""


def lzo_check(data: BytesLike, /) -> bool | None:
    """Return whether data is LZO encoded."""


def lzo_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZO encoded data (not implemented)."""


def lzo_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZO data."""


class LZW:
    """LZW codec constants."""

    available: bool
    """LZW codec is available."""


LzwError = ImcdError

lzw_version = imcd_version


def lzw_check(data: BytesLike, /) -> bool:
    """Return whether data is LZW encoded."""


def lzw_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return LZW encoded data."""


def lzw_decode(
    data: BytesLike,
    /,
    *,
    buffersize: int = 0,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded LZW data."""


# class MONO12P:
#     """MONO12P codec constants."""

#     available: bool
#     """MONO12P codec is available."""


# Mono12pError = ImcdError

# mono12p_version = imcd_version


# def mono12p_check(data: BytesLike, /) -> None:
#     """Return whether data is MONO12P encoded."""


# def mono12p_encode(
#     data: ArrayLike,
#     /,
#     msfirst: bool = False,
#     *,
#     axis: int = -1,
#     out: int | bytearray | None = None,
# ) -> bytes | bytearray:
#     """Return MONO12P packed integers."""


# def mono12p_decode(
#     data: BytesLike,
#     /,
#     msfirst: bool = False,
#     *,
#     runlen: int = 0,
#     out: NDArray[Any] | None = None,
# ) -> NDArray[Any]:
#     """Return unpacked MONO12P integers."""


class MOZJPEG:
    """MOZJPEG codec constants."""

    available: bool
    """MOZJPEG codec is available."""

    class CS(enum.IntEnum):
        """MOZJPEG codec color spaces."""

        UNKNOWN: int
        GRAYSCALE: int
        RGB: int
        YCbCr: int
        CMYK: int
        YCCK: int
        EXT_RGB: int
        EXT_RGBX: int
        EXT_BGR: int
        EXT_BGRX: int
        EXT_XBGR: int
        EXT_XRGB: int
        EXT_RGBA: int
        EXT_BGRA: int
        EXT_ABGR: int
        EXT_ARGB: int
        RGB565: int


class MozjpegError(RuntimeError):
    """MOZJPEG codec exceptions."""


def mozjpeg_version() -> str:
    """Return mozjpeg library version string."""


def mozjpeg_check(data: BytesLike, /) -> bool:
    """Return whether data is JPEG encoded image."""


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
) -> bytes | bytearray:
    """Return JPEG encoded image."""


def mozjpeg_decode(
    data: BytesLike,
    /,
    *,
    tables: bytes | None = None,
    colorspace: MOZJPEG.CS | int | str | None = None,
    outcolorspace: MOZJPEG.CS | int | str | None = None,
    shape: tuple[int, int] | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded JPEG image."""


class NONE:
    """NONE codec constants."""

    available: bool
    """NONE codec is available."""


NoneError = RuntimeError


def none_version() -> str:
    """Return empty version string."""


def none_check(data: Any, /) -> None:
    """Return None."""


def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""


def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""


class NUMPY:
    """NUMPY codec constants."""

    available: bool
    """NUMPY codec is available."""


NumpyError = RuntimeError


def numpy_version() -> str:
    """Return Numpy library version string."""


def numpy_check(data: BytesLike, /) -> bool:
    """Return whether data is NPY or NPZ encoded."""


def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes:
    """Return NPY or NPZ encoded data."""


def numpy_decode(
    data: BytesLike,
    /,
    index: int | None = None,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Return decoded NPY or NPZ data."""


class PACKBITS:
    """PACKBITS codec constants."""

    available: bool
    """PACKBITS codec is available."""


PackbitsError = ImcdError

packbits_version = imcd_version


def packbits_check(
    data: BytesLike,
    /,
) -> bool | None:
    """Return whether data is PACKBITS encoded."""


def packbits_encode(
    data: BytesLike | ArrayLike,
    /,
    *,
    axis: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return PACKBITS encoded data."""


def packbits_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded PACKBITS data."""


class PACKINTS:
    """PACKINTS codec constants."""

    available: bool
    """PACKINTS codec is available."""


PackintsError = ImcdError

packints_version = imcd_version


def packints_check(data: BytesLike, /) -> None:
    """Return whether data is PACKINTS encoded."""


def packints_encode(
    data: ArrayLike,
    bitspersample: int,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return packed integers (not implemented)."""


def packints_decode(
    data: BytesLike,
    dtype: DTypeLike,
    bitspersample: int,
    /,
    *,
    runlen: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return unpacked integers."""


class PCODEC:
    """Pcodec codec constants."""

    available: bool
    """Pcodec codec is available."""


class PcodecError(RuntimeError):
    """Pcodec codec exceptions."""


def pcodec_version() -> str:
    """Return pcodec library version string."""


def pcodec_check(data: BytesLike, /) -> None:
    """Return whether data is pcodec encoded."""


def pcodec_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return pcodec encoded data."""


def pcodec_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded pcodec data."""


class PGLZ:
    """PGLZ codec constants."""

    available: bool
    """PGLZ codec is available."""


class PglzError(RuntimeError):
    """PGLZ codec exceptions."""


def pglz_version() -> str:
    """Return PostgreSQL library version string."""


def pglz_check(data: BytesLike, /) -> bool | None:
    """Return whether data is PGLZ encoded."""


def pglz_encode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    strategy: str | tuple[int, int, int, int, int, int] | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return PGLZ encoded data."""


def pglz_decode(
    data: BytesLike,
    /,
    *,
    header: bool = False,
    checkcomplete: bool | None = None,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded PGLZ data."""


class PNG:
    """PNG codec constants."""

    available: bool
    """PNG codec is available."""

    class COLOR_TYPE(enum.IntEnum):
        """PNG codec color types."""

        GRAY: int
        GRAY_ALPHA: int
        RGB: int
        RGB_ALPHA: int

    class COMPRESSION(enum.IntEnum):
        """PNG codec compression levels."""

        DEFAULT: int
        NO: int
        BEST: int
        SPEED: int

    class STRATEGY(enum.IntEnum):
        """PNG codec compression strategies."""

        DEFAULT: int
        FILTERED: int
        HUFFMAN_ONLY: int
        RLE: int
        FIXED: int

    class FILTER(enum.IntEnum):  # IntFlag
        """PNG codec filters."""

        NO: int
        NONE: int
        SUB: int
        UP: int
        AVG: int
        PAETH: int
        FAST: int
        ALL: int


class PngError(RuntimeError):
    """PNG codec exceptions."""


def png_version() -> str:
    """Return libpng library version string."""


def png_check(data: BytesLike, /) -> bool:
    """Return whether data is PNG encoded image."""


def png_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    strategy: int | None = None,
    filter: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return PNG encoded image."""


def png_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded PNG image."""


class QOI:
    """QOI codec constants."""

    available: bool
    """QOI codec is available."""

    class COLORSPACE(enum.IntEnum):
        """QOI codec color spaces."""

        SRGB: int
        LINEAR: int


class QoiError(RuntimeError):
    """QOI codec exceptions."""


def qoi_version() -> str:
    """Return QOI library version string."""


def qoi_check(data: BytesLike, /) -> bool:
    """Return whether data is QOI encoded image."""


def qoi_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return QOI encoded image."""


def qoi_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded QOI image."""


class QUANTIZE:
    """Quantize codec constants."""

    available: bool
    """Quantize codec is available."""

    class MODE(enum.IntEnum):
        """Quantize mode."""

        NOQUANTIZE: int
        BITGROOM: int
        GRANULARBR: int
        BITROUND: int
        SCALE: int


class QuantizeError(RuntimeError):
    """Quantize codec exceptions."""


def quantize_version() -> str:
    """Return nc4var library version string."""


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
) -> NDArray[Any]:
    """Return quantized floating point array."""


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
) -> NDArray[Any]:
    """Return data if lossless else raise QuantizeError."""


class RCOMP:
    """RCOMP codec constants."""

    available: bool
    """RCOMP codec is available."""


class RcompError(RuntimeError):
    """RCOMP codec exceptions."""


def rcomp_version() -> str:
    """Return cfitsio library version string."""


def rcomp_check(data: BytesLike, /) -> bool:
    """Return whether data is RCOMP encoded."""


def rcomp_encode(
    data: ArrayLike,
    /,
    *,
    nblock: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return RCOMP encoded data."""


def rcomp_decode(
    data: BytesLike,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    nblock: int | None = None,
    out: NDArray[Any] | None = None,
) -> ArrayLike:
    """Return decoded RCOMP data."""


class RGBE:
    """RGBE codec constants."""

    available: bool
    """RGBE codec is available."""


class RgbeError(RuntimeError):
    """RBGE codec exceptions."""


def rgbe_version() -> str:
    """Return RGBE library version string."""


def rgbe_check(data: BytesLike, /) -> bool:
    """Return whether data is RGBE encoded image."""


def rgbe_encode(
    data: ArrayLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return RGBE encoded image."""


def rgbe_decode(
    data: BytesLike,
    /,
    *,
    header: bool | None = None,
    rle: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded RGBE image."""


class SNAPPY:
    """SNAPPY codec constants."""

    available: bool
    """SNAPPY codec is available."""


class SnappyError(RuntimeError):
    """SNAPPY codec exceptions."""


def snappy_version() -> str:
    """Return Snappy library version string."""


def snappy_check(data: BytesLike, /) -> None:
    """Return whether data is SNAPPY encoded."""


def snappy_encode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return SNAPPY encoded data."""


def snappy_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded SNAPPY data."""


class SPERR:
    """SPERR codec constants."""

    available: bool
    """SPERR codec is available."""

    class MODE(enum.IntEnum):
        """SPERR quality mode."""

        BPP: int
        PSNR: int
        PWE: int


class SperrError(RuntimeError):
    """SPERR codec exceptions."""


def sperr_version() -> str:
    """Return SPERR library version string."""


def sperr_check(data: BytesLike, /) -> None:
    """Return whether data is SPERR encoded."""


def sperr_encode(
    data: ArrayLike,
    /,
    level: float,
    *,
    mode: SPERR.MODE | Literal['bpp', 'psnr', 'pwe'],
    chunks: tuple[int, int, int] | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return SPERR encoded data."""


def sperr_decode(
    data: BytesLike,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    header: bool = True,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded SPERR data."""


class SPNG:
    """SPNG codec constants."""

    available: bool
    """SPNG codec is available."""

    class FMT(enum.IntEnum):
        """SPNG codec formats."""

        RGBA8: int
        RGBA16: int
        RGB8: int
        GA8: int
        GA16: int
        G8: int


class SpngError(RuntimeError):
    """SPNG codec exceptions."""


def spng_version() -> str:
    """Return libspng library version string."""


def spng_check(data: BytesLike, /) -> bool:
    """Return whether data is PNG encoded image."""


def spng_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return PNG encoded image."""


def spng_decode(
    data: BytesLike,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded PNG image."""


class SZ3:
    """SZ3 codec constants."""

    available: bool
    """SZ3 codec is available."""

    class MODE(enum.IntEnum):
        """SZ3 codec error bound modes."""

        ABS: int
        REL: int
        ABS_AND_REL: int
        ABS_OR_REL: int
        # PSNR: int
        # NORM: int
        # PW_REL: int
        # ABS_AND_PW_REL: int
        # ABS_OR_PW_REL: int
        # REL_AND_PW_REL: int
        # REL_OR_PW_REL: int


class Sz3Error(RuntimeError):
    """SZ3 codec exceptions."""


def sz3_version() -> str:
    """Return SZ3 library version string."""


def sz3_check(data: BytesLike, /) -> bool:
    """Return whether data is SZ3 encoded."""


def sz3_encode(
    data: ArrayLike,
    /,
    mode: SZ3.MODE | int | str | None = None,
    abs: float | None = None,
    rel: float | None = None,
    # pwr: float | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return SZ3 encoded data."""


def sz3_decode(
    data: BytesLike,
    /,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded SZ3 data."""


class SZIP:
    """SZIP codec constants."""

    available: bool
    """SZIP codec is available."""

    class OPTION_MASK(enum.IntEnum):
        """SZIP codec flags."""

        ALLOW_K13: int
        CHIP: int
        EC: int
        LSB: int
        MSB: int
        NN: int
        RAW: int


class SzipError(RuntimeError):
    """SZIP codec exceptions."""


def szip_version() -> str:
    """Return libaec library version string."""


def szip_check(data: BytesLike, /) -> None:
    """Return whether data is SZIP encoded."""


def szip_encode(
    data: BytesLike,
    /,
    options_mask: int,
    pixels_per_block: int,
    bits_per_pixel: int,
    pixels_per_scanline: int,
    *,
    header: bool = False,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return SZIP encoded data."""


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
) -> bytes | bytearray:
    """Return decoded SZIP data."""


def szip_params(
    data: NDArray[Any], /, options_mask: int = 4, pixels_per_block: int = 32
) -> dict[str, int]:
    """Return SZIP parameters for numpy array."""


class TIFF:
    """TIFF codec constants."""

    available: bool
    """TIFF codec is available."""

    class VERSION(enum.IntEnum):
        """TIFF codec file types."""

        CLASSIC: int
        BIG: int

    class ENDIAN(enum.IntEnum):
        """TIFF codec endian values."""

        BIG: int
        LITTLE: int

    class COMPRESSION(enum.IntEnum):
        """TIFF codec compression schemes."""

        NONE: int
        LZW: int
        JPEG: int
        PACKBITS: int
        DEFLATE: int
        ADOBE_DEFLATE: int
        LZMA: int
        ZSTD: int
        WEBP: int
        # LERC: int
        # JXL: int

    class PHOTOMETRIC(enum.IntEnum):
        """TIFF codec photometric interpretations."""

        MINISWHITE: int
        MINISBLACK: int
        RGB: int
        PALETTE: int
        MASK: int
        SEPARATED: int
        YCBCR: int

    class PLANARCONFIG(enum.IntEnum):
        """TIFF codec planar configurations."""

        CONTIG: int
        SEPARATE: int

    class PREDICTOR(enum.IntEnum):
        """TIFF codec predictor schemes."""

        NONE: int
        HORIZONTAL: int
        FLOATINGPOINT: int

    class EXTRASAMPLE(enum.IntEnum):
        """TIFF codec extrasample types."""

        UNSPECIFIED: int
        ASSOCALPHA: int
        UNASSALPHA: int


class TiffError(RuntimeError):
    """TIFF codec exceptions."""


def tiff_version() -> str:
    """Return libtiff library version string."""


def tiff_check(data: BytesLike, /) -> bool:
    """Return whether data is TIFF encoded image."""


def tiff_encode(
    data: ArrayLike,
    /,
    *,
    out: int | bytearray | None = None,
) -> None:
    """Return TIFF encoded image (not implemented)."""


def tiff_decode(
    data: BytesLike,
    /,
    index: int | Sequence[int] | slice | None = 0,
    *,
    asrgb: bool = False,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded TIFF image."""


class ULTRAHDR:
    """Ultra HDR codec constants."""

    available: bool
    """Ultra HDR codec is available."""

    class CG(enum.IntEnum):
        """Ultra HDR color gamut."""

        UNSPECIFIED: int
        BT_709: int
        DISPLAY_P3: int
        BT_2100: int

    class CT(enum.IntEnum):
        """Ultra HDR color transfer."""

        UNSPECIFIED: int
        LINEAR: int
        HLG: int
        PQ: int
        SRGB: int

    class CR(enum.IntEnum):
        """Ultra HDR color range."""

        UNSPECIFIED: int
        LIMITED_RANGE: int
        FULL_RANGE: int

    class CODEC(enum.IntEnum):
        """Ultra HDR codec."""

        JPEG: int
        HEIF: int
        AVIF: int

    class USAGE(enum.IntEnum):
        """Ultra HDR codec."""

        REALTIME: int
        QUALITY: int


class UltrahdrError(RuntimeError):
    """Ultra HDR codec exceptions."""


def ultrahdr_version() -> str:
    """Return libultrahdr library version string."""


def ultrahdr_check(data: BytesLike, /) -> bool:
    """Return whether data is Ultra HDR encoded image."""


def ultrahdr_encode(
    data: ArrayLike,
    /,
    *,
    level: int | None = None,
    scale: int | None = None,
    gamut: ULTRAHDR.CG | int | None = None,
    transfer: ULTRAHDR.CT | int | None = None,
    crange: ULTRAHDR.CR | int | None = None,
    usage: ULTRAHDR.USAGE | int | None = None,
    codec: ULTRAHDR.CODEC | int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return Ultra HDR encoded image."""


def ultrahdr_decode(
    data: BytesLike,
    /,
    *,
    dtype: DTypeLike | None = None,
    transfer: ULTRAHDR.CT | int | None = None,
    boost: float | None = None,
    gpu: bool = False,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded Ultra HDR image."""


class WEBP:
    """WEBP codec constants."""

    available: bool
    """WEBP codec is available."""


class WebpError(RuntimeError):
    """WEBP codec exceptions."""


def webp_version() -> str:
    """Return libwebp library version string."""


def webp_check(data: BytesLike, /) -> bool:
    """Return whether data is WebP encoded image."""


def webp_encode(
    data: ArrayLike,
    /,
    level: float | None = None,
    *,
    lossless: bool | None = None,
    method: int | None = None,
    numthreads: int | None = None,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return WebP encoded image."""


def webp_decode(
    data: BytesLike,
    /,
    index: int | None = 0,
    *,
    hasalpha: bool | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded WebP image."""


class XOR:
    """XOR codec constants."""

    available: bool
    """XOR codec is available."""


XorError = ImcdError

xor_version = imcd_version


def xor_check(data: Any, /) -> None:
    """Return whether data is XOR encoded."""


@overload
def xor_encode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return XOR encoded data."""


@overload
def xor_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded XOR data."""


@overload
def xor_decode(
    data: BytesLike,
    /,
    *,
    axis: int = -1,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded XOR data."""


@overload
def xor_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded XOR data."""


class ZFP:
    """ZFP codec constants."""

    available: bool
    """ZFP codec is available."""

    class EXEC(enum.IntEnum):
        """ZFP codec execution policies."""

        SERIAL: int
        OMP: int
        CUDA: int

    class MODE(enum.IntEnum):
        """ZFP codec compression modes."""

        NONE: int
        EXPERT: int
        FIXED_RATE: int
        FIXED_PRECISION: int
        FIXED_ACCURACY: int
        REVERSIBLE: int

    class HEADER(enum.IntEnum):
        """ZFP codec header types."""

        MAGIC: int
        META: int
        MODE: int
        FULL: int


class ZfpError(RuntimeError):
    """ZFP codec exceptions."""


def zfp_version() -> str:
    """Return zfp library version string."""


def zfp_check(data: BytesLike, /) -> bool:
    """Return whether data is ZFP encoded."""


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
) -> bytes | bytearray:
    """Return ZFP encoded data."""


def zfp_decode(
    data: BytesLike,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
    strides: tuple[int, ...] | None = None,
    numthreads: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return decoded ZFP data."""


class ZLIB:
    """ZLIB codec constants."""

    available: bool
    """ZLIB codec is available."""

    class COMPRESSION(enum.IntEnum):
        """ZLIB codec compression levels."""

        DEFAULT: int
        NO: int
        BEST: int
        SPEED: int

    class STRATEGY(enum.IntEnum):
        """ZLIB codec compression strategies."""

        DEFAULT: int
        FILTERED: int
        HUFFMAN_ONLY: int
        RLE: int
        FIXED: int


class ZlibError(RuntimeError):
    """ZLIB codec exceptions."""


def zlib_version() -> str:
    """Return zlib library version string."""


def zlib_check(data: BytesLike, /) -> bool | None:
    """Return whether data is DEFLATE encoded."""


def zlib_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DEFLATE encoded data."""


def zlib_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DEFLATE data."""


def zlib_crc32(data: BytesLike, /, value: int | None = None) -> int:
    """Return CRC32 checksum of data."""


def zlib_adler32(data: BytesLike, /, value: int | None = None) -> int:
    """Return Adler-32 checksum of data."""


class ZLIBNG:
    """ZLIBNG codec constants."""

    available: bool
    """ZLIBNG codec is available."""

    class COMPRESSION(enum.IntEnum):
        """ZLIBNG codec compression levels."""

        DEFAULT: int
        NO: int
        BEST: int
        SPEED: int

    class STRATEGY(enum.IntEnum):
        """ZLIBNG codec compression strategies."""

        DEFAULT: int
        FILTERED: int
        HUFFMAN_ONLY: int
        RLE: int
        FIXED: int


class ZlibngError(RuntimeError):
    """ZLIBNG codec exceptions."""


def zlibng_version() -> str:
    """Return zlibng library version string."""


def zlibng_check(data: BytesLike, /) -> bool | None:
    """Return whether data is DEFLATE encoded."""


def zlibng_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return DEFLATE encoded data."""


def zlibng_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded DEFLATE data."""


def zlibng_crc32(data: BytesLike, /, value: int | None = None) -> int:
    """Return CRC32 checksum of data."""


def zlibng_adler32(data: BytesLike, /, value: int | None = None) -> int:
    """Return Adler-32 checksum of data."""


class ZOPFLI:
    """ZOPFLI codec constants."""

    available: bool
    """ZOPFLI codec is available."""

    class FORMAT(enum.IntEnum):
        """ZOPFLI codec formats."""

        GZIP: int
        ZLIB: int
        DEFLATE: int


class ZopfliError(RuntimeError):
    """ZOPFLI codec exceptions."""


def zopfli_version() -> str:
    """Return Zopfli library version string."""


zopfli_check = zlib_check


def zopfli_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
    **kwargs: Any,
) -> bytes | bytearray:
    """Return DEFLATE encoded data."""


zopfli_decode = zlib_decode


class ZSTD:
    """ZSTD codec constants."""

    available: bool
    """ZSTD codec is available."""


class ZstdError(RuntimeError):
    """ZSTD codec exceptions."""


def zstd_version() -> str:
    """Return Zstandard library version string."""


def zstd_check(data: BytesLike, /) -> bool:
    """Return whether data is ZSTD encoded."""


def zstd_encode(
    data: BytesLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes | bytearray:
    """Return ZSTD encoded data."""


def zstd_decode(
    data: BytesLike,
    /,
    *,
    out: int | bytearray | memoryview | None = None,
) -> bytes | bytearray:
    """Return decoded ZSTD data."""
