# imagecodecs/_imagecodecs.py

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

"""Python implementations of image codecs.

This module implements a limited set of image and compression codecs using
pure Python and 3rd party Python packages.
The module is intended for testing and reference, not production code.

"""

from __future__ import annotations

__all__ = [
    'bitshuffle',
    'blosc',
    'blosc2',
    'brotli',
    'bz2',
    'lz4',
    'lzf',
    'lzfse',
    'lzham',
    'lzma',
    'pillow',
    'snappy',
    'tifffile',
    'zlib',
    'zopfli',
    'zstd',
]

import bz2
import contextlib
import functools
import gzip
import io
import lzma
import struct
import sys
import zlib
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable, Sequence
    from types import ModuleType
    from typing import Any, NoReturn

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import BytesOut, BytesResult

import numpy

from .imagecodecs import __version__

pillow: ModuleType | None
bitshuffle: ModuleType | None
blosc: ModuleType | None
blosc2: ModuleType | None
brotli: ModuleType | None
lz4: ModuleType | None
lzf: ModuleType | None
lzfse: ModuleType | None
lzham: ModuleType | None
snappy: ModuleType | None
tifffile: ModuleType | None
zfp: ModuleType | None
zopfli: ModuleType | None
zstd: ModuleType | None
zarr: ModuleType | None
numcodecs: ModuleType | None

try:
    import PIL as pillow
except ImportError:
    pillow = None

try:
    import bitshuffle
except ImportError:
    bitshuffle = None

try:
    import blosc
except ImportError:
    blosc = None

try:
    import blosc2
except ImportError:
    blosc2 = None

try:
    import brotli
except ImportError:
    brotli = None

try:
    import lz4
    import lz4.block
    import lz4.frame
except ImportError:
    lz4 = None

try:
    import lzf
except ImportError:
    lzf = None

try:
    import liblzfse as lzfse
except ImportError:
    lzfse = None

try:
    import lzham
except ImportError:
    lzham = None

try:
    import snappy
except ImportError:
    snappy = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import zfpy as zfp
except ImportError:
    zfp = None

try:
    import zopfli
except ImportError:
    zopfli = None

try:
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        from backports import zstd
except ImportError:
    zstd = None

try:
    import zarr
except ImportError:
    zarr = None

try:
    import numcodecs
except ImportError:
    numcodecs = None


def version() -> str:
    """Return detailed version information about test dependencies."""
    try:
        zstd_version: str = zstd.version()
    except Exception:
        try:
            if sys.version_info >= (3, 14):
                from compression import zstd as _zstd
            else:
                from backports import zstd as _zstd

            zstd_version = _zstd.zstd_version
        except ImportError:
            zstd_version = 'n/a'

    return ', '.join(
        f'{k}-{v}'
        for k, v in (
            ('imagecodecs.py', __version__),
            ('numpy', numpy.__version__),
            ('zlib', zlib.ZLIB_VERSION),
            ('bz2', 'stdlib'),
            ('lzma', getattr(lzma, '__version__', 'stdlib')),
            ('blosc', blosc.__version__ if blosc else 'n/a'),
            ('blosc2', blosc2.__version__ if blosc2 else 'n/a'),
            ('zstd', zstd_version),
            ('lz4', lz4.VERSION if lz4 else 'n/a'),
            ('lzf', 'unknown' if lzf else 'n/a'),
            ('lzham', 'unknown' if lzham else 'n/a'),
            ('pyliblzfse', 'unknown' if lzfse else 'n/a'),
            ('snappy', 'unknown' if snappy else 'n/a'),
            ('zopflipy', zopfli.__version__ if zopfli else 'n/a'),
            ('zfpy', zfp.__version__ if zfp else 'n/a'),
            ('bitshuffle', bitshuffle.__version__ if bitshuffle else 'n/a'),
            ('pillow', pillow.__version__ if pillow else 'n/a'),
            ('numcodecs', numcodecs.__version__ if numcodecs else 'n/a'),
            ('zarr', zarr.__version__ if zarr else 'n/a'),
            ('tifffile', tifffile.__version__ if tifffile else 'n/a'),
        )
    )


@overload
def notimplemented[F: Callable[..., Any]](arg: F, /) -> F: ...


@overload
def notimplemented[F: Callable[..., Any]](
    arg: ModuleType | bool | None = ...,  # noqa: FBT001
    /,
) -> Callable[[F], F]: ...


def notimplemented(arg: Any = 0, /) -> Any:
    """Return function decorator that raises NotImplementedError if not arg.

    >>> @notimplemented
    ... def test():
    ...     pass
    ...
    >>> test()
    Traceback (most recent call last):
    ...
    NotImplementedError: test not implemented

    >>> @notimplemented(True)
    ... def test():
    ...     pass
    ...
    >>> test()

    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def notimplemented(*args: Any, **kwargs: Any) -> NoReturn:
            msg = f'{func.__name__} not implemented'
            raise NotImplementedError(msg)

        return notimplemented

    if callable(arg):
        return wrapper(arg)
    if not arg:
        return wrapper

    def nop(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return nop


def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Decode NOP."""
    return data


def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Encode NOP."""
    return data


def numpy_decode(
    data: Buffer,
    /,
    index: int | None = 0,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Decode NPY and NPZ."""
    if index is None:
        index = 0
    with io.BytesIO(data) as fh:
        result = numpy.load(fh, **kwargs)
        if hasattr(result, 'files'):
            with contextlib.suppress(TypeError, IndexError):
                index = result.files[index]
            result = result[index]
    return cast('NDArray[Any]', result)


def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
    **kwargs: Any,
) -> bytes:
    """Encode NPY and NPZ."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data, **kwargs)
        else:
            numpy.save(fh, data, **kwargs)
        fh.seek(0)
        return fh.read()


def delta_encode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    r"""Encode Delta.

    >>> delta_encode(b'0123456789')
    b'0\x01\x01\x01\x01\x01\x01\x01\x01\x01'

    """
    if dist != 1:
        msg = f'{dist=} not implemented'
        raise NotImplementedError(msg)

    if not isinstance(data, numpy.ndarray):
        data = numpy.frombuffer(data, dtype='u1')
        diff = numpy.diff(data, axis=0)
        return numpy.insert(diff, 0, data[0]).tobytes()

    dtype = data.dtype
    if dtype.kind == 'f':
        data = data.view(f'{dtype.byteorder}u{dtype.itemsize}')

    diff = numpy.diff(data, axis=axis)
    key: list[int | slice] = [slice(None)] * data.ndim
    key[axis] = 0
    diff = numpy.insert(diff, 0, data[tuple(key)], axis=axis)
    if not data.dtype.isnative:
        diff = diff.byteswap(inplace=True)
        diff = diff.view(diff.dtype.newbyteorder())

    if dtype.kind == 'f':
        return diff.view(dtype)
    return diff


def delta_decode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    axis: int = -1,
    dist: int = 1,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    r"""Decode Delta.

    >>> delta_decode(b'0\x01\x01\x01\x01\x01\x01\x01\x01\x01')
    b'0123456789'

    """
    if dist != 1:
        msg = f'{dist=} not implemented'
        raise NotImplementedError(msg)
    if not isinstance(out, numpy.ndarray) or not out.flags.writeable:
        out = None
    if not isinstance(data, numpy.ndarray):
        data = numpy.frombuffer(data, dtype='u1')
        return numpy.cumsum(data, axis=0, dtype='u1', out=out).tobytes()
    out = numpy.cumsum(data, axis=axis, dtype=data.dtype, out=out)
    if data.dtype.isnative:
        return out
    out = out.byteswap(inplace=True)
    return out.view(out.dtype.newbyteorder())


def xor_encode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    r"""Encode XOR delta.

    >>> xor_encode(b'0123456789')
    b'0\x01\x03\x01\x07\x01\x03\x01\x0f\x01'

    """
    if not isinstance(data, numpy.ndarray):
        data = numpy.frombuffer(data, dtype='u1')
        xor = numpy.bitwise_xor(data[1:], data[:-1])
        return numpy.insert(xor, 0, data[0]).tobytes()

    dtype = data.dtype
    if dtype.kind == 'f':
        data = data.view(f'u{dtype.itemsize}')

    key: list[int | slice] = [slice(None)] * data.ndim
    key[axis] = 0
    key0 = [slice(None)] * data.ndim
    key0[axis] = slice(1, None, None)
    key1 = [slice(None)] * data.ndim
    key1[axis] = slice(0, -1, None)

    xor = numpy.bitwise_xor(data[tuple(key0)], data[tuple(key1)])
    xor = numpy.insert(xor, 0, data[tuple(key)], axis=axis)

    if dtype.kind == 'f':
        return xor.view(dtype)
    if not data.dtype.isnative:
        xor = xor.byteswap(inplace=True)
        xor = xor.view(xor.dtype.newbyteorder())
    return xor


def xor_decode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    axis: int = -1,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    r"""Decode XOR delta.

    >>> xor_decode(b'0\x01\x03\x01\x07\x01\x03\x01\x0f\x01')
    b'0123456789'

    """
    if isinstance(data, bytes | bytearray):
        prev = data[0]
        b = [chr(prev)]
        for c in data[1:]:
            prev = c ^ prev
            b.append(chr(prev))
        return ''.join(b).encode('latin1')
    raise NotImplementedError


def floatpred_decode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -2,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode floating point horizontal differencing.

    The TIFF predictor type 3 reorders the bytes of the image values and
    applies horizontal byte differencing to improve compression of floating
    point images. The ordering of interleaved color channels is preserved.

    Parameters:
        data:
            The image to be decoded. The dtype must be a floating point.
            The shape must include the number of contiguous samples per pixel
            even if 1.

    """
    if dist != 1:
        msg = f'{dist=} not implemented'
        raise NotImplementedError(msg)  # TODO
    if axis != -2:
        msg = f'{axis=!r} != -2'
        raise NotImplementedError(msg)  # TODO
    shape = data.shape
    dtype = data.dtype
    if len(shape) < 3:
        msg = 'invalid data shape'
        raise ValueError(msg)
    if dtype.char not in 'dfe':
        msg = 'not a floating point image'
        raise ValueError(msg)
    littleendian = data.dtype.byteorder == '<' or (
        sys.byteorder == 'little' and data.dtype.byteorder == '='
    )
    # undo horizontal byte differencing
    data = data.view('uint8').reshape((*shape[:-2], -1, *shape[-1:]))
    numpy.cumsum(data, axis=-2, dtype='uint8', out=data)
    # reorder bytes
    if littleendian:
        data = data.reshape((*shape[:-2], -1, *shape[-2:]))
    data = numpy.swapaxes(data, -3, -2)
    data = numpy.swapaxes(data, -2, -1)
    data = data[..., ::-1]
    # back to float
    data = numpy.asarray(data, order='C')
    return data.view(dtype).reshape(shape)


@notimplemented
def floatpred_encode(
    data: NDArray[Any],
    /,
    *,
    axis: int = -2,
    dist: int = 1,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Encode Floating Point Predictor."""
    raise NotImplementedError


def bitorder_decode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
    _bitorder: list[Any] = [],
) -> NDArray[Any] | BytesResult:
    r"""Reverse bits in each byte of byte string or numpy array.

    Decode data where pixels with lower column values are stored in the
    lower-order bits of the bytes (TIFF FillOrder is LSB2MSB).

    Parameters:
        data:
            Data to be bit reversed. If byte string, a new bit-reversed byte
            string is returned. Numpy arrays are bit-reversed in-place.

    Examples:
        >>> bitorder_decode(b'\x01\x64')
        b'\x80&'
        >>> data = numpy.array([1, 666], dtype='uint16')
        >>> bitorder_decode(data)
        array([  128, 16473], dtype=uint16)
        >>> data
        array([  128, 16473], dtype=uint16)

    """
    if not _bitorder:
        _bitorder.append(
            b'\x00\x80@\xc0 \xa0`\xe0\x10\x90P\xd00\xb0p\xf0\x08\x88H\xc8('
            b'\xa8h\xe8\x18\x98X\xd88\xb8x\xf8\x04\x84D\xc4$\xa4d\xe4\x14'
            b'\x94T\xd44\xb4t\xf4\x0c\x8cL\xcc,\xacl\xec\x1c\x9c\\\xdc<\xbc|'
            b'\xfc\x02\x82B\xc2"\xa2b\xe2\x12\x92R\xd22\xb2r\xf2\n\x8aJ\xca*'
            b'\xaaj\xea\x1a\x9aZ\xda:\xbaz\xfa\x06\x86F\xc6&\xa6f\xe6\x16'
            b'\x96V\xd66\xb6v\xf6\x0e\x8eN\xce.\xaen\xee\x1e\x9e^\xde>\xbe~'
            b'\xfe\x01\x81A\xc1!\xa1a\xe1\x11\x91Q\xd11\xb1q\xf1\t\x89I\xc9)'
            b'\xa9i\xe9\x19\x99Y\xd99\xb9y\xf9\x05\x85E\xc5%\xa5e\xe5\x15'
            b'\x95U\xd55\xb5u\xf5\r\x8dM\xcd-\xadm\xed\x1d\x9d]\xdd=\xbd}'
            b'\xfd\x03\x83C\xc3#\xa3c\xe3\x13\x93S\xd33\xb3s\xf3\x0b\x8bK'
            b'\xcb+\xabk\xeb\x1b\x9b[\xdb;\xbb{\xfb\x07\x87G\xc7\'\xa7g\xe7'
            b'\x17\x97W\xd77\xb7w\xf7\x0f\x8fO\xcf/\xafo\xef\x1f\x9f_'
            b'\xdf?\xbf\x7f\xff'
        )
        _bitorder.append(numpy.frombuffer(_bitorder[0], dtype='uint8'))
    if not isinstance(data, numpy.ndarray):
        return bytes(data).translate(_bitorder[0])
    try:
        view = data.view('uint8')
        return numpy.take(_bitorder[1], view, out=view)
    except ValueError as exc:
        msg = 'slices of arrays not supported'
        raise NotImplementedError(msg) from exc


bitorder_encode = bitorder_decode


def packbits_decode(
    encoded: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    r"""Decompress PackBits encoded byte string.

    >>> packbits_decode(b'\x80\x80')  # NOP
    b''
    >>> packbits_decode(b'\x02123')
    b'123'
    >>> packbits_decode(
    ...     b'\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa'
    ... )[:-4]
    b'\xaa\xaa\xaa\x80\x00*\xaa\xaa\xaa\xaa\x80\x00*"\xaa\xaa\xaa\xaa\xaa\xaa'

    """
    encoded = bytes(encoded)
    result: list[int] = []
    out_extend = result.extend
    i = 0
    try:
        while True:
            n = ord(encoded[i : i + 1]) + 1
            i += 1
            if n > 129:
                # replicate
                out_extend(encoded[i : i + 1] * (258 - n))
                i += 1
            elif n < 129:
                # literal
                out_extend(encoded[i : i + n])
                i += n
    except TypeError:
        pass
    return bytes(result)


def lzw_decode(
    encoded: Buffer,
    /,
    *,
    buffersize: int = 0,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    r"""Decompress LZW (Lempel-Ziv-Welch) encoded TIFF strip (byte string).

    The strip must begin with a CLEAR code and end with an EOI code.

    This implementation of the LZW decoding algorithm is described in TIFF v6
    and is not compatible with old style LZW compressed files like
    quad-lzw.tif.

    >>> lzw_decode(
    ...     b'\x80\x1c\xcc\'\x91\x01\xa0\xc2m6\x99NB\x03\xc9\xbe\x0b'
    ...     b'\x07\x84\xc2\xcd\xa68|"\x14 3\xc3\xa0\xd1c\x94\x02\x02'
    ... )
    b'say hammer yo hammer mc hammer go hammer'

    """
    encoded = bytes(encoded)
    len_encoded = len(encoded)
    bitcount_max = len_encoded * 8
    unpack = struct.unpack
    newtable = [bytes([i]) for i in range(256)]
    newtable.extend((b'\0', b'\0'))

    def next_code() -> int:
        # return integer of 'bitw' bits at 'bitcount' position in encoded
        start = bitcount // 8
        s = encoded[start : start + 4]
        try:
            code = unpack('>I', s)[0]
        except Exception:
            code = unpack('>I', s + b'\x00' * (4 - len(s)))[0]
        code <<= bitcount % 8
        code &= mask
        return code >> shr

    switchbits = {  # code: bit-width, shr-bits, bit-mask
        255: (9, 23, int(9 * '1' + '0' * 23, 2)),
        511: (10, 22, int(10 * '1' + '0' * 22, 2)),
        1023: (11, 21, int(11 * '1' + '0' * 21, 2)),
        2047: (12, 20, int(12 * '1' + '0' * 20, 2)),
    }
    bitw, shr, mask = switchbits[255]
    bitcount = 0

    if len_encoded < 4:
        msg = 'strip must be at least 4 characters long'
        raise ValueError(msg)

    if next_code() != 256:
        msg = 'strip must begin with CLEAR code'
        raise ValueError(msg)

    code = 0
    oldcode = 0
    result: list[bytes] = []
    result_append = result.append
    while True:
        code = next_code()  # ~5% faster when inlining this function
        bitcount += bitw
        if code == 257 or bitcount >= bitcount_max:  # EOI
            break
        if code == 256:  # CLEAR
            table = newtable[:]
            table_append = table.append
            lentable = 258
            bitw, shr, mask = switchbits[255]
            code = next_code()
            bitcount += bitw
            if code == 257:  # EOI
                break
            result_append(table[code])
        else:
            if code < lentable:
                decoded = table[code]
                newcode = table[oldcode] + decoded[:1]
            else:
                newcode = table[oldcode]
                newcode += newcode[:1]
                decoded = newcode
            result_append(decoded)
            table_append(newcode)
            lentable += 1
        oldcode = code
        if lentable in switchbits:
            bitw, shr, mask = switchbits[lentable]

    if code != 257:
        # logging.warning(f'unexpected end of LZW stream ({code=!r})')
        pass

    return b''.join(result)


def packints_decode(
    data: Buffer,
    dtype: DTypeLike,
    bitspersample: int,
    /,
    *,
    runlen: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decompress byte string to array of integers of any bit size <= 32.

    This Python implementation is slow and only handles itemsizes 1, 2, 4, 8,
    16, 32, and 64.

    Parameters:
        data : byte str
            Data to decompress.
        dtype : numpy.dtype or str
            A numpy boolean or integer type.
        bitspersample : int
            Number of bits per integer.
        runlen : int
            Number of consecutive integers, after which to start at next byte.

    Examples:
        >>> packints_decode(b'a', 'B', 1)
        array([0, 1, 1, 0, 0, 0, 0, 1], dtype=uint8)
        >>> packints_decode(b'ab', 'B', 2)
        array([1, 2, 0, 1, 1, 2, 0, 2], dtype=uint8)

    """
    if bitspersample == 1:  # bitarray
        data = numpy.frombuffer(data, '|B')
        data = numpy.unpackbits(data)
        if runlen % 8:
            data = data.reshape((-1, runlen + (8 - runlen % 8)))
            data = data[:, :runlen].reshape(-1)
        return data.astype(dtype)

    data = bytes(data)
    dtype = numpy.dtype(dtype)
    if bitspersample in {8, 16, 32, 64}:
        return numpy.frombuffer(data, dtype)
    if bitspersample not in {1, 2, 4, 8, 16, 32}:
        msg = f'{bitspersample=} not supported'
        raise ValueError(msg)
    if dtype.kind not in 'bu':
        msg = 'invalid dtype'
        raise ValueError(msg)

    itembytes = next(i for i in (1, 2, 4, 8) if 8 * i >= bitspersample)
    if itembytes != dtype.itemsize:
        msg = 'dtype.itemsize too small'
        raise ValueError(msg)
    if runlen == 0:
        runlen = (8 * len(data)) // bitspersample
    skipbits = runlen * bitspersample % 8
    if skipbits:
        skipbits = 8 - skipbits
    shrbits = itembytes * 8 - bitspersample
    bitmask = int(bitspersample * '1' + '0' * shrbits, 2)
    dtypestr = '>' + dtype.char  # dtype always big-endian?

    unpack = struct.unpack
    size = runlen * (len(data) * 8 // (runlen * bitspersample + skipbits))
    result = numpy.empty((size,), dtype)
    bitcount = 0
    for i in range(size):
        start = bitcount // 8
        s = data[start : start + itembytes]
        try:
            code = unpack(dtypestr, s)[0]
        except Exception:
            code = unpack(dtypestr, s + b'\x00' * (itembytes - len(s)))[0]
        code <<= bitcount % 8
        code &= bitmask
        result[i] = code >> shrbits
        bitcount += bitspersample
        if (i + 1) % runlen == 0:
            bitcount += skipbits
    return result


@notimplemented(bitshuffle)
def bitshuffle_encode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    itemsize: int | None = 1,
    blocksize: int = 0,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    """Bitshuffle."""
    if isinstance(data, numpy.ndarray):
        return bitshuffle.bitshuffle(data, blocksize)
    if itemsize is None:
        itemsize = 1
    data = numpy.frombuffer(data, dtype=f'uint{itemsize * 8}')
    data = bitshuffle.bitshuffle(data, blocksize)
    return data.tobytes()


@notimplemented(bitshuffle)
def bitshuffle_decode(
    data: NDArray[Any] | Buffer,
    /,
    *,
    itemsize: int | None = 1,
    blocksize: int = 0,
    out: NDArray[Any] | BytesOut | None = None,
) -> NDArray[Any] | BytesResult:
    """Bitunshuffle."""
    if isinstance(data, numpy.ndarray):
        return bitshuffle.bitunshuffle(data, blocksize)
    if itemsize is None:
        itemsize = 1
    data = numpy.frombuffer(data, dtype=f'uint{itemsize * 8}')
    data = bitshuffle.bitunshuffle(data, blocksize)
    return data.tobytes()


def zlib_encode(
    data: Buffer,
    /,
    level: int | None = 6,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Zlib."""
    return zlib.compress(data, 6 if level is None else level)


def zlib_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress Zlib."""
    return zlib.decompress(data)


def deflate_encode(
    data: Buffer,
    /,
    level: int | None = 6,
    *,
    raw: bool = False,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Deflate/Zlib."""
    if raw:
        raise NotImplementedError
    return zlib.compress(data, 6 if level is None else level)


def deflate_decode(
    data: Buffer,
    /,
    *,
    raw: bool = False,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress deflate/Zlib."""
    if raw:
        raise NotImplementedError
    return zlib.decompress(data)


def gzip_encode(
    data: Buffer,
    /,
    level: int | None = 6,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress GZIP."""
    level = 6 if level is None else level
    return gzip.compress(data, level)  # type: ignore[arg-type]


def gzip_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress GZIP."""
    return gzip.decompress(data)


def bz2_encode(
    data: Buffer,
    /,
    level: int | None = 9,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress BZ2."""
    return bz2.compress(data, 9 if level is None else level)


def bz2_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress BZ2."""
    return bz2.decompress(data)


@notimplemented(blosc)
def blosc_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    compressor: str | None = 'blosclz',
    numthreads: int | None = 1,
    typesize: int | None = 8,
    blocksize: int | None = 0,
    shuffle: int | None = None,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Blosc."""
    if shuffle is None:
        shuffle = blosc.SHUFFLE
    if level is None:
        level = 9
    return blosc.compress(
        data,
        typesize=8 if typesize is None else typesize,
        clevel=level,
        shuffle=shuffle,
        cname='blosclz' if compressor is None else compressor,
    )


@notimplemented(blosc)
def blosc_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress Blosc."""
    return blosc.decompress(data)


def lzma_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress LZMA."""
    return lzma.compress(data)


def lzma_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress LZMA."""
    return lzma.decompress(data)


@notimplemented(zstd)
def zstd_encode(
    data: Buffer,
    /,
    level: int | None = 5,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress ZStandard."""
    return zstd.compress(data, level)


@notimplemented(zstd)
def zstd_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress ZStandard."""
    return zstd.decompress(data)


@notimplemented(brotli)
def brotli_encode(
    data: Buffer,
    /,
    level: int | None = 11,
    *,
    mode: int | None = 0,
    lgwin: int | None = 22,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Brotli."""
    return brotli.compress(
        data,
        quality=11 if level is None else level,
        mode=0 if mode is None else mode,
        lgwin=22 if lgwin is None else lgwin,
    )


@notimplemented(brotli)
def brotli_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress Brotli."""
    return brotli.decompress(data)


@notimplemented(snappy)
def snappy_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Snappy."""
    return snappy.compress(data)


@notimplemented(snappy)
def snappy_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress Snappy."""
    return snappy.decompress(data)


@notimplemented(zopfli)
def zopfli_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress Zopfli."""
    c = zopfli.ZopfliCompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
    return c.compress(data) + c.flush()


@notimplemented(zopfli)
def zopfli_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress Zopfli."""
    d = zopfli.ZopfliDecompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
    return d.decompress(data) + d.flush()


@notimplemented(lzf)
def lzf_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress LZF."""
    return lzf.compress(data)


@notimplemented(lzf)
def lzf_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress LZF."""
    return lzf.decompress(data)


@notimplemented(lzfse)
def lzfse_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress LZFSE."""
    return lzfse.compress(data)


@notimplemented(lzfse)
def lzfse_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress LZFSE."""
    return lzfse.decompress(data)


@notimplemented(lzham)
def lzham_encode(
    data: Buffer,
    /,
    level: int | None = None,
    *,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress LZHAM."""
    return lzham.compress(data)


@notimplemented(lzham)
def lzham_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress LZHAM."""
    return lzham.decompress(data)


@notimplemented(zfp)
def zfp_encode(
    data: ArrayLike,
    /,
    level: float | tuple[int, int, int, int] | None = None,
    *,
    mode: int | str | None = None,
    execution: int | str | None = None,
    header: bool = True,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress ZFP."""
    del execution
    kwargs: dict[str, Any] = {'write_header': header}
    match mode:
        case None | zfp.mode_null | 'R' | 'reversible':  # zfp.mode_reversible
            pass
        case zfp.mode_fixed_precision | 'p' | 'precision':
            kwargs['precision'] = -1 if level is None else level
        case zfp.mode_fixed_rate | 'r' | 'rate':
            kwargs['rate'] = -1 if level is None else level
        case zfp.mode_fixed_accuracy | 'a' | 'accuracy':
            kwargs['tolerance'] = -1 if level is None else level
        case zfp.mode_expert | 'c' | 'expert':
            _minbits, _maxbits, _maxprec, _minexp = level  # type: ignore[misc]
            raise NotImplementedError
    return zfp.compress_numpy(data, **kwargs)


@notimplemented(zfp)
def zfp_decode(
    data: Buffer,
    /,
    *,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decompress ZFP."""
    return zfp.decompress_numpy(data)


@notimplemented(bitshuffle)
def bitshuffle_lz4_encode(
    data: NDArray[Any],
    /,
    level: int | None = 1,
    *,
    blocksize: int = 0,
    out: BytesOut | None = None,
) -> NDArray[Any]:
    """Compress LZ4 with Bitshuffle."""
    return bitshuffle.compress_lz4(data, blocksize)


@notimplemented(bitshuffle)
def bitshuffle_lz4_decode(
    data: NDArray[Any],
    /,
    shape: Sequence[int],
    dtype: DTypeLike,
    *,
    blocksize: int = 0,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decompress LZ4 with Bitshuffle."""
    return bitshuffle.decompress_lz4(data, shape, dtype, blocksize)


@notimplemented(lz4)
def lz4_encode(
    data: Buffer,
    /,
    level: int | None = 1,
    *,
    header: bool = False,
    out: BytesOut | None = None,
) -> BytesResult:
    """Compress LZ4."""
    return lz4.block.compress(data, store_size=header)


@notimplemented(lz4)
def lz4_decode(
    data: Buffer,
    /,
    *,
    header: bool = False,
    out: NDArray[Any] | BytesOut | None = None,
) -> BytesResult:
    """Decompress LZ4."""
    if header:
        return lz4.block.decompress(data)
    if isinstance(out, int | numpy.integer):
        return lz4.block.decompress(data, uncompressed_size=out)
    outsize = max(24, 24 + 255 * (len(memoryview(data)) - 10))  # ugh
    return lz4.block.decompress(data, uncompressed_size=outsize)


@notimplemented(tifffile)
def tiff_decode(
    data: Buffer,
    /,
    index: int | Sequence[int] | slice | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Decode TIFF."""
    with io.BytesIO(data) as fh:
        return tifffile.imread(fh, key=index, **kwargs)


@notimplemented(tifffile)
def tiff_encode(
    data: ArrayLike,
    /,
    level: int | None = 1,
    **kwargs: Any,
) -> BytesResult:
    """Encode TIFF."""
    with io.BytesIO() as fh:
        tifffile.imwrite(fh, data, **kwargs)
        fh.seek(0)
        return fh.read()


@notimplemented(pillow)
def pil_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode image data using Pillow."""
    return numpy.asarray(pillow.Image.open(io.BytesIO(data)))


@notimplemented(pillow)
def jpeg_decode(
    data: Buffer,
    /,
    *,
    tables: bytes | None = None,
    colorspace: int | str | None = None,
    outcolorspace: int | str | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode JPEG 8-bit."""
    return pil_decode(data)


@notimplemented(pillow)
def jpeg2k_decode(
    data: Buffer,
    /,
    *,
    verbose: int | None = None,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode JPEG 2000."""
    return pil_decode(data)


@notimplemented(pillow)
def webp_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode WebP."""
    return pil_decode(data)


@notimplemented(pillow)
def png_decode(
    data: Buffer,
    /,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Decode PNG."""
    return pil_decode(data)


if __name__ == '__main__':
    import doctest

    print(version())
    numpy.set_printoptions(suppress=True, precision=2)
    doctest.testmod()

# mypy: disable-error-code="union-attr, no-any-return, no-redef"
