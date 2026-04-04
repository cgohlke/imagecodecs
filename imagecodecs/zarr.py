# imagecodecs/zarr.py

# Copyright (c) 2026, Christoph Gohlke
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

"""Zarr version 3 codecs implemented using imagecodecs.

ArrayArrayCodec:
    Bitorder, Bitshuffle, Byteshuffle, Cms, Delta, Floatpred, Quantize, Xor

ArrayBytesCodec:
    Apng, Avif, Bfloat16, Bmp, Ccittfax3, Ccittfax4, Ccittrle, Dds, Dicomrle,
    Eer, Exr, Float24, Gif, Hcomp, Heif, Htj2k, Jpeg, Jpeg2k, Jpegls, Jpegxl,
    Jpegxr, Jpegxs, Lerc, Ljpeg, Meshopt, Packints, Pcodec, Pixarlog, Plio,
    Png, Qoi, Rcomp, Rgbe, Sperr, Spng, Sz3, Tiff, Ultrahdr, Webp, Wic, Zfp

BytesBytesCodec:
    Aec, Blosc, Blosc2, Brotli, Bz2, Checksum, Deflate, Lz4, Lz4f, Lz4h5,
    Lzf, Lzfse, Lzham, Lzma, Lzo, Lzw, Packbits, Pglz, Snappy, Szip, Zlib,
    Zlibng, Zopfli, Zstd

"""

from __future__ import annotations

__all__ = [
    'Aec',
    'Apng',
    'Avif',
    'Bfloat16',
    'Bitorder',
    'Bitshuffle',
    'Blosc',
    'Blosc2',
    'Bmp',
    'Brotli',
    'Byteshuffle',
    'Bz2',
    'Ccittfax3',
    'Ccittfax4',
    'Ccittrle',
    'Checksum',
    'Cms',
    'Dds',
    'Deflate',
    'Delta',
    'Dicomrle',
    'Eer',
    'Exr',
    'Float24',
    'Floatpred',
    'Gif',
    'Hcomp',
    'Heif',
    'Htj2k',
    'Jpeg',
    'Jpeg2k',
    'Jpegls',
    'Jpegxl',
    'Jpegxr',
    'Jpegxs',
    'Lerc',
    'Ljpeg',
    'Lz4',
    'Lz4f',
    'Lz4h5',
    'Lzf',
    'Lzfse',
    'Lzham',
    'Lzma',
    'Lzo',
    'Lzw',
    'Meshopt',
    'Packbits',
    'Packints',
    'Pcodec',
    'Pglz',
    'Pixarlog',
    'Plio',
    'Png',
    'Qoi',
    'Quantize',
    'Rcomp',
    'Rgbe',
    'Snappy',
    'Sperr',
    'Spng',
    'Sz3',
    'Szip',
    'Tiff',
    'Ultrahdr',
    'Wavpack',
    'Webp',
    'Wic',
    'Xor',
    'Zfp',
    'Zlib',
    'Zlibng',
    'Zopfli',
    'Zstd',
    'register_codecs',
]

import asyncio
import base64
import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy

import imagecodecs

if TYPE_CHECKING:
    from collections.abc import Container
    from typing import Any, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.common import JSON

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.core.common import parse_named_configuration


@dataclass(frozen=True)
class Aec(BytesBytesCodec):
    """AEC codec for Zarr 3."""

    is_fixed_size = False

    bitspersample: int | None = None
    flags: int | None = None
    blocksize: int | None = None
    rsi: int | None = None

    def __init__(
        self,
        *,
        bitspersample: int | None = None,
        flags: int | None = None,
        blocksize: int | None = None,
        rsi: int | None = None,
    ) -> None:
        if not imagecodecs.AEC.available:
            msg = 'imagecodecs.AEC not available'
            raise ValueError(msg)
        _setattrs(
            self,
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            flags=None if flags is None else int(flags),
            blocksize=None if blocksize is None else int(blocksize),
            rsi=None if rsi is None else int(rsi),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'aec'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.bitspersample is not None:
            cfg['bitspersample'] = self.bitspersample
        if self.flags is not None:
            cfg['flags'] = self.flags
        if self.blocksize is not None:
            cfg['blocksize'] = self.blocksize
        if self.rsi is not None:
            cfg['rsi'] = self.rsi
        if cfg:
            return {'name': 'imagecodecs_aec', 'configuration': cfg}
        return {'name': 'imagecodecs_aec'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        decoded = imagecodecs.aec_decode(
            data,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        encoded = imagecodecs.aec_encode(
            data,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Apng(ArrayBytesCodec):
    """APNG codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    strategy: str | None = None
    filter: str | None = None
    photometric: str | None = None
    delay: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        strategy: imagecodecs.APNG.STRATEGY | int | str | None = None,
        filter: imagecodecs.APNG.FILTER | int | str | None = None,
        photometric: imagecodecs.APNG.COLOR_TYPE | int | str | None = None,
        delay: int | None = None,
    ) -> None:
        if not imagecodecs.APNG.available:
            msg = 'imagecodecs.APNG not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            strategy=_enum_name(strategy, imagecodecs.APNG.STRATEGY),
            filter=_enum_name(filter, imagecodecs.APNG.FILTER),
            photometric=_enum_name(photometric, imagecodecs.APNG.COLOR_TYPE),
            delay=None if delay is None else int(delay),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'apng'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'strategy', 'filter', 'photometric', 'delay'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_apng', 'configuration': cfg}
        return {'name': 'imagecodecs_apng'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.apng_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.apng_encode(
            arr,
            level=self.level,
            strategy=self.strategy,
            filter=self.filter,
            photometric=self.photometric,
            delay=self.delay,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Avif(ArrayBytesCodec):
    """AVIF codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    speed: int | None = None
    tilelog2: tuple[int, int] | None = None
    bitspersample: int | None = None
    pixelformat: str | None = None
    codec: str | None = None
    primaries: str | None = None
    transfer: str | None = None
    matrix: str | None = None
    numthreads: int | None = None
    index: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        speed: int | None = None,
        tilelog2: tuple[int, int] | None = None,
        bitspersample: int | None = None,
        pixelformat: imagecodecs.AVIF.PIXEL_FORMAT | int | str | None = None,
        codec: imagecodecs.AVIF.CODEC_CHOICE | int | str | None = None,
        primaries: imagecodecs.AVIF.COLOR_PRIMARIES | int | str | None = None,
        transfer: (
            imagecodecs.AVIF.TRANSFER_CHARACTERISTICS | int | str | None
        ) = None,
        matrix: imagecodecs.AVIF.MATRIX_COEFFICIENTS | int | str | None = None,
        numthreads: int | None = None,
        index: int | None = None,
    ) -> None:
        if not imagecodecs.AVIF.available:
            msg = 'imagecodecs.AVIF not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            speed=None if speed is None else int(speed),
            tilelog2=(
                None
                if tilelog2 is None
                else (int(tilelog2[0]), int(tilelog2[1]))
            ),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            pixelformat=_enum_name(pixelformat, imagecodecs.AVIF.PIXEL_FORMAT),
            codec=_enum_name(codec, imagecodecs.AVIF.CODEC_CHOICE),
            primaries=_enum_name(primaries, imagecodecs.AVIF.COLOR_PRIMARIES),
            transfer=_enum_name(
                transfer, imagecodecs.AVIF.TRANSFER_CHARACTERISTICS
            ),
            matrix=_enum_name(matrix, imagecodecs.AVIF.MATRIX_COEFFICIENTS),
            numthreads=None if numthreads is None else int(numthreads),
            index=None if index is None else int(index),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'avif'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'speed',
            'tilelog2',
            'bitspersample',
            'pixelformat',
            'codec',
            'primaries',
            'transfer',
            'matrix',
            'numthreads',
            'index',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_avif', 'configuration': cfg}
        return {'name': 'imagecodecs_avif'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.avif_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.avif_encode(
            arr,
            level=self.level,
            speed=self.speed,
            tilelog2=self.tilelog2,
            bitspersample=self.bitspersample,
            pixelformat=self.pixelformat,
            codec=self.codec,
            primaries=self.primaries,
            transfer=self.transfer,
            matrix=self.matrix,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Bfloat16(ArrayBytesCodec):
    """Bfloat16 codec for Zarr 3."""

    is_fixed_size = False

    byteorder: Literal['>', '<', '='] | None = None
    rounding: str | None = None

    def __init__(
        self,
        *,
        byteorder: Literal['>', '<', '='] | None = None,
        rounding: imagecodecs.BFLOAT16.ROUND | int | str | None = None,
    ) -> None:
        if not imagecodecs.BFLOAT16.available:
            msg = 'imagecodecs.BFLOAT16 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            byteorder=byteorder,
            rounding=_enum_name(rounding, imagecodecs.BFLOAT16.ROUND),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'bfloat16'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.byteorder is not None:
            cfg['byteorder'] = self.byteorder
        if self.rounding is not None:
            cfg['rounding'] = self.rounding
        if cfg:
            return {
                'name': 'imagecodecs_bfloat16',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_bfloat16'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.bfloat16_decode(
            chunk_bytes.as_numpy_array(), byteorder=self.byteorder
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # cheap transform, no thread dispatch
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.bfloat16_encode(
            chunk_array.as_numpy_array(),
            byteorder=self.byteorder,
            rounding=self.rounding,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        # float32 (4 bytes) -> bfloat16 (2 bytes): half the input
        return input_byte_length // 2


@dataclass(frozen=True)
class Bitorder(ArrayArrayCodec):
    """Bitorder codec for Zarr 3."""

    is_fixed_size = True

    def __init__(self) -> None:
        if not imagecodecs.BITORDER.available:
            msg = 'imagecodecs.BITORDER not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'bitorder')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_bitorder'}

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.bitorder_decode(chunk_array.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.bitorder_encode(chunk_array.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Bitshuffle(ArrayArrayCodec):
    """Bitshuffle codec for Zarr 3."""

    is_fixed_size = True

    blocksize: int = 0

    def __init__(self, *, blocksize: int = 0) -> None:
        if not imagecodecs.BITSHUFFLE.available:
            msg = 'imagecodecs.BITSHUFFLE not available'
            raise ValueError(msg)
        _setattrs(self, blocksize=int(blocksize))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'bitshuffle'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_bitshuffle',
            'configuration': {'blocksize': self.blocksize},
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.bitshuffle_decode(
            chunk_array.as_numpy_array(),
            itemsize=dtype.itemsize,
            blocksize=self.blocksize,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            numpy.frombuffer(decoded, dtype=dtype).reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        dtype = chunk_spec.dtype.to_native_dtype()
        encoded = imagecodecs.bitshuffle_encode(
            chunk_array.as_numpy_array(),
            itemsize=dtype.itemsize,
            blocksize=self.blocksize,
        )
        if isinstance(encoded, numpy.ndarray):
            return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            numpy.frombuffer(encoded, dtype=numpy.uint8)
        )

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Blosc(BytesBytesCodec):
    """Blosc codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    compressor: str | None = None
    shuffle: str | None = None
    typesize: int | None = None
    blocksize: int | None = None
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        compressor: imagecodecs.BLOSC.COMPRESSOR | int | str | None = None,
        shuffle: imagecodecs.BLOSC.SHUFFLE | int | str | None = None,
        typesize: int | None = None,
        blocksize: int | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.BLOSC.available:
            msg = 'imagecodecs.BLOSC not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            compressor=_enum_name(compressor, imagecodecs.BLOSC.COMPRESSOR),
            shuffle=_enum_name(shuffle, imagecodecs.BLOSC.SHUFFLE),
            typesize=None if typesize is None else int(typesize),
            blocksize=None if blocksize is None else int(blocksize),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'blosc'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'compressor',
            'shuffle',
            'typesize',
            'blocksize',
            'numthreads',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_blosc', 'configuration': cfg}
        return {'name': 'imagecodecs_blosc'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        decoded = imagecodecs.blosc_decode(data, numthreads=self.numthreads)
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        encoded = imagecodecs.blosc_encode(
            data,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Blosc2(BytesBytesCodec):
    """Blosc2 codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    compressor: str | None = None
    shuffle: str | None = None
    splitmode: str | None = None
    typesize: int | None = None
    blocksize: int | None = None
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        compressor: imagecodecs.BLOSC2.COMPRESSOR | int | str | None = None,
        shuffle: imagecodecs.BLOSC2.FILTER | int | str | None = None,
        splitmode: imagecodecs.BLOSC2.SPLIT | int | str | None = None,
        typesize: int | None = None,
        blocksize: int | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.BLOSC2.available:
            msg = 'imagecodecs.BLOSC2 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            compressor=_enum_name(compressor, imagecodecs.BLOSC2.COMPRESSOR),
            shuffle=_enum_name(shuffle, imagecodecs.BLOSC2.FILTER),
            splitmode=_enum_name(splitmode, imagecodecs.BLOSC2.SPLIT),
            typesize=None if typesize is None else int(typesize),
            blocksize=None if blocksize is None else int(blocksize),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'blosc2'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'compressor',
            'shuffle',
            'splitmode',
            'typesize',
            'blocksize',
            'numthreads',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_blosc2', 'configuration': cfg}
        return {'name': 'imagecodecs_blosc2'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        decoded = imagecodecs.blosc2_decode(data, numthreads=self.numthreads)
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        encoded = imagecodecs.blosc2_encode(
            data,
            level=self.level,
            compressor=self.compressor,
            shuffle=self.shuffle,
            splitmode=self.splitmode,
            typesize=self.typesize,
            blocksize=self.blocksize,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Bmp(ArrayBytesCodec):
    """BMP codec for Zarr 3."""

    is_fixed_size = False

    ppm: int | None = None
    asrgb: bool | None = None

    def __init__(
        self,
        *,
        ppm: int | None = None,
        asrgb: bool | None = None,
    ) -> None:
        if not imagecodecs.BMP.available:
            msg = 'imagecodecs.BMP not available'
            raise ValueError(msg)
        _setattrs(
            self,
            ppm=None if ppm is None else max(1, int(ppm)),
            asrgb=None if asrgb is None else bool(asrgb),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'bmp'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.ppm is not None:
            cfg['ppm'] = self.ppm
        if self.asrgb is not None:
            cfg['asrgb'] = self.asrgb
        if cfg:
            return {'name': 'imagecodecs_bmp', 'configuration': cfg}
        return {'name': 'imagecodecs_bmp'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.bmp_decode(
            chunk_bytes.as_numpy_array(), asrgb=self.asrgb
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.bmp_encode(arr, ppm=self.ppm)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Brotli(BytesBytesCodec):
    """Brotli codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    mode: str | None = None
    lgwin: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        mode: imagecodecs.BROTLI.MODE | int | str | None = None,
        lgwin: int | None = None,
    ) -> None:
        if not imagecodecs.BROTLI.available:
            msg = 'imagecodecs.BROTLI not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            mode=_enum_name(mode, imagecodecs.BROTLI.MODE),
            lgwin=None if lgwin is None else int(lgwin),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'brotli'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.mode is not None:
            cfg['mode'] = self.mode
        if self.lgwin is not None:
            cfg['lgwin'] = self.lgwin
        if cfg:
            return {'name': 'imagecodecs_brotli', 'configuration': cfg}
        return {'name': 'imagecodecs_brotli'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.brotli_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.brotli_encode(
            chunk_bytes.as_numpy_array(),
            level=self.level,
            mode=self.mode,
            lgwin=self.lgwin,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Byteshuffle(ArrayArrayCodec):
    """Byteshuffle codec for Zarr 3."""

    is_fixed_size = True

    axis: int = -1
    dist: int = 1
    delta: bool = False
    reorder: bool = False

    def __init__(
        self,
        *,
        axis: int = -1,
        dist: int = 1,
        delta: bool = False,
        reorder: bool = False,
    ) -> None:
        if not imagecodecs.BYTESHUFFLE.available:
            msg = 'imagecodecs.BYTESHUFFLE not available'
            raise ValueError(msg)
        _setattrs(
            self,
            axis=int(axis),
            dist=int(dist),
            delta=bool(delta),
            reorder=bool(reorder),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'byteshuffle'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_byteshuffle',
            'configuration': {
                'axis': self.axis,
                'dist': self.dist,
                'delta': self.delta,
                'reorder': self.reorder,
            },
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.byteshuffle_decode(
            chunk_array.as_numpy_array(),
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.byteshuffle_encode(
            chunk_array.as_numpy_array(),
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Bz2(BytesBytesCodec):
    """Bz2 codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.BZ2.available:
            msg = 'imagecodecs.BZ2 not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'bz2'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_bz2',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_bz2'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.bz2_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.bz2_encode(
            chunk_bytes.as_numpy_array(), level=self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Ccittfax3(ArrayBytesCodec):
    """CCITT Fax3 codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    height: int = 0
    width: int = 0
    t4options: int = 0

    def __init__(
        self,
        *,
        height: int = 0,
        width: int = 0,
        t4options: int = 0,
    ) -> None:
        if not imagecodecs.CCITTFAX3.available:
            msg = 'imagecodecs.CCITTFAX3 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            height=int(height),
            width=int(width),
            t4options=int(t4options),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'ccittfax3'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.height:
            cfg['height'] = self.height
        if self.width:
            cfg['width'] = self.width
        if self.t4options:
            cfg['t4options'] = self.t4options
        if cfg:
            return {
                'name': 'imagecodecs_ccittfax3',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_ccittfax3'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.ccittfax3_decode(
            chunk_bytes.as_numpy_array(),
            height=self.height,
            width=self.width,
            t4options=self.t4options,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT Fax3 encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT Fax3 encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Ccittfax4(ArrayBytesCodec):
    """CCITT Fax4 codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    height: int = 0
    width: int = 0

    def __init__(
        self,
        *,
        height: int = 0,
        width: int = 0,
    ) -> None:
        if not imagecodecs.CCITTFAX4.available:
            msg = 'imagecodecs.CCITTFAX4 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            height=int(height),
            width=int(width),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'ccittfax4'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.height:
            cfg['height'] = self.height
        if self.width:
            cfg['width'] = self.width
        if cfg:
            return {
                'name': 'imagecodecs_ccittfax4',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_ccittfax4'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.ccittfax4_decode(
            chunk_bytes.as_numpy_array(),
            height=self.height,
            width=self.width,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT Fax4 encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT Fax4 encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Ccittrle(ArrayBytesCodec):
    """CCITT RLE codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    height: int = 0
    width: int = 0

    def __init__(
        self,
        *,
        height: int = 0,
        width: int = 0,
    ) -> None:
        if not imagecodecs.CCITTRLE.available:
            msg = 'imagecodecs.CCITTRLE not available'
            raise ValueError(msg)
        _setattrs(
            self,
            height=int(height),
            width=int(width),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'ccittrle'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.height:
            cfg['height'] = self.height
        if self.width:
            cfg['width'] = self.width
        if cfg:
            return {
                'name': 'imagecodecs_ccittrle',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_ccittrle'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.ccittrle_decode(
            chunk_bytes.as_numpy_array(),
            height=self.height,
            width=self.width,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT RLE encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'CCITT RLE encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Checksum(BytesBytesCodec):
    """Checksum codec for Zarr 3."""

    is_fixed_size = False

    kind: str = 'crc32'
    value: int | None = None
    prefix: bytes | None = None
    prepend: bool = True
    byteorder: Literal['little', 'big'] = 'little'
    _checksum: Any = field(default=None, init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        kind: Literal[
            'crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc'
        ] = 'crc32',
        value: int | None = None,
        prefix: bytes | None = None,
        prepend: bool | None = None,
        byteorder: Literal['<', '>', 'little', 'big'] = '<',
    ) -> None:
        # validate kind and set checksum function
        match kind:
            case 'crc32':
                if imagecodecs.ZLIBNG.available:
                    checksum = imagecodecs.zlibng_crc32
                elif imagecodecs.DEFLATE.available:
                    checksum = imagecodecs.deflate_crc32
                elif imagecodecs.ZLIB.available:
                    checksum = imagecodecs.zlib_crc32
                else:
                    msg = 'imagecodecs.ZLIB not available'
                    raise ValueError(msg)
                if prepend is None:
                    prepend = True
            case 'adler32':
                if imagecodecs.ZLIBNG.available:
                    checksum = imagecodecs.zlibng_adler32
                elif imagecodecs.DEFLATE.available:
                    checksum = imagecodecs.deflate_adler32
                elif imagecodecs.ZLIB.available:
                    checksum = imagecodecs.zlib_adler32
                else:
                    msg = 'imagecodecs.ZLIB not available'
                    raise ValueError(msg)
                if prepend is None:
                    prepend = True
            case 'fletcher32':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                checksum = imagecodecs.h5checksum_fletcher32
                if prepend is None:
                    prepend = False
            case 'lookup3':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                checksum = imagecodecs.h5checksum_lookup3
                if prepend is None:
                    prepend = False
            case 'h5crc':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                checksum = imagecodecs.h5checksum_crc
                if prepend is None:
                    prepend = False
            case _:
                msg = f'checksum {kind=!r} not supported'  # type: ignore[unreachable]
                raise ValueError(msg)

        byteorder_map: dict[str, Literal['little', 'big']] = {
            '<': 'little',
            '>': 'big',
            'little': 'little',
            'big': 'big',
        }

        _setattrs(
            self,
            kind=kind,
            value=None if value is None else int(value),
            prefix=prefix,
            prepend=bool(prepend),
            byteorder=byteorder_map[byteorder],
            _checksum=checksum,
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        cfg = _parse_config(data, 'checksum')
        # prefix may be base64-encoded
        prefix = cfg.get('prefix')
        if prefix is not None and isinstance(prefix, str):
            cfg['prefix'] = base64.b64decode(prefix.encode())
        return cls(**cfg)

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {'kind': self.kind}
        if self.value is not None:
            cfg['value'] = self.value
        if self.prefix is not None:
            cfg['prefix'] = base64.b64encode(self.prefix).decode()
        cfg['prepend'] = self.prepend
        cfg['byteorder'] = self.byteorder
        return {
            'name': 'imagecodecs_checksum',
            'configuration': cfg,
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        out = memoryview(chunk_bytes.as_numpy_array())
        if self.prepend:
            expect = int.from_bytes(out[:4], self.byteorder)
            out = out[4:]
        else:
            expect = int.from_bytes(out[-4:], self.byteorder)
            out = out[:-4]
        if self.prefix is None:
            checksum = self._checksum(out, self.value)
        else:
            checksum = self._checksum(self.prefix + out, self.value)
        if checksum != expect:
            msg = (
                f'{self._checksum.__name__} checksum mismatch '
                f'{checksum} != {expect}'
            )
            raise RuntimeError(msg)
        return chunk_spec.prototype.buffer.from_bytes(bytes(out))

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        # checksum is cheap, no thread dispatch
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        buf = bytes(chunk_bytes.as_numpy_array())
        if self.prefix is None:
            checksum = self._checksum(buf, self.value)
        else:
            checksum = self._checksum(self.prefix + buf, self.value)
        out = bytearray(len(buf) + 4)
        if self.prepend:
            out[:4] = checksum.to_bytes(4, self.byteorder)
            out[4:] = buf
        else:
            out[:-4] = buf
            out[-4:] = checksum.to_bytes(4, self.byteorder)
        return chunk_spec.prototype.buffer.from_bytes(out)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return self._encode_sync(chunk_bytes, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length + 4


@dataclass(frozen=True)
class Cms(ArrayArrayCodec):
    """CMS (Color Management System) codec for Zarr 3."""

    is_fixed_size = True

    profile: bytes = b''
    outprofile: bytes = b''
    colorspace: str | None = None
    outcolorspace: str | None = None
    planar: bool | None = None
    outplanar: bool | None = None
    intent: str | None = None
    flags: int | None = None

    def __init__(
        self,
        *,
        profile: bytes | str,
        outprofile: bytes | str,
        colorspace: str | None = None,
        outcolorspace: str | None = None,
        planar: bool | None = None,
        outplanar: bool | None = None,
        intent: imagecodecs.CMS.INTENT | int | str | None = None,
        flags: int | None = None,
    ) -> None:
        if not imagecodecs.CMS.available:
            msg = 'imagecodecs.CMS not available'
            raise ValueError(msg)
        if isinstance(profile, str):
            profile = imagecodecs.cms_profile(profile)
        if isinstance(outprofile, str):
            outprofile = imagecodecs.cms_profile(outprofile)
        _setattrs(
            self,
            profile=profile,
            outprofile=outprofile,
            colorspace=colorspace,
            outcolorspace=outcolorspace,
            planar=None if planar is None else bool(planar),
            outplanar=None if outplanar is None else bool(outplanar),
            intent=_enum_name(intent, imagecodecs.CMS.INTENT),
            flags=None if flags is None else int(flags),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        cfg = _parse_config(data, 'cms')
        for key in ('profile', 'outprofile'):
            value = cfg.get(key)
            if value is not None and isinstance(value, str):
                cfg[key] = base64.b64decode(value.encode())
        return cls(**cfg)

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('profile', 'outprofile'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = base64.b64encode(value).decode()
        for key in (
            'colorspace',
            'outcolorspace',
            'planar',
            'outplanar',
            'intent',
            'flags',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_cms', 'configuration': cfg}
        return {'name': 'imagecodecs_cms'}

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.cms_transform(
            chunk_array.as_numpy_array(),
            self.outprofile,
            self.profile,
            colorspace=self.outcolorspace,
            outcolorspace=self.colorspace,
            planar=self.outplanar,
            outplanar=self.planar,
            intent=self.intent,
            flags=self.flags,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.cms_transform(
            chunk_array.as_numpy_array(),
            self.profile,
            self.outprofile,
            colorspace=self.colorspace,
            outcolorspace=self.outcolorspace,
            planar=self.planar,
            outplanar=self.outplanar,
            intent=self.intent,
            flags=self.flags,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Dds(ArrayBytesCodec):
    """DDS codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    mipmap: int = 0

    def __init__(self, *, mipmap: int = 0) -> None:
        if not imagecodecs.DDS.available:
            msg = 'imagecodecs.DDS not available'
            raise ValueError(msg)
        _setattrs(self, mipmap=int(mipmap))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'dds'))

    def to_dict(self) -> dict[str, JSON]:
        if self.mipmap:
            return {
                'name': 'imagecodecs_dds',
                'configuration': {'mipmap': self.mipmap},
            }
        return {'name': 'imagecodecs_dds'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.dds_decode(
            chunk_bytes.as_numpy_array(), mipmap=self.mipmap
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'DDS encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'DDS encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Deflate(BytesBytesCodec):
    """Deflate codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    raw: bool = False

    def __init__(
        self,
        *,
        level: int | None = None,
        raw: bool = False,
    ) -> None:
        if not imagecodecs.DEFLATE.available:
            msg = 'imagecodecs.DEFLATE not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            raw=bool(raw),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'deflate'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.raw:
            cfg['raw'] = self.raw
        if cfg:
            return {
                'name': 'imagecodecs_deflate',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_deflate'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.deflate_decode(
            chunk_bytes.as_numpy_array(), raw=self.raw
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.deflate_encode(
            chunk_bytes.as_numpy_array(), level=self.level, raw=self.raw
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Delta(ArrayArrayCodec):
    """Delta codec for Zarr 3."""

    is_fixed_size = True

    axis: int = -1
    dist: int = 1

    def __init__(
        self,
        *,
        axis: int = -1,
        dist: int = 1,
    ) -> None:
        if not imagecodecs.DELTA.available:
            msg = 'imagecodecs.DELTA not available'
            raise ValueError(msg)
        _setattrs(
            self,
            axis=int(axis),
            dist=int(dist),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'delta'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_delta',
            'configuration': {'axis': self.axis, 'dist': self.dist},
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.delta_decode(
            chunk_array.as_numpy_array(), axis=self.axis, dist=self.dist
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.delta_encode(
            chunk_array.as_numpy_array(), axis=self.axis, dist=self.dist
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Dicomrle(ArrayBytesCodec):
    """DICOM RLE codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    dtype: str = 'uint16'

    def __init__(self, *, dtype: str = 'uint16') -> None:
        if not imagecodecs.DICOMRLE.available:
            msg = 'imagecodecs.DICOMRLE not available'
            raise ValueError(msg)
        _setattrs(self, dtype=numpy.dtype(dtype).str)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'dicomrle'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_dicomrle',
            'configuration': {'dtype': self.dtype},
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        raw = imagecodecs.dicomrle_decode(
            chunk_bytes.as_numpy_array(), self.dtype
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            numpy.frombuffer(raw, dtype=self.dtype).reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'DICOM RLE encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'DICOM RLE encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Eer(ArrayBytesCodec):
    """Electron Event Representation codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    shape: tuple[int, int]
    skipbits: int
    horzbits: int
    vertbits: int
    superres: int = 0

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        skipbits: int,
        horzbits: int,
        vertbits: int,
        superres: int = 0,
    ) -> None:
        if not imagecodecs.EER.available:
            msg = 'imagecodecs.EER not available'
            raise ValueError(msg)
        _setattrs(
            self,
            shape=(int(shape[0]), int(shape[1])),
            skipbits=int(skipbits),
            horzbits=int(horzbits),
            vertbits=int(vertbits),
            superres=int(superres),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        cfg = _parse_config(data, 'eer')
        if 'shape' in cfg:
            cfg['shape'] = tuple(cfg['shape'])
        return cls(**cfg)

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {
            'shape': list(self.shape),
            'skipbits': self.skipbits,
            'horzbits': self.horzbits,
            'vertbits': self.vertbits,
        }
        if self.superres:
            cfg['superres'] = self.superres
        return {
            'name': 'imagecodecs_eer',
            'configuration': cfg,
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.eer_decode(
            chunk_bytes.as_numpy_array(),
            self.shape,
            self.skipbits,
            self.horzbits,
            self.vertbits,
            superres=self.superres,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'EER encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'EER encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Exr(ArrayBytesCodec):
    """OpenEXR codec for Zarr 3."""

    is_fixed_size = False

    level: float | None = None
    compression: str | None = None
    planar: bool | None = None
    frames: bool | None = None
    index: int | None = None

    def __init__(
        self,
        *,
        level: float | None = None,
        compression: imagecodecs.EXR.COMPRESSION | int | str | None = None,
        planar: bool | None = None,
        frames: bool | None = None,
        index: int | None = None,
    ) -> None:
        if not imagecodecs.EXR.available:
            msg = 'imagecodecs.EXR not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else float(level),
            compression=_enum_name(compression, imagecodecs.EXR.COMPRESSION),
            planar=None if planar is None else bool(planar),
            frames=None if frames is None else bool(frames),
            index=None if index is None else int(index),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'exr'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'compression', 'planar', 'frames', 'index'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_exr', 'configuration': cfg}
        return {'name': 'imagecodecs_exr'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.exr_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            planar=self.planar,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.exr_encode(
            arr,
            level=self.level,
            compression=self.compression,
            planar=self.planar,
            frames=self.frames,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Float24(ArrayBytesCodec):
    """Float24 codec for Zarr 3."""

    is_fixed_size = False

    byteorder: Literal['>', '<', '='] | None = None
    rounding: str | None = None

    def __init__(
        self,
        *,
        byteorder: Literal['>', '<', '='] | None = None,
        rounding: imagecodecs.FLOAT24.ROUND | int | str | None = None,
    ) -> None:
        if not imagecodecs.FLOAT24.available:
            msg = 'imagecodecs.FLOAT24 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            byteorder=byteorder,
            rounding=_enum_name(rounding, imagecodecs.FLOAT24.ROUND),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'float24'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.byteorder is not None:
            cfg['byteorder'] = self.byteorder
        if self.rounding is not None:
            cfg['rounding'] = self.rounding
        if cfg:
            return {
                'name': 'imagecodecs_float24',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_float24'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.float24_decode(
            chunk_bytes.as_numpy_array(), byteorder=self.byteorder
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # cheap transform, no thread dispatch
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.float24_encode(
            chunk_array.as_numpy_array(),
            byteorder=self.byteorder,
            rounding=self.rounding,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        # cheap transform, no thread dispatch
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        # float32 (4 bytes) -> float24 (3 bytes): 3/4 of input
        return (input_byte_length // 4) * 3


@dataclass(frozen=True)
class Floatpred(ArrayArrayCodec):
    """Floating Point Predictor codec for Zarr 3."""

    is_fixed_size = True

    axis: int = -1
    dist: int = 1

    def __init__(
        self,
        *,
        axis: int = -1,
        dist: int = 1,
    ) -> None:
        if not imagecodecs.FLOATPRED.available:
            msg = 'imagecodecs.FLOATPRED not available'
            raise ValueError(msg)
        _setattrs(
            self,
            axis=int(axis),
            dist=int(dist),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'floatpred'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_floatpred',
            'configuration': {'axis': self.axis, 'dist': self.dist},
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # floatpred needs typed array; use chunk_spec dtype
        dtype = chunk_spec.dtype.to_native_dtype()
        arr = chunk_array.as_numpy_array()
        arr = numpy.frombuffer(arr, dtype=dtype).reshape(chunk_spec.shape)
        decoded = imagecodecs.floatpred_decode(
            arr, axis=self.axis, dist=self.dist
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.floatpred_encode(
            chunk_array.as_numpy_array(), axis=self.axis, dist=self.dist
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Gif(ArrayBytesCodec):
    """GIF codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.GIF.available:
            msg = 'imagecodecs.GIF not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'gif')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_gif'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.gif_decode(
            chunk_bytes.as_numpy_array(), asrgb=False
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.gif_encode(arr)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Heif(ArrayBytesCodec):
    """HEIF codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    bitspersample: int | None = None
    photometric: str | None = None
    compression: str | None = None
    index: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        bitspersample: int | None = None,
        photometric: imagecodecs.HEIF.COLORSPACE | int | str | None = None,
        compression: imagecodecs.HEIF.COMPRESSION | int | str | None = None,
        index: int | None = None,
    ) -> None:
        if not imagecodecs.HEIF.available:
            msg = 'imagecodecs.HEIF not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            photometric=_enum_name(photometric, imagecodecs.HEIF.COLORSPACE),
            compression=_enum_name(compression, imagecodecs.HEIF.COMPRESSION),
            index=None if index is None else int(index),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'heif'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'bitspersample',
            'photometric',
            'compression',
            'index',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_heif', 'configuration': cfg}
        return {'name': 'imagecodecs_heif'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.heif_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            photometric=self.photometric,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.heif_encode(
            arr,
            level=self.level,
            bitspersample=self.bitspersample,
            photometric=self.photometric,
            compression=self.compression,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Hcomp(ArrayBytesCodec):
    """Hcomp codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    smooth: int | None = None
    safe32: bool | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        smooth: int | None = None,
        safe32: bool | None = None,
    ) -> None:
        if not imagecodecs.HCOMP.available:
            msg = 'imagecodecs.HCOMP not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            smooth=None if smooth is None else int(smooth),
            safe32=None if safe32 is None else bool(safe32),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'hcomp'))

    def to_dict(self) -> dict[str, JSON]:
        conf: dict[str, JSON] = {}
        if self.level is not None:
            conf['level'] = self.level
        if self.smooth is not None:
            conf['smooth'] = self.smooth
        if self.safe32:
            conf['safe32'] = self.safe32
        if conf:
            return {'name': 'imagecodecs_hcomp', 'configuration': conf}
        return {'name': 'imagecodecs_hcomp'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.hcomp_decode(
            chunk_bytes.as_numpy_array(),
            smooth=0 if self.smooth is None else self.smooth,
            safe32=self.safe32,
        )
        decoded = decoded.astype(dtype, copy=False).reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = chunk_array.as_numpy_array()
        if arr.ndim != 2:
            arr = numpy.atleast_2d(numpy.squeeze(arr))
        encoded = imagecodecs.hcomp_encode(
            arr, level=0 if self.level is None else self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Htj2k(ArrayBytesCodec):
    """HTJ2K codec for Zarr 3."""

    is_fixed_size = False

    level: float | None = None
    rgb: bool | None = None
    planar: bool | None = None
    tile: tuple[int, int] | None = None
    resolutions: int | None = None
    reversible: bool | None = None
    tlm: bool | None = None
    tilepart: str | None = None
    skipres: int | None = None
    resilient: bool = False

    def __init__(
        self,
        *,
        level: float | None = None,
        rgb: bool | None = None,
        planar: bool | None = None,
        tile: tuple[int, int] | None = None,
        resolutions: int | None = None,
        reversible: bool | None = None,
        tlm: bool | None = None,
        tilepart: imagecodecs.HTJ2K.TILEPART | int | str | None = None,
        skipres: int | None = None,
        resilient: bool = False,
    ) -> None:
        if not imagecodecs.HTJ2K.available:
            msg = 'imagecodecs.HTJ2K not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else float(level),
            rgb=None if rgb is None else bool(rgb),
            planar=None if planar is None else bool(planar),
            tile=None if tile is None else (int(tile[0]), int(tile[1])),
            resolutions=None if resolutions is None else int(resolutions),
            reversible=None if reversible is None else bool(reversible),
            tlm=None if tlm is None else bool(tlm),
            tilepart=_enum_name(tilepart, imagecodecs.HTJ2K.TILEPART),
            skipres=None if skipres is None else int(skipres),
            resilient=bool(resilient),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'htj2k'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'rgb',
            'planar',
            'tile',
            'resolutions',
            'reversible',
            'tlm',
            'tilepart',
            'skipres',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if self.resilient:
            cfg['resilient'] = True
        if cfg:
            return {'name': 'imagecodecs_htj2k', 'configuration': cfg}
        return {'name': 'imagecodecs_htj2k'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.htj2k_decode(
            chunk_bytes.as_numpy_array(),
            planar=self.planar,
            skipres=self.skipres,
            resilient=self.resilient,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.htj2k_encode(
            arr,
            level=self.level,
            rgb=self.rgb,
            planar=self.planar,
            tile=self.tile,
            resolutions=self.resolutions,
            reversible=self.reversible,
            tlm=self.tlm,
            tilepart=self.tilepart,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpeg(ArrayBytesCodec):
    """JPEG codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    bitspersample: int | None = None
    tables: bytes | None = None
    header: bytes | None = None
    colorspace_data: str | None = None
    colorspace_jpeg: str | None = None
    subsampling: str | tuple[int, int] | None = None
    optimize: bool | None = None
    smoothing: bool | None = None
    lossless: bool | None = None
    predictor: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        bitspersample: int | None = None,
        tables: bytes | None = None,
        header: bytes | None = None,
        colorspace_data: imagecodecs.JPEG8.CS | int | str | None = None,
        colorspace_jpeg: imagecodecs.JPEG8.CS | int | str | None = None,
        subsampling: str | tuple[int, int] | None = None,
        optimize: bool | None = None,
        smoothing: bool | None = None,
        lossless: bool | None = None,
        predictor: int | None = None,
    ) -> None:
        if not imagecodecs.JPEG8.available:
            msg = 'imagecodecs.JPEG8 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            tables=tables,
            header=header,
            colorspace_data=_enum_name(colorspace_data, imagecodecs.JPEG8.CS),
            colorspace_jpeg=_enum_name(colorspace_jpeg, imagecodecs.JPEG8.CS),
            subsampling=subsampling,
            optimize=None if optimize is None else bool(optimize),
            smoothing=None if smoothing is None else bool(smoothing),
            lossless=None if lossless is None else bool(lossless),
            predictor=None if predictor is None else int(predictor),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        cfg = _parse_config(data, 'jpeg')
        for key in ('header', 'tables'):
            value = cfg.get(key)
            if value is not None and isinstance(value, str):
                cfg[key] = base64.b64decode(value.encode())
        return cls(**cfg)

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'bitspersample',
            'colorspace_data',
            'colorspace_jpeg',
            'subsampling',
            'optimize',
            'smoothing',
            'lossless',
            'predictor',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        for key in ('tables', 'header'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = base64.b64encode(value).decode()
        if cfg:
            return {'name': 'imagecodecs_jpeg', 'configuration': cfg}
        return {'name': 'imagecodecs_jpeg'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        buf: bytes | numpy.ndarray[Any, Any] = chunk_bytes.as_numpy_array()
        if self.header is not None:
            buf = b''.join((self.header, bytes(buf), b'\xff\xd9'))
        decoded = imagecodecs.jpeg8_decode(
            buf,
            tables=self.tables,
            colorspace=self.colorspace_jpeg,
            outcolorspace=self.colorspace_data,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpeg8_encode(
            arr,
            level=self.level,
            colorspace=self.colorspace_data,
            outcolorspace=self.colorspace_jpeg,
            subsampling=self.subsampling,
            optimize=self.optimize,
            smoothing=self.smoothing,
            lossless=self.lossless,
            predictor=self.predictor,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpeg2k(ArrayBytesCodec):
    """JPEG 2000 codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    codecformat: str | None = None
    colorspace: str | None = None
    planar: bool | None = None
    tile: tuple[int, int] | None = None
    bitspersample: int | None = None
    resolutions: int | None = None
    reversible: bool | None = None
    mct: bool = True
    verbose: int | None = None
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        codecformat: imagecodecs.JPEG2K.CODEC | int | str | None = None,
        colorspace: imagecodecs.JPEG2K.CLRSPC | int | str | None = None,
        planar: bool | None = None,
        tile: tuple[int, int] | None = None,
        bitspersample: int | None = None,
        resolutions: int | None = None,
        reversible: bool | None = None,
        mct: bool = True,
        verbose: int | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.JPEG2K.available:
            msg = 'imagecodecs.JPEG2K not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            codecformat=_enum_name(codecformat, imagecodecs.JPEG2K.CODEC),
            colorspace=_enum_name(colorspace, imagecodecs.JPEG2K.CLRSPC),
            planar=None if planar is None else bool(planar),
            tile=None if tile is None else (int(tile[0]), int(tile[1])),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            resolutions=None if resolutions is None else int(resolutions),
            reversible=None if reversible is None else bool(reversible),
            mct=bool(mct),
            verbose=None if verbose is None else int(verbose),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'jpeg2k'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'codecformat',
            'colorspace',
            'planar',
            'tile',
            'bitspersample',
            'resolutions',
            'reversible',
            'verbose',
            'numthreads',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if not self.mct:
            cfg['mct'] = False
        if cfg:
            return {'name': 'imagecodecs_jpeg2k', 'configuration': cfg}
        return {'name': 'imagecodecs_jpeg2k'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.jpeg2k_decode(
            chunk_bytes.as_numpy_array(),
            planar=self.planar,
            verbose=self.verbose,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpeg2k_encode(
            arr,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            planar=self.planar,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            mct=self.mct,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpegls(ArrayBytesCodec):
    """JPEG LS codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.JPEGLS.available:
            msg = 'imagecodecs.JPEGLS not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'jpegls'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_jpegls',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_jpegls'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.jpegls_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpegls_encode(arr, level=self.level)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpegxl(ArrayBytesCodec):
    """JPEG XL codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    effort: int | None = None
    distance: float | None = None
    lossless: bool | None = None
    decodingspeed: int | None = None
    photometric: str | None = None
    bitspersample: int | None = None
    planar: bool | None = None
    primaries: str | None = None
    transfer: str | None = None
    usecontainer: bool | None = None
    index: int | None = None
    keeporientation: bool | None = None
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        effort: int | None = None,
        distance: float | None = None,
        lossless: bool | None = None,
        decodingspeed: int | None = None,
        photometric: imagecodecs.JPEGXL.COLOR_SPACE | int | str | None = None,
        bitspersample: int | None = None,
        planar: bool | None = None,
        primaries: imagecodecs.JPEGXL.PRIMARIES | int | str | None = None,
        transfer: (
            imagecodecs.JPEGXL.TRANSFER_FUNCTION | int | str | None
        ) = None,
        usecontainer: bool | None = None,
        index: int | None = None,
        keeporientation: bool | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.JPEGXL.available:
            msg = 'imagecodecs.JPEGXL not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            effort=None if effort is None else int(effort),
            distance=None if distance is None else float(distance),
            lossless=None if lossless is None else bool(lossless),
            decodingspeed=(
                None if decodingspeed is None else int(decodingspeed)
            ),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            photometric=_enum_name(
                photometric, imagecodecs.JPEGXL.COLOR_SPACE
            ),
            planar=None if planar is None else bool(planar),
            primaries=_enum_name(primaries, imagecodecs.JPEGXL.PRIMARIES),
            transfer=_enum_name(
                transfer, imagecodecs.JPEGXL.TRANSFER_FUNCTION
            ),
            usecontainer=None if usecontainer is None else bool(usecontainer),
            index=None if index is None else int(index),
            keeporientation=(
                None if keeporientation is None else bool(keeporientation)
            ),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'jpegxl'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'effort',
            'distance',
            'lossless',
            'decodingspeed',
            'photometric',
            'bitspersample',
            'planar',
            'primaries',
            'transfer',
            'usecontainer',
            'index',
            'keeporientation',
            'numthreads',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_jpegxl', 'configuration': cfg}
        return {'name': 'imagecodecs_jpegxl'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.jpegxl_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            keeporientation=self.keeporientation,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpegxl_encode(
            arr,
            level=self.level,
            effort=self.effort,
            distance=self.distance,
            lossless=self.lossless,
            decodingspeed=self.decodingspeed,
            bitspersample=self.bitspersample,
            photometric=self.photometric,
            planar=self.planar,
            primaries=self.primaries,
            transfer=self.transfer,
            usecontainer=self.usecontainer,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpegxr(ArrayBytesCodec):
    """JPEG XR codec for Zarr 3."""

    is_fixed_size = False

    level: float | None = None
    photometric: str | None = None
    hasalpha: bool | None = None
    resolution: tuple[float, float] | None = None
    fp2int: bool = False

    def __init__(
        self,
        *,
        level: float | None = None,
        photometric: imagecodecs.JPEGXR.PI | int | str | None = None,
        hasalpha: bool | None = None,
        resolution: tuple[float, float] | None = None,
        fp2int: bool = False,
    ) -> None:
        if not imagecodecs.JPEGXR.available:
            msg = 'imagecodecs.JPEGXR not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else float(level),
            photometric=_enum_name(photometric, imagecodecs.JPEGXR.PI),
            hasalpha=None if hasalpha is None else bool(hasalpha),
            resolution=(
                None
                if resolution is None
                else (float(resolution[0]), float(resolution[1]))
            ),
            fp2int=bool(fp2int),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'jpegxr'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'photometric', 'hasalpha', 'resolution'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if self.fp2int:
            cfg['fp2int'] = True
        if cfg:
            return {'name': 'imagecodecs_jpegxr', 'configuration': cfg}
        return {'name': 'imagecodecs_jpegxr'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.jpegxr_decode(
            chunk_bytes.as_numpy_array(), fp2int=self.fp2int
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpegxr_encode(
            arr,
            level=self.level,
            photometric=self.photometric,
            hasalpha=self.hasalpha,
            resolution=self.resolution,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Jpegxs(ArrayBytesCodec):
    """JPEG XS codec for Zarr 3."""

    is_fixed_size = False

    config: str | None = None
    bitspersample: int | None = None
    verbose: int | None = None

    def __init__(
        self,
        *,
        config: str | None = None,
        bitspersample: int | None = None,
        verbose: int | None = None,
    ) -> None:
        if not imagecodecs.JPEGXS.available:
            msg = 'imagecodecs.JPEGXS not available'
            raise ValueError(msg)
        _setattrs(
            self,
            config=config,
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            verbose=None if verbose is None else int(verbose),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'jpegxs'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('config', 'bitspersample', 'verbose'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_jpegxs', 'configuration': cfg}
        return {'name': 'imagecodecs_jpegxs'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.jpegxs_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.jpegxs_encode(
            arr,
            config=self.config,
            bitspersample=self.bitspersample,
            verbose=self.verbose,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lerc(ArrayBytesCodec):
    """LERC codec for Zarr 3."""

    is_fixed_size = False

    level: float | None = None
    version: int | None = None
    planar: bool = False
    compression: Literal['zstd', 'deflate'] | None = None
    compressionargs: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        level: float | None = None,
        version: int | None = None,
        planar: bool = False,
        compression: Literal['zstd', 'deflate'] | None = None,
        compressionargs: dict[str, Any] | None = None,
    ) -> None:
        if not imagecodecs.LERC.available:
            msg = 'imagecodecs.LERC not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else float(level),
            version=None if version is None else int(version),
            planar=bool(planar),
            compression=compression,
            compressionargs=compressionargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lerc'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'version', 'compression', 'compressionargs'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if self.planar:
            cfg['planar'] = True
        if cfg:
            return {'name': 'imagecodecs_lerc', 'configuration': cfg}
        return {'name': 'imagecodecs_lerc'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.lerc_decode(
            chunk_bytes.as_numpy_array(), masks=False
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.lerc_encode(
            arr,
            level=self.level,
            version=self.version,
            planar=self.planar,
            compression=self.compression,
            compressionargs=self.compressionargs,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Ljpeg(ArrayBytesCodec):
    """LJPEG codec for Zarr 3."""

    is_fixed_size = False

    bitspersample: int | None = None

    def __init__(self, *, bitspersample: int | None = None) -> None:
        if not imagecodecs.LJPEG.available:
            msg = 'imagecodecs.LJPEG not available'
            raise ValueError(msg)
        _setattrs(
            self,
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'ljpeg'))

    def to_dict(self) -> dict[str, JSON]:
        if self.bitspersample is not None:
            return {
                'name': 'imagecodecs_ljpeg',
                'configuration': {'bitspersample': self.bitspersample},
            }
        return {'name': 'imagecodecs_ljpeg'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.ljpeg_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.ljpeg_encode(
            arr, bitspersample=self.bitspersample
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lz4(BytesBytesCodec):
    """LZ4 codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    hc: bool = False
    header: bool = False

    def __init__(
        self,
        *,
        level: int | None = None,
        hc: bool = False,
        header: bool = False,
    ) -> None:
        if not imagecodecs.LZ4.available:
            msg = 'imagecodecs.LZ4 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            hc=bool(hc),
            header=bool(header),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lz4'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.hc:
            cfg['hc'] = self.hc
        if self.header:
            cfg['header'] = self.header
        if cfg:
            return {'name': 'imagecodecs_lz4', 'configuration': cfg}
        return {'name': 'imagecodecs_lz4'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lz4_decode(
            chunk_bytes.as_numpy_array(), header=self.header
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lz4_encode(
            chunk_bytes.as_numpy_array(),
            level=self.level,
            hc=self.hc,
            header=self.header,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lz4f(BytesBytesCodec):
    """LZ4F codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    blocksizeid: int | None = None
    contentchecksum: bool | None = None
    blockchecksum: bool | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        blocksizeid: int | None = None,
        contentchecksum: bool | None = None,
        blockchecksum: bool | None = None,
    ) -> None:
        if not imagecodecs.LZ4F.available:
            msg = 'imagecodecs.LZ4F not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            blocksizeid=None if blocksizeid is None else int(blocksizeid),
            contentchecksum=(
                None if contentchecksum is None else bool(contentchecksum)
            ),
            blockchecksum=(
                None if blockchecksum is None else bool(blockchecksum)
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lz4f'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.blocksizeid is not None:
            cfg['blocksizeid'] = self.blocksizeid
        if self.contentchecksum is not None:
            cfg['contentchecksum'] = self.contentchecksum
        if self.blockchecksum is not None:
            cfg['blockchecksum'] = self.blockchecksum
        if cfg:
            return {'name': 'imagecodecs_lz4f', 'configuration': cfg}
        return {'name': 'imagecodecs_lz4f'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lz4f_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lz4f_encode(
            chunk_bytes.as_numpy_array(),
            level=self.level,
            blocksizeid=self.blocksizeid,
            contentchecksum=self.contentchecksum,
            blockchecksum=self.blockchecksum,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lz4h5(BytesBytesCodec):
    """LZ4H5 codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    blocksize: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        blocksize: int | None = None,
    ) -> None:
        if not imagecodecs.LZ4H5.available:
            msg = 'imagecodecs.LZ4H5 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            blocksize=None if blocksize is None else int(blocksize),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lz4h5'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.blocksize is not None:
            cfg['blocksize'] = self.blocksize
        if cfg:
            return {'name': 'imagecodecs_lz4h5', 'configuration': cfg}
        return {'name': 'imagecodecs_lz4h5'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lz4h5_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lz4h5_encode(
            chunk_bytes.as_numpy_array(),
            level=self.level,
            blocksize=self.blocksize,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzf(BytesBytesCodec):
    """LZF codec for Zarr 3."""

    is_fixed_size = False

    header: bool = True

    def __init__(self, *, header: bool = True) -> None:
        if not imagecodecs.LZF.available:
            msg = 'imagecodecs.LZF not available'
            raise ValueError(msg)
        _setattrs(self, header=bool(header))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lzf'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_lzf',
            'configuration': {'header': self.header},
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzf_decode(
            chunk_bytes.as_numpy_array(), header=self.header
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lzf_encode(
            chunk_bytes.as_numpy_array(), header=self.header
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzfse(BytesBytesCodec):
    """LZFSE codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.LZFSE.available:
            msg = 'imagecodecs.LZFSE not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'lzfse')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_lzfse'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzfse_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lzfse_encode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzham(BytesBytesCodec):
    """LZHAM codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.LZHAM.available:
            msg = 'imagecodecs.LZHAM not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lzham'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_lzham',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_lzham'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzham_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lzham_encode(
            chunk_bytes.as_numpy_array(), level=self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzma(BytesBytesCodec):
    """LZMA codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    check: str | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        check: imagecodecs.LZMA.CHECK | int | str | None = None,
    ) -> None:
        if not imagecodecs.LZMA.available:
            msg = 'imagecodecs.LZMA not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            check=_enum_name(check, imagecodecs.LZMA.CHECK),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lzma'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.check is not None:
            cfg['check'] = self.check
        if cfg:
            return {'name': 'imagecodecs_lzma', 'configuration': cfg}
        return {'name': 'imagecodecs_lzma'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzma_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lzma_encode(
            chunk_bytes.as_numpy_array(),
            level=self.level,
            check=self.check,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzo(BytesBytesCodec):
    """LZO codec for Zarr 3 (decode only)."""

    is_fixed_size = False

    header: bool = False

    def __init__(self, *, header: bool = False) -> None:
        if not imagecodecs.LZO.available:
            msg = 'imagecodecs.LZO not available'
            raise ValueError(msg)
        _setattrs(self, header=bool(header))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'lzo'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_lzo',
            'configuration': {'header': self.header},
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzo_decode(
            chunk_bytes.as_numpy_array(), header=self.header
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'LZO encode not supported'
        raise NotImplementedError(msg)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        msg = 'LZO encode not supported'
        raise NotImplementedError(msg)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Lzw(BytesBytesCodec):
    """LZW codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.LZW.available:
            msg = 'imagecodecs.LZW not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'lzw')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_lzw'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.lzw_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.lzw_encode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Meshopt(ArrayBytesCodec):
    """Meshopt codec for Zarr 3."""

    is_fixed_size = False

    items: int | None = None
    level: int | None = None

    def __init__(
        self,
        *,
        items: int | None = None,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.MESHOPT.available:
            msg = 'imagecodecs.MESHOPT not available'
            raise ValueError(msg)
        _setattrs(
            self,
            items=None if items is None else int(items),
            level=None if level is None else int(level),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'meshopt'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.items is not None:
            cfg['items'] = self.items
        if self.level is not None:
            cfg['level'] = self.level
        if cfg:
            return {
                'name': 'imagecodecs_meshopt',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_meshopt'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.meshopt_decode(
            chunk_bytes.as_numpy_array(),
            shape=chunk_spec.shape,
            dtype=dtype,
            items=self.items,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.meshopt_encode(
            chunk_array.as_numpy_array(),
            level=self.level,
            items=self.items,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Packbits(BytesBytesCodec):
    """PackBits codec for Zarr 3."""

    is_fixed_size = False

    axis: int | None = None

    def __init__(self, *, axis: int | None = None) -> None:
        if not imagecodecs.PACKBITS.available:
            msg = 'imagecodecs.PACKBITS not available'
            raise ValueError(msg)
        _setattrs(self, axis=None if axis is None else int(axis))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'packbits'))

    def to_dict(self) -> dict[str, JSON]:
        if self.axis is not None:
            return {
                'name': 'imagecodecs_packbits',
                'configuration': {'axis': self.axis},
            }
        return {'name': 'imagecodecs_packbits'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.packbits_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        if self.axis is not None:
            dtype = chunk_spec.dtype.to_native_dtype()
            data = data.view(dtype).reshape(chunk_spec.shape)
        encoded = imagecodecs.packbits_encode(data, axis=self.axis)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Packints(ArrayBytesCodec):
    """Packed integer codec for Zarr 3."""

    is_fixed_size = False

    bitspersample: int = 8
    bitorder: Literal['>', '<'] | None = None
    runlen: int = 0

    def __init__(
        self,
        *,
        bitspersample: int = 8,
        bitorder: Literal['>', '<'] | None = None,
        runlen: int = 0,
    ) -> None:
        if not imagecodecs.PACKINTS.available:
            msg = 'imagecodecs.PACKINTS not available'
            raise ValueError(msg)
        _setattrs(
            self,
            bitspersample=int(bitspersample),
            bitorder=bitorder,
            runlen=int(runlen),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'packints'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {
            'bitspersample': self.bitspersample,
            'runlen': self.runlen,
        }
        if self.bitorder is not None:
            cfg['bitorder'] = self.bitorder
        return {
            'name': 'imagecodecs_packints',
            'configuration': cfg,
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.packints_decode(
            chunk_bytes.as_numpy_array(),
            dtype,
            self.bitspersample,
            bitorder=self.bitorder,
            runlen=self.runlen,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # bit-packing can be costly; dispatch to thread
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.packints_encode(
            chunk_array.as_numpy_array(),
            self.bitspersample,
            bitorder=self.bitorder,
            runlen=self.runlen,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        # bit-packing can be costly; dispatch to thread
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Pcodec(ArrayBytesCodec):
    """Pcodec codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    pagesize: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        pagesize: int | None = None,
    ) -> None:
        if not imagecodecs.PCODEC.available:
            msg = 'imagecodecs.PCODEC not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            pagesize=None if pagesize is None else int(pagesize),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'pcodec'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.pagesize is not None:
            cfg['pagesize'] = self.pagesize
        if cfg:
            return {'name': 'imagecodecs_pcodec', 'configuration': cfg}
        return {'name': 'imagecodecs_pcodec'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.pcodec_decode(
            chunk_bytes.as_numpy_array(),
            shape=chunk_spec.shape,
            dtype=dtype,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.pcodec_encode(
            chunk_array.as_numpy_array(),
            level=self.level,
            pagesize=self.pagesize,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Pglz(BytesBytesCodec):
    """PGLZ codec for Zarr 3."""

    is_fixed_size = False

    header: bool = True
    strategy: str | tuple[int, int, int, int, int, int] | None = None
    checkcomplete: bool | None = None

    def __init__(
        self,
        *,
        header: bool = True,
        strategy: str | tuple[int, int, int, int, int, int] | None = None,
        checkcomplete: bool | None = None,
    ) -> None:
        if not imagecodecs.PGLZ.available:
            msg = 'imagecodecs.PGLZ not available'
            raise ValueError(msg)
        _setattrs(
            self,
            header=bool(header),
            strategy=strategy,
            checkcomplete=(
                None if checkcomplete is None else bool(checkcomplete)
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'pglz'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {'header': self.header}
        if self.strategy is not None:
            cfg['strategy'] = self.strategy
        if self.checkcomplete is not None:
            cfg['checkcomplete'] = self.checkcomplete
        return {
            'name': 'imagecodecs_pglz',
            'configuration': cfg,
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.pglz_decode(
            chunk_bytes.as_numpy_array(),
            header=self.header,
            checkcomplete=self.checkcomplete,
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.pglz_encode(
            chunk_bytes.as_numpy_array(),
            strategy=self.strategy,
            header=self.header,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Pixarlog(ArrayBytesCodec):
    """Pixarlog codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    deflate: bool = True

    def __init__(
        self,
        *,
        level: int | None = None,
        deflate: bool = True,
    ) -> None:
        if not imagecodecs.PIXARLOG.available:
            msg = 'imagecodecs.PIXARLOG not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            deflate=bool(deflate),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'pixarlog'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if not self.deflate:
            cfg['deflate'] = self.deflate
        if cfg:
            return {
                'name': 'imagecodecs_pixarlog',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_pixarlog'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        # pixarlog requires 2D or 3D shape; squeeze out the leading batch dim
        squeezed = tuple(s for s in chunk_spec.shape if s != 1)
        shape = squeezed if len(squeezed) >= 2 else (1, *squeezed)
        decoded = imagecodecs.pixarlog_decode(
            chunk_bytes.as_numpy_array(),
            shape=shape,
            dtype=dtype,
            deflate=self.deflate,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.pixarlog_encode(
            arr, level=self.level, deflate=self.deflate
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Plio(ArrayBytesCodec):
    """PLIO codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.PLIO.available:
            msg = 'imagecodecs.PLIO not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'plio'))

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_plio'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.plio_decode(
            chunk_bytes.as_numpy_array(),
            npix=int(numpy.prod(chunk_spec.shape)),
        )
        decoded = decoded.astype(dtype, copy=False).reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = chunk_array.as_numpy_array()
        encoded = imagecodecs.plio_encode(arr.ravel())
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Png(ArrayBytesCodec):
    """PNG codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    strategy: str | None = None
    filter: str | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        strategy: imagecodecs.PNG.STRATEGY | int | str | None = None,
        filter: imagecodecs.PNG.FILTER | int | str | None = None,
    ) -> None:
        if not imagecodecs.PNG.available:
            msg = 'imagecodecs.PNG not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            strategy=_enum_name(strategy, imagecodecs.PNG.STRATEGY),
            filter=_enum_name(filter, imagecodecs.PNG.FILTER),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'png'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'strategy', 'filter'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_png', 'configuration': cfg}
        return {'name': 'imagecodecs_png'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.png_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.png_encode(
            arr,
            level=self.level,
            strategy=self.strategy,
            filter=self.filter,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Qoi(ArrayBytesCodec):
    """QOI codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.QOI.available:
            msg = 'imagecodecs.QOI not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'qoi')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_qoi'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.qoi_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.qoi_encode(arr)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Quantize(ArrayArrayCodec):
    """Quantize codec for Zarr 3."""

    is_fixed_size = True

    mode: Literal['bitgroom', 'granularbr', 'gbr', 'bitround', 'scale'] = (
        'bitgroom'
    )
    nsd: int = 8

    def __init__(
        self,
        *,
        mode: Literal[
            'bitgroom', 'granularbr', 'gbr', 'bitround', 'scale'
        ] = 'bitgroom',
        nsd: int = 8,
    ) -> None:
        if not imagecodecs.QUANTIZE.available:
            msg = 'imagecodecs.QUANTIZE not available'
            raise ValueError(msg)
        _setattrs(
            self,
            mode=mode,
            nsd=int(nsd),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'quantize'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_quantize',
            'configuration': {'mode': self.mode, 'nsd': self.nsd},
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # quantize decode is a no-op
        return chunk_array

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return chunk_array

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.quantize_encode(
            chunk_array.as_numpy_array(), self.mode, self.nsd
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Rcomp(ArrayBytesCodec):
    """Rcomp codec for Zarr 3."""

    is_fixed_size = False

    nblock: int | None = None

    def __init__(self, *, nblock: int | None = None) -> None:
        if not imagecodecs.RCOMP.available:
            msg = 'imagecodecs.RCOMP not available'
            raise ValueError(msg)
        _setattrs(self, nblock=None if nblock is None else int(nblock))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'rcomp'))

    def to_dict(self) -> dict[str, JSON]:
        if self.nblock is not None:
            return {
                'name': 'imagecodecs_rcomp',
                'configuration': {'nblock': self.nblock},
            }
        return {'name': 'imagecodecs_rcomp'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.rcomp_decode(
            chunk_bytes.as_numpy_array(),
            shape=chunk_spec.shape,
            dtype=dtype,
            nblock=self.nblock,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.rcomp_encode(
            chunk_array.as_numpy_array(), nblock=self.nblock
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Rgbe(ArrayBytesCodec):
    """RGBE codec for Zarr 3."""

    is_fixed_size = False

    header: bool = False
    rle: bool | None = None

    def __init__(
        self,
        *,
        header: bool = False,
        rle: bool | None = None,
    ) -> None:
        if not imagecodecs.RGBE.available:
            msg = 'imagecodecs.RGBE not available'
            raise ValueError(msg)
        _setattrs(
            self,
            header=bool(header),
            rle=None if rle is None else bool(rle),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'rgbe'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.header:
            cfg['header'] = self.header
        if self.rle is not None:
            cfg['rle'] = self.rle
        if cfg:
            return {
                'name': 'imagecodecs_rgbe',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_rgbe'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # when no header, rgbe_decode needs an out buffer to know dimensions
        if not self.header:
            squeezed = tuple(s for s in chunk_spec.shape if s != 1)
            shape = squeezed if len(squeezed) >= 2 else (1, *squeezed)
            out = numpy.empty(shape, numpy.float32)
            decoded = imagecodecs.rgbe_decode(
                chunk_bytes.as_numpy_array(),
                header=self.header,
                rle=self.rle,
                out=out,
            )
        else:
            decoded = imagecodecs.rgbe_decode(
                chunk_bytes.as_numpy_array(),
                header=self.header,
                rle=self.rle,
            )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.rgbe_encode(
            arr, header=self.header, rle=self.rle
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Snappy(BytesBytesCodec):
    """Snappy codec for Zarr 3."""

    is_fixed_size = False

    def __init__(self) -> None:
        if not imagecodecs.SNAPPY.available:
            msg = 'imagecodecs.SNAPPY not available'
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _parse_config(data, 'snappy')
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {'name': 'imagecodecs_snappy'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.snappy_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        # snappy is very fast, skip thread dispatch
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.snappy_encode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return self._encode_sync(chunk_bytes, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Sperr(ArrayBytesCodec):
    """SPERR codec for Zarr 3."""

    is_fixed_size = False

    level: float = 0.0
    mode: Literal['bpp', 'psnr', 'pwe'] = 'bpp'
    chunks: tuple[int, int, int] | None = None
    header: bool = True
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: float = 0.0,
        mode: Literal['bpp', 'psnr', 'pwe'] = 'bpp',
        chunks: tuple[int, int, int] | None = None,
        header: bool = True,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.SPERR.available:
            msg = 'imagecodecs.SPERR not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=float(level),
            mode=mode,
            chunks=(
                None
                if chunks is None
                else (int(chunks[0]), int(chunks[1]), int(chunks[2]))
            ),
            header=bool(header),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'sperr'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {
            'level': self.level,
            'mode': self.mode,
            'header': self.header,
        }
        if self.chunks is not None:
            cfg['chunks'] = self.chunks
        if self.numthreads is not None:
            cfg['numthreads'] = self.numthreads
        return {
            'name': 'imagecodecs_sperr',
            'configuration': cfg,
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        if self.header:
            decoded = imagecodecs.sperr_decode(chunk_bytes.as_numpy_array())
        else:
            dtype = chunk_spec.dtype.to_native_dtype()
            decoded = imagecodecs.sperr_decode(
                chunk_bytes.as_numpy_array(),
                shape=chunk_spec.shape,
                dtype=dtype,
                header=False,
                numthreads=self.numthreads,
            )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.sperr_encode(
            arr,
            level=self.level,
            mode=self.mode,
            chunks=self.chunks,
            header=self.header,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Spng(ArrayBytesCodec):
    """SPNG codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.SPNG.available:
            msg = 'imagecodecs.SPNG not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'spng'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_spng',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_spng'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.spng_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.spng_encode(arr, level=self.level)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Sz3(ArrayBytesCodec):
    """SZ3 codec for Zarr 3."""

    is_fixed_size = False

    mode: str = 'abs'
    abs: float = 0.0
    rel: float = 0.0

    def __init__(
        self,
        *,
        mode: Literal['abs', 'rel', 'abs_or_rel', 'abs_and_rel'] | None = None,
        abs: float | None = None,
        rel: float | None = None,
    ) -> None:
        if not imagecodecs.SZ3.available:
            msg = 'imagecodecs.SZ3 not available'
            raise ValueError(msg)
        _setattrs(
            self,
            mode='abs' if mode is None else mode,
            abs=0.0 if abs is None else float(abs),
            rel=0.0 if rel is None else float(rel),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'sz3'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_sz3',
            'configuration': {
                'mode': self.mode,
                'abs': self.abs,
                'rel': self.rel,
            },
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        decoded = imagecodecs.sz3_decode(
            chunk_bytes.as_numpy_array(),
            shape=chunk_spec.shape,
            dtype=dtype,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.sz3_encode(
            chunk_array.as_numpy_array(),
            mode=self.mode,
            abs=self.abs,
            rel=self.rel,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Szip(BytesBytesCodec):
    """SZIP codec for Zarr 3."""

    is_fixed_size = False

    options_mask: int = 0
    pixels_per_block: int = 0
    bits_per_pixel: int = 0
    pixels_per_scanline: int = 0
    header: bool = True

    def __init__(
        self,
        *,
        options_mask: int = 0,
        pixels_per_block: int = 0,
        bits_per_pixel: int = 0,
        pixels_per_scanline: int = 0,
        header: bool = True,
    ) -> None:
        if not imagecodecs.SZIP.available:
            msg = 'imagecodecs.SZIP not available'
            raise ValueError(msg)
        _setattrs(
            self,
            options_mask=int(options_mask),
            pixels_per_block=int(pixels_per_block),
            bits_per_pixel=int(bits_per_pixel),
            pixels_per_scanline=int(pixels_per_scanline),
            header=bool(header),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'szip'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_szip',
            'configuration': {
                'options_mask': self.options_mask,
                'pixels_per_block': self.pixels_per_block,
                'bits_per_pixel': self.bits_per_pixel,
                'pixels_per_scanline': self.pixels_per_scanline,
                'header': self.header,
            },
        }

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.szip_decode(
            chunk_bytes.as_numpy_array(),
            options_mask=self.options_mask,
            pixels_per_block=self.pixels_per_block,
            bits_per_pixel=self.bits_per_pixel,
            pixels_per_scanline=self.pixels_per_scanline,
            header=self.header,
        )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.szip_encode(
            chunk_bytes.as_numpy_array(),
            options_mask=self.options_mask,
            pixels_per_block=self.pixels_per_block,
            bits_per_pixel=self.bits_per_pixel,
            pixels_per_scanline=self.pixels_per_scanline,
            header=self.header,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Tiff(ArrayBytesCodec):
    """TIFF codec for Zarr 3."""

    is_fixed_size = False

    index: int | None = None
    asrgb: bool = False
    bigtiff: bool | None = None
    byteorder: str | None = None
    subfiletype: str | None = None
    photometric: str | None = None
    planarconfig: str | None = None
    extrasample: str | None = None
    tile: tuple[int, int] | None = None
    rowsperstrip: int | None = None
    bitspersample: int | None = None
    compression: str | None = None
    subcodec: str | None = None
    level: int | None = None
    predictor: bool | str | None = None
    verbose: int | None = None

    def __init__(
        self,
        *,
        index: int | None = None,
        asrgb: bool = False,
        bigtiff: bool | None = None,
        byteorder: imagecodecs.TIFF.ENDIAN | int | str | None = None,
        subfiletype: imagecodecs.TIFF.FILETYPE | int | str | None = None,
        photometric: imagecodecs.TIFF.PHOTOMETRIC | int | str | None = None,
        planarconfig: imagecodecs.TIFF.PLANARCONFIG | int | str | None = None,
        extrasample: imagecodecs.TIFF.EXTRASAMPLE | int | str | None = None,
        tile: tuple[int, int] | None = None,
        rowsperstrip: int | None = None,
        bitspersample: int | None = None,
        compression: imagecodecs.TIFF.COMPRESSION | int | str | None = None,
        subcodec: imagecodecs.TIFF.COMPRESSION | int | str | None = None,
        level: int | None = None,
        predictor: bool | imagecodecs.TIFF.PREDICTOR | int | str | None = None,
        verbose: int | None = None,
    ) -> None:
        if not imagecodecs.TIFF.available:
            msg = 'imagecodecs.TIFF not available'
            raise ValueError(msg)
        _setattrs(
            self,
            index=None if index is None else int(index),
            asrgb=bool(asrgb),
            bigtiff=None if bigtiff is None else bool(bigtiff),
            byteorder=_enum_name(byteorder, imagecodecs.TIFF.ENDIAN),
            subfiletype=_enum_name(subfiletype, imagecodecs.TIFF.FILETYPE),
            photometric=_enum_name(photometric, imagecodecs.TIFF.PHOTOMETRIC),
            planarconfig=_enum_name(
                planarconfig, imagecodecs.TIFF.PLANARCONFIG
            ),
            extrasample=_enum_name(extrasample, imagecodecs.TIFF.EXTRASAMPLE),
            tile=None if tile is None else (int(tile[0]), int(tile[1])),
            rowsperstrip=None if rowsperstrip is None else int(rowsperstrip),
            bitspersample=(
                None if bitspersample is None else int(bitspersample)
            ),
            compression=_enum_name(compression, imagecodecs.TIFF.COMPRESSION),
            subcodec=_enum_name(subcodec, imagecodecs.TIFF.COMPRESSION),
            level=None if level is None else int(level),
            predictor=(
                predictor
                if isinstance(predictor, bool)
                else _enum_name(predictor, imagecodecs.TIFF.PREDICTOR)
            ),
            verbose=None if verbose is None else int(verbose),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'tiff'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'index',
            'bigtiff',
            'byteorder',
            'subfiletype',
            'photometric',
            'planarconfig',
            'extrasample',
            'tile',
            'rowsperstrip',
            'bitspersample',
            'compression',
            'subcodec',
            'level',
            'predictor',
            'verbose',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if self.asrgb:
            cfg['asrgb'] = True
        if cfg:
            return {'name': 'imagecodecs_tiff', 'configuration': cfg}
        return {'name': 'imagecodecs_tiff'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.tiff_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            asrgb=self.asrgb,
            verbose=self.verbose,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.tiff_encode(
            arr,
            level=self.level,
            bigtiff=self.bigtiff,
            byteorder=self.byteorder,
            subfiletype=self.subfiletype,
            photometric=self.photometric,
            planarconfig=self.planarconfig,
            extrasample=self.extrasample,
            tile=self.tile,
            rowsperstrip=self.rowsperstrip,
            bitspersample=self.bitspersample,
            compression=self.compression,
            subcodec=self.subcodec,
            predictor=self.predictor,
            verbose=self.verbose,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Ultrahdr(ArrayBytesCodec):
    """Ultra HDR codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    scale: int | None = None
    gamut: str | None = None
    crange: str | None = None
    transfer: str | None = None
    nits: float | None = None
    boostmin: float | None = None
    boostmax: float | None = None
    usage: str | None = None
    codec: str | None = None
    dtype: str | None = None
    boost: float | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        scale: int | None = None,
        gamut: imagecodecs.ULTRAHDR.CG | int | str | None = None,
        crange: imagecodecs.ULTRAHDR.CR | int | str | None = None,
        transfer: imagecodecs.ULTRAHDR.CT | int | str | None = None,
        nits: float | None = None,
        boostmin: float | None = None,
        boostmax: float | None = None,
        usage: imagecodecs.ULTRAHDR.USAGE | int | str | None = None,
        codec: imagecodecs.ULTRAHDR.CODEC | int | str | None = None,
        dtype: str | None = None,
        boost: float | None = None,
    ) -> None:
        if not imagecodecs.ULTRAHDR.available:
            msg = 'imagecodecs.ULTRAHDR not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            scale=None if scale is None else int(scale),
            gamut=_enum_name(gamut, imagecodecs.ULTRAHDR.CG),
            crange=_enum_name(crange, imagecodecs.ULTRAHDR.CR),
            transfer=_enum_name(transfer, imagecodecs.ULTRAHDR.CT),
            nits=None if nits is None else float(nits),
            boostmin=None if boostmin is None else float(boostmin),
            boostmax=None if boostmax is None else float(boostmax),
            usage=_enum_name(usage, imagecodecs.ULTRAHDR.USAGE),
            codec=_enum_name(codec, imagecodecs.ULTRAHDR.CODEC),
            dtype=None if dtype is None else numpy.dtype(dtype).str,
            boost=None if boost is None else float(boost),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'ultrahdr'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'scale',
            'gamut',
            'crange',
            'transfer',
            'nits',
            'boostmin',
            'boostmax',
            'usage',
            'codec',
            'dtype',
            'boost',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {
                'name': 'imagecodecs_ultrahdr',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_ultrahdr'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.ultrahdr_decode(
            chunk_bytes.as_numpy_array(),
            dtype=self.dtype,
            transfer=self.transfer,
            boost=self.boost,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.ultrahdr_encode(
            arr,
            scale=self.scale,
            level=self.level,
            gamut=self.gamut,
            crange=self.crange,
            transfer=self.transfer,
            nits=self.nits,
            boostmin=self.boostmin,
            boostmax=self.boostmax,
            usage=self.usage,
            codec=self.codec,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Webp(ArrayBytesCodec):
    """WebP codec for Zarr 3."""

    is_fixed_size = False

    level: float | None = None
    lossless: bool | None = None
    method: int | None = None
    index: int | None = None
    hasalpha: bool | None = None
    numthreads: int | None = None
    delay: int | None = None

    def __init__(
        self,
        *,
        level: float | None = None,
        lossless: bool | None = None,
        method: int | None = None,
        index: int | None = None,
        hasalpha: bool | None = None,
        numthreads: int | None = None,
        delay: int | None = None,
    ) -> None:
        if not imagecodecs.WEBP.available:
            msg = 'imagecodecs.WEBP not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else float(level),
            hasalpha=None if hasalpha is None else bool(hasalpha),
            method=None if method is None else int(method),
            index=None if index is None else int(index),
            lossless=None if lossless is None else bool(lossless),
            numthreads=None if numthreads is None else int(numthreads),
            delay=None if delay is None else int(delay),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'webp'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in (
            'level',
            'lossless',
            'method',
            'index',
            'hasalpha',
            'numthreads',
            'delay',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if cfg:
            return {'name': 'imagecodecs_webp', 'configuration': cfg}
        return {'name': 'imagecodecs_webp'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.webp_decode(
            chunk_bytes.as_numpy_array(),
            index=self.index,
            hasalpha=self.hasalpha,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.webp_encode(
            arr,
            level=self.level,
            lossless=self.lossless,
            method=self.method,
            numthreads=self.numthreads,
            delay=self.delay,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Wavpack(ArrayBytesCodec):
    """Wavpack codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    bitrate: float | None = None
    channels: bool = False
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        bitrate: float | None = None,
        channels: bool = False,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.WAVPACK.available:
            msg = 'imagecodecs.WAVPACK not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            bitrate=None if bitrate is None else float(bitrate),
            channels=bool(channels),
            numthreads=None if numthreads is None else int(numthreads),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'wavpack'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        for key in ('level', 'bitrate', 'numthreads'):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        if self.channels:
            cfg['channels'] = True
        if cfg:
            return {'name': 'imagecodecs_wavpack', 'configuration': cfg}
        return {'name': 'imagecodecs_wavpack'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.wavpack_decode(
            chunk_bytes.as_numpy_array(),
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = chunk_array.as_numpy_array()
        if arr.ndim > 1:
            arr = (
                arr.reshape(-1, arr.shape[-1])
                if self.channels
                else arr.ravel()
            )
        encoded = imagecodecs.wavpack_encode(
            arr,
            level=self.level,
            bitrate=self.bitrate,
            numthreads=self.numthreads,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Wic(ArrayBytesCodec):
    """WIC codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    format: str | None = None
    index: int = 0

    def __init__(
        self,
        *,
        level: int | None = None,
        format: imagecodecs.WIC.FORMAT | int | str | None = None,
        index: int = 0,
    ) -> None:
        if not imagecodecs.WIC.available:
            msg = 'imagecodecs.WIC not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            format=_enum_name(format, imagecodecs.WIC.FORMAT),
            index=int(index),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'wic'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.format is not None:
            cfg['format'] = self.format
        if self.index != 0:
            cfg['index'] = self.index
        if cfg:
            return {
                'name': 'imagecodecs_wic',
                'configuration': cfg,
            }
        return {'name': 'imagecodecs_wic'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.wic_decode(
            chunk_bytes.as_numpy_array(), index=self.index
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        encoded = imagecodecs.wic_encode(
            arr, level=self.level, format=self.format
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Xor(ArrayArrayCodec):
    """XOR codec for Zarr 3."""

    is_fixed_size = True

    axis: int = -1

    def __init__(self, *, axis: int = -1) -> None:
        if not imagecodecs.XOR.available:
            msg = 'imagecodecs.XOR not available'
            raise ValueError(msg)
        _setattrs(self, axis=int(axis))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'xor'))

    def to_dict(self) -> dict[str, JSON]:
        return {
            'name': 'imagecodecs_xor',
            'configuration': {'axis': self.axis},
        }

    def _decode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        decoded = imagecodecs.xor_decode(
            chunk_array.as_numpy_array(), axis=self.axis
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        encoded = imagecodecs.xor_encode(
            chunk_array.as_numpy_array(), axis=self.axis
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return input_byte_length


@dataclass(frozen=True)
class Zfp(ArrayBytesCodec):
    """ZFP codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    mode: str | None = None
    execution: str | None = None
    chunksize: int | None = None
    header: bool = True
    numthreads: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        mode: imagecodecs.ZFP.MODE | int | str | None = None,
        execution: imagecodecs.ZFP.EXEC | int | str | None = None,
        chunksize: int | None = None,
        header: bool = True,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.ZFP.available:
            msg = 'imagecodecs.ZFP not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            mode=_enum_name(mode, imagecodecs.ZFP.MODE),
            execution=_enum_name(execution, imagecodecs.ZFP.EXEC),
            numthreads=None if numthreads is None else int(numthreads),
            chunksize=None if chunksize is None else int(chunksize),
            header=bool(header),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'zfp'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {'header': self.header}
        for key in (
            'level',
            'mode',
            'execution',
            'chunksize',
            'numthreads',
        ):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        return {'name': 'imagecodecs_zfp', 'configuration': cfg}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        if self.header:
            decoded = imagecodecs.zfp_decode(chunk_bytes.as_numpy_array())
        else:
            dtype = chunk_spec.dtype.to_native_dtype()
            decoded = imagecodecs.zfp_decode(
                chunk_bytes.as_numpy_array(),
                shape=chunk_spec.shape,
                dtype=dtype,
                numthreads=self.numthreads,
            )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.zfp_encode(
            chunk_array.as_numpy_array(),
            level=self.level,
            mode=self.mode,
            execution=self.execution,
            header=self.header,
            numthreads=self.numthreads,
            chunksize=self.chunksize,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Zlib(BytesBytesCodec):
    """Zlib codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.ZLIB.available:
            msg = 'imagecodecs.ZLIB not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'zlib'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_zlib',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_zlib'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.zlib_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.zlib_encode(
            chunk_bytes.as_numpy_array(), level=self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Zlibng(BytesBytesCodec):
    """Zlibng codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.ZLIBNG.available:
            msg = 'imagecodecs.ZLIBNG not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'zlibng'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_zlibng',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_zlibng'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.zlibng_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.zlibng_encode(
            chunk_bytes.as_numpy_array(), level=self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Zopfli(BytesBytesCodec):
    """Zopfli codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None
    format: int | None = None
    blocksplitting: bool | None = None
    blocksplittingmax: int | None = None

    def __init__(
        self,
        *,
        level: int | None = None,
        format: int | None = None,
        blocksplitting: bool | None = None,
        blocksplittingmax: int | None = None,
    ) -> None:
        if not imagecodecs.ZOPFLI.available:
            msg = 'imagecodecs.ZOPFLI not available'
            raise ValueError(msg)
        _setattrs(
            self,
            level=None if level is None else int(level),
            format=None if format is None else int(format),
            blocksplitting=(
                None if blocksplitting is None else bool(blocksplitting)
            ),
            blocksplittingmax=(
                None if blocksplittingmax is None else int(blocksplittingmax)
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'zopfli'))

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {}
        if self.level is not None:
            cfg['level'] = self.level
        if self.format is not None:
            cfg['format'] = self.format
        if self.blocksplitting is not None:
            cfg['blocksplitting'] = self.blocksplitting
        if self.blocksplittingmax is not None:
            cfg['blocksplittingmax'] = self.blocksplittingmax
        if cfg:
            return {'name': 'imagecodecs_zopfli', 'configuration': cfg}
        return {'name': 'imagecodecs_zopfli'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.zopfli_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.zopfli_encode(
            chunk_bytes.as_numpy_array(),
            self.level,
            format=self.format,
            blocksplitting=self.blocksplitting,
            blocksplittingmax=self.blocksplittingmax,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Zstd(BytesBytesCodec):
    """ZStandard codec for Zarr 3."""

    is_fixed_size = False

    level: int | None = None

    def __init__(self, *, level: int | None = None) -> None:
        if not imagecodecs.ZSTD.available:
            msg = 'imagecodecs.ZSTD not available'
            raise ValueError(msg)
        _setattrs(self, level=None if level is None else int(level))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls(**_parse_config(data, 'zstd'))

    def to_dict(self) -> dict[str, JSON]:
        if self.level is not None:
            return {
                'name': 'imagecodecs_zstd',
                'configuration': {'level': self.level},
            }
        return {'name': 'imagecodecs_zstd'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        decoded = imagecodecs.zstd_decode(chunk_bytes.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        encoded = imagecodecs.zstd_encode(
            chunk_bytes.as_numpy_array(), level=self.level
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_bytes, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


# Map of codec name suffix -> codec class; computed from __all__
_CODEC_CLASSES: dict[str, type] = {
    name.lower(): globals()[name]
    for name in __all__
    if name != 'register_codecs'
}


def register_codecs(
    codecs: Container[str] | None = None,
    *,
    verbose: bool = True,
) -> None:
    """Register imagecodecs zarr 3 codecs with zarr."""
    from zarr.registry import register_codec

    for name, cls in _CODEC_CLASSES.items():
        if codecs is not None and name not in codecs:
            continue
        key = f'imagecodecs_{name}'
        try:
            register_codec(key, cls)
        except Exception:
            if verbose:
                logging.getLogger(__name__).warning(
                    'zarr codec %s registration failed', key
                )


def _parse_config(data: dict[str, JSON], name: str) -> Any:
    """Parse named configuration, returning the configuration dict."""
    _, configuration = parse_named_configuration(
        data, f'imagecodecs_{name}', require_configuration=False
    )
    return configuration if configuration is not None else {}


def _enum_name(
    value: int | str | enum.Enum | None, enum_cls: type[enum.Enum], /
) -> str | None:
    """Normalize int, str, or enum member to canonical enum name string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, enum_cls):
        return value.name
    if isinstance(value, enum.Enum):
        return enum_cls(value.value).name
    return enum_cls(value).name


def _setattrs(obj: object, /, **kwargs: Any) -> None:
    """Set attributes on a frozen dataclass instance."""
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)
