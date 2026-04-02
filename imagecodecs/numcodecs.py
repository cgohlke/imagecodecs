# imagecodecs/numcodecs.py

# Copyright (c) 2021-2026, Christoph Gohlke
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

"""Numcodecs implemented using imagecodecs."""

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
    'Codec',
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

import base64
import contextlib
import enum
import logging
from typing import TYPE_CHECKING

import numpy
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

import imagecodecs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray


class Aec(Codec):
    """AEC codec for numcodecs."""

    codec_id = 'imagecodecs_aec'

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

        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.flags = None if flags is None else int(flags)
        self.blocksize = None if blocksize is None else int(blocksize)
        self.rsi = None if rsi is None else int(rsi)

    def encode(self, buf):
        return imagecodecs.aec_encode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
        )

    def decode(self, buf, out=None):
        return imagecodecs.aec_decode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
            out=_flat(out),
        )


class Apng(Codec):
    """APNG codec for numcodecs."""

    codec_id = 'imagecodecs_apng'

    def __init__(
        self,
        *,
        level: int | None = None,
        strategy: imagecodecs.APNG.STRATEGY | int | str | None = None,
        filter: imagecodecs.APNG.FILTER | int | str | None = None,
        photometric: imagecodecs.APNG.COLOR_TYPE | int | str | None = None,
        delay: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.APNG.available:
            msg = 'imagecodecs.APNG not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.strategy = _enum_name(strategy, imagecodecs.APNG.STRATEGY)
        self.filter = _enum_name(filter, imagecodecs.APNG.FILTER)
        self.photometric = _enum_name(photometric, imagecodecs.APNG.COLOR_TYPE)
        self.delay = None if delay is None else int(delay)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.apng_encode(
            buf,
            level=self.level,
            strategy=self.strategy,
            filter=self.filter,
            photometric=self.photometric,
            delay=self.delay,
        )

    def decode(self, buf, out=None):
        return imagecodecs.apng_decode(buf, out=out)


class Avif(Codec):
    """AVIF codec for numcodecs."""

    codec_id = 'imagecodecs_avif'

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
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.AVIF.available:
            msg = 'imagecodecs.AVIF not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.speed = None if speed is None else int(speed)
        self.tilelog2 = (
            None if tilelog2 is None else (int(tilelog2[0]), int(tilelog2[1]))
        )
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.pixelformat = _enum_name(
            pixelformat, imagecodecs.AVIF.PIXEL_FORMAT
        )
        self.codec = _enum_name(codec, imagecodecs.AVIF.CODEC_CHOICE)
        self.primaries = _enum_name(
            primaries, imagecodecs.AVIF.COLOR_PRIMARIES
        )
        self.transfer = _enum_name(
            transfer, imagecodecs.AVIF.TRANSFER_CHARACTERISTICS
        )
        self.matrix = _enum_name(matrix, imagecodecs.AVIF.MATRIX_COEFFICIENTS)
        self.numthreads = None if numthreads is None else int(numthreads)
        self.index = None if index is None else int(index)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.avif_encode(
            buf,
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

    def decode(self, buf, out=None):
        return imagecodecs.avif_decode(
            buf, index=self.index, numthreads=self.numthreads, out=out
        )


class Bfloat16(Codec):
    """Bfloat16 codec for numcodecs."""

    codec_id = 'imagecodecs_bfloat16'

    def __init__(
        self,
        byteorder: Literal['>', '<', '='] | None = None,
        rounding: imagecodecs.BFLOAT16.ROUND | int | str | None = None,
    ) -> None:
        if not imagecodecs.BFLOAT16.available:
            msg = 'imagecodecs.BFLOAT16 not available'
            raise ValueError(msg)

        self.byteorder = byteorder
        self.rounding = _enum_name(rounding, imagecodecs.BFLOAT16.ROUND)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        return imagecodecs.bfloat16_encode(
            buf, byteorder=self.byteorder, rounding=self.rounding
        )

    def decode(self, buf, out=None):
        return imagecodecs.bfloat16_decode(
            buf, byteorder=self.byteorder, out=out
        )


class Bitorder(Codec):
    """Bitorder codec for numcodecs."""

    codec_id = 'imagecodecs_bitorder'

    def __init__(self) -> None:
        if not imagecodecs.BITORDER.available:
            msg = 'imagecodecs.BITORDER not available'
            raise ValueError(msg)

    def encode(self, buf):
        return imagecodecs.bitorder_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.bitorder_decode(buf, out=_flat(out))


class Bitshuffle(Codec):
    """Bitshuffle codec for numcodecs."""

    codec_id = 'imagecodecs_bitshuffle'

    def __init__(
        self,
        *,
        itemsize: int = 1,
        blocksize: int = 0,
    ) -> None:
        if not imagecodecs.BITSHUFFLE.available:
            msg = 'imagecodecs.BITSHUFFLE not available'
            raise ValueError(msg)

        self.itemsize = int(itemsize)
        self.blocksize = int(blocksize)

    def encode(self, buf):
        ret = imagecodecs.bitshuffle_encode(
            buf, itemsize=self.itemsize, blocksize=self.blocksize
        )
        if isinstance(ret, numpy.ndarray):
            return ret.tobytes()
        return ret

    def decode(self, buf, out=None):
        return imagecodecs.bitshuffle_decode(
            buf,
            itemsize=self.itemsize,
            blocksize=self.blocksize,
            out=_flat(out),
        )


class Blosc(Codec):
    """Blosc codec for numcodecs."""

    codec_id = 'imagecodecs_blosc'

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

        self.level = None if level is None else int(level)
        self.compressor = _enum_name(compressor, imagecodecs.BLOSC.COMPRESSOR)
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = _enum_name(shuffle, imagecodecs.BLOSC.SHUFFLE)
        self.numthreads = numthreads

    def encode(self, buf):
        buf = numpy.asarray(buf)
        return imagecodecs.blosc_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.blosc_decode(
            buf, numthreads=self.numthreads, out=_flat(out)
        )


class Blosc2(Codec):
    """Blosc2 codec for numcodecs."""

    codec_id = 'imagecodecs_blosc2'

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

        self.level = None if level is None else int(level)
        self.compressor = _enum_name(compressor, imagecodecs.BLOSC2.COMPRESSOR)
        self.splitmode = _enum_name(splitmode, imagecodecs.BLOSC2.SPLIT)
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = _enum_name(shuffle, imagecodecs.BLOSC2.FILTER)
        self.numthreads = numthreads

    def encode(self, buf):
        buf = numpy.asarray(buf)
        return imagecodecs.blosc2_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            shuffle=self.shuffle,
            splitmode=self.splitmode,
            typesize=self.typesize,
            blocksize=self.blocksize,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.blosc2_decode(
            buf, numthreads=self.numthreads, out=_flat(out)
        )


class Bmp(Codec):
    """BMP codec for numcodecs."""

    codec_id = 'imagecodecs_bmp'

    def __init__(
        self,
        *,
        ppm: int | None = None,
        asrgb: bool | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.BMP.available:
            msg = 'imagecodecs.BMP not available'
            raise ValueError(msg)

        self.ppm = None if ppm is None else max(1, int(ppm))
        self.asrgb = None if asrgb is None else bool(asrgb)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.bmp_encode(buf, ppm=self.ppm)

    def decode(self, buf, out=None):
        return imagecodecs.bmp_decode(buf, asrgb=self.asrgb, out=out)


class Brotli(Codec):
    """Brotli codec for numcodecs."""

    codec_id = 'imagecodecs_brotli'

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

        self.level = None if level is None else int(level)
        self.mode = _enum_name(mode, imagecodecs.BROTLI.MODE)
        self.lgwin = lgwin

    def encode(self, buf):
        return imagecodecs.brotli_encode(
            buf, level=self.level, mode=self.mode, lgwin=self.lgwin
        )

    def decode(self, buf, out=None):
        return imagecodecs.brotli_decode(buf, out=_flat(out))


class Byteshuffle(Codec):
    """Byteshuffle codec for numcodecs."""

    codec_id = 'imagecodecs_byteshuffle'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        axis: int = -1,
        dist: int = 1,
        delta: bool = False,
        reorder: bool = False,
    ) -> None:
        if not imagecodecs.BYTESHUFFLE.available:
            msg = 'imagecodecs.BYTESHUFFLE not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)
        self.delta = bool(delta)
        self.reorder = bool(reorder)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.shape != self.shape:
            msg = f'{buf.shape=} does not match {self.shape=}'
            raise ValueError(msg)
        if buf.dtype != self.dtype:
            msg = f'{buf.dtype=} does not match {self.dtype=}'
            raise ValueError(msg)
        return imagecodecs.byteshuffle_encode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        ).tobytes()

    def decode(self, buf, out=None):
        buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
        return imagecodecs.byteshuffle_decode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
            out=out,
        )


class Bz2(Codec):
    """Bz2 codec for numcodecs."""

    codec_id = 'imagecodecs_bz2'

    def __init__(
        self,
        *,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.BZ2.available:
            msg = 'imagecodecs.BZ2 not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)

    def encode(self, buf):
        return imagecodecs.bz2_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.bz2_decode(buf, out=_flat(out))


class Ccittfax3(Codec):
    """CCITT Fax3 codec for numcodecs."""

    codec_id = 'imagecodecs_ccittfax3'

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

        self.height = int(height)
        self.width = int(width)
        self.t4options = int(t4options)

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.ccittfax3_decode(
            buf,
            height=self.height,
            width=self.width,
            t4options=self.t4options,
            out=out,
        )


class Ccittfax4(Codec):
    """CCITT Fax4 codec for numcodecs."""

    codec_id = 'imagecodecs_ccittfax4'

    def __init__(
        self,
        *,
        height: int = 0,
        width: int = 0,
    ) -> None:
        if not imagecodecs.CCITTFAX4.available:
            msg = 'imagecodecs.CCITTFAX4 not available'
            raise ValueError(msg)

        self.height = int(height)
        self.width = int(width)

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.ccittfax4_decode(
            buf,
            height=self.height,
            width=self.width,
            out=out,
        )


class Ccittrle(Codec):
    """CCITT RLE codec for numcodecs."""

    codec_id = 'imagecodecs_ccittrle'

    def __init__(
        self,
        *,
        height: int = 0,
        width: int = 0,
    ) -> None:
        if not imagecodecs.CCITTRLE.available:
            msg = 'imagecodecs.CCITTRLE not available'
            raise ValueError(msg)

        self.height = int(height)
        self.width = int(width)

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.ccittrle_decode(
            buf,
            height=self.height,
            width=self.width,
            out=out,
        )


class Checksum(Codec):
    """Checksum codec for numcodecs."""

    codec_id = 'imagecodecs_checksum'

    def __init__(
        self,
        *,
        kind: Literal['crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc'],
        value: int | None = None,
        prefix: bytes | None = None,
        prepend: bool | None = None,
        byteorder: Literal['<', '>', 'little', 'big'] = '<',
    ) -> None:
        match kind:
            case 'crc32':
                if imagecodecs.ZLIBNG.available:
                    self._checksum = imagecodecs.zlibng_crc32
                elif imagecodecs.DEFLATE.available:
                    self._checksum = imagecodecs.deflate_crc32
                elif imagecodecs.ZLIB.available:
                    self._checksum = imagecodecs.zlib_crc32
                else:
                    msg = 'imagecodecs.ZLIB not available'
                    raise ValueError(msg)
                if prepend is None:
                    prepend = True
            case 'adler32':
                if imagecodecs.ZLIBNG.available:
                    self._checksum = imagecodecs.zlibng_adler32
                elif imagecodecs.DEFLATE.available:
                    self._checksum = imagecodecs.deflate_adler32
                elif imagecodecs.ZLIB.available:
                    self._checksum = imagecodecs.zlib_adler32
                else:
                    msg = 'imagecodecs.ZLIB not available'
                    raise ValueError(msg)
                if prepend is None:
                    prepend = True
            case 'fletcher32':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                self._checksum = imagecodecs.h5checksum_fletcher32
                if prepend is None:
                    prepend = False
            case 'lookup3':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                self._checksum = imagecodecs.h5checksum_lookup3
                if prepend is None:
                    prepend = False
            case 'h5crc':
                if not imagecodecs.H5CHECKSUM.available:
                    msg = 'imagecodecs.H5CHECKSUM not available'
                    raise ValueError(msg)
                self._checksum = imagecodecs.h5checksum_crc
                if prepend is None:
                    prepend = False
            case _:
                msg = (  # type: ignore[unreachable]
                    f'checksum {kind=!r} not supported'
                )
                raise ValueError(msg)

        self.kind = kind
        self.value = value
        self.prefix = prefix
        self.prepend = bool(prepend)
        self.byteorder: Any = {
            '<': 'little',
            '>': 'big',
            'little': 'little',
            'big': 'big',
        }[byteorder]

    def encode(self, buf):
        buf = _contiguous(buf)
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
        return out

    def decode(self, buf, out=None):
        out = memoryview(buf)
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
        return out


class Cms(Codec):
    """CMS codec for numcodecs."""

    codec_id = 'imagecodecs_cms'

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
        shape: tuple[int, ...] | None = None,
        dtype: numpy.dtype | str | None = None,
    ) -> None:
        if not imagecodecs.CMS.available:
            msg = 'imagecodecs.CMS not available'
            raise ValueError(msg)

        if isinstance(profile, str):
            profile = imagecodecs.cms_profile(profile)
        if isinstance(outprofile, str):
            outprofile = imagecodecs.cms_profile(outprofile)

        self.profile = profile
        self.outprofile = outprofile
        self.colorspace = colorspace
        self.outcolorspace = outcolorspace
        self.planar = None if planar is None else bool(planar)
        self.outplanar = None if outplanar is None else bool(outplanar)
        self.intent = _enum_name(intent, imagecodecs.CMS.INTENT)
        self.flags = None if flags is None else int(flags)
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str

    def _reshape(self, buf):
        """Reshape flat buffer to chunk array if shape/dtype are set."""
        buf = numpy.asarray(buf)
        if self.shape is not None and self.dtype is not None:
            buf = buf.view(numpy.dtype(self.dtype)).reshape(self.shape)
        return buf

    def encode(self, buf):
        buf = self._reshape(buf)
        return numpy.ascontiguousarray(
            imagecodecs.cms_transform(
                buf,
                self.profile,
                self.outprofile,
                colorspace=self.colorspace,
                outcolorspace=self.outcolorspace,
                planar=self.planar,
                outplanar=self.outplanar,
                intent=self.intent,
                flags=self.flags,
            )
        )

    def decode(self, buf, out=None):
        buf = self._reshape(buf)
        return numpy.ascontiguousarray(
            imagecodecs.cms_transform(
                buf,
                self.outprofile,
                self.profile,
                colorspace=self.outcolorspace,
                outcolorspace=self.colorspace,
                planar=self.outplanar,
                outplanar=self.planar,
                intent=self.intent,
                flags=self.flags,
            )
        )

    def get_config(self):
        """Return dictionary holding configuration parameters."""
        config = {'id': self.codec_id}
        for key in self.__dict__:
            if not key.startswith('_'):
                value = getattr(self, key)
                if value is not None and key in {'profile', 'outprofile'}:
                    value = base64.b64encode(value).decode()
                config[key] = value
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiate codec from configuration object."""
        config = dict(config)
        for key in ('profile', 'outprofile'):
            value = config.get(key)
            if value is not None and isinstance(value, str):
                config[key] = base64.b64decode(value.encode())
        return cls(**config)


class Dds(Codec):
    """DDS codec for numcodecs."""

    codec_id = 'imagecodecs_dds'

    def __init__(self, *, mipmap: int = 0) -> None:
        if not imagecodecs.DDS.available:
            msg = 'imagecodecs.DDS not available'
            raise ValueError(msg)
        self.mipmap = int(mipmap)

    def encode(self, buf):
        # buf = _image(buf, self.squeeze)
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.dds_decode(buf, mipmap=self.mipmap, out=out)


class Deflate(Codec):
    """Deflate codec for numcodecs."""

    codec_id = 'imagecodecs_deflate'

    def __init__(
        self,
        *,
        level: int | None = None,
        raw: bool = False,
    ) -> None:
        if not imagecodecs.DEFLATE.available:
            msg = 'imagecodecs.DEFLATE not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.raw = bool(raw)

    def encode(self, buf):
        return imagecodecs.deflate_encode(buf, level=self.level, raw=self.raw)

    def decode(self, buf, out=None):
        return imagecodecs.deflate_decode(buf, out=_flat(out), raw=self.raw)


class Delta(Codec):
    """Delta codec for numcodecs."""

    codec_id = 'imagecodecs_delta'

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        axis: int = -1,
        dist: int = 1,
    ) -> None:
        if not imagecodecs.DELTA.available:
            msg = 'imagecodecs.DELTA not available'
            raise ValueError(msg)

        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.asarray(buf)
            if buf.shape != self.shape:
                msg = f'{buf.shape=} does not match {self.shape=}'
                raise ValueError(msg)
            if buf.dtype != self.dtype:
                msg = f'{buf.dtype=} does not match {self.dtype=}'
                raise ValueError(msg)
        return imagecodecs.delta_encode(
            buf, axis=self.axis, dist=self.dist
        ).tobytes()

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype)
            if self.shape is not None:
                buf = buf.reshape(self.shape)
        return imagecodecs.delta_decode(
            buf, axis=self.axis, dist=self.dist, out=out
        )


class Dicomrle(Codec):
    """DICOMRLE codec for numcodecs."""

    codec_id = 'imagecodecs_dicomrle'

    def __init__(self, *, dtype: DTypeLike) -> None:
        if not imagecodecs.DICOMRLE.available:
            msg = 'imagecodecs.DICOMRLE not available'
            raise ValueError(msg)

        self.dtype = numpy.dtype(dtype).str  # TODO: preserve endianness

    def encode(self, buf):
        # buf = _image(buf, self.squeeze)
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.dicomrle_decode(buf, self.dtype, out=out)


class Eer(Codec):
    """Electron Event Representation codec for numcodecs."""

    codec_id = 'imagecodecs_eer'

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

        self.shape = (int(shape[0]), int(shape[1]))
        self.skipbits = int(skipbits)
        self.horzbits = int(horzbits)
        self.vertbits = int(vertbits)
        self.superres = int(superres)

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.eer_decode(
            buf,
            self.shape,
            self.skipbits,
            self.horzbits,
            self.vertbits,
            superres=self.superres,
            out=out,
        )


class Exr(Codec):
    """OpenEXR codec for numcodecs."""

    codec_id = 'imagecodecs_exr'

    def __init__(
        self,
        *,
        # encode
        level: float | None = None,
        compression: imagecodecs.EXR.COMPRESSION | int | str | None = None,
        planar: bool | None = None,
        frames: bool | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
        # decode
        index: int | None = None,
    ) -> None:
        if not imagecodecs.EXR.available:
            msg = 'imagecodecs.EXR not available'
            raise ValueError(msg)

        self.level = None if level is None else float(level)
        self.compression = _enum_name(compression, imagecodecs.EXR.COMPRESSION)
        self.planar = None if planar is None else bool(planar)
        self.frames = None if frames is None else bool(frames)
        self.index = None if index is None else int(index)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.exr_encode(
            buf,
            level=self.level,
            compression=self.compression,
            planar=self.planar,
            frames=self.frames,
        )

    def decode(self, buf, out=None):
        return imagecodecs.exr_decode(
            buf,
            index=self.index,
            planar=self.planar,
            out=out,
        )


class Float24(Codec):
    """Float24 codec for numcodecs."""

    codec_id = 'imagecodecs_float24'

    def __init__(
        self,
        byteorder: Literal['>', '<', '='] | None = None,
        rounding: imagecodecs.FLOAT24.ROUND | int | str | None = None,
    ) -> None:
        if not imagecodecs.FLOAT24.available:
            msg = 'imagecodecs.FLOAT24 not available'
            raise ValueError(msg)

        self.byteorder = byteorder
        self.rounding = _enum_name(rounding, imagecodecs.FLOAT24.ROUND)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        return imagecodecs.float24_encode(
            buf, byteorder=self.byteorder, rounding=self.rounding
        )

    def decode(self, buf, out=None):
        return imagecodecs.float24_decode(
            buf, byteorder=self.byteorder, out=out
        )


class Floatpred(Codec):
    """Floating Point Predictor codec for numcodecs."""

    codec_id = 'imagecodecs_floatpred'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        axis: int = -1,
        dist: int = 1,
    ) -> None:
        if not imagecodecs.FLOATPRED.available:
            msg = 'imagecodecs.FLOATPRED not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.shape != self.shape:
            msg = f'{buf.shape=} does not match {self.shape=}'
            raise ValueError(msg)
        if buf.dtype != self.dtype:
            msg = f'{buf.dtype=} does not match {self.dtype=}'
            raise ValueError(msg)
        return imagecodecs.floatpred_encode(
            buf, axis=self.axis, dist=self.dist
        ).tobytes()

    def decode(self, buf, out=None):
        buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
        return imagecodecs.floatpred_decode(
            buf, axis=self.axis, dist=self.dist, out=out
        )


class Gif(Codec):
    """GIF codec for numcodecs."""

    codec_id = 'imagecodecs_gif'

    def __init__(
        self,
        *,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.GIF.available:
            msg = 'imagecodecs.GIF not available'
            raise ValueError(msg)

        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.gif_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.gif_decode(buf, asrgb=False, out=out)


class Heif(Codec):
    """HEIF codec for numcodecs."""

    codec_id = 'imagecodecs_heif'

    def __init__(
        self,
        *,
        level: int | None = None,
        bitspersample: int | None = None,
        photometric: imagecodecs.HEIF.COLORSPACE | int | str | None = None,
        compression: imagecodecs.HEIF.COMPRESSION | int | str | None = None,
        index: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.HEIF.available:
            msg = 'imagecodecs.HEIF not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.photometric = _enum_name(photometric, imagecodecs.HEIF.COLORSPACE)
        self.compression = _enum_name(
            compression, imagecodecs.HEIF.COMPRESSION
        )
        self.index = None if index is None else int(index)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.heif_encode(
            buf,
            level=self.level,
            bitspersample=self.bitspersample,
            photometric=self.photometric,
            compression=self.compression,
        )

    def decode(self, buf, out=None):
        return imagecodecs.heif_decode(
            buf,
            index=self.index,
            photometric=self.photometric,
            out=out,
        )


class Htj2k(Codec):
    """HTJ2K codec for numcodecs."""

    codec_id = 'imagecodecs_htj2k'

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
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.HTJ2K.available:
            msg = 'imagecodecs.HTJ2K not available'
            raise ValueError(msg)

        self.level = None if level is None else float(level)
        self.rgb = None if rgb is None else bool(rgb)
        self.planar = None if planar is None else bool(planar)
        self.tile = None if tile is None else (int(tile[0]), int(tile[1]))
        self.resolutions = None if resolutions is None else int(resolutions)
        self.reversible = None if reversible is None else bool(reversible)
        self.tlm = None if tlm is None else bool(tlm)
        self.tilepart = _enum_name(tilepart, imagecodecs.HTJ2K.TILEPART)
        self.skipres = None if skipres is None else int(skipres)
        self.resilient = bool(resilient)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.htj2k_encode(
            buf,
            level=self.level,
            rgb=self.rgb,
            planar=self.planar,
            tile=self.tile,
            resolutions=self.resolutions,
            reversible=self.reversible,
            tlm=self.tlm,
            tilepart=self.tilepart,
        )

    def decode(self, buf, out=None):
        return imagecodecs.htj2k_decode(
            buf,
            planar=self.planar,
            skipres=self.skipres,
            resilient=self.resilient,
            out=out,
        )


class Jpeg(Codec):
    """JPEG codec for numcodecs."""

    codec_id = 'imagecodecs_jpeg'

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
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEG8.available:
            msg = 'imagecodecs.JPEG8 not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.tables = tables
        self.header = header
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )  # unused
        self.colorspace_data = _enum_name(
            colorspace_data, imagecodecs.JPEG8.CS
        )
        self.colorspace_jpeg = _enum_name(
            colorspace_jpeg, imagecodecs.JPEG8.CS
        )
        self.subsampling = subsampling
        self.optimize = None if optimize is None else bool(optimize)
        self.smoothing = None if smoothing is None else bool(smoothing)
        self.lossless = None if lossless is None else bool(lossless)
        self.predictor = None if predictor is None else int(predictor)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpeg8_encode(
            buf,
            level=self.level,
            colorspace=self.colorspace_data,
            outcolorspace=self.colorspace_jpeg,
            subsampling=self.subsampling,
            optimize=self.optimize,
            smoothing=self.smoothing,
            lossless=self.lossless,
            predictor=self.predictor,
        )

    def decode(self, buf, out=None):
        if self.header is not None:
            buf = b''.join((self.header, buf, b'\xff\xd9'))
        return imagecodecs.jpeg8_decode(
            buf,
            tables=self.tables,
            colorspace=self.colorspace_jpeg,
            outcolorspace=self.colorspace_data,
            out=out,
        )

    def get_config(self):
        """Return dictionary holding configuration parameters."""
        config = {'id': self.codec_id}
        for key in self.__dict__:
            if not key.startswith('_'):
                value = getattr(self, key)
                if value is not None and key in {'header', 'tables'}:
                    value = base64.b64encode(value).decode()
                config[key] = value
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiate codec from configuration object."""
        config = dict(config)
        for key in ('header', 'tables'):
            value = config.get(key)
            if value is not None and isinstance(value, str):
                config[key] = base64.b64decode(value.encode())
        return cls(**config)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = 'imagecodecs_jpeg2k'

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
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEG2K.available:
            msg = 'imagecodecs.JPEG2K not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.codecformat = _enum_name(codecformat, imagecodecs.JPEG2K.CODEC)
        self.colorspace = _enum_name(colorspace, imagecodecs.JPEG2K.CLRSPC)
        self.planar = None if planar is None else bool(planar)
        self.tile = None if tile is None else (int(tile[0]), int(tile[1]))
        self.reversible = None if reversible is None else bool(reversible)
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.resolutions = None if resolutions is None else int(resolutions)
        self.numthreads = None if numthreads is None else int(numthreads)
        self.mct = bool(mct)
        self.verbose = None if verbose is None else int(verbose)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpeg2k_encode(
            buf,
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

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(
            buf,
            planar=self.planar,
            verbose=self.verbose,
            numthreads=self.numthreads,
            out=out,
        )


class Jpegls(Codec):
    """JPEG LS codec for numcodecs."""

    codec_id = 'imagecodecs_jpegls'

    def __init__(
        self,
        *,
        level: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEGLS.available:
            msg = 'imagecodecs.JPEGLS not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpegls_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.jpegls_decode(buf, out=out)


class Jpegxl(Codec):
    """JPEG XL codec for numcodecs."""

    codec_id = 'imagecodecs_jpegxl'

    def __init__(
        self,
        *,
        # encode
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
        squeeze: Literal[False] | Sequence[int] | None = None,
        # decode
        index: int | None = None,
        keeporientation: bool | None = None,
        # both
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.JPEGXL.available:
            msg = 'imagecodecs.JPEGXL not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.effort = None if effort is None else int(effort)
        self.distance = None if distance is None else float(distance)
        self.lossless = lossless is None or bool(lossless)
        self.decodingspeed = (
            None if decodingspeed is None else int(decodingspeed)
        )
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.photometric = _enum_name(
            photometric, imagecodecs.JPEGXL.COLOR_SPACE
        )
        self.planar = None if planar is None else bool(planar)
        self.primaries = _enum_name(primaries, imagecodecs.JPEGXL.PRIMARIES)
        self.transfer = _enum_name(
            transfer, imagecodecs.JPEGXL.TRANSFER_FUNCTION
        )
        self.usecontainer = (
            None if usecontainer is None else bool(usecontainer)
        )
        self.index = None if index is None else int(index)
        self.keeporientation = (
            None if keeporientation is None else bool(keeporientation)
        )
        self.numthreads = None if numthreads is None else int(numthreads)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpegxl_encode(
            buf,
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

    def decode(self, buf, out=None):
        return imagecodecs.jpegxl_decode(
            buf,
            index=self.index,
            keeporientation=self.keeporientation,
            numthreads=self.numthreads,
            out=out,
        )


class Jpegxr(Codec):
    """JPEG XR codec for numcodecs."""

    codec_id = 'imagecodecs_jpegxr'

    def __init__(
        self,
        *,
        level: float | None = None,
        photometric: imagecodecs.JPEGXR.PI | int | str | None = None,
        hasalpha: bool | None = None,
        resolution: tuple[float, float] | None = None,
        fp2int: bool = False,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEGXR.available:
            msg = 'imagecodecs.JPEGXR not available'
            raise ValueError(msg)

        self.level = None if level is None else float(level)
        self.photometric = _enum_name(photometric, imagecodecs.JPEGXR.PI)
        self.hasalpha = None if hasalpha is None else bool(hasalpha)
        self.resolution = (
            None
            if resolution is None
            else (float(resolution[0]), float(resolution[1]))
        )
        self.fp2int = bool(fp2int)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpegxr_encode(
            buf,
            level=self.level,
            photometric=self.photometric,
            hasalpha=self.hasalpha,
            resolution=self.resolution,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpegxr_decode(buf, fp2int=self.fp2int, out=out)


class Jpegxs(Codec):
    """JPEG XS codec for numcodecs."""

    codec_id = 'imagecodecs_jpegxs'

    def __init__(
        self,
        *,
        config: str | None = None,
        bitspersample: int | None = None,
        verbose: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEGXS.available:
            msg = 'imagecodecs.JPEGXS not available'
            raise ValueError(msg)

        self.config = config
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.verbose = None if verbose is None else int(verbose)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpegxs_encode(
            buf,
            config=self.config,
            bitspersample=self.bitspersample,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpegxs_decode(buf, out=out)


class Lerc(Codec):
    """LERC codec for numcodecs."""

    codec_id = 'imagecodecs_lerc'

    def __init__(
        self,
        *,
        level: float | None = None,
        # masks: ArrayLike | None = None,
        version: int | None = None,
        planar: bool | None = None,
        compression: Literal['zstd', 'deflate'] | None = None,
        compressionargs: dict[str, Any] | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.LERC.available:
            msg = 'imagecodecs.LERC not available'
            raise ValueError(msg)

        self.level = None if level is None else float(level)
        self.version = None if version is None else int(version)
        self.planar = None if planar is None else bool(planar)
        self.squeeze = squeeze
        self.compression = compression
        self.compressionargs = compressionargs
        # TODO: support mask?
        # self.mask = None

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.lerc_encode(
            buf,
            level=self.level,
            version=self.version,
            planar=self.planar,
            compression=self.compression,
            compressionargs=self.compressionargs,
        )

    def decode(self, buf, out=None):
        return imagecodecs.lerc_decode(buf, masks=False, out=out)


class Ljpeg(Codec):
    """LJPEG codec for numcodecs."""

    codec_id = 'imagecodecs_ljpeg'

    def __init__(
        self,
        *,
        bitspersample: int | None = None,
        # delinearize: ArrayLike | None = None,
        # linearize: ArrayLike | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.LJPEG.available:
            msg = 'imagecodecs.LJPEG not available'
            raise ValueError(msg)

        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.ljpeg_encode(buf, bitspersample=self.bitspersample)

    def decode(self, buf, out=None):
        return imagecodecs.ljpeg_decode(buf, out=out)


class Lz4(Codec):
    """LZ4 codec for numcodecs."""

    codec_id = 'imagecodecs_lz4'

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

        self.level = None if level is None else int(level)
        self.hc = bool(hc)
        self.header = bool(header)

    def encode(self, buf):
        return imagecodecs.lz4_encode(
            buf, level=self.level, hc=self.hc, header=self.header
        )

    def decode(self, buf, out=None):
        return imagecodecs.lz4_decode(buf, header=self.header, out=_flat(out))


class Lz4f(Codec):
    """LZ4F codec for numcodecs."""

    codec_id = 'imagecodecs_lz4f'

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

        self.level = None if level is None else int(level)
        self.blocksizeid = None if blocksizeid is None else int(blocksizeid)
        self.contentchecksum = (
            None if contentchecksum is None else bool(contentchecksum)
        )
        self.blockchecksum = (
            None if blockchecksum is None else bool(blockchecksum)
        )

    def encode(self, buf):
        return imagecodecs.lz4f_encode(
            buf,
            level=self.level,
            blocksizeid=self.blocksizeid,
            contentchecksum=self.contentchecksum,
            blockchecksum=self.blockchecksum,
        )

    def decode(self, buf, out=None):
        return imagecodecs.lz4f_decode(buf, out=_flat(out))


class Lz4h5(Codec):
    """LZ4H5 codec for numcodecs."""

    codec_id = 'imagecodecs_lz4h5'

    def __init__(
        self,
        *,
        level: int | None = None,
        blocksize: int | None = None,
    ) -> None:
        if not imagecodecs.LZ4H5.available:
            msg = 'imagecodecs.LZ4H5 not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.blocksize = None if blocksize is None else int(blocksize)

    def encode(self, buf):
        return imagecodecs.lz4h5_encode(
            buf, level=self.level, blocksize=self.blocksize
        )

    def decode(self, buf, out=None):
        return imagecodecs.lz4h5_decode(buf, out=_flat(out))


class Lzf(Codec):
    """LZF codec for numcodecs."""

    codec_id = 'imagecodecs_lzf'

    def __init__(
        self,
        *,
        header: bool = True,
    ) -> None:
        if not imagecodecs.LZF.available:
            msg = 'imagecodecs.LZF not available'
            raise ValueError(msg)

        self.header = bool(header)

    def encode(self, buf):
        return imagecodecs.lzf_encode(buf, header=self.header)

    def decode(self, buf, out=None):
        return imagecodecs.lzf_decode(buf, header=self.header, out=_flat(out))


class Lzfse(Codec):
    """LZFSE codec for numcodecs."""

    codec_id = 'imagecodecs_lzfse'

    def __init__(self) -> None:
        if not imagecodecs.LZFSE.available:
            msg = 'imagecodecs.LZFSE not available'
            raise ValueError(msg)

    def encode(self, buf):
        return imagecodecs.lzfse_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.lzfse_decode(buf, out=_flat(out))


class Lzham(Codec):
    """LZHAM codec for numcodecs."""

    codec_id = 'imagecodecs_lzham'

    def __init__(
        self,
        *,
        level: int | str | None = None,
    ) -> None:
        if not imagecodecs.LZHAM.available:
            msg = 'imagecodecs.LZHAM not available'
            raise ValueError(msg)

        self.level = level

    def encode(self, buf):
        return imagecodecs.lzham_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.lzham_decode(buf, out=_flat(out))


class Lzma(Codec):
    """LZMA codec for numcodecs."""

    codec_id = 'imagecodecs_lzma'

    def __init__(
        self,
        *,
        level: int | None = None,
        check: imagecodecs.LZMA.CHECK | int | str | None = None,
    ) -> None:
        if not imagecodecs.LZMA.available:
            msg = 'imagecodecs.LZMA not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.check = _enum_name(check, imagecodecs.LZMA.CHECK)

    def encode(self, buf):
        return imagecodecs.lzma_encode(buf, level=self.level, check=self.check)

    def decode(self, buf, out=None):
        return imagecodecs.lzma_decode(buf, out=_flat(out))


class Lzo(Codec):
    """LZO codec for numcodecs."""

    codec_id = 'imagecodecs_lzo'

    def __init__(self, *, header: bool = False) -> None:
        if not imagecodecs.LZO.available:
            msg = 'imagecodecs.LZO not available'
            raise ValueError(msg)
        self.header = bool(header)

    def encode(self, buf):
        # return imagecodecs.lzo_encode(buf)
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.lzo_decode(buf, header=self.header, out=_flat(out))


class Lzw(Codec):
    """LZW codec for numcodecs."""

    codec_id = 'imagecodecs_lzw'

    def __init__(self) -> None:
        if not imagecodecs.LZW.available:
            msg = 'imagecodecs.LZW not available'
            raise ValueError(msg)

    def encode(self, buf):
        return imagecodecs.lzw_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.lzw_decode(buf, out=_flat(out))


class Meshopt(Codec):
    """MESHOPT codec for numcodecs."""

    codec_id = 'imagecodecs_meshopt'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        items: int | None = None,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.MESHOPT.available:
            msg = 'imagecodecs.MESHOPT not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.items = None if items is None else int(items)
        self.level = None if level is None else int(level)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.shape != self.shape:
            msg = f'{buf.shape=} does not match {self.shape=}'
            raise ValueError(msg)
        if buf.dtype != self.dtype:
            msg = f'{buf.dtype=} does not match {self.dtype=}'
            raise ValueError(msg)
        return imagecodecs.meshopt_encode(
            buf, level=self.level, items=self.items
        )

    def decode(self, buf, out=None):
        return imagecodecs.meshopt_decode(
            buf,
            shape=self.shape,
            dtype=self.dtype,
            items=self.items,
            out=out,
        )


class Packbits(Codec):
    """PackBits codec for numcodecs."""

    codec_id = 'imagecodecs_packbits'

    def __init__(
        self,
        *,
        axis: int | None = None,
    ) -> None:
        if not imagecodecs.PACKBITS.available:
            msg = 'imagecodecs.PACKBITS not available'
            raise ValueError(msg)

        self.axis = None if axis is None else int(axis)

    def encode(self, buf):
        if not isinstance(buf, (bytes, bytearray)):
            buf = numpy.asarray(buf)
        return imagecodecs.packbits_encode(buf, axis=self.axis)

    def decode(self, buf, out=None):
        return imagecodecs.packbits_decode(buf, out=_flat(out))


class Packints(Codec):
    """Packed integer codec for numcodecs."""

    codec_id = 'imagecodecs_packints'

    def __init__(
        self,
        *,
        dtype: DTypeLike,
        bitspersample: int,
        bitorder: Literal['>', '<'] | None = None,
        runlen: int = 0,
    ) -> None:
        if not imagecodecs.PACKINTS.available:
            msg = 'imagecodecs.PACKINTS not available'
            raise ValueError(msg)

        self.dtype = numpy.dtype(dtype).str
        self.bitspersample = int(bitspersample)
        self.bitorder = bitorder
        self.runlen = int(runlen)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.dtype != self.dtype:
            msg = f'{buf.dtype=} does not match {self.dtype=}'
            raise ValueError(msg)
        return imagecodecs.packints_encode(
            buf,
            self.bitspersample,
            bitorder=self.bitorder,
            runlen=self.runlen,
        )

    def decode(self, buf, out=None):
        return imagecodecs.packints_decode(
            buf,
            self.dtype,
            self.bitspersample,
            bitorder=self.bitorder,
            runlen=self.runlen,
            out=out,
        )


class Pcodec(Codec):
    """Pcodec codec for numcodecs."""

    codec_id = 'imagecodecs_pcodec'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        level: int | None = None,
        pagesize: int | None = None,
    ) -> None:
        if not imagecodecs.PCODEC.available:
            msg = 'imagecodecs.PCODEC not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.level = None if level is None else int(level)
        self.pagesize = None if pagesize is None else int(pagesize)

    def encode(self, buf):
        return imagecodecs.pcodec_encode(
            buf, level=self.level, pagesize=self.pagesize
        )

    def decode(self, buf, out=None):
        return imagecodecs.pcodec_decode(
            buf, shape=self.shape, dtype=self.dtype, out=out
        )


class Pglz(Codec):
    """PGLZ codec for numcodecs."""

    codec_id = 'imagecodecs_pglz'

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

        self.header = bool(header)
        self.strategy = strategy
        self.checkcomplete = (
            None if checkcomplete is None else bool(checkcomplete)
        )

    def encode(self, buf):
        return imagecodecs.pglz_encode(
            buf, strategy=self.strategy, header=self.header
        )

    def decode(self, buf, out=None):
        return imagecodecs.pglz_decode(
            buf,
            header=self.header,
            checkcomplete=self.checkcomplete,
            out=_flat(out),
        )


class Pixarlog(Codec):
    """Pixarlog codec for numcodecs."""

    codec_id = 'imagecodecs_pixarlog'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
        level: int | None = None,
        deflate: bool = True,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.PIXARLOG.available:
            msg = 'imagecodecs.PIXARLOG not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.level = None if level is None else int(level)
        self.deflate = bool(deflate)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.pixarlog_encode(
            buf, level=self.level, deflate=self.deflate
        )

    def decode(self, buf, out=None):
        shape = _squeeze_shape(self.shape, self.squeeze)
        return imagecodecs.pixarlog_decode(
            buf,
            shape=shape,
            dtype=self.dtype,
            deflate=self.deflate,
            out=out,
        )


class Plio(Codec):
    """PLIO codec for numcodecs."""

    codec_id = 'imagecodecs_plio'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> None:
        if not imagecodecs.PLIO.available:
            msg = 'imagecodecs.PLIO not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str

    def encode(self, buf):
        return imagecodecs.plio_encode(numpy.asarray(buf).ravel())

    def decode(self, buf, out=None):
        npix = int(numpy.prod(self.shape))
        decoded = imagecodecs.plio_decode(buf, npix=npix)
        decoded = decoded.reshape(self.shape)
        if self.dtype is not None:
            decoded = decoded.astype(self.dtype)
        return decoded


class Png(Codec):
    """PNG codec for numcodecs."""

    codec_id = 'imagecodecs_png'

    def __init__(
        self,
        *,
        level: int | None = None,
        strategy: imagecodecs.PNG.STRATEGY | int | str | None = None,
        filter: imagecodecs.PNG.FILTER | int | str | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.PNG.available:
            msg = 'imagecodecs.PNG not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.strategy = _enum_name(strategy, imagecodecs.PNG.STRATEGY)
        self.filter = _enum_name(filter, imagecodecs.PNG.FILTER)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.png_encode(
            buf,
            level=self.level,
            strategy=self.strategy,
            filter=self.filter,
        )

    def decode(self, buf, out=None):
        return imagecodecs.png_decode(buf, out=out)


class Qoi(Codec):
    """QOI codec for numcodecs."""

    codec_id = 'imagecodecs_qoi'

    def __init__(
        self,
        *,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.QOI.available:
            msg = 'imagecodecs.QOI not available'
            raise ValueError(msg)

        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.qoi_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.qoi_decode(buf, out=out)


class Quantize(Codec):
    """Quantize codec for numcodecs."""

    codec_id = 'imagecodecs_quantize'

    def __init__(
        self,
        *,
        mode: Literal['bitgroom', 'granularbr', 'gbr', 'bitround', 'scale'],
        nsd: int,  # number of significant digits
    ) -> None:
        if not imagecodecs.QUANTIZE.available:
            msg = 'imagecodecs.QUANTIZE not available'
            raise ValueError(msg)

        self.nsd = int(nsd)
        self.mode = mode

    def encode(self, buf):
        return imagecodecs.quantize_encode(buf, self.mode, self.nsd)

    def decode(self, buf, out=None):
        return buf
        # return imagecodecs.quantize_decode(buf, self.mode, self.nsd, out=out)


class Hcomp(Codec):
    """Hcomp codec for numcodecs."""

    codec_id = 'imagecodecs_hcomp'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        level: int = 0,
        smooth: int = 0,
        safe32: bool | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.HCOMP.available:
            msg = 'imagecodecs.HCOMP not available'
            raise ValueError(msg)

        self.shape = _squeeze_shape(shape, squeeze)
        self.level = int(level)
        self.smooth = int(smooth)
        self.safe32 = None if safe32 is None else bool(safe32)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.hcomp_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.hcomp_decode(
            buf,
            smooth=self.smooth,
            safe32=self.safe32,
            out=out,
        )


class Rcomp(Codec):
    """Rcomp codec for numcodecs."""

    codec_id = 'imagecodecs_rcomp'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike | None,
        nblock: int | None = None,
    ) -> None:
        if not imagecodecs.RCOMP.available:
            msg = 'imagecodecs.RCOMP not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.nblock = None if nblock is None else int(nblock)

    def encode(self, buf):
        return imagecodecs.rcomp_encode(buf, nblock=self.nblock)

    def decode(self, buf, out=None):
        return imagecodecs.rcomp_decode(
            buf,
            shape=self.shape,
            dtype=self.dtype,
            nblock=self.nblock,
            out=out,
        )


class Rgbe(Codec):
    """RGBE codec for numcodecs."""

    codec_id = 'imagecodecs_rgbe'

    def __init__(
        self,
        *,
        header: bool | None = None,
        rle: bool | None = None,
        shape: tuple[int, ...] | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.RGBE.available:
            msg = 'imagecodecs.RGBE not available'
            raise ValueError(msg)

        if not header and shape is None:
            msg = 'must specify data shape if no header'
            raise ValueError(msg)
        if shape and shape[-1] != 3:
            msg = 'invalid shape'
            raise ValueError(msg)

        self.header = bool(header)
        self.rle = None if rle is None else bool(rle)
        self.shape = None if shape is None else tuple(shape)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.rgbe_encode(buf, header=self.header, rle=self.rle)

    def decode(self, buf, out=None):
        if out is None and not self.header:
            out = numpy.empty(self.shape, numpy.float32)
        return imagecodecs.rgbe_decode(
            buf, header=self.header, rle=self.rle, out=out
        )


class Snappy(Codec):
    """Snappy codec for numcodecs."""

    codec_id = 'imagecodecs_snappy'

    def __init__(self) -> None:
        if not imagecodecs.SNAPPY.available:
            msg = 'imagecodecs.SNAPPY not available'
            raise ValueError(msg)

    def encode(self, buf):
        return imagecodecs.snappy_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.snappy_decode(buf, out=_flat(out))


class Sperr(Codec):
    """SPERR codec for numcodecs."""

    codec_id = 'imagecodecs_sperr'

    def __init__(
        self,
        *,
        level: float,
        mode: Literal['bpp', 'psnr', 'pwe'],
        dtype: DTypeLike | None = None,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, int, int] | None = None,
        header: bool = True,
        numthreads: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.SPERR.available:
            msg = 'imagecodecs.SPERR not available'
            raise ValueError(msg)

        if header:
            self.shape = None
            self.dtype = None
        elif shape is None or dtype is None:
            msg = 'invalid shape or dtype'
            raise ValueError(msg)
        else:
            self.shape = tuple(shape)
            self.dtype = numpy.dtype(dtype).str
        self.mode = mode
        self.level = float(level)
        self.chunks = (
            None
            if chunks is None
            else (int(chunks[0]), int(chunks[1]), int(chunks[2]))
        )
        self.header = bool(header)
        self.numthreads = None if numthreads is None else int(numthreads)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        if not self.header:
            if buf.shape != self.shape:
                msg = f'{buf.shape=} does not match {self.shape=}'
                raise ValueError(msg)
            if buf.dtype != self.dtype:
                msg = f'{buf.dtype=} does not match {self.dtype=}'
                raise ValueError(msg)
        return imagecodecs.sperr_encode(
            buf,
            level=self.level,
            mode=self.mode,
            chunks=self.chunks,  # mypy: ignore[arg-type]
            header=self.header,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        if self.header:
            return imagecodecs.sperr_decode(buf, out=out)
        return imagecodecs.sperr_decode(
            buf,
            shape=self.shape,
            dtype=numpy.dtype(self.dtype),
            header=self.header,
            numthreads=self.numthreads,
            out=out,
        )


class Spng(Codec):
    """SPNG codec for numcodecs."""

    codec_id = 'imagecodecs_spng'

    def __init__(
        self,
        *,
        level: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.SPNG.available:
            msg = 'imagecodecs.SPNG not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.spng_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.spng_decode(buf, out=out)


class Sz3(Codec):
    """SZ3 codec for numcodecs."""

    codec_id = 'imagecodecs_sz3'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        mode: Literal['abs', 'rel', 'abs_or_rel', 'abs_and_rel'] | None = None,
        abs: float | None = None,
        rel: float | None = None,
    ) -> None:
        if not imagecodecs.SZ3.available:
            msg = 'imagecodecs.SZ3 not available'
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.mode = 'abs' if mode is None else mode
        self.abs = 0.0 if abs is None else float(abs)
        self.rel = 0.0 if rel is None else float(rel)

    def encode(self, buf):
        return imagecodecs.sz3_encode(
            buf, mode=self.mode, abs=self.abs, rel=self.rel
        )

    def decode(self, buf, out=None):
        return imagecodecs.sz3_decode(
            buf, shape=self.shape, dtype=self.dtype, out=out
        )


class Szip(Codec):
    """SZIP codec for numcodecs."""

    codec_id = 'imagecodecs_szip'

    def __init__(
        self,
        options_mask: int,
        pixels_per_block: int,
        bits_per_pixel: int,
        pixels_per_scanline: int,
        *,
        header: bool = True,
    ) -> None:
        if not imagecodecs.SZIP.available:
            msg = 'imagecodecs.SZIP not available'
            raise ValueError(msg)

        self.options_mask = int(options_mask)
        self.pixels_per_block = int(pixels_per_block)
        self.bits_per_pixel = int(bits_per_pixel)
        self.pixels_per_scanline = int(pixels_per_scanline)
        self.header = bool(header)

    def encode(self, buf):
        return imagecodecs.szip_encode(
            buf,
            options_mask=self.options_mask,
            pixels_per_block=self.pixels_per_block,
            bits_per_pixel=self.bits_per_pixel,
            pixels_per_scanline=self.pixels_per_scanline,
            header=self.header,
        )

    def decode(self, buf, out=None):
        return imagecodecs.szip_decode(
            buf,
            options_mask=self.options_mask,
            pixels_per_block=self.pixels_per_block,
            bits_per_pixel=self.bits_per_pixel,
            pixels_per_scanline=self.pixels_per_scanline,
            header=self.header,
            out=_flat(out),
        )


class Tiff(Codec):
    """TIFF codec for numcodecs."""

    codec_id = 'imagecodecs_tiff'

    def __init__(
        self,
        *,
        # decode
        index: int | None = None,
        asrgb: bool = False,
        # encode
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
        colormap: ArrayLike | None = None,
        iccprofile: bytes | None = None,
        description: str | None = None,
        datetime: str | None = None,
        resolution: tuple[float, float] | None = None,
        resolutionunit: imagecodecs.TIFF.RESUNIT | int | str | None = None,
        software: str | None = None,
        verbose: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.TIFF.available:
            msg = 'imagecodecs.TIFF not available'
            raise ValueError(msg)

        self.bigtiff = None if bigtiff is None else bool(bigtiff)
        self.byteorder = _enum_name(byteorder, imagecodecs.TIFF.ENDIAN)
        self.subfiletype = _enum_name(subfiletype, imagecodecs.TIFF.FILETYPE)
        self.photometric = _enum_name(
            photometric, imagecodecs.TIFF.PHOTOMETRIC
        )
        self.planarconfig = _enum_name(
            planarconfig, imagecodecs.TIFF.PLANARCONFIG
        )
        self.extrasample = _enum_name(
            extrasample, imagecodecs.TIFF.EXTRASAMPLE
        )
        self.tile = None if tile is None else (int(tile[0]), int(tile[1]))
        self.rowsperstrip = None if rowsperstrip is None else int(rowsperstrip)
        self.bitspersample = (
            None if bitspersample is None else int(bitspersample)
        )
        self.compression = _enum_name(
            compression, imagecodecs.TIFF.COMPRESSION
        )
        self.subcodec = _enum_name(subcodec, imagecodecs.TIFF.COMPRESSION)
        self.level = None if level is None else int(level)
        self.predictor = (
            predictor
            if isinstance(predictor, bool)
            else _enum_name(predictor, imagecodecs.TIFF.PREDICTOR)
        )
        self.colormap = (
            None
            if colormap is None
            else numpy.ascontiguousarray(colormap, dtype=numpy.uint16)
        )
        self.iccprofile = iccprofile
        self.resolution = (
            None
            if resolution is None
            else (float(resolution[0]), float(resolution[1]))
        )
        self.resolutionunit = _enum_name(
            resolutionunit, imagecodecs.TIFF.RESUNIT
        )
        self.description = description
        self.datetime = datetime
        self.software = software

        self.index = None if index is None else int(index)
        self.asrgb = bool(asrgb)
        self.verbose = None if verbose is None else int(verbose)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.tiff_encode(
            buf,
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
            colormap=self.colormap,
            iccprofile=self.iccprofile,
            resolution=self.resolution,
            resolutionunit=self.resolutionunit,
            description=self.description,
            datetime=self.datetime,
            software=self.software,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.tiff_decode(
            buf,
            index=self.index,
            asrgb=self.asrgb,
            verbose=self.verbose,
            out=out,
        )


class Ultrahdr(Codec):
    """Ultra HDR codec for numcodecs."""

    codec_id = 'imagecodecs_ultrahdr'

    def __init__(
        self,
        *,
        # encode
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
        # decode
        dtype: DTypeLike | None = None,
        boost: float | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.ULTRAHDR.available:
            msg = 'imagecodecs.ULTRAHDR not available'
            raise ValueError(msg)

        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.level = None if level is None else int(level)
        self.scale = None if scale is None else int(scale)
        self.gamut = _enum_name(gamut, imagecodecs.ULTRAHDR.CG)
        self.crange = _enum_name(crange, imagecodecs.ULTRAHDR.CR)
        self.transfer = _enum_name(transfer, imagecodecs.ULTRAHDR.CT)
        self.nits = None if nits is None else float(nits)
        self.boostmin = None if boostmin is None else float(boostmin)
        self.boostmax = None if boostmax is None else float(boostmax)
        self.usage = _enum_name(usage, imagecodecs.ULTRAHDR.USAGE)
        self.codec = _enum_name(codec, imagecodecs.ULTRAHDR.CODEC)
        self.boost = None if boost is None else float(boost)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        if self.dtype is not None and buf.dtype != self.dtype:
            msg = f'{buf.dtype=} != {self.dtype}'
            raise ValueError(msg)

        return imagecodecs.ultrahdr_encode(
            buf,
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

    def decode(self, buf, out=None):
        return imagecodecs.ultrahdr_decode(
            buf,
            dtype=self.dtype,
            transfer=self.transfer,
            boost=self.boost,
            out=out,
        )


class Webp(Codec):
    """WebP codec for numcodecs."""

    codec_id = 'imagecodecs_webp'

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
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.WEBP.available:
            msg = 'imagecodecs.WEBP not available'
            raise ValueError(msg)

        self.level = None if level is None else float(level)
        self.hasalpha = None if hasalpha is None else bool(hasalpha)
        self.method = None if method is None else int(method)
        self.index = None if index is None else int(index)
        self.lossless = None if lossless is None else bool(lossless)
        self.numthreads = None if numthreads is None else int(numthreads)
        self.delay = None if delay is None else int(delay)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.webp_encode(
            buf,
            level=self.level,
            lossless=self.lossless,
            method=self.method,
            numthreads=self.numthreads,
            delay=self.delay,
        )

    def decode(self, buf, out=None):
        return imagecodecs.webp_decode(
            buf, index=self.index, hasalpha=self.hasalpha, out=out
        )


class Wavpack(Codec):
    """Wavpack codec for numcodecs."""

    codec_id = 'imagecodecs_wavpack'

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

        self.level = None if level is None else int(level)
        self.bitrate = None if bitrate is None else float(bitrate)
        self.channels = bool(channels)
        self.numthreads = None if numthreads is None else int(numthreads)

    def encode(self, buf):
        arr = numpy.asarray(buf)
        if arr.ndim > 1:
            arr = (
                arr.reshape(-1, arr.shape[-1])
                if self.channels
                else arr.ravel()
            )
        return imagecodecs.wavpack_encode(
            arr,
            level=self.level,
            bitrate=self.bitrate,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.wavpack_decode(
            buf, numthreads=self.numthreads, out=out
        )


class Wic(Codec):
    """WIC codec for numcodecs."""

    codec_id = 'imagecodecs_wic'

    def __init__(
        self,
        *,
        level: int | None = None,
        format: imagecodecs.WIC.FORMAT | int | str | None = None,
        index: int = 0,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.WIC.available:
            msg = 'imagecodecs.WIC not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)
        self.format = _enum_name(format, imagecodecs.WIC.FORMAT)
        self.index = int(index)
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.wic_encode(
            buf, level=self.level, format=self.format
        )

    def decode(self, buf, out=None):
        return imagecodecs.wic_decode(buf, index=self.index, out=out)


class Xor(Codec):
    """XOR codec for numcodecs."""

    codec_id = 'imagecodecs_xor'

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        axis: int = -1,
    ) -> None:
        if not imagecodecs.XOR.available:
            msg = 'imagecodecs.XOR not available'
            raise ValueError(msg)

        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = int(axis)

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.asarray(buf)
            if buf.shape != self.shape:
                msg = f'{buf.shape=} does not match {self.shape=}'
                raise ValueError(msg)
            if buf.dtype != self.dtype:
                msg = f'{buf.dtype=} does not match {self.dtype=}'
                raise ValueError(msg)
        return imagecodecs.xor_encode(buf, axis=self.axis).tobytes()

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype)
            if self.shape is not None:
                buf = buf.reshape(self.shape)
        return imagecodecs.xor_decode(buf, axis=self.axis, out=_flat(out))


class Zfp(Codec):
    """ZFP codec for numcodecs."""

    codec_id = 'imagecodecs_zfp'

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        strides: tuple[int, ...] | None = None,
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

        if header:
            self.shape = None
            self.dtype = None
            self.strides = None
        elif shape is None or dtype is None:
            msg = 'invalid shape or dtype'
            raise ValueError(msg)
        else:
            self.shape = tuple(shape)
            self.dtype = numpy.dtype(dtype).str
            self.strides = None if strides is None else tuple(strides)
        self.level = None if level is None else int(level)
        self.mode = _enum_name(mode, imagecodecs.ZFP.MODE)
        self.execution = _enum_name(execution, imagecodecs.ZFP.EXEC)
        self.numthreads = numthreads
        self.chunksize = chunksize
        self.header = bool(header)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if not self.header:
            if buf.shape != self.shape:
                msg = f'{buf.shape=} does not match {self.shape=}'
                raise ValueError(msg)
            if buf.dtype != self.dtype:
                msg = f'{buf.dtype=} does not match {self.dtype=}'
                raise ValueError(msg)
        return imagecodecs.zfp_encode(
            buf,
            level=self.level,
            mode=self.mode,
            execution=self.execution,
            header=self.header,
            numthreads=self.numthreads,
            chunksize=self.chunksize,
        )

    def decode(self, buf, out=None):
        if self.header:
            return imagecodecs.zfp_decode(buf, out=out)
        return imagecodecs.zfp_decode(
            buf,
            shape=self.shape,
            dtype=numpy.dtype(self.dtype),
            strides=self.strides,
            numthreads=self.numthreads,
            out=out,
        )


class Zlib(Codec):
    """Zlib codec for numcodecs."""

    codec_id = 'imagecodecs_zlib'

    def __init__(
        self,
        *,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.ZLIB.available:
            msg = 'imagecodecs.ZLIB not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)

    def encode(self, buf):
        return imagecodecs.zlib_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zlib_decode(buf, out=_flat(out))


class Zlibng(Codec):
    """Zlibng codec for numcodecs."""

    codec_id = 'imagecodecs_zlibng'

    def __init__(
        self,
        *,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.ZLIBNG.available:
            msg = 'imagecodecs.ZLIBNG not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)

    def encode(self, buf):
        return imagecodecs.zlibng_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zlibng_decode(buf, out=_flat(out))


class Zopfli(Codec):
    """Zopfli codec for numcodecs."""

    codec_id = 'imagecodecs_zopfli'

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

        self.level = None if level is None else int(level)
        self.format = None if format is None else int(format)
        self.blocksplitting = (
            None if blocksplitting is None else bool(blocksplitting)
        )
        self.blocksplittingmax = (
            None if blocksplittingmax is None else int(blocksplittingmax)
        )

    def encode(self, buf):
        return imagecodecs.zopfli_encode(
            buf,
            self.level,
            format=self.format,
            blocksplitting=self.blocksplitting,
            blocksplittingmax=self.blocksplittingmax,
        )

    def decode(self, buf, out=None):
        return imagecodecs.zopfli_decode(buf, out=_flat(out))


class Zstd(Codec):
    """ZStandard codec for numcodecs."""

    codec_id = 'imagecodecs_zstd'

    def __init__(
        self,
        *,
        level: int | None = None,
    ) -> None:
        if not imagecodecs.ZSTD.available:
            msg = 'imagecodecs.ZSTD not available'
            raise ValueError(msg)

        self.level = None if level is None else int(level)

    def encode(self, buf):
        return imagecodecs.zstd_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zstd_decode(buf, out=_flat(out))


def _flat(buf: Any, /) -> memoryview | None:
    """Return contiguous bytes-view of numpy array if possible, else None."""
    if buf is None:
        return None
    view = memoryview(buf)
    if view.readonly or not view.contiguous:
        return None
    return view.cast('B')


def _contiguous(buf: Any, /) -> memoryview:
    """Return buffer as contiguous view of bytes."""
    view = memoryview(buf)
    if not view.contiguous:
        view = memoryview(numpy.ascontiguousarray(buf))
    return view.cast('B')


def _image(
    buf: Any,
    squeeze: Literal[False] | Sequence[int] | None = None,
    /,
) -> NDArray[Any]:
    """Return buffer as squeezed numpy array with at least 2 dimensions."""
    if squeeze is None:
        return numpy.atleast_2d(numpy.squeeze(buf))
    arr = numpy.asarray(buf)
    if not squeeze:
        return arr
    shape = tuple(i for i, j in zip(buf.shape, squeeze, strict=True) if not j)
    return arr.reshape(shape)


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


def _squeeze_shape(
    shape: tuple[int, ...],
    squeeze: Literal[False] | Sequence[int] | None = None,
    /,
) -> tuple[int, ...]:
    """Return squeezed shape with at least 2 dimensions."""
    if squeeze is None:
        shape = tuple(s for s in shape if s > 1)
        if len(shape) < 2:
            shape = (1,) * (2 - len(shape)) + shape
        return shape
    if not squeeze:
        return shape
    return tuple(i for i, j in zip(shape, squeeze, strict=True) if not j)


def register_codecs(
    codecs: Any = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """Register imagecodecs.numcodecs codecs with numcodecs."""
    for name, cls in list(globals().items()):
        if not (
            isinstance(cls, type)
            and issubclass(cls, Codec)
            and name != 'Codec'
        ):
            continue
        assert hasattr(cls, 'codec_id')
        if codecs is not None and cls.codec_id not in codecs:
            continue
        try:
            with contextlib.suppress(TypeError):
                # registered, but failed
                get_codec({'id': cls.codec_id})
        except ValueError:
            # not registered yet
            pass
        else:
            if not force:
                if verbose:
                    logging.getLogger(__name__).warning(
                        'numcodec %s already registered', cls.codec_id
                    )
                continue
            if verbose:
                logging.getLogger(__name__).warning(
                    'replacing registered numcodec %s', cls.codec_id
                )
        register_codec(cls)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="misc"
