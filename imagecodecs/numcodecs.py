# imagecodecs/numcodecs.py

# Copyright (c) 2021-2024, Christoph Gohlke
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

"""Additional numcodecs implemented using imagecodecs."""

from __future__ import annotations

__all__ = ['register_codecs']

from typing import TYPE_CHECKING

import imagecodecs
import numpy
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from numpy.typing import DTypeLike, NDArray


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
            raise ValueError('imagecodecs.AEC not available')

        self.bitspersample = bitspersample
        self.flags = flags
        self.blocksize = blocksize
        self.rsi = rsi

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
        strategy: int | None = None,
        filter: int | None = None,
        photometric: int | None = None,
        delay: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.APNG.available:
            raise ValueError('imagecodecs.APNG not available')

        self.level = level
        self.strategy = strategy
        self.filter = filter
        self.photometric = photometric
        self.delay = delay
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
        pixelformat: int | str | None = None,
        codec: int | str | None = None,
        numthreads: int | None = None,
        index: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.AVIF.available:
            raise ValueError('imagecodecs.AVIF not available')

        self.level = level
        self.speed = speed
        self.tilelog2 = tilelog2
        self.bitspersample = bitspersample
        self.pixelformat = pixelformat
        self.codec = codec
        self.numthreads = numthreads
        self.index = index
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
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.avif_decode(
            buf, index=self.index, numthreads=self.numthreads, out=out
        )


class Bitorder(Codec):
    """Bitorder codec for numcodecs."""

    codec_id = 'imagecodecs_bitorder'

    def __init__(self) -> None:
        if not imagecodecs.BITORDER.available:
            raise ValueError('imagecodecs.BITORDER not available')

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
            raise ValueError('imagecodecs.BITSHUFFLE not available')

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
        compressor: int | str | None = None,
        shuffle: int | str | None = None,
        typesize: int | None = None,
        blocksize: int | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.BLOSC.available:
            raise ValueError('imagecodecs.BLOSC not available')

        self.level = level
        self.compressor = compressor
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = shuffle
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
        compressor: int | str | None = None,
        shuffle: int | str | None = None,
        splitmode: int | str | None = None,
        typesize: int | None = None,
        blocksize: int | None = None,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.BLOSC2.available:
            raise ValueError('imagecodecs.BLOSC2 not available')

        self.level = level
        self.compressor = compressor
        self.splitmode = splitmode
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = shuffle
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
            raise ValueError('imagecodecs.BMP not available')

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
        mode: int | None = None,
        lgwin: int | None = None,
    ) -> None:
        if not imagecodecs.BROTLI.available:
            raise ValueError('imagecodecs.BROTLI not available')

        self.level = level
        self.mode = mode
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
            raise ValueError('imagecodecs.BYTESHUFFLE not available')

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)
        self.delta = bool(delta)
        self.reorder = bool(reorder)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.shape != self.shape:
            raise ValueError(f'{buf.shape=} does not match {self.shape=}')
        if buf.dtype != self.dtype:
            raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
        return imagecodecs.byteshuffle_encode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        ).tobytes()

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
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
            raise ValueError('imagecodecs.BZ2 not available')

        self.level = level

    def encode(self, buf):
        return imagecodecs.bz2_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.bz2_decode(buf, out=_flat(out))


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
        if kind == 'crc32':
            if imagecodecs.ZLIBNG.available:
                self._checksum = imagecodecs.zlibng_crc32
            elif imagecodecs.DEFLATE.available:
                self._checksum = imagecodecs.deflate_crc32
            elif imagecodecs.ZLIB.available:
                self._checksum = imagecodecs.zlib_crc32
            else:
                raise ValueError('imagecodecs.ZLIB not available')
            if prepend is None:
                prepend = True
        elif kind == 'adler32':
            if imagecodecs.ZLIBNG.available:
                self._checksum = imagecodecs.zlibng_adler32
            elif imagecodecs.DEFLATE.available:
                self._checksum = imagecodecs.deflate_adler32
            if imagecodecs.ZLIB.available:
                self._checksum = imagecodecs.zlib_adler32
            else:
                raise ValueError('imagecodecs.ZLIB not available')
            if prepend is None:
                prepend = True
        elif kind == 'fletcher32':
            if not imagecodecs.H5CHECKSUM.available:
                raise ValueError('imagecodecs.H5CHECKSUM not available')
            self._checksum = imagecodecs.h5checksum_fletcher32
            if prepend is None:
                prepend = False
        elif kind == 'lookup3':
            if not imagecodecs.H5CHECKSUM.available:
                raise ValueError('imagecodecs.H5CHECKSUM not available')
            self._checksum = imagecodecs.h5checksum_lookup3
            if prepend is None:
                prepend = False
        elif kind == 'h5crc':
            if not imagecodecs.H5CHECKSUM.available:
                raise ValueError('imagecodecs.H5CHECKSUM not available')
            self._checksum = imagecodecs.h5checksum_crc
            if prepend is None:
                prepend = False
        else:
            raise ValueError(f'checksum {kind=!r} not supported')

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
            raise RuntimeError(
                f'{self._checksum.__name__} checksum mismatch '
                f'{checksum} != {expect}'
            )
        return out


class Cms(Codec):
    """CMS codec for numcodecs."""

    codec_id = 'imagecodecs_cms'

    def __init__(self) -> None:
        if not imagecodecs.CMS.available:
            raise ValueError('imagecodecs.CMS not available')

    def encode(self, buf, out=None):
        # return imagecodecs.cms_transform(buf)
        raise NotImplementedError

    def decode(self, buf, out=None):
        # return imagecodecs.cms_transform(buf)
        raise NotImplementedError


class Dds(Codec):
    """DDS codec for numcodecs."""

    codec_id = 'imagecodecs_dds'

    def __init__(self, *, mipmap: int = 0) -> None:
        if not imagecodecs.DDS.available:
            raise ValueError('imagecodecs.DDS not available')
        self.mipmap = mipmap

    def encode(self, buf, out=None):
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
            raise ValueError('imagecodecs.DEFLATE not available')

        self.level = level
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
        dtype: DTypeLike = None,
        axis: int = -1,
        dist: int = 1,
    ) -> None:
        if not imagecodecs.DELTA.available:
            raise ValueError('imagecodecs.DELTA not available')

        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.asarray(buf)
            if buf.shape != self.shape:
                raise ValueError(f'{buf.shape=} does not match {self.shape=}')
            if buf.dtype != self.dtype:
                raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
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
            raise ValueError('imagecodecs.DICOMRLE not available')

        self.dtype = numpy.dtype(dtype).str  # TODO: preserve endianness

    def encode(self, buf, out=None):
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
        rlebits: int,
        horzbits: int,
        vertbits: int,
        superres: bool = False,
    ) -> None:
        if not imagecodecs.EER.available:
            raise ValueError('imagecodecs.EER not available')

        self.shape = shape
        self.rlebits = rlebits
        self.horzbits = horzbits
        self.vertbits = vertbits
        self.superres = bool(superres)

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.eer_decode(
            buf,
            self.shape,
            rlebits=self.rlebits,
            horzbits=self.horzbits,
            vertbits=self.vertbits,
            superres=self.superres,
            out=out,
        )


class Float24(Codec):
    """Float24 codec for numcodecs."""

    codec_id = 'imagecodecs_float24'

    def __init__(
        self,
        byteorder: Literal['>', '<', '='] | None = None,
        rounding: int | None = None,
    ) -> None:
        if not imagecodecs.FLOAT24.available:
            raise ValueError('imagecodecs.FLOAT24 not available')

        self.byteorder = byteorder
        self.rounding = rounding

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
            raise ValueError('imagecodecs.FLOATPRED not available')

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = int(axis)
        self.dist = int(dist)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if buf.shape != self.shape:
            raise ValueError(f'{buf.shape=} does not match {self.shape=}')
        if buf.dtype != self.dtype:
            raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
        return imagecodecs.floatpred_encode(
            buf, axis=self.axis, dist=self.dist
        ).tobytes()

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
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
            raise ValueError('imagecodecs.GIF not available')

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
        photometric: int | str | None = None,
        compression: int | str | None = None,
        index: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.HEIF.available:
            raise ValueError('imagecodecs.HEIF not available')

        self.level = level
        self.bitspersample = bitspersample
        self.photometric = photometric
        self.compression = compression
        self.index = index
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


class Jetraw(Codec):
    """Jetraw codec for numcodecs."""

    codec_id = 'imagecodecs_jetraw'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        identifier: str,
        parameters: str | None = None,
        verbose: int | None = None,
        errorbound: float | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JETRAW.available:
            raise ValueError('imagecodecs.JETRAW not available')

        self.shape = shape
        self.identifier = identifier
        self.errorbound = errorbound
        self.squeeze = squeeze
        if not verbose:
            verbose = 0
        imagecodecs.jetraw_init(parameters, verbose=verbose)

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jetraw_encode(
            buf, self.identifier, errorbound=self.errorbound
        )

    def decode(self, buf, out=None):
        if out is None:
            out = numpy.empty(self.shape, numpy.uint16)
        return imagecodecs.jetraw_decode(buf, out=out)


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
        colorspace_data: int | str | None = None,
        colorspace_jpeg: int | str | None = None,
        subsampling: str | tuple[int, int] | None = None,
        optimize: bool | None = None,
        smoothing: bool | None = None,
        lossless: bool | None = None,
        predictor: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEG.available:
            raise ValueError('imagecodecs.JPEG not available')

        self.level = level
        self.tables = tables
        self.header = header
        self.bitspersample = bitspersample
        self.colorspace_data = colorspace_data
        self.colorspace_jpeg = colorspace_jpeg
        self.subsampling = subsampling
        self.optimize = optimize
        self.smoothing = smoothing
        self.lossless = lossless
        self.predictor = predictor
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.jpeg_encode(
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
        return imagecodecs.jpeg_decode(
            buf,
            bitspersample=self.bitspersample,
            tables=self.tables,
            header=self.header,
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
                    import base64

                    value = base64.b64encode(value).decode()
                config[key] = value
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiate codec from configuration object."""
        for key in ('header', 'tables'):
            value = config.get(key, None)
            if value is not None and isinstance(value, str):
                import base64

                config[key] = base64.b64decode(value.encode())
        return cls(**config)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = 'imagecodecs_jpeg2k'

    def __init__(
        self,
        *,
        level: int | None = None,
        codecformat: int | str | None = None,
        colorspace: int | str | None = None,
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
            raise ValueError('imagecodecs.JPEG2K not available')

        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.planar = planar
        self.tile = None if tile is None else tile
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.mct = mct
        self.verbose = verbose
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
            raise ValueError('imagecodecs.JPEGLS not available')

        self.level = level
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
        photometric: int | str | None = None,
        bitspersample: int | None = None,
        # extrasamples: Sequence[int] | None = None,
        planar: bool | None = None,
        usecontainer: bool | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
        # decode
        index: int | None = None,
        keeporientation: bool | None = None,
        # both
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.JPEGXL.available:
            raise ValueError('imagecodecs.JPEGXL not available')

        self.level = level
        self.effort = effort
        self.distance = distance
        self.lossless = lossless is None or bool(lossless)
        self.decodingspeed = decodingspeed
        self.bitspersample = bitspersample
        self.photometric = photometric
        self.planar = planar
        self.usecontainer = usecontainer
        self.index = index
        self.keeporientation = keeporientation
        self.numthreads = numthreads
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
        photometric: int | str | None = None,
        hasalpha: bool | None = None,
        resolution: tuple[float, float] | None = None,
        fp2int: bool = False,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.JPEGXR.available:
            raise ValueError('imagecodecs.JPEGXR not available')

        self.level = level
        self.photometric = photometric
        self.hasalpha = hasalpha
        self.resolution = resolution
        self.fp2int = fp2int
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
            raise ValueError('imagecodecs.JPEGXS not available')

        self.config = config
        self.bitspersample = bitspersample
        self.verbose = verbose
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
            raise ValueError('imagecodecs.LERC not available')

        self.level = level
        self.version = version
        self.planar = bool(planar)
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
            raise ValueError('imagecodecs.LJPEG not available')

        self.bitspersample = bitspersample
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
            raise ValueError('imagecodecs.LZ4 not available')

        self.level = level
        self.hc = hc
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
            raise ValueError('imagecodecs.LZ4F not available')

        self.level = level
        self.blocksizeid = blocksizeid
        self.contentchecksum = contentchecksum
        self.blockchecksum = blockchecksum

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
            raise ValueError('imagecodecs.LZ4H5 not available')

        self.level = level
        self.blocksize = blocksize

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
            raise ValueError('imagecodecs.LZF not available')

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
            raise ValueError('imagecodecs.LZFSE not available')

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
        level: int | None = None,
    ) -> None:
        if not imagecodecs.LZHAM.available:
            raise ValueError('imagecodecs.LZHAM not available')

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
        check: int | None = None,
    ) -> None:
        if not imagecodecs.LZMA.available:
            raise ValueError('imagecodecs.LZMA not available')

        self.level = level
        self.check = check

    def encode(self, buf):
        return imagecodecs.lzma_encode(buf, level=self.level, check=self.check)

    def decode(self, buf, out=None):
        return imagecodecs.lzma_decode(buf, out=_flat(out))


class Lzo(Codec):
    """LZO codec for numcodecs."""

    codec_id = 'imagecodecs_lzo'

    def __init__(self, *, header: bool = False) -> None:
        if not imagecodecs.LZO.available:
            raise ValueError('imagecodecs.LZO not available')
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
            raise ValueError('imagecodecs.LZW not available')

    def encode(self, buf):
        return imagecodecs.lzw_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.lzw_decode(buf, out=_flat(out))


class Packbits(Codec):
    """PackBits codec for numcodecs."""

    codec_id = 'imagecodecs_packbits'

    def __init__(
        self,
        *,
        axis: int | None = None,
    ) -> None:
        if not imagecodecs.PACKBITS.available:
            raise ValueError('imagecodecs.PACKBITS not available')

        self.axis = axis

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
        self, *, dtype: DTypeLike, bitspersample: int, runlen: int = 0
    ) -> None:
        if not imagecodecs.PACKINTS.available:
            raise ValueError('imagecodecs.PACKINTS not available')

        self.dtype = numpy.dtype(dtype).str
        self.bitspersample = bitspersample
        self.runlen = runlen

    def encode(self, buf):
        raise NotImplementedError

    def decode(self, buf, out=None):
        return imagecodecs.packints_decode(
            buf,
            self.dtype,
            self.bitspersample,
            runlen=self.runlen,
            out=_flat(out),  # type: ignore[arg-type]
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
    ) -> None:
        if not imagecodecs.PCODEC.available:
            raise ValueError('imagecodecs.PCODEC not available')

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.level = level

    def encode(self, buf):
        return imagecodecs.pcodec_encode(buf, level=self.level)

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
            raise ValueError('imagecodecs.PGLZ not available')

        self.header = bool(header)
        self.strategy = strategy
        self.checkcomplete = checkcomplete

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


class Png(Codec):
    """PNG codec for numcodecs."""

    codec_id = 'imagecodecs_png'

    def __init__(
        self,
        *,
        level: int | None = None,
        strategy: int | None = None,
        filter: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.PNG.available:
            raise ValueError('imagecodecs.PNG not available')

        self.level = level
        self.strategy = strategy
        self.filter = filter
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
            raise ValueError('imagecodecs.QOI not available')

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
            raise ValueError('imagecodecs.QUANTIZE not available')

        self.nsd = nsd
        self.mode = mode

    def encode(self, buf):
        return imagecodecs.quantize_encode(buf, self.mode, self.nsd)

    def decode(self, buf, out=None):
        return buf
        # return imagecodecs.quantize_decode(buf, self.mode, self.nsd, out=out)


class Rcomp(Codec):
    """Rcomp codec for numcodecs."""

    codec_id = 'imagecodecs_rcomp'

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        nblock: int | None = None,
    ) -> None:
        if not imagecodecs.RCOMP.available:
            raise ValueError('imagecodecs.RCOMP not available')

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.nblock = nblock

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
            raise ValueError('imagecodecs.RGBE not available')

        if not header and shape is None:
            raise ValueError('must specify data shape if no header')
        if shape and shape[-1] != 3:
            raise ValueError('invalid shape')
        assert shape is not None
        self.shape = tuple(shape)
        self.header = bool(header)
        self.rle = None if rle is None else bool(rle)
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
            raise ValueError('imagecodecs.SNAPPY not available')

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
        dtype: DTypeLike = None,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, int, int] | None = None,
        header: bool = True,
        numthreads: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.SPERR.available:
            raise ValueError('imagecodecs.SPERR not available')

        if header:
            self.shape = None
            self.dtype = None
        elif shape is None or dtype is None:
            raise ValueError('invalid shape or dtype')
        else:
            self.shape = tuple(shape)
            self.dtype = numpy.dtype(dtype).str
        self.mode = mode
        self.level = float(level)
        self.chunks = None if chunks is None else tuple(chunks)
        self.header = bool(header)
        self.numthreads = numthreads
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        if not self.header:
            if buf.shape != self.shape:
                raise ValueError(f'{buf.shape=} does not match {self.shape=}')
            if buf.dtype != self.dtype:
                raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
        return imagecodecs.sperr_encode(
            buf,
            level=self.level,
            mode=self.mode,
            chunks=self.chunks,  # type: ignore[arg-type]
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
            raise ValueError('imagecodecs.SPNG not available')

        self.level = level
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
            raise ValueError('imagecodecs.SZ3 not available')

        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.mode = 'abs' if mode is None else mode
        self.abs = 0.0 if abs is None else abs
        self.rel = 0.0 if rel is None else rel

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
            raise ValueError('imagecodecs.SZIP not available')

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
        index: int | None = None,
        asrgb: bool = False,
        verbose: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.TIFF.available:
            raise ValueError('imagecodecs.TIFF not available')

        self.index = index
        self.asrgb = bool(asrgb)
        self.verbose = verbose
        self.squeeze = squeeze

    def encode(self, buf):
        # TODO: not implemented
        buf = _image(buf, self.squeeze)
        return imagecodecs.tiff_encode(buf)

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
        gamut: int | None = None,
        crange: int | None = None,
        transfer: int | None = None,
        usage: int | None = None,
        codec: int | None = None,
        # decode
        dtype: DTypeLike | None = None,
        boost: float | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.ULTRAHDR.available:
            raise ValueError('imagecodecs.ULTRAHDR not available')

        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.level = level
        self.scale = scale
        self.gamut = gamut
        self.crange = crange
        self.transfer = transfer
        self.usage = usage
        self.codec = codec
        self.boost = boost
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        if self.dtype is not None and buf.dtype != self.dtype:
            raise ValueError(f'{buf.dtype=} != {self.dtype}')

        return imagecodecs.ultrahdr_encode(
            buf,
            scale=self.scale,
            level=self.level,
            gamut=self.gamut,
            crange=self.crange,
            transfer=self.transfer,
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
        level: int | None = None,
        lossless: bool | None = None,
        method: int | None = None,
        index: int | None = 0,
        hasalpha: bool | None = None,
        numthreads: int | None = None,
        squeeze: Literal[False] | Sequence[int] | None = None,
    ) -> None:
        if not imagecodecs.WEBP.available:
            raise ValueError('imagecodecs.WEBP not available')

        self.level = level
        self.hasalpha = bool(hasalpha)
        self.method = method
        self.index = index
        self.lossless = lossless
        self.numthreads = numthreads
        self.squeeze = squeeze

    def encode(self, buf):
        buf = _image(buf, self.squeeze)
        return imagecodecs.webp_encode(
            buf,
            level=self.level,
            lossless=self.lossless,
            method=self.method,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.webp_decode(
            buf, index=self.index, hasalpha=self.hasalpha, out=out
        )


class Xor(Codec):
    """XOR codec for numcodecs."""

    codec_id = 'imagecodecs_xor'

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike = None,
        axis: int = -1,
    ) -> None:
        if not imagecodecs.XOR.available:
            raise ValueError('imagecodecs.XOR not available')

        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = int(axis)

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.asarray(buf)
            if buf.shape != self.shape:
                raise ValueError(f'{buf.shape=} does not match {self.shape=}')
            if buf.dtype != self.dtype:
                raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
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
        dtype: DTypeLike = None,
        strides: tuple[int, ...] | None = None,
        level: int | None = None,
        mode: int | str | None = None,
        execution: int | str | None = None,
        chunksize: int | None = None,
        header: bool = True,
        numthreads: int | None = None,
    ) -> None:
        if not imagecodecs.ZFP.available:
            raise ValueError('imagecodecs.ZFP not available')

        if header:
            self.shape = None
            self.dtype = None
            self.strides = None
        elif shape is None or dtype is None:
            raise ValueError('invalid shape or dtype')
        else:
            self.shape = tuple(shape)
            self.dtype = numpy.dtype(dtype).str
            self.strides = None if strides is None else tuple(strides)
        self.level = level
        self.mode = mode
        self.execution = execution
        self.numthreads = numthreads
        self.chunksize = chunksize
        self.header = bool(header)

    def encode(self, buf):
        buf = numpy.asarray(buf)
        if not self.header:
            if buf.shape != self.shape:
                raise ValueError(f'{buf.shape=} does not match {self.shape=}')
            if buf.dtype != self.dtype:
                raise ValueError(f'{buf.dtype=} does not match {self.dtype=}')
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
            raise ValueError('imagecodecs.ZLIB not available')

        self.level = level

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
            raise ValueError('imagecodecs.ZLIBNG not available')

        self.level = level

    def encode(self, buf):
        return imagecodecs.zlibng_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zlibng_decode(buf, out=_flat(out))


class Zopfli(Codec):
    """Zopfli codec for numcodecs."""

    codec_id = 'imagecodecs_zopfli'

    def __init__(self) -> None:
        if not imagecodecs.ZOPFLI.available:
            raise ValueError('imagecodecs.ZOPFLI not available')

    def encode(self, buf):
        return imagecodecs.zopfli_encode(buf)

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
            raise ValueError('imagecodecs.ZSTD not available')

        self.level = level

    def encode(self, buf):
        return imagecodecs.zstd_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zstd_decode(buf, out=_flat(out))


def _flat(buf: Any, /) -> memoryview | None:
    """Return numpy array as contiguous view of bytes if possible."""
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
        return numpy.atleast_2d(  # type: ignore[no-any-return]
            numpy.squeeze(buf)
        )
    arr = numpy.asarray(buf)
    if not squeeze:
        return arr
    shape = tuple(i for i, j in zip(buf.shape, squeeze) if not j)
    return arr.reshape(shape)


def register_codecs(
    codecs: Any = None,
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
            try:
                get_codec({'id': cls.codec_id})
            except TypeError:
                # registered, but failed
                pass
        except ValueError:
            # not registered yet
            pass
        else:
            if not force:
                if verbose:
                    log_warning(
                        f'numcodec {cls.codec_id!r} already registered'
                    )
                continue
            if verbose:
                log_warning(f'replacing registered numcodec {cls.codec_id!r}')
        register_codec(cls)


def log_warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="misc"
