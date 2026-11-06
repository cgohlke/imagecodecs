# imagecodecs/tests/test_zarr.py

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

"""Unittests for the imagecodecs.zarr module."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import numpy
import pytest
from conftest import datafiles
from numpy import array_equal  # faster than assert_array_equal
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

try:
    import imagecodecs
except ImportError:
    pytest.skip('imagecodecs not found', allow_module_level=True)

try:
    import zarr
except ImportError:
    pytest.skip('zarr not found', allow_module_level=True)

try:
    import imagecodecs.zarr as zarr3codecs
except ImportError:
    pytest.skip('imagecodecs.zarr not found', allow_module_level=True)

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
)
from zarr.buffer import default_buffer_prototype
from zarr.buffer.cpu import Buffer
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.dtype import get_data_type_from_native_dtype

zarr3codecs.register_codecs()


def _array_spec(shape: tuple[int, ...], dtype: DTypeLike) -> ArraySpec:
    """Return an ArraySpec for decode-only zarr codec tests."""
    return ArraySpec(
        shape=shape,
        dtype=get_data_type_from_native_dtype(numpy.dtype(dtype)),
        fill_value=0,
        config=ArrayConfig(order='C', write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )


@pytest.mark.parallel_threads(1)
@pytest.mark.iterations(1)
def test_zarr_register(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test register_codecs function."""
    from zarr.registry import get_codec_class

    zarr3codecs.register_codecs()
    assert 'zarr codec' not in caplog.text

    assert get_codec_class('imagecodecs_lzw') is zarr3codecs.Lzw
    assert (
        zarr3codecs.JpegCodec
        in zarr.registry._codec_registries['jpeg'].values()
    )

    def register_codec(_key: Any, _cls: Any) -> None:
        raise RuntimeError

    monkeypatch.setattr(zarr.registry, 'register_codec', register_codec)
    zarr3codecs.register_codecs(['lzw'], verbose=False)
    assert 'zarr codec' not in caplog.text
    zarr3codecs.register_codecs(['lzw'])
    assert 'zarr codec imagecodecs_lzw registration failed' in caplog.text


@pytest.mark.parametrize(
    ('names', 'expected'),
    [
        (['jpeg2k'], [('imagecodecs_jpeg2k', zarr3codecs.Jpeg2k)]),
        (['imagecodecs_jpeg2k'], [('imagecodecs_jpeg2k', zarr3codecs.Jpeg2k)]),
        (
            ['jpeg2k', 'imagecodecs_jpeg2k'],
            [('imagecodecs_jpeg2k', zarr3codecs.Jpeg2k)],
        ),
        ([], []),
    ],
)
def test_zarr_register_names(
    names: list[str],
    expected: list[tuple[str, type]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test registration by short codec name or full codec ID."""
    registered = []
    monkeypatch.setattr(
        zarr.registry,
        'register_codec',
        lambda key, cls: registered.append((key, cls)),
    )
    zarr3codecs.register_codecs(names, verbose=False)
    assert registered == expected


@pytest.mark.parametrize(
    'names',
    [['unknown'], ['imagecodecs_unknown'], ['jpeg2k', 'unknown']],
)
def test_zarr_register_unknown(
    names: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test unknown names fail without partial registration."""
    registered = []
    monkeypatch.setattr(
        zarr.registry,
        'register_codec',
        lambda key, cls: registered.append((key, cls)),
    )
    with pytest.raises(ValueError, match='unknown codec names'):
        zarr3codecs.register_codecs(names, verbose=False)
    assert registered == []


@pytest.mark.skipif(not imagecodecs.LZO.available, reason='LZO missing')
def test_lzo_zarr() -> None:
    """Test LZO decoding with zarr."""
    spec = _array_spec((64,), 'uint8')
    result = zarr3codecs.Lzo(header=True)._decode_sync(
        Buffer.from_bytes(
            b'\xf1\x00\x00\x00@\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00'
        ),
        spec,
    )
    assert bytes(result.to_bytes()) == b'a\x01\x0ca' * 16


@pytest.mark.parametrize('dtype', ['uint8', 'float32'])
@pytest.mark.parametrize(
    'kind', ['crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc']
)
def test_checksum_zarr(kind: str, dtype: DTypeLike) -> None:
    """Test zarr3codecs.Checksum roundtrips."""
    if kind in {'crc32', 'adler32'}:
        if not (
            imagecodecs.ZLIBNG.available
            or imagecodecs.DEFLATE.available
            or imagecodecs.ZLIB.available
        ):
            pytest.skip('ZLIB missing')
    elif not imagecodecs.H5CHECKSUM.available:
        pytest.skip('H5CHECKSUM missing')

    data = numpy.arange(255, dtype=dtype).reshape((15, 17))[1:14, 2:15]
    codec_obj = zarr3codecs.Checksum(kind=kind)  # type: ignore[arg-type]
    assert isinstance(codec_obj, BytesBytesCodec)

    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=data.shape,
            chunks=data.shape,
            dtype=data.dtype,
            compressors=[codec_obj],
        )
        z[:] = data
        del z

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    assert array_equal(decoded, data)


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available, reason='DICOMRLE missing'
)
def test_dicomrle_zarr() -> None:
    """Test DICOMRLE decoding with zarr."""
    encoded = (
        b'\x0c\x00\x00\x00@\x00\x00\x00G\x00\x00\x00N\x00\x00\x00U\x00'
        b'\x00\x00\\\x00\x00\x00c\x00\x00\x00j\x00\x00\x00q\x00\x00\x00x'
        b'\x00\x00\x00\x7f\x00\x00\x00\x86\x00\x00\x00\x8d\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x01'
        b'\x00\x00\x00\xff\x05\x00\x00\x01\x00\x00\xff\x05\x00\x00\x00'
        b'\x01\x00\xff\x05\x00\x00\x00\x00\x01\xff\x05\xff\x01\x00\x00'
        b'\x00\x00\x05\xff\x00\x01\x00\x00\x00\x05\xff\x00\x00\x01\x00'
        b'\x00\x05\xff\x00\x00\x00\x01\x00\x05\x00\x01\x00\x00\x00\xff'
        b'\x05\x00\x00\x01\x00\x00\xff\x05\x00\x00\x00\x01\x00\xff\x05'
        b'\x01\x00\x00\x00\x01\xfe'
    )
    expected = numpy.array(
        [
            [0, 16777216, 65536, 256, 1, 4294967295],
            [4294967295, 16777216, 65536, 256, 1, 0],
            [1, 16777216, 65536, 256, 1, 4294967294],
        ],
        dtype='u4',
    )
    spec = _array_spec(expected.shape, '<u4')
    decoded = zarr3codecs.Dicomrle(dtype='<u4')._decode_sync(
        Buffer.from_bytes(encoded), spec
    )
    assert array_equal(decoded.as_numpy_array(), expected)


@pytest.mark.skipif(not imagecodecs.DDS.available, reason='DDS missing')
def test_dds_zarr() -> None:
    """Test DDS decoding with zarr."""
    try:
        with open(datafiles('bcn/bc1.dds'), 'rb') as fh:
            encoded = fh.read()
    except FileNotFoundError:
        pytest.skip('bcn/bc1.dds not found')
    expected = imagecodecs.dds_decode(encoded)
    decoded = zarr3codecs.Dds()._decode_sync(
        Buffer.from_bytes(encoded), _array_spec(expected.shape, expected.dtype)
    )
    assert array_equal(decoded.as_numpy_array(), expected)


@pytest.mark.skipif(not imagecodecs.OPENZL.available, reason='OPENZL missing')
def test_openzl_zarr() -> None:
    """Test OPENZL decoding with zarr."""
    with open(datafiles('rgb_u2.zl'), 'rb') as fh:
        encoded = fh.read()
    spec = _array_spec((32, 31, 3), 'uint16')
    decoded = zarr3codecs.Openzl()._decode_sync(
        Buffer.from_bytes(encoded), spec
    )
    assert bytes(decoded.to_bytes()) == bytes(
        imagecodecs.openzl_decode(encoded)
    )


@pytest.mark.skipif(not imagecodecs.EER.available, reason='EER missing')
def test_eer_zarr() -> None:
    """Test EER decoding with zarr."""
    encoded = b'\x03\x1b\xfc\xb1\x35\xfb'  # from EER specification
    expected = imagecodecs.eer_decode(encoded, (1, 312), 7, 1, 1)
    decoded = zarr3codecs.Eer(
        shape=(1, 312), skipbits=7, horzbits=1, vertbits=1
    )._decode_sync(
        Buffer.from_bytes(encoded), _array_spec(expected.shape, expected.dtype)
    )
    assert array_equal(decoded.as_numpy_array(), expected)
    assert decoded.as_numpy_array()[0, 3]
    assert decoded.as_numpy_array()[0, 17]


@pytest.mark.skipif(not imagecodecs.ZSTD1.available, reason='ZSTD1 missing')
def test_zstd1_zarr() -> None:
    """Test Zstd1 zarr codec."""
    data = numpy.arange(256, dtype=numpy.uint16)
    raw = data.tobytes()
    spec = _array_spec(data.shape, data.dtype)

    # auto hilo=False (no shuffle): Zstd1() on uint8
    spec8 = _array_spec(data.shape, numpy.dtype('uint8'))
    raw8 = data.astype(numpy.uint8).tobytes()
    codec = zarr3codecs.Zstd1()
    encoded_nohilo = codec._encode_sync(Buffer.from_bytes(raw8), spec8)
    assert encoded_nohilo is not None
    decoded = codec._decode_sync(encoded_nohilo, spec8)
    assert decoded.as_numpy_array().tobytes() == raw8

    # auto hilo=True (shuffle): Zstd1() on uint16 -> hilo enabled
    codec = zarr3codecs.Zstd1()
    encoded_hilo = codec._encode_sync(Buffer.from_bytes(raw), spec)
    assert encoded_hilo is not None
    decoded = codec._decode_sync(encoded_hilo, spec)
    assert decoded.as_numpy_array().tobytes() == raw
    codec_nohilo = zarr3codecs.Zstd1(hilo=False)
    encoded_no = codec_nohilo._encode_sync(Buffer.from_bytes(raw), spec)
    assert encoded_no is not None
    assert (
        encoded_hilo.as_numpy_array().tobytes()
        != encoded_no.as_numpy_array().tobytes()
    )

    # explicit hilo=False
    codec = zarr3codecs.Zstd1(hilo=False)
    encoded = codec._encode_sync(Buffer.from_bytes(raw), spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    assert decoded.as_numpy_array().tobytes() == raw

    # explicit level
    codec = zarr3codecs.Zstd1(level=3)
    encoded = codec._encode_sync(Buffer.from_bytes(raw), spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    assert decoded.as_numpy_array().tobytes() == raw


@pytest.mark.skipif(
    not imagecodecs.CCITTRLE.available or not imagecodecs.TIFF.available,
    reason='CCITT or TIFF missing',
)
@pytest.mark.parametrize('compression', ['ccittrle', 'ccittfax3', 'ccittfax4'])
def test_ccitt_zarr(compression: str) -> None:
    """Test CCITT decoding with zarr."""
    tifffile = pytest.importorskip('tifffile')
    codec: ArrayBytesCodec
    shape = (8, 16)
    data = numpy.zeros(shape, dtype=numpy.bool_)
    data[0, 1] = data[4, 8] = True
    tiff_bytes = imagecodecs.tiff_encode(
        data,
        compression=compression,
        photometric='minisblack',
        bitspersample=1,
    )
    with tifffile.TiffFile(io.BytesIO(tiff_bytes)) as tif:
        page = tif.pages.first
        fh = tif.filehandle
        fh.seek(page.dataoffsets[0])
        t4options = (
            int(page.tags['T4Options'].value)
            if 'T4Options' in page.tags
            else 0
        )
        encoded = bytes(fh.read(page.databytecounts[0]))
    spec = _array_spec(shape, 'uint8')
    if compression == 'ccittrle':
        codec = zarr3codecs.Ccittrle()
        expected = imagecodecs.ccittrle_decode(
            encoded, height=shape[0], width=shape[1]
        )
    elif compression == 'ccittfax3':
        codec = zarr3codecs.Ccittfax3(t4options=t4options)
        expected = imagecodecs.ccittfax3_decode(
            encoded, height=shape[0], width=shape[1], t4options=t4options
        )
    else:
        codec = zarr3codecs.Ccittfax4()
        expected = imagecodecs.ccittfax4_decode(
            encoded, height=shape[0], width=shape[1]
        )
    decoded = codec._decode_sync(Buffer.from_bytes(encoded), spec)
    assert array_equal(decoded.as_numpy_array(), expected)
    assert array_equal(decoded.as_numpy_array(), data.view('uint8'))


@pytest.mark.parametrize(
    'pipeline',
    [
        pytest.param(
            'delta_lzw',
            marks=pytest.mark.skipif(
                not (
                    imagecodecs.DELTA.available and imagecodecs.LZW.available
                ),
                reason='DELTA or LZW missing',
            ),
        ),
        pytest.param(
            'bitshuffle_zstd',
            marks=pytest.mark.skipif(
                not (
                    imagecodecs.BITSHUFFLE.available
                    and imagecodecs.ZSTD.available
                ),
                reason='BITSHUFFLE or ZSTD missing',
            ),
        ),
        pytest.param(
            'delta_bitshuffle_zstd',
            marks=pytest.mark.skipif(
                not (
                    imagecodecs.DELTA.available
                    and imagecodecs.BITSHUFFLE.available
                    and imagecodecs.ZSTD.available
                ),
                reason='DELTA, BITSHUFFLE, or ZSTD missing',
            ),
        ),
        pytest.param(
            'floatpred_deflate',
            marks=pytest.mark.skipif(
                not (
                    imagecodecs.FLOATPRED.available
                    and imagecodecs.DEFLATE.available
                ),
                reason='FLOATPRED or DEFLATE missing',
            ),
        ),
        pytest.param(
            'xor_lzma',
            marks=pytest.mark.skipif(
                not (imagecodecs.XOR.available and imagecodecs.LZMA.available),
                reason='XOR or LZMA missing',
            ),
        ),
    ],
)
def test_zarr_pipeline(pipeline: str) -> None:
    """Test imagecodecs.zarr codec pipelines through zarr 3 roundtrips."""
    data = numpy.load(datafiles('rgb.u1.npy'))
    data = numpy.stack([data, data])
    shape = data.shape
    chunks = (1, 128, 128, 3)
    serializer: ArrayBytesCodec | None = None
    compressors: list[BytesBytesCodec]
    filters: list[ArrayArrayCodec]

    match pipeline:
        case 'delta_lzw':
            filters = [zarr3codecs.Delta(axis=-1)]
            compressors = [zarr3codecs.Lzw()]
        case 'bitshuffle_zstd':
            filters = []
            serializer = zarr3codecs.Bitshuffle(blocksize=0)
            compressors = [zarr3codecs.Zstd(level=5)]
        case 'delta_bitshuffle_zstd':
            filters = [zarr3codecs.Delta(axis=-1)]
            serializer = zarr3codecs.Bitshuffle(blocksize=0)
            compressors = [zarr3codecs.Zstd(level=5)]
        case 'floatpred_deflate':
            data = data.astype('float32')
            shape = data.shape
            filters = []
            serializer = zarr3codecs.Floatpred(axis=-1)
            compressors = [zarr3codecs.Deflate(level=6)]
        case 'xor_lzma':
            filters = [zarr3codecs.Xor(axis=-1)]
            compressors = [zarr3codecs.Lzma(level=3)]
        case _:
            raise RuntimeError

    kwargs: dict[str, Any] = {}
    if serializer is not None:
        kwargs['serializer'] = serializer
    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=shape,
            chunks=chunks,
            dtype=data.dtype,
            filters=filters,
            compressors=compressors,
            **kwargs,
        )
        z[:] = data
        del z

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    assert array_equal(decoded, data)


@pytest.mark.parametrize(
    'codec', ['jpeg2k', 'jpegxl', 'htj2k', 'exr', 'lerc', 'tiff']
)
@pytest.mark.parametrize('planar', [True, False])
def test_planar(codec: str, *, planar: bool) -> None:
    """Test zarr planar option."""
    # https://github.com/cgohlke/imagecodecs/issues/145
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not found')

    data = numpy.load(datafiles('rgb.u1.npy'))
    if planar:
        data = numpy.moveaxis(data, -1, 0)
        chunks = (3, 128, 128)
    else:
        chunks = (128, 128, 3)
    kwargs: dict[str, Any] = {'planar': planar}

    if codec == 'exr':
        data = data.astype(numpy.float32)
    elif codec == 'tiff':
        kwargs = {'planarconfig': 'separate' if planar else 'contig'}

    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            serializer=getattr(zarr3codecs, codec.capitalize())(**kwargs),
        )
        z[:] = data
        del z

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    assert array_equal(decoded, data)


@pytest.mark.parametrize('photometric', ['gray', 'rgb', 'stack'])
@pytest.mark.parametrize(
    'codec',
    [
        'aec',
        'apng',
        'avif',
        'b2nd',
        'bfloat16',
        'bitorder',
        'bitshuffle',
        'blosc',
        'blosc2',
        'bmp',
        'brotli',
        'byteshuffle',
        'bz2',
        'chunked',
        'cms',
        'deflate',
        'delta',
        'exr',
        'float24',
        'floatpred',
        'gif',
        'hcomp',
        'heif',
        'htj2k',
        'isal',
        'jpeg',
        'jpeg2k',
        'jpegli',
        'jpegls',
        'jpegxl',
        'jpegxr',
        'jpegxs',
        'lerc',
        'ljpeg',
        'lz4',
        'lz4f',
        'lz4h5',
        'lzf',
        'lzfse',
        'lzham',
        'lzma',
        'lzw',
        'meshopt',
        # 'openzl',  # read-only
        'packbits',
        'packints',
        'pcodec',
        'pcx',
        'pglz',
        'pixarlog',
        'plio',
        'png',
        'qoi',
        'quantize',
        'rgbe',
        'rcomp',
        'snappy',
        'sperr',
        'spng',
        'sz3',
        'szip',
        'tga',
        'tiff',
        'ultrahdr',
        'wavpack',
        'webp',
        'wic',
        'xor',
        'zfp',
        'zlib',
        'zlibng',
        'zopfli',
        'zstd',
        'zstd1',
    ],
)
def test_zarr(codec: str, photometric: str) -> None:
    """Test imagecodecs.zarr codecs through zarr 3 roundtrips."""
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not found')

    data = numpy.load(datafiles('rgb.u1.npy'))
    data = numpy.stack([data, data])
    chunks: tuple[int, ...]
    if photometric == 'rgb':
        shape = data.shape
        chunks = (1, 128, 128, 3)
        axis = -2
    elif photometric == 'gray':
        data = data[..., 1].copy()
        shape = data.shape
        chunks = (1, 128, 128)
        axis = -1
    else:
        # https://github.com/cgohlke/imagecodecs/issues/98
        data = data[:, :128, :128].copy()
        photometric = 'rgb'
        shape = data.shape
        chunks = (1, *data.shape[1:])
        axis = -1

    codec_obj: ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
    compression: str | None
    lossless = True
    atol = 0.0
    match codec:
        case 'aec':
            codec_obj = zarr3codecs.Aec()
        case 'apng':
            codec_obj = zarr3codecs.Apng(photometric=photometric, delay=100)
        case 'avif':
            if photometric != 'rgb':
                pytest.skip('xfail - AVIF does not support grayscale')
            codec_obj = zarr3codecs.Avif(level=100, numthreads=2)
        case 'b2nd':
            data = data.astype('float32')
            codec_obj = zarr3codecs.B2nd(
                level=6,
                codec='zstd',
                filters=['shuffle', 'bytedelta'],
                filters_meta=[0, 4],
                chunkshape=chunks,
                numthreads=2,
            )
        case 'bfloat16':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Bfloat16()
        case 'bitorder':
            codec_obj = zarr3codecs.Bitorder()
        case 'bitshuffle':
            codec_obj = zarr3codecs.Bitshuffle(blocksize=0)
        case 'blosc':
            codec_obj = zarr3codecs.Blosc(
                level=9, compressor='blosclz', numthreads=2
            )
        case 'blosc2':
            codec_obj = zarr3codecs.Blosc2(
                level=9,
                codec='blosclz',
                filters=['shuffle', 'bytedelta'],
                numthreads=2,
            )
        case 'bmp':
            codec_obj = zarr3codecs.Bmp()
        case 'brotli':
            codec_obj = zarr3codecs.Brotli(level=5)
        case 'byteshuffle':
            data = data.astype('int16')
            codec_obj = zarr3codecs.Byteshuffle(axis=axis)
        case 'bz2':
            codec_obj = zarr3codecs.Bz2(level=9)
        case 'chunked':
            codec_obj = zarr3codecs.Chunked(codec='lz4')
        case 'cms':
            if photometric == 'gray':
                codec_obj = zarr3codecs.Cms(
                    profile=imagecodecs.cms_profile('gray', gamma=1.0),
                    outprofile=imagecodecs.cms_profile('gray', gamma=2.2),
                )
                lossless = False
                atol = 20
            else:
                data = data.astype('float32') / 255.0
                codec_obj = zarr3codecs.Cms(
                    profile='linearrgb', outprofile='srgb'
                )
                lossless = False
                atol = 1e-4
        case 'deflate':
            codec_obj = zarr3codecs.Deflate(level=8)
        case 'delta':
            codec_obj = zarr3codecs.Delta(axis=axis)
        case 'exr':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Exr(compression='PIZ')
        case 'float24':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Float24()
        case 'floatpred':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Floatpred(axis=axis)
        case 'gif':
            codec_obj = zarr3codecs.Gif()
        case 'hcomp':
            if photometric != 'gray':
                pytest.skip('hcomp only supports 2D data')
            codec_obj = zarr3codecs.Hcomp(safe32=True)
        case 'heif':
            codec_obj = zarr3codecs.Heif(photometric=photometric)
            lossless = False
            atol = 1
        case 'htj2k':
            codec_obj = zarr3codecs.Htj2k(reversible=True)
        case 'isal':
            codec_obj = zarr3codecs.Isal(level=2)
        case 'jpeg':
            lossless = False
            atol = 4
            codec_obj = zarr3codecs.Jpeg(level=99, subsampling='444')
        case 'jpeg2k':
            codec_obj = zarr3codecs.Jpeg2k(level=0)
        case 'jpegli':
            lossless = False
            atol = 6
            codec_obj = zarr3codecs.Jpegli(level=99, subsampling='444')
        case 'jpegls':
            codec_obj = zarr3codecs.Jpegls(level=0)
        case 'jpegxl':
            codec_obj = zarr3codecs.Jpegxl(level=101)
        case 'jpegxr':
            codec_obj = zarr3codecs.Jpegxr(
                level=1.0, photometric='RGB' if photometric == 'rgb' else None
            )
        case 'jpegxs':
            if photometric != 'rgb':
                pytest.skip(f'xfail - JPEGXS does not support {photometric=}')
            codec_obj = zarr3codecs.Jpegxs(config='p=MLS.12')
        case 'lerc':
            if imagecodecs.ZSTD.available:
                compression = 'zstd'
                compressionargs = {'level': 10}
            else:
                compression = None
                compressionargs = None
            codec_obj = zarr3codecs.Lerc(
                level=0.0,
                compression=compression,  # type: ignore[arg-type]
                compressionargs=compressionargs,
            )
        case 'ljpeg':
            if photometric == 'rgb':
                pytest.skip('xfail - LJPEG does not support rgb')
            data = data.astype('uint16') << 2
            codec_obj = zarr3codecs.Ljpeg(bitspersample=10)
        case 'lz4':
            codec_obj = zarr3codecs.Lz4(level=10, hc=True, header=True)
        case 'lz4f':
            codec_obj = zarr3codecs.Lz4f(
                level=12, contentchecksum=True, blockchecksum=True
            )
        case 'lz4h5':
            codec_obj = zarr3codecs.Lz4h5(level=10, blocksize=100)
        case 'lzf':
            codec_obj = zarr3codecs.Lzf(header=True)
        case 'lzfse':
            codec_obj = zarr3codecs.Lzfse()
        case 'lzham':
            codec_obj = zarr3codecs.Lzham(level=6)
        case 'lzma':
            codec_obj = zarr3codecs.Lzma(
                level=6, check=imagecodecs.LZMA.CHECK.SHA256
            )
        case 'lzw':
            codec_obj = zarr3codecs.Lzw()
        case 'meshopt':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Meshopt(
                level=3,
                items=1 if photometric == 'gray' else None,
            )
        # case 'packbits' if photometric == 'rgb':
        #     codec_obj = zarr3codecs.Packbits(axis=-2)
        case 'packbits':
            codec_obj = zarr3codecs.Packbits()
        case 'packints':
            data //= 2
            codec_obj = zarr3codecs.Packints(
                bitspersample=data.dtype.itemsize * 8 - 1
            )
        case 'pcodec':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Pcodec(level=8)
            lossless = False
            atol = 1e-2
        case 'pglz':
            codec_obj = zarr3codecs.Pglz()
        case 'pixarlog':
            data = data.astype('float32')
            data /= 255.0
            codec_obj = zarr3codecs.Pixarlog(level=6)
            lossless = False
            atol = 2e-3
        case 'plio':
            data = data.astype('int32')
            codec_obj = zarr3codecs.Plio()
        case 'pcx':
            codec_obj = zarr3codecs.Pcx()
        case 'png':
            codec_obj = zarr3codecs.Png()
        case 'qoi':
            if photometric != 'rgb':
                pytest.skip('xfail - QOI does not support grayscale')
            codec_obj = zarr3codecs.Qoi()
        case 'quantize':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Quantize(mode='bitgroom', nsd=6)
            lossless = False
            atol = 1.0
        case 'rgbe':
            if photometric != 'rgb':
                pytest.skip('xfail - RGBE does not support grayscale')
            data = data.astype('float32')
            codec_obj = zarr3codecs.Rgbe(header=False, rle=True)
        case 'rcomp':
            codec_obj = zarr3codecs.Rcomp()
        case 'snappy':
            codec_obj = zarr3codecs.Snappy()
        case 'sperr':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Sperr(
                level=100.0,
                mode='psnr',
                header=False,
            )
            lossless = False
            atol = 1e-2
        case 'spng':
            codec_obj = zarr3codecs.Spng()
        case 'sz3':
            data = data.astype('float32')
            atol = 1e-2
            codec_obj = zarr3codecs.Sz3(mode='abs', abs=atol)
            lossless = False
        case 'szip':
            codec_obj = zarr3codecs.Szip(
                header=True, **imagecodecs.szip_params(data)
            )
        case 'tga':
            codec_obj = zarr3codecs.Tga()
        case 'tiff':
            codec_obj = zarr3codecs.Tiff(level=6, predictor=True)
        case 'ultrahdr':
            if photometric != 'rgb':
                pytest.skip(
                    f'xfail - ULTRAHDR does not support {photometric=}'
                )
            data = data.astype('float32')
            data /= data.max()
            data = data.astype('float16')
            codec_obj = zarr3codecs.Ultrahdr()
            lossless = False
            atol = 1e-2
            pytest.skip('xfail - ultrahdr_encode not working')
        case 'wavpack':
            data = data.astype('int16')
            codec_obj = zarr3codecs.Wavpack(
                level=2, channels=photometric == 'rgb'
            )
        case 'webp':
            if photometric != 'rgb':
                pytest.skip('xfail - WebP does not support grayscale')
            codec_obj = zarr3codecs.Webp(level=-1)
        case 'wic':
            codec_obj = zarr3codecs.Wic(format='png')
        case 'xor':
            codec_obj = zarr3codecs.Xor(axis=axis)
        case 'zfp':
            data = data.astype('float32')
            codec_obj = zarr3codecs.Zfp(header=True)
        case 'zlib':
            codec_obj = zarr3codecs.Zlib(level=6)
        case 'zlibng':
            codec_obj = zarr3codecs.Zlibng(level=6)
        case 'zopfli':
            codec_obj = zarr3codecs.Zopfli(level=1, blocksplitting=False)
        case 'zstd':
            codec_obj = zarr3codecs.Zstd(level=10)
        case 'zstd1':
            codec_obj = zarr3codecs.Zstd1(level=10)
        case _:
            raise RuntimeError

    with zarr.storage.MemoryStore() as store:
        if isinstance(codec_obj, ArrayArrayCodec):
            z = zarr.create_array(
                store,
                shape=shape,
                chunks=chunks,
                dtype=data.dtype,
                filters=[codec_obj],
            )
        elif isinstance(codec_obj, ArrayBytesCodec):
            z = zarr.create_array(
                store,
                shape=shape,
                chunks=chunks,
                dtype=data.dtype,
                serializer=codec_obj,
            )
        else:  # BytesBytesCodec
            z = zarr.create_array(
                store,
                shape=shape,
                chunks=chunks,
                dtype=data.dtype,
                compressors=[codec_obj],
            )
        z[:] = data
        del z

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    if lossless:
        assert array_equal(decoded, data)
    else:
        assert_allclose(decoded, data, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ('shape', 'planar', 'expected'),
    [
        ((1, 1), None, (1, 1)),
        ((1, 5), None, (1, 5)),
        ((5, 1), None, (5, 1)),
        ((5, 1, 1), None, (5, 1)),
        ((1, 4, 5), None, (4, 5)),
        ((1, 4, 5, 3), None, (4, 5, 3)),
        ((4, 5), True, (1, 4, 5)),
        ((1, 3, 4, 5), True, (3, 4, 5)),
    ],
)
def test_zarr_squeeze_shape(
    shape: tuple[int, ...], *, planar: bool | None, expected: tuple[int, ...]
) -> None:
    """Test squeezing chunk shapes for image codecs."""
    assert zarr3codecs._squeeze_shape(shape, planar=planar) == expected


@pytest.mark.parametrize(
    ('codec', 'kwargs'),
    [
        ('hcomp', {'safe32': False}),
        ('jpeg', {'subsampling': (2, 2)}),
        ('jpegli', {'subsampling': (2, 2), 'progressive': 2}),
        ('jpegls', {'spiff': True}),
        ('lerc', {'compressionargs': {'level': 1}}),
        ('openzl', {'level': 3}),
        ('pglz', {'strategy': (32, 1024, 25, 128, 10, 100)}),
        ('qoi', {'colorspace': 'linear'}),
        ('snappy', {'level': 1}),
        ('tiff', {'planarconfig': 'separate'}),
        (
            'tiff',
            {
                'iccprofile': b'\x00icc',
                'colormap': [[0, 1], [2, 3], [4, 5]],
                'resolution': (72, 72),
                'resolutionunit': 'inch',
            },
        ),
    ],
)
def test_zarr_metadata_roundtrip(codec: str, kwargs: dict[str, Any]) -> None:
    """Test codec configuration survives JSON metadata roundtrip."""
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not found')
    codec_obj = zarr3codecs._CODEC_CLASSES[codec](**kwargs)
    restored = type(codec_obj).from_dict(
        json.loads(json.dumps(codec_obj.to_dict()))
    )
    assert restored == codec_obj
    assert hash(restored) == hash(codec_obj)


@pytest.mark.skipif(not imagecodecs.JPEG.available, reason='JPEG missing')
@pytest.mark.parametrize(
    ('kwargs', 'samples', 'atol'),
    [
        ({'encoded_color_space': 'grayscale'}, 0, 8),
        ({'encoded_color_space': 'grayscale'}, 1, 8),
        ({'encoded_color_space': 'rgb'}, 3, 8),
        ({'encoded_color_space': 'ycbcr', 'subsampling': '4:2:0'}, 3, 64),
        (
            {
                'encoded_color_space': 'ycbcr',
                'decoded_color_space': 'ycbcr',
                'subsampling': '4:4:4',
            },
            3,
            8,
        ),
    ],
)
def test_jpegcodec_zarr(
    kwargs: dict[str, Any], samples: int, atol: int
) -> None:
    """Test JpegCodec implementing the zarr-extensions jpeg spec."""
    data = numpy.load(datafiles('rgb.u1.npy'))[:64, :96]
    if samples == 0:
        data = data[..., 0].copy()
    elif samples == 1:
        data = data[..., :1].copy()
    codec_obj = zarr3codecs.JpegCodec(**kwargs)
    assert codec_obj.to_dict()['name'] == 'jpeg'
    assert zarr3codecs.JpegCodec.from_dict(codec_obj.to_dict()) == codec_obj

    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=data.shape,
            chunks=data.shape,
            dtype=data.dtype,
            serializer=codec_obj,
            compressors=None,
        )
        z[:] = data
        del z

        chunk = store._store_dict['c/' + '/'.join('0' * data.ndim)].to_bytes()
        assert (b'Adobe' in chunk) == (kwargs['encoded_color_space'] == 'rgb')

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    assert_allclose(decoded, data, atol=atol, rtol=0)


@pytest.mark.skipif(not imagecodecs.JPEG.available, reason='JPEG missing')
@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'encoded_color_space': 'ycbcr'}, 'subsampling required'),
        (
            {'encoded_color_space': 'rgb', 'subsampling': '4:2:0'},
            r'must be all \[1, 1\]',
        ),
        (
            {'encoded_color_space': 'grayscale', 'decoded_color_space': 'rgb'},
            'invalid combination',
        ),
        (
            {'encoded_color_space': 'grayscale', 'subsampling': [[1, 1]] * 3},
            'must have 1 components',
        ),
        ({'encoded_color_space': 'grayscale', 'quality': 0}, 'range 1-100'),
        (
            {
                'encoded_color_space': 'ycbcr',
                'subsampling': [[2.0, 2], [1, 1]],
            },
            'pairs of int',
        ),
    ],
)
def test_jpegcodec_invalid(kwargs: dict[str, Any], match: str) -> None:
    """Test JpegCodec rejects arguments invalid per spec."""
    with pytest.raises(ValueError, match=match):
        zarr3codecs.JpegCodec(**kwargs)


@pytest.mark.skipif(not imagecodecs.JPEG.available, reason='JPEG missing')
@pytest.mark.parametrize(
    ('cfg', 'match'),
    [
        ({'encoded_color_space': 'grayscale'}, "requires 'quality'"),
        (
            {'quality': 90, 'encoded_color_space': 'ycbcr'},
            'subsampling required',
        ),
        (
            {
                'quality': 90,
                'encoded_color_space': 'grayscale',
                'decoded_color_space': None,
            },
            'must not be null',
        ),
        (
            {'quality': 90, 'encoded_color_space': 'grayscale', 'level': 1},
            'unknown jpeg configuration keys',
        ),
    ],
)
def test_jpegcodec_invalid_config(cfg: dict[str, Any], match: str) -> None:
    """Test JpegCodec rejects configurations invalid per spec."""
    with pytest.raises(ValueError, match=match):
        zarr3codecs.JpegCodec.from_dict({'name': 'jpeg', 'configuration': cfg})


@pytest.mark.skipif(not imagecodecs.JPEG.available, reason='JPEG missing')
@pytest.mark.parametrize(
    'cfg',
    [
        {'quality': 90, 'encoded_color_space': 'grayscale'},
        {'quality': 90, 'encoded_color_space': 'rgb'},
        {
            'quality': 90,
            'encoded_color_space': 'ycbcr',
            'subsampling': [[2, 2], [1, 1], [1, 1]],
        },
        {
            'quality': 90,
            'encoded_color_space': 'ycbcr',
            'decoded_color_space': 'ycbcr',
            'subsampling': [[2, 2], [1, 1], [1, 1]],
        },
    ],
)
def test_jpegcodec_spec_examples(cfg: dict[str, Any]) -> None:
    """Test JpegCodec roundtrips configuration examples from spec."""
    data: dict[str, Any] = {'name': 'jpeg', 'configuration': cfg}
    assert zarr3codecs.JpegCodec.from_dict(data).to_dict() == data


@pytest.mark.skipif(not imagecodecs.JPEG.available, reason='JPEG missing')
def test_jpegcodec_unsupported_subsampling() -> None:
    """Test JpegCodec fails encoding subsampling valid per spec."""
    codec_obj = zarr3codecs.JpegCodec(
        encoded_color_space='ycbcr', subsampling=[[4, 2], [1, 1], [1, 1]]
    )
    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=(16, 16, 3),
            dtype='uint8',
            serializer=codec_obj,
            compressors=None,
        )
        with pytest.raises(ValueError, match='not supported'):
            z[:] = 1  # chunks equal to fill value are not encoded


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='CMS missing')
@pytest.mark.parametrize(
    'kwargs', [{'outplanar': True}, {'outcolorspace': 'rgba'}]
)
def test_cms_zarr_layout(kwargs: dict[str, Any]) -> None:
    """Test Cms filter changing chunk layout or number of samples."""
    data = numpy.load(datafiles('rgb.u1.npy'))[:32, :32]
    data = numpy.stack([data, data])
    codec_obj = zarr3codecs.Cms(
        profile='srgb', outprofile='srgb', colorspace='rgb', **kwargs
    )
    with zarr.storage.MemoryStore() as store:
        z = zarr.create_array(
            store,
            shape=data.shape,
            chunks=(1, *data.shape[1:]),
            dtype=data.dtype,
            filters=[codec_obj],
            compressors=None,
        )
        z[:] = data
        del z

        z = zarr.open_array(store, mode='r')
        decoded = z[:]

    assert isinstance(decoded, numpy.ndarray)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype
    assert array_equal(decoded, data)
