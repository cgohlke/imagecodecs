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

import contextlib
import io

import numpy
import pytest
from conftest import datafiles
from numpy.testing import assert_allclose, assert_array_equal

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

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
from zarr.buffer import default_buffer_prototype
from zarr.buffer.cpu import Buffer
from zarr.core.array import ArrayConfig
from zarr.core.array_spec import ArraySpec
from zarr.core.dtype import get_data_type_from_native_dtype

zarr3codecs.register_codecs()


def _array_spec(shape, dtype):
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
def test_zarr_register(caplog):
    """Test register_codecs function."""
    from zarr.registry import get_codec_class

    zarr3codecs.register_codecs(verbose=False)
    assert 'zarr codec' not in caplog.text
    zarr3codecs.register_codecs()
    assert 'zarr codec' not in caplog.text

    assert get_codec_class('imagecodecs_lzw') is zarr3codecs.Lzw


@pytest.mark.skipif(not imagecodecs.LZO.available, reason='LZO missing')
def test_lzo_zarr():
    """Test LZO decoding with zarr."""
    spec = _array_spec((64,), 'uint8')
    result = zarr3codecs.Lzo(header=True)._decode_sync(
        Buffer.from_bytes(
            b'\xf1\x00\x00\x00@\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00'
        ),
        spec,
    )
    assert bytes(result.to_bytes()) == b'a\01\fa' * 16


@pytest.mark.parametrize('dtype', ['uint8', 'float32'])
@pytest.mark.parametrize(
    'kind', ['crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc']
)
def test_checksum_zarr(kind, dtype):
    """Test zarr3codecs.Checksum roundtrips."""
    if kind in {'crc32', 'adler32'} and not imagecodecs.ZLIB.available:
        pytest.skip('ZLIB missing')
    elif not imagecodecs.H5CHECKSUM.available:
        pytest.skip('H5CHECKSUM missing')
    from zarr.abc.codec import BytesBytesCodec

    data = numpy.arange(255, dtype=dtype).reshape((15, 17))[1:14, 2:15]
    codec_obj = zarr3codecs.Checksum(kind=kind)
    assert isinstance(codec_obj, BytesBytesCodec)

    store = zarr.storage.MemoryStore()
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
    assert_array_equal(z[:], data)
    with contextlib.suppress(Exception):
        store.close()


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available, reason='DICOMRLE missing'
)
def test_dicomrle_zarr():
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
    assert_array_equal(decoded.as_numpy_array(), expected)


@pytest.mark.skipif(not imagecodecs.DDS.available, reason='DDS missing')
def test_dds_zarr():
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
    assert_array_equal(decoded.as_numpy_array(), expected)


@pytest.mark.skipif(not imagecodecs.EER.available, reason='EER missing')
def test_eer_zarr():
    """Test EER decoding with zarr."""
    encoded = b'\x03\x1b\xfc\xb1\x35\xfb'  # from EER specification
    expected = imagecodecs.eer_decode(encoded, (1, 312), 7, 1, 1)
    decoded = zarr3codecs.Eer(
        shape=(1, 312), skipbits=7, horzbits=1, vertbits=1
    )._decode_sync(
        Buffer.from_bytes(encoded), _array_spec(expected.shape, expected.dtype)
    )
    assert_array_equal(decoded.as_numpy_array(), expected)
    assert decoded.as_numpy_array()[0, 3]
    assert decoded.as_numpy_array()[0, 17]


@pytest.mark.skipif(
    not imagecodecs.CCITTRLE.available or not imagecodecs.TIFF.available,
    reason='CCITT or TIFF missing',
)
@pytest.mark.parametrize('compression', ['ccittrle', 'ccittfax3', 'ccittfax4'])
def test_ccitt_zarr(compression):
    """Test CCITT decoding with zarr."""
    tifffile = pytest.importorskip('tifffile')
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
        codec = zarr3codecs.Ccittrle(height=shape[0], width=shape[1])
        expected = imagecodecs.ccittrle_decode(
            encoded, height=shape[0], width=shape[1]
        )
    elif compression == 'ccittfax3':
        codec = zarr3codecs.Ccittfax3(
            height=shape[0], width=shape[1], t4options=t4options
        )
        expected = imagecodecs.ccittfax3_decode(
            encoded, height=shape[0], width=shape[1], t4options=t4options
        )
    else:
        codec = zarr3codecs.Ccittfax4(height=shape[0], width=shape[1])
        expected = imagecodecs.ccittfax4_decode(
            encoded, height=shape[0], width=shape[1]
        )
    decoded = codec._decode_sync(Buffer.from_bytes(encoded), spec)
    assert_array_equal(decoded.as_numpy_array(), expected)
    assert_array_equal(decoded.as_numpy_array(), data.view('uint8'))


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
def test_zarr_pipeline(pipeline):
    """Test imagecodecs.zarr codec pipelines through zarr 3 roundtrips."""
    data = numpy.load(datafiles('rgb.u1.npy'))
    data = numpy.stack([data, data])
    shape = data.shape
    chunks = (1, 128, 128, 3)

    match pipeline:
        case 'delta_lzw':
            filters = [zarr3codecs.Delta(axis=-1)]
            compressors = [zarr3codecs.Lzw()]
        case 'bitshuffle_zstd':
            filters = [zarr3codecs.Bitshuffle(blocksize=0)]
            compressors = [zarr3codecs.Zstd(level=5)]
        case 'delta_bitshuffle_zstd':
            filters = [
                zarr3codecs.Delta(axis=-1),
                zarr3codecs.Bitshuffle(blocksize=0),
            ]
            compressors = [zarr3codecs.Zstd(level=5)]
        case 'floatpred_deflate':
            data = data.astype('float32')
            shape = data.shape
            filters = [zarr3codecs.Floatpred(axis=-1)]
            compressors = [zarr3codecs.Deflate(level=6)]
        case 'xor_lzma':
            filters = [zarr3codecs.Xor(axis=-1)]
            compressors = [zarr3codecs.Lzma(level=3)]
        case _:
            raise RuntimeError

    store = zarr.storage.MemoryStore()
    z = zarr.create_array(
        store,
        shape=shape,
        chunks=chunks,
        dtype=data.dtype,
        filters=filters,
        compressors=compressors,
    )
    z[:] = data
    del z

    z = zarr.open_array(store, mode='r')
    assert_array_equal(z[:], data)
    with contextlib.suppress(Exception):
        store.close()


@pytest.mark.parametrize('photometric', ['gray', 'rgb', 'stack'])
@pytest.mark.parametrize(
    'codec',
    [
        'aec',
        'apng',
        'avif',
        'bfloat16',
        'bitorder',
        'bitshuffle',
        'blosc',
        'blosc2',
        'bmp',
        'brotli',
        'byteshuffle',
        'bz2',
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
        'jpeg',
        'jpeg2k',
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
        'packbits',
        'packints',
        'pcodec',
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
    ],
)
def test_zarr(codec, photometric):
    """Test imagecodecs.zarr codecs through zarr 3 roundtrips."""
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not found')

    data = numpy.load(datafiles('rgb.u1.npy'))
    data = numpy.stack([data, data])
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

    lossless = True
    atol = 0
    match codec:
        case 'aec':
            codec_obj = zarr3codecs.Aec()
        case 'apng':
            codec_obj = zarr3codecs.Apng(photometric=photometric, delay=100)
        case 'avif':
            if photometric != 'rgb':
                pytest.skip('xfail - AVIF does not support grayscale')
            codec_obj = zarr3codecs.Avif(level=100, numthreads=2)
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
                level=9, compressor='blosclz', numthreads=2
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
        case 'jpeg':
            lossless = False
            atol = 4
            codec_obj = zarr3codecs.Jpeg(level=99, subsampling='444')
        case 'jpeg2k':
            codec_obj = zarr3codecs.Jpeg2k(level=0)
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
            codec_obj = zarr3codecs.Jpegxs(config='p=MLS.12', verbose=1)
        case 'lerc':
            if imagecodecs.ZSTD.available:
                compression = 'zstd'
                compressionargs = {'level': 10}
            else:
                compression = None
                compressionargs = None
            codec_obj = zarr3codecs.Lerc(
                level=0.0,
                compression=compression,
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
        case 'packbits' if photometric == 'rgb':
            codec_obj = zarr3codecs.Packbits(axis=-2)
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
            atol = 1e-3
            codec_obj = zarr3codecs.Sz3(mode='abs', abs=atol)
            lossless = False
        case 'szip':
            codec_obj = zarr3codecs.Szip(
                header=True, **imagecodecs.szip_params(data)
            )
        case 'tiff':
            codec_obj = zarr3codecs.Tiff(level=6, predictor=True)
        case 'ultrahdr':
            if photometric != 'rgb':
                pytest.skip(
                    f'xfail - ULTRAHDR does not support {photometric=}'
                )
            else:
                pytest.skip('xfail - ultrahdr_encode not working')
            data = data.astype('float32')
            data /= data.max()
            data = data.astype('float16')
            codec_obj = zarr3codecs.Ultrahdr()
            lossless = False
            atol = 1e-2
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
        case _:
            raise RuntimeError

    store = zarr.storage.MemoryStore()
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
    if lossless:
        assert_array_equal(z[:], data)
    else:
        assert_allclose(z[:], data, atol=atol)
    with contextlib.suppress(Exception):
        store.close()


# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
