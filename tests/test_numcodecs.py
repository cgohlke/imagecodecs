# imagecodecs/tests/test_numcodecs.py

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

"""Unittests for the imagecodecs.numcodecs module."""

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
    from imagecodecs import numcodecs
except ImportError:
    pytest.skip('imagecodecs.numcodecs not found', allow_module_level=True)

numcodecs.register_codecs()


@pytest.mark.parallel_threads(1)
@pytest.mark.iterations(1)
def test_numcodecs_register(caplog):
    """Test register_codecs function."""
    numcodecs.register_codecs(verbose=False)
    assert 'already registered' not in caplog.text
    numcodecs.register_codecs(force=True, verbose=False)
    assert 'already registered' not in caplog.text
    numcodecs.register_codecs()
    assert 'already registered' in caplog.text
    numcodecs.register_codecs(force=True)
    assert 'replacing registered numcodec' in caplog.text

    assert isinstance(
        numcodecs.get_codec({'id': 'imagecodecs_lzw'}), numcodecs.Lzw
    )


@pytest.mark.skipif(not imagecodecs.LZO.available, reason='LZO missing')
def test_lzo_numcodecs():
    """Test LZO decoding with numcodecs."""
    assert (
        imagecodecs.numcodecs.Lzo(header=True).decode(
            b'\xf1\x00\x00\x00@\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00'
        )
        == b'a\01\fa' * 16
    )


@pytest.mark.parametrize('dtype', ['uint8', 'float32'])
@pytest.mark.parametrize(
    'kind', ['crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc']
)
def test_checksum_roundtrip(kind, dtype):
    """Test numcodecs.Checksum roundtrips."""
    if kind in {'crc32', 'adler32'} and not imagecodecs.ZLIB.available:
        pytest.skip('ZLIB missing')
    elif not imagecodecs.H5CHECKSUM.available:
        pytest.skip('H5CHECKSUM missing')

    data = numpy.arange(255, dtype=dtype).reshape((15, 17))[1:14, 2:15]
    codec = imagecodecs.numcodecs.Checksum(kind=kind)
    encoded = codec.encode(data)
    decoded = numpy.frombuffer(codec.decode(encoded), dtype=dtype)
    assert_array_equal(data, decoded.reshape((13, 13)))

    encoded = bytearray(encoded)
    encoded[10] += 1
    with pytest.raises(RuntimeError):
        codec.decode(encoded)


@pytest.mark.skipif(
    not imagecodecs.QUANTIZE.available, reason='QUANTIZE missing'
)
@pytest.mark.parametrize('nsd', [1, 4])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_quantize_bitround(dtype, nsd):
    """Test BitRound quantize against numcodecs."""
    from numcodecs import BitRound

    # TODO: 31.4 fails
    data = numpy.linspace(-2.1, 31.5, 51, dtype=dtype).reshape((3, 17))
    encoded = imagecodecs.numcodecs.Quantize(
        mode=imagecodecs.QUANTIZE.MODE.BITROUND,
        nsd=nsd,
    ).encode(data)
    nc = BitRound(keepbits=nsd)
    encoded2 = nc.decode(nc.encode(data))
    assert_array_equal(encoded, encoded2)


@pytest.mark.skipif(
    not imagecodecs.QUANTIZE.available, reason='QUANTIZE missing'
)
@pytest.mark.parametrize('nsd', [1, 4])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_quantize_scale(dtype, nsd):
    """Test Scale quantize against numcodecs."""
    from numcodecs import Quantize

    data = numpy.linspace(-2.1, 31.4, 51, dtype=dtype).reshape((3, 17))
    encoded = imagecodecs.numcodecs.Quantize(
        mode=imagecodecs.QUANTIZE.MODE.SCALE,
        nsd=nsd,
    ).encode(data)
    encoded2 = Quantize(digits=nsd, dtype=dtype).encode(data)
    assert_array_equal(encoded, encoded2)


@pytest.mark.skipif(
    not imagecodecs.H5CHECKSUM.available, reason='H5CHECKSUM missing'
)
def test_checksum_fletcher32():
    """Test numcodecs Checksum fletcher32 roundtrip."""
    data = numpy.arange(40, dtype='<i8')
    codec = imagecodecs.numcodecs.Checksum(kind='fletcher32')
    encoded = codec.encode(data)
    decoded = numpy.frombuffer(codec.decode(encoded), dtype='<i8')
    assert_array_equal(decoded, data)


@pytest.mark.skipif(
    not imagecodecs.H5CHECKSUM.available, reason='H5CHECKSUM missing'
)
def test_checksum_lookup3():
    """Test numcodecs Checksum lookup3 roundtrip."""
    data = b'Four score and seven years ago'
    codec = imagecodecs.numcodecs.Checksum(kind='lookup3')
    result = codec.encode(data)
    assert result[-4:] == b'\x51\x05\x77\x17'
    assert bytes(codec.decode(result)) == data

    codec = imagecodecs.numcodecs.Checksum(
        kind='lookup3', value=1230, prefix=b'Hello world'
    )
    result = codec.encode(data)
    assert bytes(codec.decode(result)) == data


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available, reason='DICOMRLE missing'
)
def test_dicomrle_numcodecs():
    """Test DICOMRLE decoding with numcodecs."""
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
    result = numpy.array(
        [
            [0, 16777216, 65536, 256, 1, 4294967295],
            [4294967295, 16777216, 65536, 256, 1, 0],
            [1, 16777216, 65536, 256, 1, 4294967294],
        ],
        dtype='u4',
    )
    decoded = imagecodecs.numcodecs.Dicomrle(dtype='<u4').decode(encoded)
    assert_array_equal(
        numpy.frombuffer(decoded, '<u4').reshape(result.shape), result
    )


@pytest.mark.skipif(not imagecodecs.DDS.available, reason='DDS missing')
def test_dds_numcodecs():
    """Test DDS decoding with numcodecs."""
    try:
        with open(datafiles('bcn/bc1.dds'), 'rb') as fh:
            encoded = fh.read()
    except FileNotFoundError:
        pytest.skip('bcn/bc1.dds not found')
    expected = imagecodecs.dds_decode(encoded)
    result = imagecodecs.numcodecs.Dds().decode(encoded)
    assert_array_equal(result, expected)


@pytest.mark.skipif(not imagecodecs.EER.available, reason='EER missing')
def test_eer_numcodecs():
    """Test EER decoding with numcodecs."""
    encoded = b'\x03\x1b\xfc\xb1\x35\xfb'  # from EER specification
    codec = imagecodecs.numcodecs.Eer(
        shape=(1, 312), skipbits=7, horzbits=1, vertbits=1
    )
    result = codec.decode(encoded)
    expected = imagecodecs.eer_decode(encoded, (1, 312), 7, 1, 1)
    assert_array_equal(result, expected)
    assert result[0, 3]
    assert result[0, 17]


@pytest.mark.skipif(
    not imagecodecs.CCITTRLE.available or not imagecodecs.TIFF.available,
    reason='CCITT or TIFF missing',
)
@pytest.mark.parametrize('compression', ['ccittrle', 'ccittfax3', 'ccittfax4'])
def test_ccitt_numcodecs(compression):
    """Test CCITT decoding with numcodecs."""
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
    if compression == 'ccittrle':
        codec = imagecodecs.numcodecs.Ccittrle(height=shape[0], width=shape[1])
        expected = imagecodecs.ccittrle_decode(
            encoded, height=shape[0], width=shape[1]
        )
    elif compression == 'ccittfax3':
        codec = imagecodecs.numcodecs.Ccittfax3(
            height=shape[0], width=shape[1], t4options=t4options
        )
        expected = imagecodecs.ccittfax3_decode(
            encoded, height=shape[0], width=shape[1], t4options=t4options
        )
    else:
        codec = imagecodecs.numcodecs.Ccittfax4(
            height=shape[0], width=shape[1]
        )
        expected = imagecodecs.ccittfax4_decode(
            encoded, height=shape[0], width=shape[1]
        )
    result = codec.decode(encoded)
    assert_array_equal(result, expected)
    assert_array_equal(result, data.view('uint8'))


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
        'checksum',
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
        'jpeg12',
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
def test_numcodecs(codec, photometric):
    """Test numcodecs though roundtrips."""
    if not getattr(
        imagecodecs,
        {'checksum': 'ZLIB', 'jpeg12': 'JPEG'}.get(codec, codec.upper()),
    ).available:
        pytest.skip(f'{codec} not found')
    if codec == 'jpeg12' and imagecodecs.JPEG.legacy:
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

    match codec:
        case 'aec':
            compressor = numcodecs.Aec(
                bitspersample=None, flags=None, blocksize=None, rsi=None
            )
        case 'apng':
            compressor = numcodecs.Apng(photometric=photometric, delay=100)
        case 'avif':
            if photometric != 'rgb':
                pytest.skip('xfail - AVIF does not support grayscale')
            compressor = numcodecs.Avif(
                level=100,
                speed=None,
                tilelog2=None,
                bitspersample=None,
                pixelformat=None,
                numthreads=2,
            )  # lossless
        case 'bfloat16':
            data = data.astype('float32')
            compressor = numcodecs.Bfloat16()
        case 'bitorder':
            compressor = numcodecs.Bitorder()
        case 'bitshuffle':
            compressor = numcodecs.Bitshuffle(
                itemsize=data.dtype.itemsize, blocksize=0
            )
        case 'blosc':
            compressor = numcodecs.Blosc(
                level=9,
                compressor='blosclz',
                typesize=data.dtype.itemsize * 8,
                blocksize=None,
                shuffle=None,
                numthreads=2,
            )
        case 'blosc2':
            compressor = numcodecs.Blosc2(
                level=9,
                compressor='blosclz',
                typesize=data.dtype.itemsize * 8,
                blocksize=None,
                shuffle=None,
                numthreads=2,
            )
        case 'bmp':
            compressor = numcodecs.Bmp(asrgb=None)
        case 'brotli':
            compressor = numcodecs.Brotli(level=5, mode=None, lgwin=None)
        case 'byteshuffle':
            data = data.astype('int16')
            compressor = numcodecs.Byteshuffle(
                shape=chunks, dtype=data.dtype, axis=axis
            )
        case 'bz2':
            compressor = numcodecs.Bz2(level=9)
        case 'checksum':
            compressor = numcodecs.Checksum(kind='crc32')
        case 'cms':
            if photometric == 'gray':
                compressor = numcodecs.Cms(
                    profile=imagecodecs.cms_profile('gray', gamma=1.0),
                    outprofile=imagecodecs.cms_profile('gray', gamma=2.2),
                    shape=chunks,
                    dtype=data.dtype,
                )
                lossless = False
                atol = 20
            else:
                data = data.astype('float32') / 255.0
                compressor = numcodecs.Cms(
                    profile='linearrgb',
                    outprofile='srgb',
                    shape=chunks,
                    dtype=data.dtype,
                )
                lossless = False
                atol = 1e-4
        case 'deflate':
            compressor = numcodecs.Deflate(level=8)
        case 'delta':
            compressor = numcodecs.Delta(
                shape=chunks, dtype=data.dtype, axis=axis
            )
        case 'exr':
            data = data.astype('float32')
            compressor = numcodecs.Exr(compression='PIZ')
        case 'float24':
            data = data.astype('float32')
            compressor = numcodecs.Float24()
        case 'floatpred':
            data = data.astype('float32')
            compressor = numcodecs.Floatpred(
                shape=chunks, dtype=data.dtype, axis=axis
            )
        case 'gif':
            compressor = numcodecs.Gif()
        case 'hcomp':
            if photometric == 'rgb':
                pytest.skip('xfail - HCOMP does not support RGB')
            data = data.astype('int32')
            compressor = numcodecs.Hcomp(shape=chunks)
        case 'heif':
            compressor = numcodecs.Heif(photometric=photometric)
            lossless = False  # TODO: lossless not working
            atol = 1
        case 'htj2k':
            compressor = numcodecs.Htj2k()  # lossless
        case 'jpeg':
            lossless = False
            atol = 4
            compressor = numcodecs.Jpeg(level=99)
        case 'jpeg12':
            lossless = False
            atol = 4 << 4
            data = data.astype('uint16') << 4
            compressor = numcodecs.Jpeg(level=99, bitspersample=12)
        case 'jpeg2k':
            compressor = numcodecs.Jpeg2k(level=0)  # lossless
        case 'jpegls':
            compressor = numcodecs.Jpegls(level=0)  # lossless
        case 'jpegxl':
            compressor = numcodecs.Jpegxl(level=101)  # lossless
        case 'jpegxr':
            compressor = numcodecs.Jpegxr(
                level=1.0, photometric='RGB' if photometric == 'rgb' else None
            )  # lossless
        case 'jpegxs':
            if photometric != 'rgb':
                pytest.skip(f'xfail - JPEGXS does not support {photometric=}')
            compressor = numcodecs.Jpegxs(config='p=MLS.12', verbose=1)
        case 'lerc':
            if imagecodecs.ZSTD.available:
                compression = 'zstd'
                compressionargs = {'level': 10}
            else:
                compression = None
                compressionargs = None
            compressor = numcodecs.Lerc(
                level=0.0,
                compression=compression,
                compressionargs=compressionargs,
            )
        case 'ljpeg':
            if photometric == 'rgb':
                pytest.skip('xfail - LJPEG does not support rgb')
            data = data.astype('uint16') << 2
            compressor = numcodecs.Ljpeg(bitspersample=10)
        case 'lz4':
            compressor = numcodecs.Lz4(level=10, hc=True, header=True)
        case 'lz4f':
            compressor = numcodecs.Lz4f(
                level=12,
                blocksizeid=False,
                contentchecksum=True,
                blockchecksum=True,
            )
        case 'lz4h5':
            compressor = numcodecs.Lz4h5(level=10, blocksize=100)
        case 'lzf':
            compressor = numcodecs.Lzf(header=True)
        case 'lzfse':
            compressor = numcodecs.Lzfse()
        case 'lzham':
            compressor = numcodecs.Lzham(level=6)
        case 'lzma':
            compressor = numcodecs.Lzma(
                level=6, check=imagecodecs.LZMA.CHECK.SHA256
            )
        case 'lzw':
            compressor = numcodecs.Lzw()
        case 'meshopt':
            data = data.astype('float32')
            compressor = numcodecs.Meshopt(
                level=3,
                shape=chunks,
                dtype=data.dtype,
                items=1 if photometric == 'gray' else None,
            )
        case 'packbits' if photometric == 'rgb':
            compressor = numcodecs.Packbits(axis=-2)
        case 'packbits':
            compressor = numcodecs.Packbits()
        case 'packints':
            data //= 2
            compressor = numcodecs.Packints(
                dtype=data.dtype, bitspersample=data.itemsize * 8 - 1
            )
        case 'pcodec':
            data = data.astype('float32')
            compressor = numcodecs.Pcodec(
                level=8,
                shape=chunks[1:],
                dtype=data.dtype,
            )
            lossless = False
            atol = 1e-2
        case 'pglz':
            compressor = numcodecs.Pglz(strategy=None)
        case 'pixarlog':
            data = data.astype('float32')
            data /= 255.0
            compressor = numcodecs.Pixarlog(
                shape=chunks, dtype=data.dtype, level=6
            )
            lossless = False
            atol = 1e-3
        case 'plio':
            data = data.astype('int32')
            compressor = numcodecs.Plio(shape=chunks, dtype=data.dtype)
        case 'png':
            compressor = numcodecs.Png()
        case 'qoi':
            if photometric != 'rgb':
                pytest.skip('xfail - QOI does not support grayscale')
            compressor = numcodecs.Qoi()
        case 'quantize':
            data = data.astype('float32')
            compressor = numcodecs.Quantize(mode='bitgroom', nsd=6)
            lossless = False
            atol = 1.0
        case 'rgbe':
            if photometric != 'rgb':
                pytest.skip('xfail - RGBE does not support grayscale')
            data = data.astype('float32')
            # lossless = False
            compressor = numcodecs.Rgbe(
                shape=chunks[-3:], header=False, rle=True
            )
        case 'rcomp':
            compressor = numcodecs.Rcomp(shape=chunks, dtype=data.dtype)
        case 'snappy':
            compressor = numcodecs.Snappy()
        case 'sperr':
            data = data.astype('float32')
            compressor = numcodecs.Sperr(
                level=100.0,
                mode='psnr',
                shape=chunks[1:],
                dtype=data.dtype,
                header=False,
            )
            lossless = False
            atol = 1e-2
        case 'spng':
            compressor = numcodecs.Spng()
        case 'sz3':
            data = data.astype('float32')
            atol = 1e-3
            compressor = numcodecs.Sz3(
                mode='abs',
                abs=atol,
                shape=chunks[1:],
                dtype=data.dtype,
            )
            lossless = False
        case 'szip':
            compressor = numcodecs.Szip(
                header=True, **imagecodecs.szip_params(data)
            )
        case 'tiff':
            compressor = numcodecs.Tiff(level=6, predictor=True)
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
            compressor = numcodecs.Ultrahdr()
            lossless = False
            atol = 1e-2
        case 'wavpack':
            data = data.astype('int16')
            compressor = numcodecs.Wavpack(
                level=2, channels=photometric == 'rgb'
            )
        case 'webp':
            if photometric != 'rgb':
                pytest.skip('xfail - WebP does not support grayscale')
            compressor = numcodecs.Webp(level=-1)
        case 'wic':
            compressor = numcodecs.Wic(format='png')
        case 'xor':
            compressor = numcodecs.Xor(
                shape=chunks, dtype=data.dtype, axis=axis
            )
        case 'zfp':
            data = data.astype('float32')
            compressor = numcodecs.Zfp(
                shape=chunks, dtype=data.dtype, header=False
            )
        case 'zlib':
            compressor = numcodecs.Zlib(level=6)
        case 'zlibng':
            compressor = numcodecs.Zlibng(level=6)
        case 'zopfli':
            compressor = numcodecs.Zopfli(level=1, blocksplitting=False)
        case 'zstd':
            compressor = numcodecs.Zstd(level=10)
        case _:
            raise RuntimeError

    if 0:
        # use ZIP file on disk
        filename = f'test_{codec}.{photometric}.{data.dtype.str[1:]}.zarr.zip'
        store = zarr.ZipStore(filename, mode='w')
    else:
        try:
            store = zarr.MemoryStore()
            store_args = {}
        except AttributeError:
            # zarr 3
            store = zarr.storage.MemoryStore()
            store_args = {'zarr_format': 2}
    z = zarr.create(
        store=store,
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype=data.dtype,
        compressor=compressor,
        **store_args,
    )
    z[:] = data
    del z

    z = zarr.open(store, mode='r')
    if codec == 'jetraw':
        pass  # it does not make sense to test Jetraw on tiled, synthetic data
    elif lossless:
        assert_array_equal(z[:, :150, :150], data[:, :150, :150])
    else:
        assert_allclose(
            z[:, :150, :150], data[:, :150, :150], atol=atol, rtol=0.001
        )
    with contextlib.suppress(Exception):
        store.close()


# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
