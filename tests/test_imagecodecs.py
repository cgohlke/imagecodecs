# -*- coding: utf-8 -*-
# test_imagecodecs.py

# Copyright (c) 2018-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.31

"""

from __future__ import division, print_function

import sys
import os
import io
import re
import glob
import pathlib
import tempfile
import os.path as osp

import pytest
import numpy
from numpy.testing import assert_array_equal, assert_allclose

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import czifile
except Exception:
    czifile = None


if (
    'imagecodecs_lite' in os.getcwd() or
    osp.exists(osp.join(osp.dirname(__file__), '..', 'imagecodecs_lite'))
):
    try:
        import imagecodecs_lite as imagecodecs
        from imagecodecs_lite import _imagecodecs_lite
        from imagecodecs_lite import imagecodecs as imagecodecs_py
    except ImportError:
        pytest.exit('the imagecodec-lite package is not installed')
    lzma = zlib = bz2 = zstd = lz4 = lzf = blosc = brotli = snappy = None
    zopfli = bitshuffle = None
    _imagecodecs = _jpeg12 = _jpegls = _jpegxl = _zfp = None
else:
    try:
        import imagecodecs
        import imagecodecs.imagecodecs as imagecodecs_py
        from imagecodecs.imagecodecs import (
            lzma, zlib, bz2, zstd, lz4, lzf, blosc, brotli, snappy, zopfli,
            bitshuffle
        )
        from imagecodecs import _imagecodecs
        from imagecodecs import _imagecodecs_lite
    except ImportError:
        pytest.exit('the imagecodec package is not installed')

    try:
        from imagecodecs import _jpeg12
    except ImportError:
        _jpeg12 = None

    try:
        from imagecodecs import _jpegls
    except ImportError:
        _jpegls = None

    try:
        from imagecodecs import _jpegxl
    except ImportError:
        _jpegxl = None

    try:
        from imagecodecs import _zfp
    except ImportError:
        _zfp = None


TEST_DIR = osp.dirname(__file__)
IS_PY2 = sys.version_info[0] == 2
IS_32BIT = sys.maxsize < 2**32
IS_WIN = sys.platform == 'win32'
# running on Windows development computer?
IS_CG = os.environ.get('COMPUTERNAME', '').startswith('CG-')
# running in cibuildwheel environment?
IS_CI = os.environ.get('CIBUILDWHEEL', False)


###############################################################################

def test_version():
    """Assert imagecodecs versions match docstrings."""
    ver = ':Version: ' + imagecodecs.__version__
    assert ver in __doc__
    assert ver in imagecodecs.__doc__
    assert imagecodecs.version().startswith('imagecodecs')
    assert _imagecodecs_lite.version().startswith('imagecodecs')
    assert ver in imagecodecs_py.__doc__
    if zlib:
        assert imagecodecs.version(dict)['zlib'].startswith('1.')


@pytest.mark.skipif(_imagecodecs is None, reason='Testing imagecodecs-lite')
@pytest.mark.parametrize('name', [
    # optional Cython extension modules
    '_jpeg12', '_jpegls', '_jpegxl', '_zfp',
    # third-party Python packages
    'lzma', 'zstd', 'lz4', 'lzf', 'blosc', 'brotli', 'zopfli', 'bitshuffle',
    'tifffile', 'czifile'])
def test_module_exist(name):
    """Test that required modules are present."""
    if not IS_CG and _jpeg12 is None and name == '_jpeg12':
        pytest.skip('_jpeg12 not supported in this build')
    if not (IS_CG or IS_CI) and name in ('_jpegls', '_jpegxl', '_zfp'):
        pytest.skip(name + ' not supported in this build')
    if IS_WIN and IS_PY2 and name in ('bitshuffle', '_jpegls', '_jpegxl'):
        pytest.skip(name + ' not supported on this platform')
    assert globals()[name] is not None, "no module named '%s'" % name


@pytest.mark.skipif(not hasattr(imagecodecs, 'imread'),
                    reason='imread function missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('codec', ['none', 'str', 'ext', 'codec', 'list',
                                   'fail'])
@pytest.mark.parametrize('filearg', ['str', 'pathlib', 'bytesio', 'bytes'])
def test_imread_imwrite(filearg, codec):
    """Test imread and imwrite functions."""
    if IS_PY2 and filearg == 'bytes':
        pytest.skip('bytes input not supported on Python 2')

    imread = imagecodecs.imread
    imwrite = imagecodecs.imwrite
    data = image_data('rgba', 'uint8')

    if codec == 'ext':
        # auto detect codec from file extension or trial&error
        with TempFileName(suffix='.npy') as fileobj:
            if filearg == 'pathlib':
                fileobj = pathlib.Path(fileobj)
            if filearg == 'bytes':
                fileobj = imagecodecs.numpy_encode(data)
            elif filearg == 'bytesio':
                # must specify codec
                fileobj = io.BytesIO()
                imwrite(fileobj, data, codec=imagecodecs.numpy_encode)
            else:
                imwrite(fileobj, data, level=99)
            if filearg == 'bytesio':
                fileobj.seek(0)
            im, codec = imread(fileobj, return_codec=True)
            assert codec == imagecodecs.numpy_decode
            assert_array_equal(data, im)
        return

    if codec == 'none':
        encode = None
        decode = None
    elif codec == 'str':
        encode = 'numpy'
        decode = 'numpy'
    elif codec == 'list':
        encode = 'npz'
        decode = ['npz']
    elif codec == 'fail':
        encode = 'fail'
        decode = 'fail'
    elif codec == 'codec':
        encode = imagecodecs.numpy_encode
        decode = imagecodecs.numpy_decode

    with TempFileName() as fileobj:
        if filearg == 'pathlib':
            fileobj = pathlib.Path(fileobj)
        elif filearg == 'bytesio':
            fileobj = io.BytesIO()

        if filearg == 'bytes':
            fileobj = imagecodecs.numpy_encode(data)
        elif encode in (None, 'fail'):
            with pytest.raises(ValueError):
                imwrite(fileobj, data, codec=encode)
            imwrite(fileobj, data, codec=imagecodecs.numpy_encode)
        else:
            imwrite(fileobj, data, codec=encode)

        if filearg == 'bytesio':
            fileobj.seek(0)

        if codec == 'fail':
            with pytest.raises(ValueError):
                im = imread(fileobj, codec=decode)
            return

        im, ret = imread(fileobj, codec=decode, return_codec=True)
        assert ret == imagecodecs.numpy_decode
        assert_array_equal(data, im)


def test_none():
    """Test NOP codec."""
    encode = imagecodecs.none_encode
    decode = imagecodecs.none_decode
    data = b'None'
    assert encode(data) is data
    assert decode(data) is data


def test_bitorder():
    """Test BitOrder codec with bytes."""
    decode = imagecodecs.bitorder_decode
    data = b'\x01\x00\x9a\x02'
    reverse = b'\x80\x00Y@'
    # return new string
    assert decode(data) == reverse
    assert data == b'\x01\x00\x9a\x02'
    # provide output
    out = bytearray(len(data))
    decode(data, out=out)
    assert out == reverse
    assert data == b'\x01\x00\x9a\x02'
    # inplace
    decode(data, out=data)
    assert data == reverse
    # bytes range
    assert BYTES == decode(readfile('bytes.bitorder.bin'))


def test_bitorder_ndarray():
    """Test BitOrder codec with ndarray."""
    decode = imagecodecs.bitorder_decode
    data = numpy.array([1, 666], dtype='uint16')
    reverse = numpy.array([128, 16473], dtype='uint16')
    # return new array
    assert_array_equal(decode(data), reverse)
    # inplace
    decode(data, out=data)
    assert_array_equal(data, numpy.array([128, 16473], dtype='uint16'))
    # array view
    data = numpy.array([[1, 666, 1431655765, 62],
                        [2, 667, 2863311530, 32],
                        [3, 668, 1431655765, 30]], dtype='uint32')
    reverse = numpy.array([[1, 666, 1431655765, 62],
                           [2, 16601, 1431655765, 32],
                           [3, 16441, 2863311530, 30]], dtype='uint32')
    assert_array_equal(decode(data[1:, 1:3]), reverse[1:, 1:3])
    # array view inplace
    decode(data[1:, 1:3], out=data[1:, 1:3])
    assert_array_equal(data, reverse)


def test_packints_decode():
    """Test PackInts decoder."""
    decode = imagecodecs.packints_decode

    decoded = decode(b'', 'B', 1)
    assert len(decoded) == 0

    decoded = decode(b'a', 'B', 1)
    assert tuple(decoded) == (0, 1, 1, 0, 0, 0, 0, 1)

    decoded = decode(b'ab', 'B', 2)
    assert tuple(decoded) == (1, 2, 0, 1, 1, 2, 0, 2)

    decoded = decode(b'abcd', 'B', 3)
    assert tuple(decoded) == (3, 0, 2, 6, 1, 1, 4, 3, 3, 1)

    decoded = decode(numpy.frombuffer(b'abcd', dtype='uint8'), 'B', 3)
    assert tuple(decoded) == (3, 0, 2, 6, 1, 1, 4, 3, 3, 1)


PACKBITS_DATA = [
    (b'', b''),
    (b'X', b'\x00X'),
    (b'123', b'\x02123'),
    (b'112112', b'\xff1\x002\xff1\x002'),
    (b'1122', b'\xff1\xff2'),
    (b'1' * 126, b'\x831'),
    (b'1' * 127, b'\x821'),
    (b'1' * 128, b'\x811'),
    (b'1' * 127 + b'foo', b'\x821\x00f\xffo'),
    (b'12345678' * 16,  # literal 128
     b'\x7f1234567812345678123456781234567812345678123456781234567812345678'
     b'1234567812345678123456781234567812345678123456781234567812345678'),
    (b'12345678' * 17,
     b'~1234567812345678123456781234567812345678123456781234567812345678'
     b'123456781234567812345678123456781234567812345678123456781234567\x08'
     b'812345678'),
    (b'1' * 128 + b'12345678' * 17,
     b'\x821\xff1~2345678123456781234567812345678123456781234567812345678'
     b'1234567812345678123456781234567812345678123456781234567812345678'
     b'12345678\x0712345678'),
    (b'\xaa\xaa\xaa\x80\x00\x2a\xaa\xaa\xaa\xaa\x80\x00'
     b'\x2a\x22\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa',
     b'\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa')]


@pytest.mark.parametrize('data', range(len(PACKBITS_DATA)))
@pytest.mark.parametrize('codec', ['encode', 'decode'])
def test_packbits(codec, data):
    """Test PackBits codec."""
    encode = imagecodecs.packbits_encode
    decode = imagecodecs.packbits_decode
    uncompressed, compressed = PACKBITS_DATA[data]
    if codec == 'decode':
        assert decode(compressed) == uncompressed
    elif codec == 'encode':
        try:
            assert encode(uncompressed) == compressed
        except AssertionError:
            # roundtrip
            assert decode(encode(uncompressed)) == uncompressed


def test_packbits_nop():
    """Test PackBits decoding empty data."""
    decode = imagecodecs.packbits_decode
    assert decode(b'\x80') == b''
    assert decode(b'\x80\x80') == b''


@pytest.mark.parametrize('output', [None, 'array'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
def test_packbits_array(codec, output):
    """Test PackBits codec with arrays."""
    encode = imagecodecs.packbits_encode
    decode = imagecodecs.packbits_decode
    uncompressed, compressed = PACKBITS_DATA[-1]
    shape = (2, 7, len(uncompressed))
    data = numpy.empty(shape, dtype='uint8')
    data[..., :] = numpy.frombuffer(uncompressed, dtype='uint8')
    compressed = compressed * (shape[0] * shape[1])
    if codec == 'encode':
        if output == 'array':
            out = numpy.empty(data.size, data.dtype)
            assert_array_equal(encode(data, out=out),
                               numpy.frombuffer(compressed, dtype='uint8'))
        else:
            assert encode(data) == compressed
    else:
        if output == 'array':
            out = numpy.empty(data.size, data.dtype)
            assert_array_equal(decode(compressed, out=out), data.flat)
        else:
            assert decode(compressed) == data.tobytes()


@pytest.mark.parametrize('output', ['new', 'out', 'inplace'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
@pytest.mark.parametrize(
    'kind', ['u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'B',
             pytest.param('b', marks=pytest.mark.skipif(
                 sys.version_info[0] == 2, reason='Python 2'))])
@pytest.mark.parametrize('func', ['delta', 'xor'])
def test_delta(output, kind, codec, func):
    """Test Delta codec."""
    if func == 'delta':
        encode = imagecodecs.delta_encode
        decode = imagecodecs.delta_decode
        encode_py = imagecodecs_py.delta_encode
        # decode_py = imagecodecs_py.imagecodecs.delta_decode
    elif func == 'xor':
        encode = imagecodecs.xor_encode
        decode = imagecodecs.xor_decode
        encode_py = imagecodecs_py.xor_encode
        # decode_py = imagecodecs_py.imagecodecs.xor_decode

    bytetype = bytearray
    if kind == 'b':
        bytetype = bytes
        kind = 'B'

    axis = -2  # do not change
    dtype = numpy.dtype(kind)
    if kind[0] in 'iuB':
        low = numpy.iinfo(dtype).min
        high = numpy.iinfo(dtype).max
        data = numpy.random.randint(low, high, size=33 * 31 * 3,
                                    dtype=dtype).reshape(33, 31, 3)
    else:
        low, high = -1e5, 1e5
        data = numpy.random.randint(low, high, size=33 * 31 * 3,
                                    dtype='i4').reshape(33, 31, 3)
        data = data.astype(dtype)

    data[16, 14] = [0, 0, 0]
    data[16, 15] = [low, high, low]
    data[16, 16] = [high, low, high]
    data[16, 17] = [low, high, low]
    data[16, 18] = [high, low, high]
    data[16, 19] = [0, 0, 0]

    if kind == 'B':
        # data = data.reshape(-1)
        data = data.tobytes()
        diff = encode_py(data, axis=0)
        if output == 'new':
            if codec == 'encode':
                encoded = encode(data, out=bytetype)
                assert encoded == diff
            elif codec == 'decode':
                decoded = decode(diff, out=bytetype)
                assert decoded == data
        elif output == 'out':
            if codec == 'encode':
                encoded = bytetype(len(data))
                encode(data, out=encoded)
                assert encoded == diff
            elif codec == 'decode':
                decoded = bytetype(len(data))
                decode(diff, out=decoded)
                assert decoded == data
        elif output == 'inplace':
            if codec == 'encode':
                encoded = bytetype(data)
                encode(encoded, out=encoded)
                assert encoded == diff
            elif codec == 'decode':
                decoded = bytetype(diff)
                decode(decoded, out=decoded)
                assert decoded == data
    else:
        # if func == 'xor' and kind in ('f4', 'f8'):
        #      with pytest.raises(ValueError):
        #          encode(data, axis=axis)
        #      pytest.skip("XOR codec not implemented for float data")
        diff = encode_py(data, axis=-2)

        if output == 'new':
            if codec == 'encode':
                encoded = encode(data, axis=axis)
                assert_array_equal(encoded, diff)
            elif codec == 'decode':
                decoded = decode(diff, axis=axis)
                assert_array_equal(decoded, data)
        elif output == 'out':
            if codec == 'encode':
                encoded = numpy.zeros_like(data)
                encode(data, axis=axis, out=encoded)
                assert_array_equal(encoded, diff)
            elif codec == 'decode':
                decoded = numpy.zeros_like(data)
                decode(diff, axis=axis, out=decoded)
                assert_array_equal(decoded, data)
        elif output == 'inplace':
            if codec == 'encode':
                encoded = data.copy()
                encode(encoded, axis=axis, out=encoded)
                assert_array_equal(encoded, diff)
            elif codec == 'decode':
                decoded = diff.copy()
                decode(decoded, axis=axis, out=decoded)
                assert_array_equal(decoded, data)


@pytest.mark.parametrize('output', ['new', 'out'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
@pytest.mark.parametrize('endian', ['le', 'be'])
@pytest.mark.parametrize('planar', ['rgb', 'rrggbb'])
def test_floatpred(planar, endian, output, codec):
    """Test FloatPred codec."""
    encode = imagecodecs.floatpred_encode
    decode = imagecodecs.floatpred_decode
    data = numpy.fromfile(
        datafiles('rgb.bin'), dtype='<f4').reshape(33, 31, 3)

    if planar == 'rgb':
        axis = -2
        if endian == 'le':
            encoded = numpy.fromfile(
                datafiles('rgb.floatpred_le.bin'), dtype='<f4')
            encoded = encoded.reshape(33, 31, 3)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.empty_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.empty_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
        elif endian == 'be':
            data = data.astype('>f4')
            encoded = numpy.fromfile(
                datafiles('rgb.floatpred_be.bin'), dtype='>f4')
            encoded = encoded.reshape(33, 31, 3)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.empty_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.empty_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
    elif planar == 'rrggbb':
        axis = -1
        data = numpy.ascontiguousarray(numpy.moveaxis(data, 2, 0))
        if endian == 'le':
            encoded = numpy.fromfile(
                datafiles('rrggbb.floatpred_le.bin'), dtype='<f4')
            encoded = encoded.reshape(3, 33, 31)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.empty_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.empty_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
        elif endian == 'be':
            data = data.astype('>f4')
            encoded = numpy.fromfile(
                datafiles('rrggbb.floatpred_be.bin'), dtype='>f4')
            encoded = encoded.reshape(3, 33, 31)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.empty_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.empty_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)


def test_lzw_msb():
    """Test LZW decoder with MSB."""
    # TODO: add test_lzw_lsb
    decode = imagecodecs.lzw_decode
    for data, decoded in [
            (b'\x80\x1c\xcc\'\x91\x01\xa0\xc2m6\x99NB\x03\xc9\xbe\x0b'
             b'\x07\x84\xc2\xcd\xa68|"\x14 3\xc3\xa0\xd1c\x94\x02\x02\x80',
             b'say hammer yo hammer mc hammer go hammer'),
            (b'\x80\x18M\xc6A\x01\xd0\xd0e\x10\x1c\x8c\xa73\xa0\x80\xc7\x02'
             b'\x10\x19\xcd\xe2\x08\x14\x10\xe0l0\x9e`\x10\x10\x80',
             b'and the rest can go and play'),
            (b'\x80\x18\xcc&\xe19\xd0@t7\x9dLf\x889\xa0\xd2s',
             b"can't touch this"),
            (b'\x80@@', b'')]:
        assert decode(data) == decoded


@pytest.mark.parametrize('output', ['new', 'size', 'ndarray', 'bytearray'])
def test_lzw_decode(output):
    """Test LZW decoder of input with horizontal differencing."""
    decode = imagecodecs.lzw_decode
    delta_decode = imagecodecs.delta_decode
    data = readfile('bytes.lzw_horizontal.bin')
    decoded_size = len(BYTES)

    if output == 'new':
        decoded = decode(data)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
    elif output == 'size':
        decoded = decode(data, out=decoded_size)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
        # with pytest.raises(RuntimeError):
        decode(data, buffersize=32, out=decoded_size)
    elif output == 'ndarray':
        decoded = numpy.empty_like(BYTESIMG)
        decode(data, out=decoded.reshape(-1))
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
    elif output == 'bytearray':
        decoded = bytearray(decoded_size)
        decode(data, out=decoded)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)


def test_lzw_decode_image_noeoi():
    """Test LZW decoder of input without EOI 512x512u2."""
    decode = imagecodecs.lzw_decode
    fname = datafiles('image_noeoi.lzw.bin')
    with open(fname, 'rb') as fh:
        encoded = fh.read()
    fname = datafiles('image_noeoi.bin')
    with open(fname, 'rb') as fh:
        decoded_known = fh.read()
    # new output
    decoded = decode(encoded)
    assert decoded == decoded_known
    # provide output
    decoded = bytearray(len(decoded))
    decode(encoded, out=decoded)
    assert decoded == decoded_known
    # truncated output
    decoded = bytearray(100)
    decode(encoded, out=decoded)
    assert len(decoded) == 100


@pytest.mark.skipif(not hasattr(imagecodecs, 'blosc_decode'),
                    reason='compression codecs missing')
@pytest.mark.parametrize('output', [
    'new', 'bytearray', 'out', 'size', 'excess', 'trunc'])
@pytest.mark.parametrize('length', [0, 2, 31 * 33 * 3])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
@pytest.mark.parametrize('module', [
    'zlib', 'bz2',
    pytest.param('blosc',
                 marks=pytest.mark.skipif(blosc is None,
                                          reason='import blosc')),
    pytest.param('lzma',
                 marks=pytest.mark.skipif(lzma is None,
                                          reason='import lzma')),
    pytest.param('zstd',
                 marks=pytest.mark.skipif(zstd is None,
                                          reason='import zstd')),
    pytest.param('lzf',
                 marks=pytest.mark.skipif(lzf is None,
                                          reason='import lzf')),
    pytest.param('lz4',
                 marks=pytest.mark.skipif(lz4 is None,
                                          reason='import lz4')),
    pytest.param('lz4h',
                 marks=pytest.mark.skipif(lz4 is None,
                                          reason='import lz4')),
    pytest.param('bitshuffle',
                 marks=pytest.mark.skipif(bitshuffle is None,
                                          reason='import bitshuffle')),
    pytest.param('brotli',
                 marks=pytest.mark.skipif(brotli is None,
                                          reason='import brotli')),
    pytest.param('zopfli',
                 marks=pytest.mark.skipif(zopfli is None,
                                          reason='import zopfli')),
    pytest.param('snappy',
                 marks=pytest.mark.skipif(snappy is None,
                                          reason='import snappy')),
])
def test_compressors(module, codec, output, length):
    """Test various non-image codecs."""
    if length:
        data = numpy.random.randint(255, size=length, dtype='uint8').tobytes()
    else:
        data = b''

    if module == 'blosc':
        encode = imagecodecs.blosc_encode
        decode = imagecodecs.blosc_decode
        level = 9
        encoded = blosc.compress(data, clevel=level)
    elif module == 'zlib':
        encode = imagecodecs.zlib_encode
        decode = imagecodecs.zlib_decode
        level = 5
        encoded = zlib.compress(data, level)
    elif module == 'lzma':
        encode = imagecodecs.lzma_encode
        decode = imagecodecs.lzma_decode
        level = 6
        encoded = lzma.compress(data)
    elif module == 'zstd':
        encode = imagecodecs.zstd_encode
        decode = imagecodecs.zstd_decode
        level = 5
        if length == 0:
            # bug in zstd.compress?
            encoded = encode(data, level)
        else:
            encoded = zstd.compress(data, level)
    elif module == 'lzf':
        encode = imagecodecs.lzf_encode
        decode = imagecodecs.lzf_decode
        level = 1
        encoded = lzf.compress(data, ((len(data) * 33) >> 5) + 1)
        if encoded is None:
            pytest.skip("lzf can't compress empty input")
    elif module == 'lz4':
        encode = imagecodecs.lz4_encode
        decode = imagecodecs.lz4_decode
        level = 1
        encoded = lz4.block.compress(data, store_size=False)
    elif module == 'lz4h':
        def encode(*args, **kwargs):
            return imagecodecs.lz4_encode(*args, header=True, **kwargs)

        def decode(*args, **kwargs):
            return imagecodecs.lz4_decode(*args, header=True, **kwargs)

        level = 1
        encoded = lz4.block.compress(data, store_size=True)
    elif module == 'bz2':
        encode = imagecodecs.bz2_encode
        decode = imagecodecs.bz2_decode
        level = 9
        encoded = bz2.compress(data, compresslevel=level)
    elif module == 'bitshuffle':
        encode = imagecodecs.bitshuffle_encode
        decode = imagecodecs.bitshuffle_decode
        level = 0
        encoded = bitshuffle.bitshuffle(
            numpy.frombuffer(data, 'uint8')).tobytes()
    elif module == 'brotli':
        if codec == 'encode' and length == 0:
            # TODO: why?
            pytest.skip('python-brotli returns different valid results')
        encode = imagecodecs.brotli_encode
        decode = imagecodecs.brotli_decode
        level = 11
        encoded = brotli.compress(data)
    elif module == 'zopfli':
        encode = imagecodecs.zopfli_encode
        decode = imagecodecs.zopfli_decode
        level = 1
        c = zopfli.ZopfliCompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
        encoded = c.compress(data) + c.flush()
    elif module == 'snappy':
        encode = imagecodecs.snappy_encode
        decode = imagecodecs.snappy_decode
        level = 1
        encoded = snappy.compress(data)
    else:
        raise ValueError(module)

    if codec == 'encode':
        size = len(encoded)
        if output == 'new':
            assert encoded == encode(data, level)
        elif output == 'bytearray':
            ret = encode(data, level, out=bytearray)
            assert encoded == ret
        elif output == 'size':
            ret = encode(data, level, out=size)
            assert encoded == ret
        elif output == 'out':
            if module == 'zstd':
                out = bytearray(max(size, 64))
            # elif module == 'blosc':
            #     out = bytearray(max(size, 17))  # bug in blosc ?
            elif module == 'lzf':
                out = bytearray(size + 1)  # bug in liblzf ?
            else:
                out = bytearray(size)
            ret = encode(data, level, out=out)
            assert encoded == out[:size]
            assert encoded == ret
        elif output == 'excess':
            out = bytearray(size + 1021)
            ret = encode(data, level, out=out)
            if module == 'blosc':
                # pytest.skip("blosc output depends on output size")
                assert data == decode(ret)
            else:
                assert ret == out[:size]
                assert encoded == ret
        elif output == 'trunc':
            size = max(0, size - 1)
            out = bytearray(size)
            if size == 0 and module == 'bitshuffle':
                encode(data, level, out=out) == b''
            else:
                with pytest.raises(RuntimeError):
                    encode(data, level, out=out)
        else:
            raise ValueError(output)
    elif codec == 'decode':
        size = len(data)
        if output == 'new':
            assert data == decode(encoded)
        elif output == 'bytearray':
            ret = decode(encoded, out=bytearray)
            assert data == ret
        elif output == 'size':
            ret = decode(encoded, out=size)
            assert data == ret
        elif output == 'out':
            out = bytearray(size)
            ret = decode(encoded, out=out)
            assert data == out
            assert data == ret
        elif output == 'excess':
            out = bytearray(size + 1021)
            ret = decode(encoded, out=out)
            assert data == out[:size]
            assert data == ret
        elif output == 'trunc':
            size = max(0, size - 1)
            out = bytearray(size)
            if length == 0 or module in ('bz2', 'lzma'):
                decode(encoded, out=out)
                assert data[:size] == out
            else:
                # most codecs don't support truncated output
                with pytest.raises(RuntimeError):
                    decode(encoded, out=out)
        else:
            raise ValueError(output)
    else:
        raise ValueError(codec)


@pytest.mark.skipif(not hasattr(imagecodecs, 'bitshuffle_decode'),
                    reason='bitshuffle codec missing')
@pytest.mark.parametrize('dtype', ['bytes', 'ndarray'])
@pytest.mark.parametrize('itemsize', [1, 2, 4, 8])
@pytest.mark.parametrize('blocksize', [0, 8, 64])
def test_bitshuffle_roundtrip(dtype, itemsize, blocksize):
    """Test Bitshuffle codec."""
    encode = imagecodecs.bitshuffle_encode
    decode = imagecodecs.bitshuffle_decode
    if dtype == 'bytes':
        data = numpy.random.randint(255, size=1024, dtype='uint8').tobytes()
    else:
        data = numpy.random.randint(255, size=1024, dtype='u%i' % itemsize)
        data.shape = 2, 4, 128
    encoded = encode(data, itemsize=itemsize, blocksize=blocksize)
    decoded = decode(encoded, itemsize=itemsize, blocksize=blocksize)
    if dtype == 'bytes':
        assert data == decoded
    else:
        assert_array_equal(data, decoded)


@pytest.mark.skipif(not hasattr(imagecodecs, 'blosc_decode'),
                    reason='blosc codec missing')
@pytest.mark.parametrize('numthreads', [1, 6])
@pytest.mark.parametrize('level', [None, 1])
@pytest.mark.parametrize('shuffle', ['noshuffle', 'shuffle', 'bitshuffle'])
@pytest.mark.parametrize('compressor', ['blosclz', 'lz4', 'lz4hc', 'snappy',
                                        'zlib', 'zstd'])
def test_blosc_roundtrip(compressor, shuffle, level, numthreads):
    """Test Blosc codec."""
    encode = imagecodecs.blosc_encode
    decode = imagecodecs.blosc_decode
    data = numpy.random.randint(255, size=2021, dtype='uint8').tobytes()
    encoded = encode(data, level=level, compressor=compressor,
                     shuffle=shuffle, numthreads=numthreads)
    decoded = decode(encoded, numthreads=numthreads)
    assert data == decoded


# test data from libaec https://gitlab.dkrz.de/k202009/libaec/tree/master/data
AEC_TEST_DIR = osp.join(TEST_DIR, 'libaec/121B2TestData')

AEC_TEST_OPTIONS = list(
    osp.split(f)[-1][5:-3] for f in glob.glob(osp.join(
        AEC_TEST_DIR, 'AllOptions', '*.rz')))

AEC_TEST_EXTENDED = list(
    osp.split(f)[-1][:-3] for f in glob.glob(osp.join(
        AEC_TEST_DIR, 'ExtendedParameters', '*.rz')))


@pytest.mark.skipif(not hasattr(imagecodecs, 'aec_decode'),
                    reason='aec codec missing')
@pytest.mark.parametrize('dtype', ['bytes', 'numpy'])
@pytest.mark.parametrize('name', AEC_TEST_EXTENDED)
def test_aec_extended(name, dtype):
    """Test AEC codec with libaec ExtendedParameters."""
    if (
        name == 'sar32bit.j16.r256' and
        not (IS_CG or os.environ.get('AEC_TEST_EXTENDED', False))
    ):
        pytest.skip('aec extension not built with ENABLE_RSI_PADDING')

    encode = imagecodecs.aec_encode
    decode = imagecodecs.aec_decode

    size = 512 * 512 * 4
    bitspersample = 32
    flags = imagecodecs.AEC_DATA_PREPROCESS | imagecodecs.AEC_PAD_RSI

    matches = re.search(r'j(\d+)\.r(\d+)', name).groups()
    blocksize = int(matches[0])
    rsi = int(matches[1])

    filename = osp.join(AEC_TEST_DIR, 'ExtendedParameters', '%s.rz' % name)
    with open(filename, 'rb') as fh:
        rz = fh.read()

    filename = osp.join(AEC_TEST_DIR, 'ExtendedParameters',
                        '%s.dat' % name.split('.')[0])
    if dtype == 'bytes':
        with open(filename, 'rb') as fh:
            dat = fh.read()
        out = size
    else:
        dat = numpy.fromfile(filename, 'uint32').reshape(512, 512)
        out = numpy.empty_like(dat)

    # decode
    decoded = decode(rz, bitspersample=bitspersample, flags=flags,
                     blocksize=blocksize, rsi=rsi, out=out)
    if dtype == 'bytes':
        assert decoded == dat
    else:
        pass

    # roundtrip
    if dtype == 'bytes':
        encoded = encode(dat, bitspersample=bitspersample, flags=flags,
                         blocksize=blocksize, rsi=rsi)
        # fails with AEC_DATA_ERROR if libaec wasn't built with libaec.diff
        decoded = decode(encoded, bitspersample=bitspersample, flags=flags,
                         blocksize=blocksize, rsi=rsi, out=size)
        assert decoded == dat
    else:
        encoded = encode(dat, flags=flags, blocksize=blocksize, rsi=rsi)
        # fails with AEC_DATA_ERROR if libaec wasn't built with libaec.diff
        decoded = decode(encoded, flags=flags, blocksize=blocksize, rsi=rsi,
                         out=out)
        assert_array_equal(decoded, out)


@pytest.mark.skipif(not hasattr(imagecodecs, 'aec_decode'),
                    reason='aec codec missing')
@pytest.mark.parametrize('name', AEC_TEST_OPTIONS)
def test_aec_options(name):
    """Test AEC codec with libaec 121B2TestData."""
    encode = imagecodecs.aec_encode
    decode = imagecodecs.aec_decode

    rsi = 128
    blocksize = 16
    flags = imagecodecs.AEC_DATA_PREPROCESS
    if 'restricted' in name:
        flags |= imagecodecs.AEC_RESTRICTED
    matches = re.search(r'p(\d+)n(\d+)', name).groups()
    size = int(matches[0])
    bitspersample = int(matches[1])

    if bitspersample > 8:
        size *= 2
    if bitspersample > 16:
        size *= 2

    filename = osp.join(AEC_TEST_DIR, 'AllOptions', 'test_%s.rz' % name)
    with open(filename, 'rb') as fh:
        rz = fh.read()

    filename = filename.replace('.rz', '.dat'
                                ).replace('-basic', ''
                                          ).replace('-restricted', '')
    with open(filename, 'rb') as fh:
        dat = fh.read()
    out = size

    # decode
    decoded = decode(rz, bitspersample=bitspersample, flags=flags,
                     blocksize=blocksize, rsi=rsi, out=out)
    assert decoded == dat

    # roundtrip
    encoded = encode(dat, bitspersample=bitspersample, flags=flags,
                     blocksize=blocksize, rsi=rsi)
    decoded = decode(encoded, bitspersample=bitspersample, flags=flags,
                     blocksize=blocksize, rsi=rsi, out=out)
    assert decoded == dat


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg_encode'),
                    reason='jpeg codecs missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('smoothing', [0, 25])
@pytest.mark.parametrize('subsampling', ['444', '422', '420', '411', '440'])
@pytest.mark.parametrize('itype', ['rgb', 'rgba', 'gray'])
@pytest.mark.parametrize('codec', ['jpeg8', 'jpeg12'])
def test_jpeg_encode(codec, itype, subsampling, smoothing, optimize):
    """Test various JPEG encode options."""
    # general and default options are tested in test_image_roundtrips
    if codec == 'jpeg8':
        dtype = 'uint8'
        decode = imagecodecs.jpeg8_decode
        encode = imagecodecs.jpeg8_encode
        atol = 24
    elif codec == 'jpeg12':
        if _jpeg12 is None:
            pytest.skip('_jpeg12 module missing')
        if not optimize:
            pytest.skip('jpeg12 fails without optimize')
        dtype = 'uint16'
        decode = imagecodecs.jpeg12_decode
        encode = imagecodecs.jpeg12_encode
        atol = 24 * 16
    else:
        raise ValueError(codec)

    dtype = numpy.dtype(dtype)
    data = image_data(itype, dtype)
    data = data[:32, :16].copy()  # make divisable by subsamples

    encoded = encode(data, level=95, subsampling=subsampling,
                     smoothing=smoothing, optimize=optimize)
    decoded = decode(encoded)

    if itype == 'gray':
        decoded = decoded.reshape(data.shape)

    assert_allclose(data, decoded, atol=atol)


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg8_decode'),
                    reason='jpeg8 codec missing')
@pytest.mark.parametrize('output', ['new', 'out'])
def test_jpeg8_decode(output):
    """Test JPEG 8-bit decoder with separate tables."""
    decode = imagecodecs.jpeg8_decode
    data = readfile('bytes.jpeg8.bin')
    tables = readfile('bytes.jpeg8_tables.bin')

    if output == 'new':
        decoded = decode(data, tables)
    elif output == 'out':
        decoded = numpy.empty_like(BYTESIMG)
        decode(data, tables, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(BYTESIMG.size * BYTESIMG.itemsize)
        decoded = decode(data, out=decoded)
    assert_array_equal(BYTESIMG, decoded)


@pytest.mark.skipif(_jpeg12 is None, reason='_jpeg12 module missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpeg12_decode(output):
    """Test JPEG 12-bit decoder with separate tables."""
    decode = imagecodecs.jpeg12_decode
    data = readfile('words.jpeg12.bin')
    tables = readfile('words.jpeg12_tables.bin')

    if output == 'new':
        decoded = decode(data, tables)
    elif output == 'out':
        decoded = numpy.empty_like(WORDSIMG)
        decode(data, tables, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(WORDSIMG.size * WORDSIMG.itemsize)
        decoded = decode(data, out=decoded)

    assert numpy.max(numpy.abs(WORDSIMG.astype('int32') -
                               decoded.astype('int32'))) < 2


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpegsof3_decode'),
                    reason='jpegsof3 codec missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('fname', ['gray8.sof3.jpg', 'gray16.sof3.jpg'])
def test_jpegsof3(fname, output):
    """Test JPEG SOF3 decoder with 8 and 16-bit images."""
    decode = imagecodecs.jpegsof3_decode

    shape = 535, 800
    if fname == 'gray8.sof3.jpg':
        dtype = 'uint8'
        value = 75
    elif fname == 'gray16.sof3.jpg':
        dtype = 'uint16'
        value = 19275

    data = readfile(fname)

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(535 * 800 * numpy.dtype(dtype).itemsize)
        decoded = decode(data, out=decoded)

    assert decoded.shape == shape
    assert decoded.dtype == dtype
    assert decoded[500, 600] == value


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpegxr_decode'),
                    reason='jpegxr codec missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpegxr_decode(output):
    """Test JPEG XR decoder with RGBA32 image."""
    decode = imagecodecs.jpegxr_decode
    image = readfile('rgba32.jxr.bin')
    image = numpy.frombuffer(image, dtype='uint8').reshape(100, 100, -1)
    data = readfile('rgba32.jxr')

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.empty_like(image)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(image.size * image.itemsize)
        decoded = decode(data, out=decoded)
    assert_array_equal(image, decoded)


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg2k_decode'),
                    reason='jpeg2k codec missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpeg2k_int8_4bit(output):
    """Test JPEG 2000 decoder with int8, 4-bit image."""
    decode = imagecodecs.jpeg2k_decode
    data = readfile('int8_4bit.j2k')
    dtype = 'int8'
    shape = 256, 256

    if output == 'new':
        decoded = decode(data, verbose=2)
    elif output == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[0, 0] == -6
    assert decoded[-1, -1] == 2


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg2k_decode'),
                    reason='jpeg2k codec missing')
def test_jpeg2k_ycbc():
    """Test JPEG 2000 decoder with subsampling."""
    decode = imagecodecs.jpeg2k_decode
    data = readfile('ycbc.j2k')
    decoded = decode(data, verbose=2)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (256, 256, 3)
    assert tuple(decoded[0, 0]) == (243, 243, 240)
    assert tuple(decoded[-1, -1]) == (0, 0, 0)


@pytest.mark.skipif(_jpegls is None, reason='_jpegls module missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpegls_decode(output):
    """Test JPEG LS decoder with RGBA32 image."""
    decode = imagecodecs.jpegls_decode
    data = readfile('rgba.u1.jls')
    dtype = 'uint8'
    shape = 32, 31, 4

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 97
    assert decoded[-1, -1, -1] == 63


@pytest.mark.skipif(_jpegxl is None, reason='_jpegxl module missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpegxl_decode(output):
    """Test JPEG XL decoder with RGBA32 image."""
    decode = imagecodecs.jpegxl_decode
    data = readfile('rgba.u1.jxl')
    dtype = 'uint8'
    shape = 32, 31, 4

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 100
    assert decoded[-1, -1, -1] == 81


@pytest.mark.skipif(_jpegxl is None, reason='_jpegxl module missing')
def test_jpegxl_encode_jpeg():
    """Test JPEG XL encoder with JPEG input."""
    encode = imagecodecs.jpegxl_encode
    decode = imagecodecs.jpegxl_decode
    jpg = readfile('rgba.u1.jpg')
    jxl = readfile('rgba.u1.jxl')

    encoded = encode(jpg)
    assert encoded == jxl

    decoded = decode(encoded)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (32, 31, 4)
    assert decoded[25, 25, 1] == 100
    assert decoded[-1, -1, -1] == 81


@pytest.mark.skipif(not hasattr(imagecodecs, 'webp_decode'),
                    reason='webp codec missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_webp_decode(output):
    """Test WebpP  decoder with RGBA32 image."""
    decode = imagecodecs.webp_decode
    data = readfile('rgba.u1.webp')
    dtype = 'uint8'
    shape = 32, 31, 4

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 94  # lossy
    assert decoded[-1, -1, -1] == 63


@pytest.mark.skipif(_zfp is None, reason='_zfp module missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('execution', [None, 'omp'])
@pytest.mark.parametrize('mode', [(None, None), ('p', None)])  # ('r', 24)
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])  # 'view',
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgba', 'view', 'gray', 'line'])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_zfp(dtype, itype, enout, deout, mode, execution):
    """Test ZFP codecs."""
    if execution == 'omp' and os.environ.get('SKIP_OMP', False):
        pytest.skip('omp test skip because of enviroment variable')
    decode = imagecodecs.zfp_decode
    encode = imagecodecs.zfp_encode
    mode, level = mode
    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape

    kwargs = dict(mode=mode, level=level, execution=execution)
    encoded = encode(data, **kwargs)

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.empty(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(encoded, out=decoded)
    elif deout == 'view':
        temp = numpy.empty((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2:2 + shape[0], 3:3 + shape[1], :]
        decode(encoded, out=decoded)
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    if dtype.char == 'f':
        atol = 1e-6
    else:
        atol = 20
    assert_allclose(data, decoded, atol=atol, rtol=0)


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpegxr_decode'),
                    reason='jpegxr codec missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('level', [None, 90, 0.4])
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])  # 'view',
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', [
    'gray uint8', 'gray uint16', 'gray float16', 'gray float32',
    'rgb uint8', 'rgb uint16', 'rgb float16', 'rgb float32',
    'rgba uint8', 'rgba uint16', 'rgba float16', 'rgba float32',
    'channels uint8', 'channelsa uint8', 'channels uint16', 'channelsa uint16',
    'cmyk uint8', 'cmyka uint8'])
def test_jpegxr(itype, enout, deout, level):
    """Test JPEG XR codecs."""
    decode = imagecodecs.jpegxr_decode
    encode = imagecodecs.jpegxr_encode
    itype, dtype = itype.split()
    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape

    kwargs = dict(level=level)
    if itype.startswith('cmyk'):
        kwargs['photometric'] = 'cmyk'
    if itype.endswith('a'):
        kwargs['hasalpha'] = True
    print(data.shape, data.dtype, data.strides)
    encoded = encode(data, **kwargs)

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.empty(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'view':
        temp = numpy.empty((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2:2 + shape[0], 3:3 + shape[1], :]
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    if itype == 'gray':
        decoded = decoded.reshape(shape)

    if level is None:
        atol = 0.00001 if dtype.kind == 'f' else 1
    if level == 90:
        atol = 0.005 if dtype.kind == 'f' else 8 if dtype == 'uint8' else 12
    else:
        atol = 0.1 if dtype.kind == 'f' else 64 if dtype == 'uint8' else 700
    assert_allclose(data, decoded, atol=atol, rtol=0)


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg_decode'),
                    reason='image codecs missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('level', [None, 5, -1])
@pytest.mark.parametrize('deout', ['new', 'out', 'view', 'bytearray'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgb', 'rgba', 'view', 'gray', 'graya'])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('codec', [
    'webp', 'png', 'jpeg8', 'jpeg12', 'jpegls', 'jpegxl', 'jpegxr', 'jpeg2k'])
def test_image_roundtrips(codec, dtype, itype, enout, deout, level):
    """Test various image codecs."""
    if codec == 'jpeg8':
        if itype in ('view', 'graya') or deout == 'view' or dtype == 'uint16':
            pytest.skip("jpeg8 doesn't support these cases")
        decode = imagecodecs.jpeg8_decode
        encode = imagecodecs.jpeg8_encode
        atol = 24
        if level:
            level += 95
    elif codec == 'jpeg12':
        if _jpeg12 is None:
            pytest.skip('_jpeg12 module missing')
        if itype in ('view', 'graya') or deout == 'view' or dtype == 'uint8':
            pytest.skip("jpeg12 doesn't support these cases")
        decode = imagecodecs.jpeg12_decode
        encode = imagecodecs.jpeg12_encode
        atol = 24 * 16
        if level:
            level += 95
    elif codec == 'jpegls':
        if _jpegls is None:
            pytest.skip('_jpegls module missing')
        if itype in ('view', 'graya') or deout == 'view':
            pytest.skip("jpegls doesn't support these cases")
        decode = imagecodecs.jpegls_decode
        encode = imagecodecs.jpegls_encode
    elif codec == 'webp':
        decode = imagecodecs.webp_decode
        encode = imagecodecs.webp_encode
        if dtype != 'uint8' or itype.startswith('gray'):
            pytest.skip("webp doesn't support these cases")
    elif codec == 'png':
        decode = imagecodecs.png_decode
        encode = imagecodecs.png_encode
    elif codec == 'jpeg2k':
        if itype == 'view' or deout == 'view':
            pytest.skip("jpeg2k doesn't support these cases")
        decode = imagecodecs.jpeg2k_decode
        encode = imagecodecs.jpeg2k_encode
        if level:
            level += 95
    elif codec == 'jpegxl':
        if _jpegxl is None:
            pytest.skip('_jpegxl module missing')
        if itype in ('view', 'graya') or deout == 'view' or dtype == 'uint16':
            pytest.skip("jpegxl doesn't support these cases")
        decode = imagecodecs.jpegxl_decode
        encode = imagecodecs.jpegxl_encode
        atol = 24
        if level:
            level += 95
    elif codec == 'jpegxr':
        if itype == 'graya' or deout == 'view':
            pytest.skip("jpegxr doesn't support these cases")
        decode = imagecodecs.jpegxr_decode
        encode = imagecodecs.jpegxr_encode
        atol = 10
        if level:
            level = (level + 95) / 100
    else:
        raise ValueError(codec)

    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape

    if enout == 'new':
        encoded = encode(data, level=level)
    elif enout == 'out':
        encoded = numpy.empty(2 * shape[0] * shape[1] * shape[2] * itemsize,
                              'uint8')
        ret = encode(data, level=level, out=encoded)
        if codec == 'jpegxl':
            # Brunsli doesn't like extra bytes
            encoded = encoded[:len(ret)]
    elif enout == 'bytearray':
        encoded = bytearray(2 * shape[0] * shape[1] * shape[2] * itemsize)
        ret = encode(data, level=level, out=encoded)
        if codec == 'jpegxl':
            # Brunsli doesn't like extra bytes
            encoded = encoded[:len(ret)]

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.empty(shape, dtype)
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'view':
        temp = numpy.empty((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2:2 + shape[0], 3:3 + shape[1], :]
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    if itype == 'gray':
        decoded = decoded.reshape(shape)

    if codec == 'webp' and (level != -1 or itype == 'rgba'):
        # RGBA roundtip doesn't work for A=0
        assert_allclose(data, decoded, atol=255)
    elif codec in ('jpeg8', 'jpeg12', 'jpegxl', 'jpegxr'):
        assert_allclose(data, decoded, atol=atol)
    elif codec == 'jpegls' and level == 5:
        assert_allclose(data, decoded, atol=6)
    else:
        assert_array_equal(data, decoded, verbose=True)


@pytest.mark.skipif(not hasattr(imagecodecs, 'png_decode'),
                    reason='png codec missing')
def test_png_rgba_palette():
    """Test decoding indexed PNG with transparency."""
    png = readfile('rgba.u1.pal.png')
    image = imagecodecs.png_decode(png)
    assert tuple(image[6, 15]) == (255, 255, 255, 0)
    assert tuple(image[6, 16]) == (141, 37, 52, 255)


@pytest.mark.skipif(not hasattr(imagecodecs, 'lzma_decode'),
                    reason='codecs missing')
@pytest.mark.skipif(tifffile is None, reason='tifffile module missing')
@pytest.mark.filterwarnings('ignore:Possible precision loss')
@pytest.mark.parametrize('dtype', ['u1', 'u2', 'f4'])
@pytest.mark.parametrize('codec', ['deflate', 'lzma', 'zstd', 'packbits'])
def test_tifffile(dtype, codec):
    """Test tifffile compression."""
    if codec == 'packbits' and dtype != 'u1':
        pytest.skip('dtype not supported')

    data = image_data('rgb', dtype)
    with io.BytesIO() as fh:
        tifffile.imwrite(fh, data, compress=codec)
        fh.seek(0)
        image = tifffile.imread(fh)
    assert_array_equal(data, image, verbose=True)


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpegxr_decode'),
                    reason='jpegxr codec missing')
@pytest.mark.skipif(czifile is None, reason='czifile module missing')
def test_czifile():
    """Test JpegXR compressed CZI file."""
    fname = datafiles('jpegxr.czi')
    if not osp.exists(fname):
        pytest.skip('large file not included with source distribution')

    with czifile.CziFile(fname) as czi:
        assert czi.shape == (1, 1, 15, 404, 356, 1)
        assert czi.axes == 'BCZYX0'
        # verify data
        data = czi.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1, 1, 15, 404, 356, 1)
        assert data.dtype == 'uint16'
        assert data[0, 0, 14, 256, 146, 0] == 38086


@pytest.mark.skipif(not hasattr(imagecodecs, 'jpeg8_decode'),
                    reason='jpeg8 codec missing')
@pytest.mark.skipif(IS_32BIT, reason='data too large for 32-bit')
def test_jpeg8_large():
    """Test JPEG 8-bit decoder with dimensions > 65000."""
    decode = imagecodecs.jpeg8_decode

    try:
        data = readfile('33792x79872.jpg')
    except IOError:
        pytest.skip('large file not included with source distribution')

    # fails if libjpeg-turbo wasn't compiled with libjpeg-turbo.diff
    # Jpeg8Error: Empty JPEG image (DNL not supported)
    decoded = decode(data, shape=(33792, 79872))
    assert decoded.shape == (33792, 79872, 3)
    assert decoded.dtype == 'uint8'
    assert tuple(decoded[33791, 79871]) == (204, 195, 180)


###############################################################################

class TempFileName():
    """Temporary file name context manager."""
    def __init__(self, name=None, suffix='', remove=True):
        self.remove = bool(remove)
        if not name:
            self.name = tempfile.NamedTemporaryFile(prefix='test_',
                                                    suffix=suffix).name
        else:
            self.name = osp.join(tempfile.gettempdir(),
                                 'test_%s%s' % (name, suffix))

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            try:
                os.remove(self.name)
            except Exception:
                pass


def datafiles(pathname, base=None):
    """Return path to data file(s)."""
    if base is None:
        base = osp.dirname(__file__)
    path = osp.join(base, *pathname.split('/'))
    if any(i in path for i in '*?'):
        return glob.glob(path)
    return path


def readfile(fname):
    """Return content of data file."""
    with open(datafiles(fname), 'rb') as fh:
        return fh.read()


def image_data(itype, dtype):
    """Return test image array."""
    if itype in ('rgb', 'view'):
        data = DATA[..., [0, 2, 4]]
    elif itype == 'rgba':
        data = DATA[..., [0, 2, 4, -1]]
    elif itype == 'cmyk':
        data = DATA[..., [0, 2, 4, 6]]
    elif itype == 'cmyka':
        data = DATA[..., [0, 2, 4, 6, -1]]
    elif itype == 'gray':
        data = DATA[..., 0:1]
    elif itype == 'graya':
        data = DATA[..., [0, -1]]
    elif itype == 'rrggbbaa':
        data = numpy.moveaxis(DATA[..., [0, 2, 4, -1]], -1, 0)
    elif itype == 'rrggbb':
        data = numpy.moveaxis(DATA[..., [0, 2, 4]], -1, 0)
    elif itype == 'channels':
        data = DATA[..., :-1]
    elif itype == 'channelsa':
        data = DATA[..., :]
    elif itype == 'line':
        data = DATA[0:1, :, 0:1]
    else:
        raise ValueError('itype not found')

    data = data.copy()

    dtype = numpy.dtype(dtype)
    if dtype.kind in 'iu':
        iinfo = numpy.iinfo(dtype)
        if dtype.kind == 'u':
            data *= iinfo.max + 1
        else:
            data *= (iinfo.max - iinfo.max) / 2
            data -= 1.0 / 2.0
        data = numpy.rint(data)
        data = numpy.clip(data, iinfo.min, iinfo.max)
    elif dtype.kind != 'f':
        raise NotImplementedError('dtype not supported')

    data = data.astype(dtype)

    if dtype == 'uint16':
        # 12-bit
        data //= 16

    if itype == 'view':
        shape = data.shape
        temp = numpy.empty((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        temp[2:2 + shape[0], 3:3 + shape[1], :] = data
        data = temp[2:2 + shape[0], 3:3 + shape[1], :]

    return data


DATA = numpy.load(datafiles('testdata.npy'))  # (32, 31, 9) float64
BYTES = readfile('bytes.bin')
BYTESIMG = numpy.frombuffer(BYTES, 'uint8').reshape(16, 16)
WORDS = readfile('words.bin')
WORDSIMG = numpy.frombuffer(WORDS, 'uint16').reshape(36, 36, 3)

if __name__ == '__main__':
    import warnings
    # warnings.simplefilter('always')  # noqa
    warnings.filterwarnings('ignore', category=ImportWarning)  # noqa
    argv = sys.argv
    argv.append('-vv')
    pytest.main(argv)
