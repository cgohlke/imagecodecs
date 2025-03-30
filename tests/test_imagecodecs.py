# test_imagecodecs.py

# Copyright (c) 2018-2025, Christoph Gohlke
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
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, TEST_DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False

"""Unittests for the imagecodecs package.

:Version: 2025.3.30

"""

from __future__ import annotations

import glob
import importlib
import io
import mmap
import os
import pathlib
import platform
import re
import sys
import tempfile
from typing import Any

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

try:
    import imagecodecs
    from imagecodecs import _imagecodecs
    from imagecodecs._imagecodecs import (
        bitshuffle,
        blosc,
        blosc2,
        brotli,
        bz2,
        czifile,
        liffile,
        lz4,
        lzf,
        lzfse,
        lzma,
        snappy,
        tifffile,
        zlib,
        zopfli,
        zstd,
    )
    from imagecodecs.imagecodecs import _add_codec, _extensions
except ImportError as _exc:
    pytest.exit(str(_exc))


try:
    import zarr
    from imagecodecs import numcodecs
except ImportError:
    SKIP_NUMCODECS = True
else:
    SKIP_NUMCODECS = False

    numcodecs.register_codecs()

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / 'data'

IS_32BIT = sys.maxsize < 2**32
IS_WIN = sys.platform == 'win32'
IS_MAC = sys.platform == 'darwin'
IS_AARCH64 = platform.machine() == 'aarch64'
IS_ARM64 = platform.machine() == 'ARM64'
IS_PYPY = 'pypy' in sys.version.lower()
# running on Windows development computer?
IS_CG = os.environ.get('COMPUTERNAME', '').startswith('CG-')
# running in cibuildwheel environment?
IS_CIBW = bool(os.environ.get('IMAGECODECS_CIBW', False))
SKIP_DEBUG = bool(os.environ.get('IMAGECODECS_DEBUG', False))

numpy.set_printoptions(suppress=True, precision=5)


###############################################################################


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert imagecodecs versions match docstrings."""
    ver = ':Version: ' + imagecodecs.__version__
    assert ver in __doc__
    assert ver in imagecodecs.__doc__


@pytest.mark.parametrize('name', _extensions())
def test_module_exist(name):
    """Assert extension modules are present."""
    name = name[1:]  # remove underscore
    try:
        exists = bool(importlib.import_module('._' + name, 'imagecodecs'))
    except ImportError:
        exists = False
    if exists:
        return
    if IS_CG:
        if IS_32BIT and name in {'heif', 'jetraw', 'sperr'}:
            pytest.xfail(f'imagecodecs._{name} may be missing')
        elif IS_ARM64 and name == 'jetraw':
            pytest.xfail(f'imagecodecs._{name} may be missing')
    elif IS_CIBW:
        if (
            (name == 'jpegxl' and (IS_MAC or IS_32BIT or IS_AARCH64))
            or (name == 'lzham' and IS_MAC)
            or name == 'mozjpeg'
            or name == 'heif'
            or name == 'jetraw'
            or name == 'jpegxs'
            or name == 'pcodec'
            # or name == 'nvjpeg'
            # or name == 'nvjpeg2k'
        ):
            pytest.xfail(f'imagecodecs._{name} may be missing')
    else:
        pytest.xfail(f'imagecodecs._{name} may be missing')
    assert exists, f'no module named imagecodecs._{name}'


@pytest.mark.parametrize(
    'name',
    [
        'bitshuffle',
        'blosc',
        'blosc2',
        'brotli',
        'czifile',
        'liffile',
        'lz4',
        'lzf',
        'liblzfse',
        # 'lzham',
        'lzma',
        'numcodecs',
        'snappy',
        'tifffile',
        'zopfli',
        'zstd',
        'zarr',
    ],
)
def test_dependency_exist(name):
    """Assert third-party Python packages are present."""
    mayfail = not IS_CG and not IS_CIBW
    if SKIP_NUMCODECS and IS_PYPY:
        mayfail = True
    elif name in {'blosc', 'blosc2', 'snappy', 'liffile'}:
        if IS_PYPY or not IS_CG:
            mayfail = True
    try:
        importlib.import_module(name)
    except ImportError:
        if mayfail:
            pytest.skip(f'{name} may be missing')
        raise


def test_module_attributes():
    """Test __module__ attributes are set to 'imagecodecs'."""
    assert imagecodecs.NoneError.__module__ == 'imagecodecs'
    assert imagecodecs.JPEG.__module__ == 'imagecodecs'
    assert imagecodecs.LZW.__module__ == 'imagecodecs'
    assert imagecodecs.jpeg_decode.__module__ == 'imagecodecs'
    assert imagecodecs.lzw_decode.__module__ == 'imagecodecs'


def test_version_functions():
    """Test imagecodecs version functions."""
    assert imagecodecs.version().startswith('imagecodecs')
    assert 'imagecodecs' in imagecodecs.version(dict)
    assert imagecodecs.version(tuple)[1].startswith('cython')
    assert _imagecodecs.version().startswith('imagecodecs.py')
    assert 'imagecodecs.py' in _imagecodecs.version(dict)


def test_stubs():
    """Test stub attributes for non-existing extension."""
    with pytest.raises(AttributeError):
        assert imagecodecs._STUB
    _add_codec('_stub')
    assert not imagecodecs._STUB  # typing: ignore
    assert not imagecodecs._STUB.available
    assert not imagecodecs._stub_check(b'')
    assert imagecodecs._stub_version() == '_stub n/a'
    with pytest.raises(imagecodecs.DelayedImportError):
        imagecodecs._STUB.attr
    with pytest.raises(imagecodecs.DelayedImportError):
        imagecodecs._stub_encode(b'')
    with pytest.raises(imagecodecs.DelayedImportError):
        imagecodecs._stub_decode(b'')
    with pytest.raises(imagecodecs.DelayedImportError):
        raise imagecodecs._stubError()


def test_dir():
    """Assert __dir__ contains delay-loaded attributes."""
    d = dir(imagecodecs)
    assert 'NONE' in d
    assert 'LZW' in d
    assert 'jxr_decode' in d


@pytest.mark.skipif(
    not imagecodecs.NUMPY.available, reason='Numpy codec missing'
)
@pytest.mark.parametrize(
    'codec', ['none', 'str', 'ext', 'codec', 'list', 'fail']
)
@pytest.mark.parametrize('filearg', ['str', 'pathlib', 'bytesio', 'bytes'])
def test_imread_imwrite(filearg, codec):
    """Test imread and imwrite functions."""
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
        elif encode in {None, 'fail'}:
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
    data = b'None'
    assert imagecodecs.none_encode(data) is data
    assert imagecodecs.none_decode(data) is data


@pytest.mark.skipif(
    not imagecodecs.BITORDER.available, reason='Bitorder missing'
)
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
    with pytest.raises(TypeError):
        decode(data, out=data)
    data = bytearray(data)
    decode(data, out=data)
    assert data == reverse
    # bytes range
    assert BYTES == decode(readfile('bytes.bitorder.bin'))


@pytest.mark.skipif(
    not imagecodecs.BITORDER.available, reason='Bitorder missing'
)
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
    data = numpy.array(
        [
            [1, 666, 1431655765, 62],
            [2, 667, 2863311530, 32],
            [3, 668, 1431655765, 30],
        ],
        dtype='uint32',
    )
    reverse = numpy.array(
        [
            [1, 666, 1431655765, 62],
            [2, 16601, 1431655765, 32],
            [3, 16441, 2863311530, 30],
        ],
        dtype='uint32',
    )
    assert_array_equal(decode(data[1:, 1:3]), reverse[1:, 1:3])
    # array view inplace
    decode(data[1:, 1:3], out=data[1:, 1:3])
    assert_array_equal(data, reverse)


@pytest.mark.skipif(
    not imagecodecs.PACKINTS.available, reason='Packints missing'
)
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
    ([], b''),
    ([0] * 1, b'\x00\x00'),  # literal
    ([0] * 2, b'\xff\x00'),  # replicate
    ([0] * 3, b'\xfe\x00'),
    ([0] * 64, b'\xc1\x00'),
    ([0] * 127, b'\x82\x00'),
    ([0] * 128, b'\x81\x00'),  # max replicate
    ([0] * 129, b'\x81\x00\x00\x00'),
    ([0] * 130, b'\x81\x00\xff\x00'),
    ([0] * 128 * 3, b'\x81\x00' * 3),
    ([255] * 1, b'\x00\xff'),  # literal
    ([255] * 2, b'\xff\xff'),  # replicate
    ([0, 1], b'\x01\x00\x01'),
    ([0, 1, 2], b'\x02\x00\x01\x02'),
    ([0, 1] * 32, b'\x3f' + b'\x00\x01' * 32),
    ([0, 1] * 63 + [2], b'\x7e' + b'\x00\x01' * 63 + b'\x02'),
    ([0, 1] * 64, b'\x7f' + b'\x00\x01' * 64),  # max literal
    ([0, 1] * 64 + [2], b'\x7f' + b'\x00\x01' * 64 + b'\x00\x02'),
    ([0, 1] * 64 * 5, (b'\x7f' + b'\x00\x01' * 64) * 5),
    ([0, 1, 1], b'\x00\x00\xff\x01'),  # or b'\x02\x00\x01\x01'
    ([0] + [1] * 128, b'\x00\x00\x81\x01'),  # or b'\x01\x00\x01\x82\x01'
    ([0] + [1] * 129, b'\x00\x00\x81\x01\x00\x01'),  # b'\x01\x00\x01\x81\x01'
    ([0, 1] * 64 + [2] * 2, b'\x7f' + b'\x00\x01' * 64 + b'\xff\x02'),
    ([0, 1] * 64 + [2] * 128, b'\x7f' + b'\x00\x01' * 64 + b'\x81\x02'),
    ([0, 0, 1], b'\x02\x00\x00\x01'),  # or b'\xff\x00\x00\x01'
    ([0, 0] + [1, 2] * 64, b'\xff\x00\x7f' + b'\x01\x02' * 64),
    ([0] * 128 + [1], b'\x81\x00\x00\x01'),
    ([0] * 128 + [1, 2] * 64, b'\x81\x00\x7f' + b'\x01\x02' * 64),
    (
        # one literal run is shortest
        [0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8],
        b'\x0c\x00\x01\x02\x03\x03\x03\x04\x05\x06\x07\x07\x07\x08',
    ),
    (
        # use replicate runs
        [0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8],
        b'\x02\x00\x01\x02\xfe\x03\x02\x04\x05\x06\xfd\x07\x00\x08',
    ),
    (
        # skip short replicate run
        [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8],
        b'\x07\x00\x01\x02\x03\x03\x04\x05\x06\xfd\x07\x00\x08',
        # not b'\x0c\x00\x01\x02\x03\x03\x04\x05\x06\x07\x07\x07\x07\x08'
    ),
    (
        b'\xaa\xaa\xaa\x80\x00\x2a\xaa\xaa\xaa\xaa\x80\x00'
        b'\x2a\x22\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa',
        b'\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa',
    ),
]


@pytest.mark.skipif(
    not imagecodecs.PACKBITS.available, reason='Packbits missing'
)
@pytest.mark.parametrize('data', range(len(PACKBITS_DATA)))
@pytest.mark.parametrize('codec', ['encode', 'decode'])
def test_packbits(codec, data):
    """Test PackBits codec."""
    encode = imagecodecs.packbits_encode
    decode = imagecodecs.packbits_decode
    uncompressed, compressed = PACKBITS_DATA[data]
    uncompressed = bytes(uncompressed)
    if codec == 'decode':
        assert decode(compressed) == uncompressed
    elif codec == 'encode':
        assert len(encode(uncompressed)) <= len(compressed)
        assert encode(uncompressed) == compressed


@pytest.mark.parametrize('data', range(len(PACKBITS_DATA)))
def test_packbits_py(data):
    """Test pure Python PackBits decoder."""
    uncompressed, compressed = PACKBITS_DATA[data]
    uncompressed = bytes(uncompressed)
    assert _imagecodecs.packbits_decode(compressed) == uncompressed


@pytest.mark.skipif(
    not imagecodecs.PACKBITS.available, reason='Packbits missing'
)
def test_packbits_nop():
    """Test PackBits decoding empty data."""
    decode = imagecodecs.packbits_decode
    assert decode(b'\x80') == b''
    assert decode(b'\x80\x80') == b''


@pytest.mark.skipif(
    not imagecodecs.PACKBITS.available, reason='Packbits missing'
)
@pytest.mark.parametrize('output', [None, 'array'])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
def test_packbits_array(codec, output, dtype):
    """Test PackBits codec with arrays."""
    dtype = numpy.dtype(dtype)
    encode = imagecodecs.packbits_encode
    decode = imagecodecs.packbits_decode
    uncompressed, compressed = PACKBITS_DATA[-1]
    shape = (2, 7, len(uncompressed) // dtype.itemsize)
    data = numpy.zeros(shape, dtype=dtype)
    data[..., :] = numpy.frombuffer(uncompressed, dtype=dtype)
    compressed = compressed * (shape[0] * shape[1])
    if codec == 'encode':
        if output == 'array':
            out = numpy.zeros(data.nbytes, 'uint8')
            assert_array_equal(
                encode(data, out=out),
                numpy.frombuffer(compressed, dtype=dtype).view('uint8'),
            )
        else:
            assert encode(data) == compressed
    else:
        if output == 'array':
            out = numpy.zeros(data.nbytes, 'uint8')
            assert_array_equal(
                decode(compressed, out=out), data.flatten().view('uint8')
            )
        else:
            assert decode(compressed) == data.tobytes()


@pytest.mark.skipif(
    not imagecodecs.PACKBITS.available, reason='Packbits missing'
)
def test_packbits_encode_axis():
    """Test PackBits encoder with samples."""
    data = numpy.zeros((97, 67, 3), dtype=numpy.int16)
    data[10:20, 11:21, 1] = -1
    encoded = imagecodecs.packbits_encode(data, axis=-1)  # very inefficient
    assert len(encoded) > 10000
    assert imagecodecs.packbits_decode(encoded) == data.tobytes()
    encoded = imagecodecs.packbits_encode(data, axis=-2)
    assert len(encoded) < 1200
    assert imagecodecs.packbits_decode(encoded) == data.tobytes()


@pytest.mark.skipif(
    not imagecodecs.PACKBITS.available, reason='Packbits missing'
)
def test_packbits_padbyte():
    """Test PackBits decoding with pad byte."""
    # https://github.com/cgohlke/imagecodecs/issues/86
    encode = imagecodecs.packbits_encode
    decode = imagecodecs.packbits_decode

    data = numpy.array([[121, 121], [27, 63]], dtype=numpy.uint8)
    encoded = encode(data)
    assert encoded == b'\xffy\x01\x1b?'
    assert decode(encoded) == data.tobytes()
    assert decode(encoded + b'\x00') == data.tobytes()
    with pytest.raises(imagecodecs.PackbitsError):
        assert decode(encoded + b'\x01') == data.tobytes()


DICOMRLE_DATA = [
    (
        (
            b'\x01\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x05\x00@\x80\xa0\xc0\xff'
        ),
        numpy.array([0, 64, 128, 160, 192, 255], dtype='u1'),
    ),
    (
        (
            b'\x03\x00\x00\x00@\x00\x00\x00G\x00\x00\x00N\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x05\x00@\x80\xa0\xc0\xff\x05\xff\xc0\x80@\x00\xff'
            b'\x05\x01@\x80\xa0\xc0\xfe'
        ),
        numpy.array(
            [
                [0, 64, 128, 160, 192, 255],
                [255, 192, 128, 64, 0, 255],
                [1, 64, 128, 160, 192, 254],
            ],
            dtype='u1',
        ),
    ),
    (
        (
            b'\x02\x00\x00\x00@\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x05\x00\x00\x01\x00\xff\xff\x05\x00\x01\x00\xff\x00\xff'
        ),
        numpy.array([0, 1, 256, 255, 65280, 65535], dtype='u2'),
    ),
    (
        (
            b'\x06\x00\x00\x00@\x00\x00\x00G\x00\x00\x00N\x00\x00\x00U\x00'
            b'\x00\x00\\\x00\x00\x00c\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x05\x00\x00\x01\x00\xff\xff\x05\x00\x01\x00\xff\x00\xff\x05'
            b'\xff\x00\x01\x00\xff\x00\x05\xff\x01\x00\xff\x00\x00\x05\x00'
            b'\x00\x01\x00\xff\xff\x05\x01\x01\x00\xff\x00\xfe'
        ),
        numpy.array(
            [
                [0, 1, 256, 255, 65280, 65535],
                [65535, 1, 256, 255, 65280, 0],
                [1, 1, 256, 255, 65280, 65534],
            ],
            dtype='u2',
        ),
    ),
    (
        (
            b'\x04\x00\x00\x00@\x00\x00\x00G\x00\x00\x00N\x00\x00\x00U\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x05\x00\x01\x00\x00\x00\xff\x05\x00\x00\x01\x00\x00\xff'
            b'\x05\x00\x00\x00\x01\x00\xff\x05\x00\x00\x00\x00\x01\xff'
        ),
        numpy.array([0, 16777216, 65536, 256, 1, 4294967295], dtype='u4'),
    ),
    (
        (
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
        ),
        numpy.array(
            [
                [0, 16777216, 65536, 256, 1, 4294967295],
                [4294967295, 16777216, 65536, 256, 1, 0],
                [1, 16777216, 65536, 256, 1, 4294967294],
            ],
            dtype='u4',
        ),
    ),
]


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available, reason='DICOMRLE missing'
)
@pytest.mark.parametrize('data', range(len(DICOMRLE_DATA)))
@pytest.mark.parametrize('byteorder', ['=', '<', '>'])
@pytest.mark.parametrize('output', [False, True])
def test_dicomrle(data, byteorder, output):
    """Test DICOMRLE decoding."""
    encoded, result = DICOMRLE_DATA[data]
    out = result.nbytes if output else None
    dtype = byteorder + result.dtype.char
    assert imagecodecs.dicomrle_check(encoded)
    decoded = imagecodecs.dicomrle_decode(encoded, dtype, out=out)
    assert_array_equal(
        numpy.frombuffer(decoded, dtype).reshape(result.shape), result
    )
    if out is not None:
        with pytest.raises(imagecodecs.DicomrleError):
            imagecodecs.dicomrle_decode(encoded, dtype, out=out - 1)


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available or SKIP_NUMCODECS,
    reason='zarr or numcodecs missing',
)
def test_dicomrle_numcodecs():
    """Test DICOMRLE decoding with numcodecs."""
    from imagecodecs.numcodecs import Dicomrle

    encoded, result = DICOMRLE_DATA[-1]
    decoded = Dicomrle(dtype='<u4').decode(encoded)
    assert_array_equal(
        numpy.frombuffer(decoded, '<u4').reshape(result.shape), result
    )


@pytest.mark.skipif(
    not imagecodecs.DICOMRLE.available, reason='DICOMRLE missing'
)
def test_dicomrle_raises():
    """Test DICOMRLE decoding exceptions."""
    decode = imagecodecs.dicomrle_decode
    with pytest.raises(ValueError):
        decode(b'\x00' * 63, 'u1')
    with pytest.raises(ValueError):
        decode(b'\x00' * 64, 'u1')
    with pytest.raises(ValueError):
        decode(DICOMRLE_DATA[0][0], 'u4')
    assert not imagecodecs.dicomrle_check(b'\x00' * 63)
    assert not imagecodecs.dicomrle_check(b'\x00' * 64)


@pytest.mark.filterwarnings('ignore:invalid value encountered')
@pytest.mark.parametrize('output', ['new', 'out', 'inplace'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
@pytest.mark.parametrize(
    'kind',
    ['u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'B', 'b'],
)
@pytest.mark.parametrize('byteorder', ['>', '<'])
@pytest.mark.parametrize('func', ['delta', 'xor'])
def test_delta(output, byteorder, kind, codec, func):
    """Test Delta codec."""
    # if byteorder == '>' and numpy.dtype(kind).itemsize == 1:
    #     pytest.skip('duplicate test')

    if func == 'delta':
        if not imagecodecs.DELTA.available:
            pytest.skip('Delta missing')
        encode = imagecodecs.delta_encode
        decode = imagecodecs.delta_decode
        encode_py = _imagecodecs.delta_encode
        # decode_py = _imagecodecs.delta_decode
    elif func == 'xor':
        if not imagecodecs.XOR.available:
            pytest.skip('Xor missing')
        encode = imagecodecs.xor_encode
        decode = imagecodecs.xor_decode
        encode_py = _imagecodecs.xor_encode
        # decode_py = _imagecodecs.xor_decode

    bytetype = bytearray
    if kind == 'b':
        bytetype = bytes
        kind = 'B'

    axis = -2  # do not change
    if kind[0] in 'iuB':
        low = numpy.iinfo(kind).min
        high = numpy.iinfo(kind).max
        data = numpy.random.randint(
            low, high, size=33 * 31 * 3, dtype=kind
        ).reshape(33, 31, 3)
    else:
        # floating point
        low, high = -1e5, 1e5
        data = numpy.random.randint(
            low, high, size=33 * 31 * 3, dtype='i4'
        ).reshape(33, 31, 3)
    data = data.astype(byteorder + kind)

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
                if bytetype is bytes:
                    with pytest.raises(TypeError):
                        encode(data, out=encoded)
                else:
                    encode(data, out=encoded)
                    assert encoded == diff
            elif codec == 'decode':
                decoded = bytetype(len(data))
                if bytetype is bytes:
                    with pytest.raises(TypeError):
                        decode(diff, out=decoded)
                else:
                    decode(diff, out=decoded)
                    assert decoded == data
        elif output == 'inplace':
            if codec == 'encode':
                encoded = bytetype(data)
                if bytetype is bytes:
                    with pytest.raises(TypeError):
                        encode(encoded, out=encoded)
                else:
                    encode(encoded, out=encoded)
                    assert encoded == diff
            elif codec == 'decode':
                decoded = bytetype(diff)
                if bytetype is bytes:
                    with pytest.raises(TypeError):
                        decode(decoded, out=decoded)
                else:
                    decode(decoded, out=decoded)
                    assert decoded == data
    else:
        # if func == 'xor' and kind in {'f4', 'f8'}:
        #      with pytest.raises(ValueError):
        #          encode(data, axis=axis)
        #      pytest.xfail("XOR codec not implemented for float data")
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


@pytest.mark.skipif(
    not imagecodecs.FLOATPRED.available, reason='FloatPred missing'
)
@pytest.mark.parametrize('output', ['new', 'out'])
@pytest.mark.parametrize('codec', ['encode', 'decode'])
@pytest.mark.parametrize('endian', ['le', 'be'])
@pytest.mark.parametrize('planar', [False, True])
def test_floatpred(planar, endian, output, codec):
    """Test FloatPred codec."""
    encode = imagecodecs.floatpred_encode
    decode = imagecodecs.floatpred_decode
    data = numpy.fromfile(datafiles('rgb.bin'), dtype='<f4').reshape(33, 31, 3)

    if not planar:
        axis = -2
        if endian == 'le':
            encoded = numpy.fromfile(
                datafiles('rgb.floatpred_le.bin'), dtype='<f4'
            )
            encoded = encoded.reshape(33, 31, 3)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.zeros_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.zeros_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
        elif endian == 'be':
            data = data.astype('>f4')
            encoded = numpy.fromfile(
                datafiles('rgb.floatpred_be.bin'), dtype='>f4'
            )
            encoded = encoded.reshape(33, 31, 3)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.zeros_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.zeros_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
    else:
        axis = -1
        data = numpy.ascontiguousarray(numpy.moveaxis(data, 2, 0))
        if endian == 'le':
            encoded = numpy.fromfile(
                datafiles('rrggbb.floatpred_le.bin'), dtype='<f4'
            )
            encoded = encoded.reshape(3, 33, 31)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.zeros_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.zeros_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)
        elif endian == 'be':
            data = data.astype('>f4')
            encoded = numpy.fromfile(
                datafiles('rrggbb.floatpred_be.bin'), dtype='>f4'
            )
            encoded = encoded.reshape(3, 33, 31)
            if output == 'new':
                if codec == 'decode':
                    assert_array_equal(decode(encoded, axis=axis), data)
                elif codec == 'encode':
                    assert_array_equal(encode(data, axis=axis), encoded)
            elif output == 'out':
                out = numpy.zeros_like(data)
                if codec == 'decode':
                    decode(encoded, axis=axis, out=out)
                    assert_array_equal(out, data)
                elif codec == 'encode':
                    out = numpy.zeros_like(data)
                    encode(data, axis=axis, out=out)
                    assert_array_equal(out, encoded)


@pytest.mark.skipif(
    not imagecodecs.FLOAT24.available, reason='Float24 missing'
)
@pytest.mark.parametrize(
    'f3, f4, f3_expected',
    [
        # +/-/signalling NaN
        ((0x7F, 0x80, 0x00), (0x7F, 0xC0, 0x00, 0x00), numpy.nan),
        ((0xFF, 0x80, 0x00), (0xFF, 0xC0, 0x00, 0x00), numpy.nan),
        ((0xFF, 0x80, 0x00), (0xFF, 0x80, 0x00, 0x01), numpy.nan),
        # +/- inf
        ((0x7F, 0x00, 0x00), (0x7F, 0x80, 0x00, 0x00), numpy.inf),
        ((0xFF, 0x00, 0x00), (0xFF, 0x80, 0x00, 0x00), -numpy.inf),
        # +/- zero
        ((0x00, 0x00, 0x00), (0x00, 0x00, 0x00, 0x00), 0.0),
        ((0x80, 0x00, 0x00), (0x80, 0x00, 0x00, 0x00), -0.0),
        # +/- one
        ((0x3F, 0x00, 0x00), (0x3F, 0x80, 0x00, 0x00), 1.0),
        ((0xBF, 0x00, 0x00), (0xBF, 0x80, 0x00, 0x00), -1.0),
        # pi
        ((0x40, 0x92, 0x20), (0x40, 0x49, 0x0F, 0xDB), 3.1416016),
        # pi, no rounding
        # ((0x40, 0x92, 0x1F), (0x40, 0x49, 0x0F, 0xDB), 3.141571),
        # pi * 10**-6
        ((0x2C, 0xA5, 0xA8), (0x36, 0x52, 0xD4, 0x27), 3.1415839e-06),
        # pi * 10**6
        ((0x2C, 0xA5, 0xA8), (0x36, 0x52, 0xD4, 0x27), 3.1415839e-06),
        # subnormal 1e-19
        ((0x00, 0x76, 0x0F), (0x1F, 0xEC, 0x1E, 0x4A), 1e-19),
        # overflow 1.85e19
        ((0x7F, 0x00, 0x00), (0x5F, 0x80, 0x5E, 0x9A), numpy.inf),
        # subnormal shift 0
        ((0x00, 0x80, 0x00), (0x20, 0x00, 0x00, 0x00), 1.0842022e-19),
        # encode normal to denormal with rounding
        ((0x00, 0x80, 0x00), (0x1F, 0xFF, 0xFF, 0xFF), 1.0842021e-19),
        # subnormal shift 1
        ((0x00, 0x40, 0x00), (0x1F, 0x80, 0x00, 0x00), 5.421011e-20),
        # minimum positive subnormal, shift 15
        ((0x00, 0x00, 0x01), (0x18, 0x80, 0x00, 0x00), 3.3087225e-24),
        # minimum positive normal
        ((0x01, 0x00, 0x00), (0x20, 0x80, 0x00, 0x00), 2.1684043e-19),
        # round minimum normal float32 to zero; 1.1754943508222875e-38
        ((0x00, 0x00, 0x00), (0x00, 0x80, 0x00, 0x00), 0.0),
        ((0x80, 0x00, 0x00), (0x80, 0x80, 0x00, 0x00), 0.0),
        # round largest denormal float32 to zero; 5.877471754111438e-39
        ((0x00, 0x00, 0x00), (0x00, 0x40, 0x00, 0x00), 0.0),
        ((0x80, 0x00, 0x00), (0x80, 0x40, 0x00, 0x00), 0.0),
    ],
)
@pytest.mark.parametrize('byteorder', ['>', '<'])
@pytest.mark.parametrize('mode', ['encode', 'decode'])
def test_float24(f3, f4, f3_expected, byteorder, mode):
    """Test float24 special numbers."""
    decode = imagecodecs.float24_decode
    encode = imagecodecs.float24_encode

    f3_bytes = bytes(f3)
    f4_bytes = bytes(f4)

    if byteorder == '<':
        f3_bytes = f3_bytes[::-1]

    if mode == 'decode':
        f3_decoded = decode(f3_bytes, byteorder=byteorder)[0]
        if f3_expected is numpy.nan:
            assert numpy.isnan([f3_decoded])[0]
        elif f3_expected in {-numpy.inf, numpy.inf}:
            assert f3_decoded == f3_expected
        else:
            assert abs(f3_decoded - f3_expected) < 4e-8
    else:
        f4_array = numpy.frombuffer(f4_bytes, dtype='>f4').astype('=f4')
        f3_encoded = encode(f4_array, byteorder=byteorder)
        assert f3_encoded == f3_bytes


@pytest.mark.skipif(
    not imagecodecs.FLOAT24.available, reason='Float24 missing'
)
@pytest.mark.parametrize('byteorder', ['>', '<'])
def test_float24_roundtrip(byteorder):
    """Test all float24 numbers."""
    f3_bytes = numpy.arange(2**24, dtype='>u4').astype('u1').reshape((-1, 4))
    if byteorder == '>':
        f3_bytes = f3_bytes[:, :3].tobytes()
    else:
        f3_bytes = f3_bytes[:, 2::-1].tobytes()
    f3_decoded = imagecodecs.float24_decode(f3_bytes, byteorder=byteorder)
    f3_encoded = imagecodecs.float24_encode(f3_decoded, byteorder=byteorder)
    assert f3_bytes == f3_encoded


@pytest.mark.skipif(not imagecodecs.EER.available, reason='EER missing')
def test_eer():
    """Test EER decoder."""
    encoded = b'\x03\x1b\xfc\xb1\x35\xfb'  # from EER specification
    assert imagecodecs.eer_check(encoded) is None

    im = imagecodecs.eer_decode(encoded, (1, 312), 7, 1, 1)
    assert im[0, 3]
    assert im[0, 17]  # 17 = 3 + 13 + 1
    assert im[0, 233]  # 233 = 3 + 13 + 127 + 88 + 2
    assert im[0, 311]  # 311 = 3 + 13 + 127 + 88 + 77 + 3

    im = imagecodecs.eer_decode(encoded, (20, 16), 7, 1, 1)
    assert im[0, 3]
    assert im[1, 1]  # 17 // 16, 17 % 16
    assert im[14, 9]  # 233 // 16, 233 % 16
    assert im[19, 7]  # 311 // 16, 311 % 16

    with pytest.raises(RuntimeError):
        # shape too small
        im = imagecodecs.eer_decode(encoded, (19, 15), 7, 1, 1)


@pytest.mark.skipif(not imagecodecs.EER.available, reason='EER missing')
def test_eer_superres():
    """Test EER decoder superresolution mode."""
    encoded = b'\x03\x1b\xfc\xb1\x35\xfb'  # from EER specification

    im = imagecodecs.eer_decode(encoded, (40, 32), 7, 1, 1, superres=True)
    assert im[1, 6]
    assert im[3, 3]
    assert im[28, 19]
    assert im[38, 15]

    out = numpy.zeros((40, 32), numpy.bool_)
    imagecodecs.eer_decode(encoded, (40, 32), 7, 1, 1, superres=True, out=out)
    assert im[1, 6]
    assert im[3, 3]
    assert im[28, 19]
    assert im[38, 15]

    with pytest.raises(ValueError):
        # shape not compatible with superresolution
        im = imagecodecs.eer_decode(encoded, (40, 33), 7, 1, 1, superres=True)

    with pytest.raises(RuntimeError):
        # shape too small
        im = imagecodecs.eer_decode(encoded, (40, 30), 7, 1, 1, superres=True)


@pytest.mark.skipif(SKIP_NUMCODECS, reason='zarr or numcodecs missing')
@pytest.mark.parametrize('dtype', ['uint8', 'float32'])
@pytest.mark.parametrize(
    'kind', ['crc32', 'adler32', 'fletcher32', 'lookup3', 'h5crc']
)
def test_checksum_roundtrip(kind, dtype):
    """Test numcodecs.Checksum roundtrips."""
    from imagecodecs.numcodecs import Checksum

    if kind in {'crc32', 'adler32'} and not imagecodecs.ZLIB.available:
        pytest.skip('ZLIB missing')
    elif not imagecodecs.H5CHECKSUM.available:
        pytest.skip('H5CHECKSUM missing')

    data = numpy.arange(255, dtype=dtype).reshape(15, 17)[1:14, 2:15]
    codec = Checksum(kind=kind)
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
@pytest.mark.parametrize(
    'mode', ['bitgroom', 'granularbr', 'bitround', 'scale']
)
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_quantize_roundtrip(mode, dtype):
    """Test quantize roundtrips."""
    nsd = 12
    atol = 0.006
    if mode == 'bitgroom':
        nsd = (nsd - 1) // 3  # bits = math.ceil(nsd * 3.32) + 1
    if dtype == 'f4':
        nsd //= 2
        atol = 0.5
    data = numpy.linspace(-2.1, 31.4, 51, dtype=dtype).reshape((3, 17))
    encoded = imagecodecs.quantize_encode(data, mode, nsd)
    out = data.copy()
    imagecodecs.quantize_encode(data, mode, nsd, out=out)
    assert_array_equal(out, encoded)
    assert_allclose(data, encoded, atol=atol)


@pytest.mark.skipif(
    SKIP_NUMCODECS or not imagecodecs.QUANTIZE.available,
    reason='QUANTIZE missing',
)
@pytest.mark.parametrize('nsd', [1, 4])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_quantize_bitround(dtype, nsd):
    """Test BitRound quantize against numcodecs."""
    from imagecodecs.numcodecs import Quantize
    from numcodecs import BitRound

    # TODO: 31.4 fails
    data = numpy.linspace(-2.1, 31.5, 51, dtype=dtype).reshape((3, 17))
    encoded = Quantize(
        mode=imagecodecs.QUANTIZE.MODE.BITROUND,
        nsd=nsd,
    ).encode(data)
    nc = BitRound(keepbits=nsd)
    encoded2 = nc.decode(nc.encode(data))
    assert_array_equal(encoded, encoded2)


@pytest.mark.skipif(
    SKIP_NUMCODECS or not imagecodecs.QUANTIZE.available,
    reason='QUANTIZE missing',
)
@pytest.mark.parametrize('nsd', [1, 4])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_quantize_scale(dtype, nsd):
    """Test Scale quantize against numcodecs."""
    from imagecodecs.numcodecs import Quantize
    from numcodecs import Quantize as Quantize2

    data = numpy.linspace(-2.1, 31.4, 51, dtype=dtype).reshape((3, 17))
    encoded = Quantize(
        mode=imagecodecs.QUANTIZE.MODE.SCALE,
        nsd=nsd,
    ).encode(data)
    encoded2 = Quantize2(digits=nsd, dtype=dtype).encode(data)
    assert_array_equal(encoded, encoded2)


@pytest.mark.skipif(
    SKIP_NUMCODECS or not imagecodecs.H5CHECKSUM.available,
    reason='H5CHECKSUM, zarr, or numcodecs missing',
)
def test_checksum_fletcher32():
    """Test numcodecs.Checksum fletcher32 kind."""
    from imagecodecs.numcodecs import Checksum

    codec = Checksum(kind='fletcher32')
    data = (
        b'w\x07\x00\x00\x00\x00\x00\x00\x85\xf6\xff\xff\xff\xff\xff\xff'
        b'i\x07\x00\x00\x00\x00\x00\x00\x94\xf6\xff\xff\xff\xff\xff\xff'
        b'\x88\t\x00\x00\x00\x00\x00\x00i\x03\x00\x00\x00\x00\x00\x00'
        b'\x93\xfd\xff\xff\xff\xff\xff\xff\xc3\xfc\xff\xff\xff\xff\xff\xff'
        b"'\x02\x00\x00\x00\x00\x00\x00\xba\xf7\xff\xff\xff\xff\xff\xff"
        b'\xfd%\x86d'
    )
    decoded = codec.decode(data)
    assert_array_equal(
        numpy.frombuffer(decoded, dtype='<i8'),
        [1911, -2427, 1897, -2412, 2440, 873, -621, -829, 551, -2118],
    )
    encoded = codec.encode(decoded)
    assert encoded == data


@pytest.mark.skipif(
    SKIP_NUMCODECS or not imagecodecs.H5CHECKSUM.available,
    reason='H5CHECKSUM, zarr, or numcodecs missing',
)
def test_checksum_lookup3():
    """Test numcodecs.Checksum lookup3 kind."""
    from imagecodecs.numcodecs import Checksum

    data = b'Four score and seven years ago'
    codec = Checksum(kind='lookup3')
    result = codec.encode(data)
    assert result[-4:] == b'\x51\x05\x77\x17'
    assert bytes(codec.decode(result)) == data

    codec = Checksum(kind='lookup3', value=0xDEADBEEF)
    result = codec.encode(data)
    assert bytes(codec.decode(result)) == data

    codec = Checksum(kind='lookup3', value=1230)
    result = codec.encode(data)
    assert result[-4:] == b'\xd7Z\xe2\x0e'
    assert bytes(codec.decode(result)) == data

    codec = Checksum(kind='lookup3', value=1230, prefix=b'Hello world')
    result = codec.encode(data)
    assert bytes(codec.decode(result)) == data

    data = (
        b"\x00\x08\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xf7\x17\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xee'\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xe57\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xdcG\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xd3W\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xcag\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xc1w\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xb8\x87\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xaf\x97\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xa6\xa7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x9d\xb7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x94\xc7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x8b\xd7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x82\xe7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"y\xf7\x00\x00\x00\x00\x00\x00\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"n\x96\x07\x85"
    )
    codec = Checksum(
        kind='lookup3', prefix=b'FADB\x00\x01\xcf\x01\x00\x00\x00\x00\x00\x00'
    )
    encoded = codec.encode(data[:-4])
    codec.decode(encoded)
    assert encoded == data


@pytest.mark.skipif(
    not imagecodecs.H5CHECKSUM.available, reason='H5CHECKSUM missing'
)
def test_h5checksum_lookup3():
    """Test h5checksum_lookup3 function."""
    from imagecodecs import h5checksum_lookup3 as lookup3

    assert lookup3(b'', 0) == 0xDEADBEEF
    assert lookup3(b'', 0xDEADBEEF) == 0xBD5B7DDE
    assert lookup3(b'Four score and seven years ago', 0) == 0x17770551
    assert lookup3(b'Four score and seven years ago', 1) == 0xCD628161
    assert lookup3(b'jenkins', 0) == 0xC0E7DF9

    checksum_list = [0]
    for _ in range(9):
        checksum = lookup3(b'', checksum_list[-1])
        assert checksum not in checksum_list
        checksum_list.append(checksum)

    a = numpy.frombuffer(b'Four score and seven years ago', dtype=numpy.uint8)
    assert lookup3(a, 0) == 0x17770551


@pytest.mark.skipif(
    not imagecodecs.H5CHECKSUM.available, reason='H5CHECKSUM missing'
)
def test_h5checksum_other():
    """Test h5checksum metadata and hash_string functions."""
    from imagecodecs import h5checksum_hash_string, h5checksum_metadata

    assert h5checksum_metadata(b'', 0) == 3735928559
    assert h5checksum_hash_string(b'', 0) == 5381

    data = b'Four score and seven years ago'
    assert h5checksum_metadata(data) == 393676113
    assert h5checksum_hash_string(data) == 316074956


@pytest.mark.parametrize('value', [None, 2])
@pytest.mark.parametrize('kind', ['crc32', 'adler32'])
@pytest.mark.parametrize('codec', ['zlib', 'zlibng', 'deflate'])
def test_zlib_checksums(codec, kind, value):
    """Test crc32 and adler32 checksum functions."""
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not available')

    args = tuple() if value is None else (value,)
    data = b'Four score and seven years ago'
    expected = getattr(zlib, kind)(data, *args) & 0xFFFFFFFF
    checksum = getattr(imagecodecs, codec + '_' + kind)(data, *args)
    assert checksum == expected


@pytest.mark.skipif(not imagecodecs.LZO.available, reason='LZO missing')
def test_lzo_decode():
    """Test LZO decoder."""
    decode = imagecodecs.lzo_decode

    assert imagecodecs.lzo_check(b'\x11\x00\x00') is None
    assert decode(b'\x11\x00\x00', out=0) == b''
    assert decode(b'\x11\x00\x00', out=10) == b''
    assert decode(b'\x12  \x1e\x00\x00\x11\x00\x00', out=65) == b' ' * 64
    assert (
        decode(
            b'\x02      \x0b\x10\x00\x0c               \x11\x00\x00', out=64
        )
        == b' ' * 64
    )
    assert (
        decode(b'\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00', out=64)
        == b'a\01\fa' * 16
    )

    # with header
    assert decode(b'\xf0\x00\x00\x00\x00\x11\x00\x00', header=True) == b''
    assert (
        decode(b'\xf1\x00\x00\x00@\x12  \x1e\x00\x00\x11\x00\x00', header=True)
        == b' ' * 64
    )

    with pytest.raises(imagecodecs.LzoError):
        decode(b'\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00', out=63)
    with pytest.raises(TypeError):
        decode(b'\x11\x00\x00')


@pytest.mark.skipif(
    not imagecodecs.LZO.available or SKIP_NUMCODECS,
    reason='zarr or numcodecs missing',
)
def test_lzo_numcodecs():
    """Test LZO decoding with numcodecs."""
    from imagecodecs.numcodecs import Lzo

    assert (
        Lzo(header=True).decode(
            b'\xf1\x00\x00\x00@\x15a\x01\x0ca \x1b\x0c\x00\x11\x00\x00'
        )
        == b'a\01\fa' * 16
    )


@pytest.mark.skipif(not imagecodecs.LZW.available, reason='LZW missing')
def test_lzw_corrupt():
    """Test LZW decoder with corrupt stream."""
    # reported by S Richter on 2020.2.17
    fname = datafiles('corrupt.lzw.bin')
    with open(fname, 'rb') as fh:
        encoded = fh.read()
    assert imagecodecs.lzw_check(encoded)
    with pytest.raises(RuntimeError):
        imagecodecs.lzw_decode(encoded, out=655360)


@pytest.mark.skipif(not imagecodecs.LZW.available, reason='LZW missing')
def test_lzw_msb():
    """Test LZW decoder with MSB."""
    # TODO: add test_lzw_lsb
    decode = imagecodecs.lzw_decode
    for encoded, decoded in [
        (
            b'\x80\x1c\xcc\'\x91\x01\xa0\xc2m6\x99NB\x03\xc9\xbe\x0b'
            b'\x07\x84\xc2\xcd\xa68|"\x14 3\xc3\xa0\xd1c\x94\x02\x02\x80',
            b'say hammer yo hammer mc hammer go hammer',
        ),
        (
            b'\x80\x18M\xc6A\x01\xd0\xd0e\x10\x1c\x8c\xa73\xa0\x80\xc7\x02'
            b'\x10\x19\xcd\xe2\x08\x14\x10\xe0l0\x9e`\x10\x10\x80',
            b'and the rest can go and play',
        ),
        (
            b'\x80\x18\xcc&\xe19\xd0@t7\x9dLf\x889\xa0\xd2s',
            b"can't touch this",
        ),
        (b'\x80@@', b''),
    ]:
        assert imagecodecs.lzw_check(encoded)
        assert decode(encoded) == decoded


@pytest.mark.skipif(
    not (imagecodecs.LZW.available and imagecodecs.DELTA.available),
    reason='skip',
)
@pytest.mark.parametrize('output', ['new', 'size', 'ndarray', 'bytearray'])
def test_lzw_decode(output):
    """Test LZW decoder of input with horizontal differencing."""
    decode = imagecodecs.lzw_decode
    delta_decode = imagecodecs.delta_decode
    encoded = readfile('bytes.lzw_horizontal.bin')
    assert imagecodecs.lzw_check(encoded)
    decoded_size = len(BYTES)

    if output == 'new':
        decoded = decode(encoded)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
    elif output == 'size':
        decoded = decode(encoded, out=decoded_size)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
        # with pytest.raises(RuntimeError):
        decode(encoded, buffersize=32, out=decoded_size)
    elif output == 'ndarray':
        decoded = numpy.zeros_like(BYTESIMG)
        decode(encoded, out=decoded.reshape(-1))
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)
    elif output == 'bytearray':
        decoded = bytearray(decoded_size)
        decode(encoded, out=decoded)
        decoded = numpy.frombuffer(decoded, 'uint8').reshape(16, 16)
        delta_decode(decoded, out=decoded, axis=-1)
        assert_array_equal(BYTESIMG, decoded)


@pytest.mark.skipif(not imagecodecs.LZW.available, reason='LZW missing')
def test_lzw_decode_image_noeoi():
    """Test LZW decoder of input without EOI 512x512u2."""
    decode = imagecodecs.lzw_decode
    fname = datafiles('image_noeoi.lzw.bin')
    with open(fname, 'rb') as fh:
        encoded = fh.read()
    fname = datafiles('image_noeoi.bin')
    with open(fname, 'rb') as fh:
        decoded_known = fh.read()
    assert imagecodecs.lzw_check(encoded)
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


@pytest.mark.skipif(not imagecodecs.LZ4H5.available, reason='LZ4H5 missing')
def test_lz4h5():
    """Test lz4h5 codec with input from HDF5."""
    # https://github.com/cgohlke/imagecodecs/issues/126
    # with (
    #     h5py.File('h5ex_d_nplz4.h5', 'r') as h5,
    #     open('lz4h5.bin', 'wb') as fh,
    # ):
    #     filter_mask, chunk = ds.id.read_direct_chunk(
    #         ds.id.get_chunk_info(0).chunk_offset
    #     )
    #     fh.write(chunk)
    with open(datafiles('lz4h5.bin'), 'rb') as fh:
        data = fh.read()
    assert len(data) == 4792  # why this large?

    decoded = imagecodecs.lz4h5_decode(data)
    assert len(decoded) == 2048  # 16x32 uint32
    assert decoded[:8] == b'\x00\x00\x00\x00\xff\xff\xff\xff'
    encoded = imagecodecs.lz4h5_encode(decoded)
    assert len(encoded) == 1508
    decoded == imagecodecs.lz4h5_decode(encoded)

    with pytest.raises(imagecodecs.Lz4h5Error):
        imagecodecs.lz4h5_decode(data[:-1])


@pytest.mark.filterwarnings('ignore: PY_SSIZE_T_CLEAN')
@pytest.mark.parametrize(
    'output', ['new', 'bytearray', 'out', 'size', 'excess', 'trunc']
)
@pytest.mark.parametrize('length', [0, 2, 31 * 33 * 3])
@pytest.mark.parametrize('func', ['encode', 'decode'])
@pytest.mark.parametrize(
    'codec',
    [
        'bitshuffle',
        'brotli',
        'blosc',
        'blosc2',
        'bz2',
        'deflate',
        'gzip',
        'lz4',
        'lz4h',
        'lz4h5',
        'lz4f',
        'lzf',
        'lzfse',
        'lzham',
        'lzma',
        'lzw',
        'snappy',
        'szip',
        'zlib',
        'zlibng',
        'zopfli',
        'zstd',
    ],
)
def test_compressors(codec, func, output, length):
    """Test various non-image codecs."""
    if length:
        data = numpy.random.randint(255, size=length, dtype='uint8').tobytes()
    else:
        data = b''

    level = None
    if codec == 'blosc':
        if not imagecodecs.BLOSC.available or blosc is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.blosc_encode
        decode = imagecodecs.blosc_decode
        check = imagecodecs.blosc_check
        level = 9
        encoded = blosc.compress(data, clevel=level, typesize=1)
    elif codec == 'blosc2':
        if not imagecodecs.BLOSC2.available or blosc2 is None:
            pytest.skip(f'{codec} missing')
        if IS_PYPY:
            pytest.xfail('blosc2.compress fails under PyPy')
        encode = imagecodecs.blosc2_encode
        decode = imagecodecs.blosc2_decode
        check = imagecodecs.blosc2_check
        level = 5
        encoded = blosc2.compress2(data, clevel=level, typesize=8)
    elif codec == 'zlib':
        if not imagecodecs.ZLIB.available or zlib is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.zlib_encode
        decode = imagecodecs.zlib_decode
        check = imagecodecs.zlib_check
        level = 5
        encoded = zlib.compress(data, level)
    elif codec == 'zlibng':
        if not imagecodecs.ZLIBNG.available or zlib is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.zlibng_encode
        decode = imagecodecs.zlibng_decode
        check = imagecodecs.zlibng_check
        level = 5
        encoded = zlib.compress(data, level)
    elif codec == 'deflate':
        if not imagecodecs.DEFLATE.available:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.deflate_encode
        decode = imagecodecs.deflate_decode
        check = imagecodecs.deflate_check
        level = 8
        # TODO: use a 3rd party libdeflate wrapper
        # encoded = deflate.compress(data, level)
        encoded = encode(data, level)
    elif codec == 'gzip':
        if not imagecodecs.GZIP.available:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.gzip_encode
        decode = imagecodecs.gzip_decode
        check = imagecodecs.gzip_check
        level = 8
        encoded = encode(data, level)
        # encoded = gzip.compress(data, level)
    elif codec == 'lzma':
        if not imagecodecs.LZMA.available or lzma is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lzma_encode
        decode = imagecodecs.lzma_decode
        check = imagecodecs.lzma_check
        level = 6
        encoded = lzma.compress(data)
    elif codec == 'lzw':
        if not imagecodecs.LZW.available:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lzw_encode
        decode = imagecodecs.lzw_decode
        check = imagecodecs.lzw_check
        encoded = encode(data)
    elif codec == 'zstd':
        if not imagecodecs.ZSTD.available or zstd is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.zstd_encode
        decode = imagecodecs.zstd_decode
        check = imagecodecs.zstd_check
        level = 5
        if length == 0:
            # bug in zstd.compress?
            encoded = encode(data, level)
        else:
            encoded = zstd.compress(data, level)
    elif codec == 'lzf':
        if not imagecodecs.LZF.available or lzf is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lzf_encode
        decode = imagecodecs.lzf_decode
        check = imagecodecs.lzf_check
        encoded = lzf.compress(data, ((len(data) * 33) >> 5) + 1)
        if encoded is None:
            pytest.xfail("lzf can't compress empty input")
    elif codec == 'lzfse':
        if not imagecodecs.LZFSE.available or lzfse is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lzfse_encode
        decode = imagecodecs.lzfse_decode
        check = imagecodecs.lzfse_check
        encoded = lzfse.compress(data)
    elif codec == 'lzham':
        # TODO: test against pylzham?
        if not imagecodecs.LZHAM.available:  # or lzham is None
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lzham_encode
        decode = imagecodecs.lzham_decode
        check = imagecodecs.lzham_check
        level = 5
        encoded = encode(data, level)
    elif codec == 'lz4':
        if not imagecodecs.LZ4.available or lz4 is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lz4_encode
        decode = imagecodecs.lz4_decode
        check = imagecodecs.lz4_check
        level = 1
        encoded = lz4.block.compress(data, store_size=False)
    elif codec == 'lz4h':
        if not imagecodecs.LZ4.available or lz4 is None:
            pytest.skip(f'{codec} missing')

        def encode(*args, **kwargs):
            return imagecodecs.lz4_encode(*args, header=True, **kwargs)

        def decode(*args, **kwargs):
            return imagecodecs.lz4_decode(*args, header=True, **kwargs)

        check = imagecodecs.lz4_check
        level = 1
        encoded = lz4.block.compress(data, store_size=True)
    elif codec == 'lz4h5':
        if not imagecodecs.LZ4H5.available:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lz4h5_encode
        decode = imagecodecs.lz4h5_decode
        check = imagecodecs.lz4h5_check
        level = 1
        encoded = encode(data)
    elif codec == 'lz4f':
        if not imagecodecs.LZ4F.available or lz4 is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.lz4f_encode
        decode = imagecodecs.lz4f_decode
        check = imagecodecs.lz4f_check
        level = 0
        encoded = lz4.frame.compress(data)
    elif codec == 'bz2':
        if not imagecodecs.BZ2.available or bz2 is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.bz2_encode
        decode = imagecodecs.bz2_decode
        check = imagecodecs.bz2_check
        level = 9
        encoded = bz2.compress(data, compresslevel=level)
    elif codec == 'bitshuffle':
        if not imagecodecs.BITSHUFFLE.available or bitshuffle is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.bitshuffle_encode
        decode = imagecodecs.bitshuffle_decode
        check = imagecodecs.bitshuffle_check
        encoded = bitshuffle.bitshuffle(
            numpy.frombuffer(data, 'uint8')
        ).tobytes()
    elif codec == 'brotli':
        if not imagecodecs.BROTLI.available or brotli is None:
            pytest.skip(f'{codec} missing')
        if func == 'encode' and length == 0:
            # TODO: why?
            pytest.xfail('python-brotli returns different valid results')
        encode = imagecodecs.brotli_encode
        decode = imagecodecs.brotli_decode
        check = imagecodecs.brotli_check
        level = 11
        encoded = brotli.compress(data)
    elif codec == 'zopfli':
        if not imagecodecs.ZOPFLI.available or zopfli is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.zopfli_encode
        decode = imagecodecs.zopfli_decode
        check = imagecodecs.zopfli_check
        level = 1
        c = zopfli.ZopfliCompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
        encoded = c.compress(data) + c.flush()
    elif codec == 'snappy':
        if not imagecodecs.SNAPPY.available or snappy is None:
            pytest.skip(f'{codec} missing')
        encode = imagecodecs.snappy_encode
        decode = imagecodecs.snappy_decode
        check = imagecodecs.snappy_check
        encoded = snappy.compress(data)
    elif codec == 'szip':
        if not imagecodecs.SZIP.available:
            pytest.skip(f'{codec} missing')

        def encode(*args, **kwargs):
            return imagecodecs.szip_encode(
                *args,
                options_mask=141,
                pixels_per_block=32,
                bits_per_pixel=8,
                pixels_per_scanline=256,
                header=True,
                **kwargs,
            )

        def decode(*args, **kwargs):
            return imagecodecs.szip_decode(
                *args,
                options_mask=141,
                pixels_per_block=32,
                bits_per_pixel=8,
                pixels_per_scanline=256,
                header=True,
                **kwargs,
            )

        check = imagecodecs.szip_check
        encoded = encode(data)
    else:
        raise ValueError(codec)

    assert check(encoded) in {None, True}

    if level is None:

        def encode(data, level, encode=encode, **kwargs):
            return encode(data, **kwargs)

    if func == 'encode':
        size = len(encoded)
        if output == 'new':
            assert encoded == encode(data, level)
        elif output == 'bytearray':
            ret = encode(data, level, out=bytearray)
            assert encoded == ret
        elif output == 'size':
            if codec == 'lz4f':
                pytest.xfail(
                    'LZ4F_compressFrame cannot compress to exact output size'
                )
                encode(data, level, out=size)
            elif codec in {'gzip'}:
                ret = encode(data, level, out=size + 9)
            else:
                ret = encode(data, level, out=size)
            assert encoded == ret[:size]
        elif output == 'out':
            if codec == 'lz4f':
                pytest.xfail(
                    'LZ4F_compressFrame cannot compress to exact output size'
                )
                out = bytearray(size)
            elif codec == 'zstd':
                out = bytearray(max(size, 64))
            # elif codec == 'blosc':
            #     out = bytearray(max(size, 17))  # bug in blosc ?
            elif codec == 'lzf':
                out = bytearray(size + 1)  # bug in liblzf ?
            elif codec in {'gzip'}:
                out = bytearray(size + 9)
            else:
                out = bytearray(size)
            ret = encode(data, level, out=out)
            assert encoded == out[:size]
            assert encoded == ret
        elif output == 'excess':
            out = bytearray(size + 1021)
            ret = encode(data, level, out=out)
            if codec in {'blosc', 'blosc2'}:
                # pytest.xfail('blosc output depends on output size')
                assert data == decode(ret)
            else:
                assert ret == out[:size]
                assert encoded == ret
        elif output == 'trunc':
            size = max(0, size - 1)
            out = bytearray(size)
            if size == 0 and codec == 'bitshuffle':
                assert encode(data, level, out=out) == b''
            else:
                with pytest.raises(RuntimeError):
                    encode(data, level, out=out)
        else:
            raise ValueError(output)
    elif func == 'decode':
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
            if length == 0 or codec in {'bz2', 'lzma', 'lz4f', 'lzfse', 'lzw'}:
                decode(encoded, out=out)
                assert data[:size] == out
            else:
                # most codecs don't support truncated output
                with pytest.raises(RuntimeError):
                    decode(encoded, out=out)
        else:
            raise ValueError(output)
    else:
        raise ValueError(func)


@pytest.mark.skipif(
    not imagecodecs.BITSHUFFLE.available, reason='bitshuffle missing'
)
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
        data = numpy.random.randint(255, size=1024, dtype=f'u{itemsize}')
        data.shape = 2, 4, 128
    encoded = encode(data, itemsize=itemsize, blocksize=blocksize)
    decoded = decode(encoded, itemsize=itemsize, blocksize=blocksize)
    if dtype == 'bytes':
        assert data == decoded
    else:
        assert_array_equal(data, decoded)


@pytest.mark.parametrize('numthreads', [1, 6])
@pytest.mark.parametrize('level', [None, 1])
@pytest.mark.parametrize('shuffle', ['noshuffle', 'shuffle', 'bitshuffle'])
@pytest.mark.parametrize(
    'compressor', ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']
)
@pytest.mark.parametrize('version', [1, 2])
def test_blosc_roundtrip(version, compressor, shuffle, level, numthreads):
    """Test Blosc codecs."""
    if version == 1:
        if not imagecodecs.BLOSC.available:
            pytest.skip('blosc missing')
        encode = imagecodecs.blosc_encode
        decode = imagecodecs.blosc_decode
        if compressor == 'zstd':
            compressor = imagecodecs.BLOSC.COMPRESSOR.ZSTD
        if shuffle == 'bitshuffle':
            shuffle = imagecodecs.BLOSC.SHUFFLE.BITSHUFFLE
    else:
        if not imagecodecs.BLOSC2.available:
            pytest.skip('blosc2 missing')
        encode = imagecodecs.blosc2_encode
        decode = imagecodecs.blosc2_decode
        if compressor == 'zstd':
            compressor = imagecodecs.BLOSC2.COMPRESSOR.ZSTD
        if shuffle == 'bitshuffle':
            shuffle = imagecodecs.BLOSC2.FILTER.BITSHUFFLE
    data = numpy.random.randint(255, size=2021, dtype='uint8').tobytes()
    encoded = encode(
        data,
        level=level,
        compressor=compressor,
        shuffle=shuffle,
        numthreads=numthreads,
    )
    decoded = decode(encoded, numthreads=numthreads)
    assert data == decoded


# test data from libaec https://gitlab.dkrz.de/k202009/libaec/tree/master/data
AEC_PATH = DATA_PATH / 'libaec/121B2TestData'

AEC_TEST_OPTIONS = [
    os.path.split(f)[-1][5:-3]
    for f in glob.glob(str(AEC_PATH / 'AllOptions' / '*.rz'))
]

AEC_TEST_EXTENDED = [
    os.path.split(f)[-1][:-3]
    for f in glob.glob(str(AEC_PATH / 'ExtendedParameters' / '*.rz'))
]


@pytest.mark.skipif(not imagecodecs.AEC.available, reason='aec missing')
@pytest.mark.parametrize('dtype', ['bytes', 'numpy'])
@pytest.mark.parametrize('name', AEC_TEST_EXTENDED)
def test_aec_extended(name, dtype):
    """Test AEC codec with libaec ExtendedParameters."""
    if name == 'sar32bit.j16.r256' and not (
        IS_CG or os.environ.get('AEC_TEST_EXTENDED', False)
    ):
        pytest.xfail('aec extension not built with ENABLE_RSI_PADDING')

    encode = imagecodecs.aec_encode
    decode = imagecodecs.aec_decode

    size = 512 * 512 * 4
    bitspersample = 32
    flags = imagecodecs.AEC.FLAG.DATA_PREPROCESS | imagecodecs.AEC.FLAG.PAD_RSI

    matches = re.search(r'j(\d+)\.r(\d+)', name).groups()
    blocksize = int(matches[0])
    rsi = int(matches[1])

    filename = AEC_PATH / 'ExtendedParameters' / f'{name}.rz'
    with open(filename, 'rb') as fh:
        rz = fh.read()

    filename = (
        AEC_PATH / 'ExtendedParameters' / '{}.dat'.format(name.split('.')[0])
    )
    if dtype == 'bytes':
        with open(filename, 'rb') as fh:
            dat = fh.read()
        out = size
    else:
        dat = numpy.fromfile(filename, 'uint32').reshape(512, 512)
        out = numpy.zeros_like(dat)

    # decode
    decoded = decode(
        rz,
        bitspersample=bitspersample,
        flags=flags,
        blocksize=blocksize,
        rsi=rsi,
        out=out,
    )
    if dtype == 'bytes':
        assert decoded == dat
    else:
        pass

    # roundtrip
    if dtype == 'bytes':
        encoded = encode(
            dat,
            bitspersample=bitspersample,
            flags=flags,
            blocksize=blocksize,
            rsi=rsi,
        )
        # fails with AEC_DATA_ERROR if libaec wasn't built with libaec.diff
        decoded = decode(
            encoded,
            bitspersample=bitspersample,
            flags=flags,
            blocksize=blocksize,
            rsi=rsi,
            out=size,
        )
        assert decoded == dat
    else:
        encoded = encode(dat, flags=flags, blocksize=blocksize, rsi=rsi)
        # fails with AEC_DATA_ERROR if libaec wasn't built with libaec.diff
        decoded = decode(
            encoded, flags=flags, blocksize=blocksize, rsi=rsi, out=out
        )
        assert_array_equal(decoded, out)


@pytest.mark.skipif(not imagecodecs.AEC.available, reason='aec missing')
@pytest.mark.parametrize('name', AEC_TEST_OPTIONS)
def test_aec_options(name):
    """Test AEC codec with libaec 121B2TestData."""
    encode = imagecodecs.aec_encode
    decode = imagecodecs.aec_decode

    rsi = 128
    blocksize = 16
    flags = imagecodecs.AEC.FLAG.DATA_PREPROCESS
    if 'restricted' in name:
        flags |= imagecodecs.AEC.FLAG.RESTRICTED
    matches = re.search(r'p(\d+)n(\d+)', name).groups()
    size = int(matches[0])
    bitspersample = int(matches[1])

    if bitspersample > 8:
        size *= 2
    if bitspersample > 16:
        size *= 2

    filename = str(AEC_PATH / 'AllOptions' / f'test_{name}.rz')
    with open(filename, 'rb') as fh:
        rz = fh.read()

    filename = (
        filename.replace('.rz', '.dat')
        .replace('-basic', '')
        .replace('-restricted', '')
    )
    with open(filename, 'rb') as fh:
        dat = fh.read()
    out = size

    # decode
    decoded = decode(
        rz,
        bitspersample=bitspersample,
        flags=flags,
        blocksize=blocksize,
        rsi=rsi,
        out=out,
    )
    assert decoded == dat

    # roundtrip
    encoded = encode(
        dat,
        bitspersample=bitspersample,
        flags=flags,
        blocksize=blocksize,
        rsi=rsi,
    )
    decoded = decode(
        encoded,
        bitspersample=bitspersample,
        flags=flags,
        blocksize=blocksize,
        rsi=rsi,
        out=out,
    )
    assert decoded == dat


@pytest.mark.skipif(not imagecodecs.BCN.available, reason='bcn missing')
@pytest.mark.parametrize(
    'name,format,shape',
    [
        ('bc1', 1, (512, 768, 4)),
        ('bc2', 2, (256, 256, 4)),
        ('bc3', 3, (600, 800, 4)),
        ('bc4', 4, (600, 800)),
        ('bc5', 5, (600, 800, 2)),
        ('bc6hs', -6, (512, 1024, 3)),
        ('bc7', 7, (600, 800, 4)),
        ('cubemap', 1, (6, 512, 512, 4)),
        # TODO: DDS with depthmap, mipmap, texture array
    ],
)
def test_bcn(name, format, shape):
    """Test BCN and DDS codecs."""
    fname = DATA_PATH / f'bcn/{name}.dds'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')
    with open(fname, 'rb') as fh:
        dds = fh.read()
    data = imagecodecs.dds_decode(dds)
    assert data.shape == shape
    assert data.dtype == 'uint8' if name != 'bc6hs' else 'float16'

    offset = 128 if format in {1, 2, 3} else 148
    out = imagecodecs.bcn_decode(dds[offset:], format, shape=shape)
    assert_array_equal(data, out)
    out = numpy.zeros_like(data)
    imagecodecs.bcn_decode(dds[offset:], format, out=out)
    assert_array_equal(data, out)


@pytest.mark.skipif(not imagecodecs.SZIP.available, reason='szip missing')
def test_szip_canonical():
    """Test SZIP codec."""
    # from https://github.com/zarr-developers/numcodecs/pull/422
    decode = imagecodecs.szip_decode
    encode = imagecodecs.szip_encode
    data = (
        b'\x00\x02\x00\x00\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU'
        b'\x15UUUUUUUQUUUUUUUU\x15UUUUUUUQUUUUUUUU'
    )
    param = dict(
        options_mask=141,
        pixels_per_block=32,
        bits_per_pixel=16,
        pixels_per_scanline=256,
        header=True,
    )
    decoded = decode(data, **param)
    arr = numpy.frombuffer(decoded, dtype=numpy.uint16)
    assert (arr == 1).all()
    encoded = encode(arr, **param)
    assert encoded == data

    del param['header']
    assert param == imagecodecs.szip_params(
        arr, options_mask=1 | 4 | 128, pixels_per_block=32
    )
    # positional arguments
    assert encoded == encode(
        arr, *imagecodecs.szip_params(arr, 141, 32).values(), header=True
    )


@pytest.mark.skipif(not imagecodecs.SZIP.available, reason='szip missing')
def test_szip_params():
    """Test szip_params function."""
    MASK = imagecodecs.SZIP.OPTION_MASK

    assert imagecodecs.szip_params(
        numpy.empty((256, 256), numpy.uint16),
        pixels_per_block=16,
        options_mask=MASK.ALLOW_K13 | MASK.EC | MASK.RAW,
    ) == {
        'options_mask': MASK.ALLOW_K13 | MASK.EC | MASK.LSB | MASK.RAW,
        'pixels_per_block': 16,
        'bits_per_pixel': 16,
        'pixels_per_scanline': 256,
    }

    assert imagecodecs.szip_params(numpy.empty((2, 8192, 4), numpy.uint8)) == {
        'options_mask': 12,
        'pixels_per_block': 32,
        'bits_per_pixel': 32,
        'pixels_per_scanline': 4096,  # max blocks per line
    }

    assert imagecodecs.szip_params(numpy.empty((4,), numpy.float64)) == {
        'options_mask': 12,
        'pixels_per_block': 4,  # reduced from default
        'bits_per_pixel': 64,
        'pixels_per_scanline': 4,
    }


@pytest.mark.skipif(not imagecodecs.ZSTD.available, reason='zstd missing')
def test_zstd_stream():
    """Test ZSTD decoder on stream with unknown decoded size."""
    # https://github.com/fsspec/kerchunk/issues/317
    data = readfile('zstd_stream.bin')
    decoded = imagecodecs.zstd_decode(data)
    arr = numpy.frombuffer(decoded, dtype=numpy.uint32).reshape(256, 256, 5)
    assert arr[86, 97, 4] == 1092705


@pytest.mark.skipif(not imagecodecs.LZF.available, reason='lzf missing')
def test_lzf_exceptions():
    """Test LZF codec exceptions codec."""
    # https://github.com/cgohlke/imagecodecs/issues/103
    encoded = imagecodecs.lzf_encode(b'0123456789')
    with pytest.raises(imagecodecs.LzfError):
        imagecodecs.lzf_decode(encoded, out=9)
    with pytest.raises(imagecodecs.LzfError):
        imagecodecs.lzf_decode(encoded[:2] + encoded[4:], out=10)


@pytest.mark.skipif(not imagecodecs.PGLZ.available, reason='pglz missing')
def test_pglz():
    """Test PGLZ codec."""
    decode = imagecodecs.pglz_decode
    encode = imagecodecs.pglz_encode
    data = b'111111181111111111111111121111111111111111111111111'

    with pytest.raises(RuntimeError):
        # not compressible
        encode(b'')
    with pytest.raises(ValueError):
        # output must be len(data) + 4
        encode(data, out=len(data))
    with pytest.raises(RuntimeError):
        # not enough output
        decode(encode(data), checkcomplete=True, out=4)
    with pytest.raises(RuntimeError):
        # default output too large
        assert decode(encode(data), checkcomplete=True) == data

    assert decode(b'') == b''
    assert decode(encode(data)) == data
    assert decode(encode(data, header=True), header=True) == data
    assert decode(encode(data), checkcomplete=True, out=len(data)) == data
    assert decode(encode(data), out=len(data)) == data
    assert decode(encode(data), out=len(data) + 7) == data
    assert decode(encode(data, strategy='always'), out=len(data)) == data

    data = data[:8]
    assert decode(encode(data, strategy='always'), out=len(data)) == data
    assert (
        decode(encode(data, strategy=[6, 100, 0, 100, 128, 6]), out=len(data))
        == data
    )
    with pytest.raises(RuntimeError):
        # data too short for default strategy
        encode(data)
    data = b'1234567890abcdefghijklmnopqrstuvwxyz'
    with pytest.raises(RuntimeError):
        # data not compressible
        encode(data, strategy='always')
    assert encode(data, header=True)[4:] == data
    assert decode(encode(data, header=True), header=True) == data


@pytest.mark.skipif(not imagecodecs.RCOMP.available, reason='rcomp missing')
@pytest.mark.parametrize('dtype', ['u1', 'u2', 'u4', 'i1', 'i2', 'i4'])
@pytest.mark.parametrize('case', [1, 2, 3, 4])
def test_rcomp(dtype, case):
    """Test RCOMP codec."""
    decode = imagecodecs.rcomp_decode
    encode = imagecodecs.rcomp_encode

    data = numpy.load(datafiles('rgb.u1.npy'))
    if dtype[0] == 'i':
        data = data.astype('i2')
        data -= 128
    data = data.astype(dtype)

    encoded = encode(data)
    if case == 1:
        assert_array_equal(
            data, decode(encoded, shape=data.shape, dtype=data.dtype)
        )
    elif case == 2:
        decoded = decode(encoded, shape=data.size, dtype=data.dtype)
        decoded = decoded.reshape(data.shape)
        assert_array_equal(data, decoded)
    elif case == 3:
        out = numpy.zeros_like(data)
        decode(encoded, out=out)
        assert_array_equal(data, out)
    elif case == 4:
        out = numpy.zeros_like(data)
        decode(encoded, shape=data.shape, dtype=data.dtype, out=out)
        assert_array_equal(data, out)


@pytest.mark.skipif(not imagecodecs.JETRAW.available, reason='jetraw missing')
def test_jetraw():
    """Test Jetraw codec."""
    data = readfile('jetraw.bin')
    im = numpy.empty((2304, 2304), numpy.uint16)
    imagecodecs.jetraw_decode(data, out=im)
    assert im[1490, 1830] == 36569

    imagecodecs.jetraw_init()
    try:
        encoded = imagecodecs.jetraw_encode(im, '500202_fast_bin1x')
    except imagecodecs.JetrawError as exc:
        errmsg = str(exc)
        if 'license' in errmsg:  # or 'identifier' in errmsg:
            # encoding requires a license
            return
        raise exc
    decoded = numpy.empty((2304, 2304), numpy.uint16)
    imagecodecs.jetraw_decode(encoded, out=decoded)
    assert im[1490, 1830] == 36569


@pytest.mark.skipif(not imagecodecs.PCODEC.available, reason='pcodec missing')
@pytest.mark.parametrize(
    'dtype',
    ['uint16', 'uint32', 'int32', 'int64', 'float32', 'float64', 'float16'],
)
def test_pcodec(dtype):
    """Test pcodec codec."""
    dtype = numpy.dtype(dtype)
    data = image_data('rgb', dtype)
    shape = data.shape

    encoded = imagecodecs.pcodec_encode(data)
    decoded = imagecodecs.pcodec_decode(encoded, shape, dtype)
    assert_array_equal(data, decoded)

    encoded = imagecodecs.pcodec_encode(data, level=10)
    out = numpy.empty_like(data)
    imagecodecs.pcodec_decode(encoded, out=out)
    assert_array_equal(data, out)


@pytest.mark.skipif(not imagecodecs.RGBE.available, reason='rgbe missing')
def test_rgbe_decode():
    """Test RGBE decoding."""
    decode = imagecodecs.rgbe_decode
    encoded = readfile('384x256.hdr')

    out = decode(encoded)
    assert out.shape == (384, 256, 3)
    assert out.dtype == 'float32'
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    out[:] = 0.0
    decode(encoded, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    with pytest.raises(ValueError):
        decode(encoded, header=False)

    with pytest.raises(ValueError):
        decode(encoded[77:])

    out[:] = 0.0
    decode(encoded[77:], out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    out[:] = 0.0
    decode(encoded[77:], rle=True, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    with pytest.raises(ValueError):
        decode(encoded[77:], rle=False, out=out)

    with pytest.raises(imagecodecs.RgbeError):
        # RGBE_ReadPixels_RLE returned READ_ERROR
        decode(encoded, header=False, out=out)

    # no header, no rle
    encoded = readfile('384x256.rgbe.bin')
    with pytest.raises(ValueError):
        # output required if no header
        decode(encoded)

    out[:] = 0.0
    decode(encoded, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)
    image = out.copy()

    out[:] = 0.0
    decode(encoded, header=False, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    out[:] = 0.0
    decode(encoded, header=False, rle=False, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    # TODO: not sure why this succeeds
    # decoding non-rle data with rle=True
    out[:] = 0.0
    decode(encoded, header=False, rle=True, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)
    assert_array_equal(out, image)

    encoded_array = numpy.frombuffer(
        encoded, count=-1, dtype=numpy.uint8
    ).reshape((384, 256, 4))
    out = decode(encoded_array)
    assert out.shape == (384, 256, 3)
    assert out.dtype == 'float32'
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)

    out[:] = 0.0
    decode(encoded_array, out=out)
    assert tuple(out[227, 201]) == (133.0, 73.0, 39.0)


@pytest.mark.skipif(not imagecodecs.RGBE.available, reason='rgbe missing')
def test_rgbe_roundtrip():
    """Test RGBE roundtrips."""
    encode = imagecodecs.rgbe_encode
    decode = imagecodecs.rgbe_decode

    data = decode(readfile('384x256.hdr'))
    assert data.shape == (384, 256, 3)
    assert data.dtype == 'float32'
    assert tuple(data[227, 201]) == (133.0, 73.0, 39.0)

    # encode to bytes
    encoded = encode(data)
    assert encoded[:10] == b'#?RADIANCE'
    assert_array_equal(data, decode(encoded))
    assert_array_equal(data, decode(encode(data, header=True)))
    assert_array_equal(data, decode(encode(data, header=True, rle=True)))
    assert_array_equal(data, decode(encode(data, out=bytearray)))

    assert_array_equal(data[0, 0], numpy.squeeze(decode(encode(data[0, 0]))))

    assert_array_equal(data, decode(encode(data, out=len(encoded))))
    assert_array_equal(data, decode(encode(data, out=len(encoded) + 4)))
    assert_array_equal(data, decode(encode(data, out=bytearray(len(encoded)))))

    with pytest.raises(imagecodecs.RgbeError):
        encode(data, out=1)

    with pytest.raises(imagecodecs.RgbeError):
        encode(data, out=len(encoded) - 1)

    # encode to bytes without header
    encoded = encode(data, header=False)
    assert encoded[:10] != b'#?RADIANCE'
    out = numpy.zeros_like(data)
    decode(encoded, out=out)
    assert_array_equal(data, out)

    encoded = encode(data, header=False, rle=True)
    out = numpy.zeros_like(data)
    decode(encoded, out=out)
    assert_array_equal(data, out)

    encoded = encode(data, header=False, rle=True)
    out = numpy.zeros_like(data)
    decode(encoded, out=out)
    assert_array_equal(data, out)

    # encode to output
    out = numpy.zeros((384, 256, 4), numpy.uint8)
    encode(data, out=out)
    assert_array_equal(decode(out), data)

    with pytest.raises(ValueError):
        encode(data, out=numpy.zeros((384, 256, 3), numpy.uint8))

    with pytest.raises(ValueError):
        encode(data, out=numpy.zeros((384, 255, 4), numpy.uint8))

    out = numpy.zeros((384 * 256 * 4), numpy.uint8)
    encode(data, out=out)
    assert_array_equal(decode(out.reshape((384, 256, 4))), data)


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='cms missing')
def test_cms_profile():
    """Test cms_profile function."""
    from imagecodecs import CmsError, cms_profile, cms_profile_validate

    with pytest.raises(CmsError):
        cms_profile_validate(b'12345')

    profile = cms_profile(None)
    cms_profile_validate(profile)
    profile = cms_profile('null')
    cms_profile_validate(profile)
    profile = cms_profile('gray')
    cms_profile_validate(profile)
    profile = cms_profile('rgb')
    cms_profile_validate(profile)
    profile = cms_profile('srgb')
    cms_profile_validate(profile)
    profile = cms_profile('xyz')
    cms_profile_validate(profile)
    profile = cms_profile('lab2')
    cms_profile_validate(profile)
    profile = cms_profile('lab4')
    cms_profile_validate(profile)
    profile = cms_profile('adobergb')
    cms_profile_validate(profile)

    primaries = [
        2748779008 / 4294967295,
        1417339264 / 4294967295,
        1.0,
        1288490240 / 4294967295,
        2576980480 / 4294967295,
        1.0,
        644245120 / 4294967295,
        257698032 / 4294967295,
        1.0,
    ]
    whitepoint = [1343036288 / 4294967295, 1413044224 / 4294967295, 1.0]
    transferfunction = numpy.arange(1024, dtype=numpy.uint16)

    profile = cms_profile(
        'gray', whitepoint=whitepoint, transferfunction=transferfunction
    )
    cms_profile_validate(profile)

    transferfunction = transferfunction.astype(numpy.float32)
    transferfunction /= 1024
    profile = cms_profile(
        'rgb',
        whitepoint=whitepoint,
        primaries=primaries,
        transferfunction=transferfunction,
    )
    cms_profile_validate(profile)

    transferfunction = [transferfunction, transferfunction, transferfunction]
    profile = cms_profile(
        'rgb',
        whitepoint=whitepoint,
        primaries=primaries,
        transferfunction=transferfunction,
    )
    cms_profile_validate(profile)

    # xyY
    profile1 = cms_profile(
        'rgb',
        whitepoint=whitepoint,
        primaries=primaries,
        gamma=2.19921875,
    )
    cms_profile_validate(profile1)

    # xy
    profile2 = cms_profile(
        'rgb',
        whitepoint=[1343036288 / 4294967295, 1413044224 / 4294967295],
        primaries=[
            2748779008 / 4294967295,
            1417339264 / 4294967295,
            1288490240 / 4294967295,
            2576980480 / 4294967295,
            644245120 / 4294967295,
            257698032 / 4294967295,
        ],
        gamma=2.19921875,
    )
    cms_profile_validate(profile2)

    # xy rationals
    profile3 = cms_profile(
        'rgb',
        whitepoint=[1343036288, 4294967295, 1413044224, 4294967295],
        primaries=[
            2748779008,
            4294967295,
            1417339264,
            4294967295,
            1288490240,
            4294967295,
            2576980480,
            4294967295,
            644245120,
            4294967295,
            257698032,
            4294967295,
        ],
        gamma=2.19921875,
    )
    cms_profile_validate(profile3)

    assert profile1 == profile2
    assert profile1 == profile3


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='cms missing')
def test_cms_output_shape():
    """Test _cms_output_shape function."""
    from imagecodecs._cms import _cms_format, _cms_output_shape

    for args, colorspace, planar, expected in (
        (((6, 7), 'u1', 'gray'), 'gray', 0, (6, 7)),
        (((6, 7, 2), 'u1', 'graya'), 'graya', 0, (6, 7, 2)),
        (((5, 6, 7), 'u1', 'gray'), 'gray', 0, (5, 6, 7)),
        (((6, 7, 3), 'u1', 'rgb'), 'gray', 0, (6, 7)),
        (((6, 7, 3), 'u1', 'rgb'), 'rgb', 0, (6, 7, 3)),
        (((6, 7, 3), 'u1', 'rgb'), 'rgba', 0, (6, 7, 4)),
        (((6, 7, 4), 'u1', 'rgba'), 'rgb', 0, (6, 7, 3)),
        (((6, 7, 4), 'u1', 'rgba'), 'cmyk', 0, (6, 7, 4)),
        (((6, 7), 'u1', 'gray'), 'rgb', 0, (6, 7, 3)),
        (((6, 7), 'u1', 'gray'), 'rgba', 0, (6, 7, 4)),
        # planar
        (((6, 7), 'u1', 'gray'), 'rgb', 1, (3, 6, 7)),
        (((6, 7, 2), 'u1', 'graya'), 'graya', 1, (2, 6, 7)),
        (((6, 7, 2), 'u1', 'graya'), 'rgba', 1, (4, 6, 7)),
        (((3, 6, 7), 'u1', 'rgb', 1), 'gray', 0, (6, 7)),
        (((3, 6, 7), 'u1', 'rgb', 1), 'rgb', 0, (6, 7, 3)),
        (((3, 6, 7), 'u1', 'rgb', 1), 'cmyk', 1, (4, 6, 7)),
        (((6, 7, 3), 'u1', 'rgb', 0), 'rgba', 1, (4, 6, 7)),
        (((5, 6, 7), 'u1', 'gray'), 'rgb', 1, (5, 3, 6, 7)),
        (((5, 3, 6, 7), 'u1', 'rgb', 1), 'rgba', 1, (5, 4, 6, 7)),
        (((5, 3, 6, 7), 'u1', 'rgb', 1), 'gray', 0, (5, 6, 7)),
    ):
        fmt = _cms_format(*args)
        # print(imagecodecs._cms._cms_format_decode(fmt))
        shape = _cms_output_shape(fmt, args[0], colorspace, planar)
        assert shape == expected

    fmt = _cms_format((6, 7), 'u1', 'gray')
    with pytest.raises(RuntimeError):
        # output planar with samples < 2
        _cms_output_shape(fmt, (6, 7), 'gray', 1)

    fmt = _cms_format((3, 6, 7), 'u1', 'rgb')
    with pytest.raises(RuntimeError):
        # input planar with ndim < 2
        _cms_output_shape(fmt, (6, 7), 'gray', 1)


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='cms missing')
def test_cms_format():
    """Test _cms_format function."""
    from imagecodecs._cms import _cms_format, _cms_format_decode

    for args, (dtype, pixeltype, samples, planar, swap, swapfirst) in (
        # data types
        (((1, 1), 'u1'), ('u1', 3, 1, False, False, False)),
        (((1, 1), 'u2'), ('u2', 3, 1, False, False, False)),
        (((1, 1), '>u1'), ('u1', 3, 1, False, False, False)),
        (((1, 1), '>u2'), ('>u2', 3, 1, False, False, False)),
        # (((1, 1), '<f2'), ('f2', 3, 1, False, False, False)),
        (((1, 1), '>f4'), ('>f4', 3, 1, False, False, False)),
        (((1, 1), '<f8'), ('f8', 3, 1, False, False, False)),
        # auto detect pixeltype
        # always gray, except uint8 with 3|4 contig samples are RGB
        (((1, 1, 1), 'u1'), ('u1', 3, 1, False, False, False)),
        (((1, 1, 2), 'u1'), ('u1', 3, 1, False, False, False)),  # not GA
        (((1, 1, 3), 'u1'), ('u1', 4, 3, False, False, False)),
        (((1, 1, 4), 'u1'), ('u1', 4, 4, False, False, False)),
        (((1, 1, 5), 'u1'), ('u1', 3, 1, False, False, False)),  # not CMYKA
        (((1, 1, 6), 'u1'), ('u1', 3, 1, False, False, False)),
        (((1, 1, 3), 'u2'), ('u2', 3, 1, False, False, False)),  # not RGB
        (((1, 1, 4), 'u2'), ('u2', 3, 1, False, False, False)),  # not RGBA
        (((1, 1, 5), 'u2'), ('u2', 3, 1, False, False, False)),  # not CMYKA
        (((2, 1, 1), 'u1'), ('u1', 3, 1, False, False, False)),  # not GA
        (((3, 1, 1), 'u1'), ('u1', 3, 1, False, False, False)),  # not RGB
        (((1, 3), 'u1'), ('u1', 3, 1, False, False, False)),  # not RGB
        (((1, 1, 1, 3), 'u1'), ('u1', 3, 1, False, False, False)),  # not RGB
        # auto detect pixeltype with planar set
        (((1, 1, 3), 'u1', None, True), ('u1', 3, 1, True, False, False)),
        (((1, 1, 3), 'u1', None, False), ('u1', 4, 3, False, False, False)),
        (((1, 1, 4), 'u1', None, False), ('u1', 4, 4, False, False, False)),
        (((2, 1, 3), 'u1', None, True), ('u1', 3, 2, True, False, False)),
        (((2, 1, 3), 'u1', None, False), ('u1', 4, 3, False, False, False)),
        (((2, 1, 3), 'u2', None, False), ('u2', 4, 3, False, False, False)),
        (((2, 1, 4), 'u1', None, False), ('u1', 4, 4, False, False, False)),
        (((3, 1, 3), 'u1', None, True), ('u1', 4, 3, True, False, False)),
        (((4, 1, 3), 'u1', None, True), ('u1', 4, 4, True, False, False)),
        (((4, 1, 3), 'u2', None, True), ('u2', 4, 4, True, False, False)),
        # auto detect planar with colorspace set
        (((2, 1, 1), 'u1', 'gray'), ('u1', 3, 1, False, False, False)),
        (((2, 1, 2), 'u1', 'gray'), ('u1', 3, 1, False, False, False)),
        (((2, 1, 3), 'u1', 'gray'), ('u1', 3, 1, False, False, False)),
        (((2, 1, 2), 'u1', 'graya'), ('u1', 3, 2, False, False, False)),
        (((2, 1, 3), 'u1', 'graya'), ('u1', 3, 2, True, False, False)),
        (((3, 1, 3), 'u1', 'rgb'), ('u1', 4, 3, False, False, False)),
        (((3, 1, 4), 'u1', 'rgb'), ('u1', 4, 4, False, False, False)),
        (((3, 1, 4), 'u1', 'rgba'), ('u1', 4, 4, False, False, False)),
        (((4, 1, 3), 'u1', 'rgba'), ('u1', 4, 4, True, False, False)),
        (((3, 1, 2), 'u1', 'rgb'), ('u1', 4, 3, True, False, False)),
        (((4, 1, 3), 'u1', 'cmy'), ('u1', 5, 3, False, False, False)),
        (((4, 1, 3), 'u1', 'cmyk'), ('u1', 6, 4, True, False, False)),
        (((4, 1, 4), 'u1', 'cmyk'), ('u1', 6, 4, False, False, False)),
        (((4, 1, 5), 'u1', 'cmyk'), ('u1', 6, 5, False, False, False)),
        (((4, 1, 5), 'u1', 'cmyka'), ('u1', 6, 5, False, False, False)),
        (((5, 1, 4), 'u1', 'cmyka'), ('u1', 6, 5, True, False, False)),
        # colorspace and planar set
        (((2, 1, 1), 'u1', 'gray', False), ('u1', 3, 1, False, False, False)),
        (((2, 1, 2), 'u1', 'gray', False), ('u1', 3, 1, False, False, False)),
        (((2, 1, 3), 'u1', 'gray', False), ('u1', 3, 1, False, False, False)),
        (((2, 1, 2), 'u1', 'graya', False), ('u1', 3, 2, False, False, False)),
        (((2, 1, 3), 'u1', 'graya', True), ('u1', 3, 2, True, False, False)),
        (((3, 1, 3), 'u1', 'rgb', False), ('u1', 4, 3, False, False, False)),
        (((3, 1, 4), 'u1', 'rgb', False), ('u1', 4, 4, False, False, False)),
        (((3, 1, 4), 'u1', 'rgba', False), ('u1', 4, 4, False, False, False)),
        (((4, 1, 3), 'u1', 'rgba', True), ('u1', 4, 4, True, False, False)),
        (((4, 1, 3), 'u1', 'rgb', True), ('u1', 4, 4, True, False, False)),
        (((4, 1, 3), 'u1', 'cmy', False), ('u1', 5, 3, False, False, False)),
        (((3, 1, 3), 'u1', 'cmy', True), ('u1', 5, 3, True, False, False)),
        (((4, 1, 3), 'u1', 'cmyk', True), ('u1', 6, 4, True, False, False)),
        (((4, 1, 4), 'u1', 'cmyk', False), ('u1', 6, 4, False, False, False)),
        (((4, 1, 5), 'u1', 'cmyk', False), ('u1', 6, 5, False, False, False)),
        (((4, 1, 5), 'u1', 'cmyka', False), ('u1', 6, 5, False, False, False)),
        (((5, 1, 4), 'u1', 'cmyka', True), ('u1', 6, 5, True, False, False)),
        (((5, 1, 4), 'u1', 'cmyk', True), ('u1', 6, 5, True, False, False)),
        # swapped colorspaces
        (((3, 1, 1), 'u1', 'bgr'), ('u1', 4, 3, True, True, False)),
        (((4, 1, 1), 'u1', 'bgr'), ('u1', 4, 4, True, True, False)),
        (((4, 1, 1), 'u1', 'abgr'), ('u1', 4, 4, True, True, False)),
        (((4, 1, 1), 'u1', 'bgra'), ('u1', 4, 4, True, True, True)),
        (((4, 1, 1), 'u1', 'kymc'), ('u1', 6, 4, True, True, False)),
        (((4, 1, 1), 'u1', 'kcmy'), ('u1', 6, 4, True, False, True)),
    ):
        fmt = _cms_format(*args)
        fmt = _cms_format_decode(fmt)
        assert fmt.dtype == dtype
        assert fmt.pixeltype == pixeltype
        assert fmt.samples == samples
        assert fmt.planar == planar
        assert fmt.swap == swap
        assert fmt.swapfirst == swapfirst

    for args in (
        ((5, 1, 5), 'u1', None, False),  # cannot guess 5 noncontig samples
        ((5, 1, 5), 'u1', None, True),  # cannot guess 5 contig samples
        ((5, 1, 5), 'u1', 'rgb'),  # not rgb(a)
    ):
        with pytest.raises(ValueError):
            fmt = _cms_format(*args)
            fmt = _cms_format_decode(fmt)


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='cms missing')
def test_cms():
    """Test planar sRGB float to uint16 transform."""
    # https://github.com/mm2/Little-CMS/issues/420
    from imagecodecs import cms_profile, cms_transform

    data = image_data('rgb', 'f4', planar=True)
    out = cms_transform(
        data,
        profile=cms_profile('srgb'),
        colorspace='rgb',
        outprofile=cms_profile('srgb'),
        outcolorspace='rgb',
        planar=True,
        outplanar=True,
        outdtype='u2',
        verbose=True,
    )
    assert out.shape == data.shape
    assert out.dtype == 'u2'
    assert out[:, -1, -1].tolist() == [36947, 34042, 10460]


@pytest.mark.skipif(not imagecodecs.CMS.available, reason='cms missing')
@pytest.mark.parametrize('outdtype', list('BHfde'))
@pytest.mark.parametrize('dtype', list('BHfde'))
@pytest.mark.parametrize('planar', [False, True])
@pytest.mark.parametrize('outplanar', [False, True])
@pytest.mark.parametrize('out', [None, True])
def test_cms_identity_transforms(dtype, outdtype, planar, outplanar, out):
    """Test CMS identity transforms."""
    from imagecodecs import cms_profile, cms_transform

    if outplanar and outdtype == 'e':
        pytest.xfail('half float planar output not supported')

    shape = (3, 256, 253) if planar else (256, 253, 3)
    dtype = numpy.dtype(dtype)

    outshape = (3, 256, 253) if outplanar else (256, 253, 3)
    outdtype = numpy.dtype(outdtype)
    if out:
        out = numpy.zeros(outshape, outdtype)
        outshape = None
        outdtype = None

    if dtype.kind == 'u':
        data = numpy.random.randint(
            0, 2 ** (dtype.itemsize * 8) - 1, shape, dtype
        )
    else:
        data = numpy.random.rand(*shape).astype(dtype)

    output = cms_transform(
        data,
        profile=cms_profile('srgb'),
        colorspace='rgb',
        outprofile=cms_profile('srgb'),
        outcolorspace='rgb',
        planar=planar,
        outplanar=outplanar,
        outdtype=outdtype,
        verbose=True,
        out=out,
    )
    if out is None:
        out = output
    if dtype == out.dtype or (dtype.kind == 'f' and out.dtype.kind == 'f'):
        if shape != out.shape:
            if planar:
                out = numpy.moveaxis(out, -1, 0)
            else:
                out = numpy.moveaxis(out, 0, -1)
        if dtype.kind == 'u':
            assert_array_equal(data, out)
        else:
            assert_allclose(data, out, atol=0.001)
    else:
        # TODO: how to verify?
        pass


# test data from https://entropymine.com/jason/bmpsuite
BMP_PATH = DATA_PATH / 'bmpsuite/g'
BMP_FILES = [
    os.path.split(f)[-1][:-4] for f in glob.glob(str(BMP_PATH / '*.bmp'))
]


@pytest.mark.skipif(not imagecodecs.BMP.available, reason='bmp missing')
@pytest.mark.parametrize('name', BMP_FILES)
def test_bmpsuite(name):
    """Test BMP codec with bmpsuite files."""
    filename = BMP_PATH / f'{name}.bmp'
    with open(filename, 'rb') as fh:
        encoded = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    assert imagecodecs.bmp_check(encoded)
    try:
        decoded = imagecodecs.bmp_decode(encoded)
    except imagecodecs.BmpError as exc:
        pytest.xfail(str(exc))

    if 'nonsquare' in name:
        pass
    elif decoded.shape == 2:
        assert decoded[19, 98] == 0
    else:
        assert tuple(decoded[19, 98]) == (0, 0, 0)

    assert_array_equal(
        decoded,
        imagecodecs.bmp_decode(imagecodecs.bmp_encode(decoded, ppm=3780)),
    )


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('smoothing', [0, 25])
@pytest.mark.parametrize('subsampling', ['444', '422', '420', '411', '440'])
@pytest.mark.parametrize('itype', ['rgb', 'rgba', 'gray'])
@pytest.mark.parametrize('codec', ['jpeg8', 'jpeg12', 'mozjpeg'])
def test_jpeg_encode(codec, itype, subsampling, smoothing, optimize):
    """Test various JPEG encode options."""
    # general and default options are tested in test_image_roundtrips
    atol = 24 if subsampling != '411' else 48
    if codec == 'jpeg8':
        if not imagecodecs.JPEG8.available:
            pytest.skip('jpeg8 missing')
        dtype = 'uint8'
        decode = imagecodecs.jpeg8_decode
        encode = imagecodecs.jpeg8_encode
    elif codec == 'jpeg12':
        if not imagecodecs.JPEG8.available:
            pytest.skip('jpeg8 missing')
        if imagecodecs.JPEG.legacy:
            pytest.skip('JPEG12 not supported')
        # if not optimize:
        #     pytest.xfail('jpeg12 fails without optimize')
        dtype = 'uint16'
        decode = imagecodecs.jpeg8_decode
        encode = imagecodecs.jpeg8_encode
        atol = atol * 16
    elif codec == 'mozjpeg':
        if not imagecodecs.MOZJPEG.available:
            pytest.skip('mozjpeg missing')
        dtype = 'uint8'
        decode = imagecodecs.mozjpeg_decode
        encode = imagecodecs.mozjpeg_encode
    else:
        raise ValueError(codec)

    dtype = numpy.dtype(dtype)
    data = image_data(itype, dtype)
    data = data[:32, :16].copy()  # make divisible by subsamples

    encoded = encode(
        data,
        level=95,
        subsampling=subsampling,
        smoothing=smoothing,
        optimize=optimize,
    )
    decoded = decode(encoded)

    if itype == 'gray':
        decoded = decoded.reshape(data.shape)

    assert_allclose(data, decoded, atol=atol)


@pytest.mark.skipif(not imagecodecs.JPEG8.available, reason='jpeg8 missing')
@pytest.mark.parametrize('output', ['new', 'out'])
def test_jpeg8_decode(output):
    """Test JPEG 8-bit decoder with separate tables."""
    decode = imagecodecs.jpeg8_decode
    data = readfile('bytes.jpeg8.bin')
    tables = readfile('bytes.jpeg8_tables.bin')

    if output == 'new':
        decoded = decode(data, tables=tables)
    elif output == 'out':
        decoded = numpy.zeros_like(BYTESIMG)
        decode(data, tables=tables, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(BYTESIMG.size * BYTESIMG.itemsize)
        decoded = decode(data, out=decoded)
    assert_array_equal(BYTESIMG, decoded)


@pytest.mark.skipif(
    imagecodecs.JPEG.legacy or not imagecodecs.JPEG8.available,
    reason='jpeg12 missing',
)
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpeg12_decode(output):
    """Test JPEG 12-bit decoder with separate tables."""
    decode = imagecodecs.jpeg8_decode
    data = readfile('words.jpeg12.bin')
    tables = readfile('words.jpeg12_tables.bin')

    if output == 'new':
        decoded = decode(data, tables=tables)
    elif output == 'out':
        decoded = numpy.zeros_like(WORDSIMG)
        decode(data, tables=tables, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(WORDSIMG.size * WORDSIMG.itemsize)
        decoded = decode(data, out=decoded)

    assert (
        numpy.max(
            numpy.abs(WORDSIMG.astype('int32') - decoded.astype('int32'))
        )
        < 2
    )


@pytest.mark.skipif(not imagecodecs.JPEG8.available, reason='jpeg8 missing')
def test_jpeg_rgb_mode():
    """Test JPEG encoder in RGBA mode."""
    # https://github.com/cgohlke/tifffile/issues/146
    RGB = imagecodecs.JPEG8.CS.RGB
    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.jpeg_encode(
        data, colorspace=RGB, outcolorspace=RGB, subsampling='444', level=99
    )
    assert b'JFIF' not in encoded[:16]
    decoded = imagecodecs.jpeg_decode(
        encoded,
        colorspace=RGB,
        outcolorspace=RGB,
    )
    assert_allclose(data, decoded, atol=8)


@pytest.mark.skipif(not imagecodecs.JPEG8.available, reason='jpeg8 missing')
def test_jpeg_bitspersample():
    """Test JPEG encoder with bitspersample=8, lossless=False."""
    # https://github.com/cgohlke/imagecodecs/issues/116
    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.jpeg_encode(
        data,
        colorspace='RGB',
        outcolorspace='RGB',
        subsampling='444',
        level=99,
        bitspersample=8,
        lossless=False,
    )
    assert b'JFIF' not in encoded[:16]
    decoded = imagecodecs.jpeg_decode(
        encoded,
        colorspace='RGB',
        outcolorspace='RGB',
    )
    assert_allclose(data, decoded, atol=8)


@pytest.mark.skipif(
    not imagecodecs.MOZJPEG.available, reason='mozjpeg missing'
)
def test_mozjpeg():
    """Test MOZJPEG codec parameters."""
    data = readfile('bytes.jpeg8.bin')
    tables = readfile('bytes.jpeg8_tables.bin')
    decoded = imagecodecs.mozjpeg_decode(data, tables=tables)
    assert_array_equal(BYTESIMG, decoded)

    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.mozjpeg_encode(
        data,
        level=90,
        outcolorspace='ycbcr',
        subsampling='444',
        quanttable=2,
        notrellis=True,
    )
    decoded = imagecodecs.mozjpeg_decode(encoded)
    assert_allclose(data, decoded, atol=16, rtol=0)


@pytest.mark.parametrize('codec', ['jpeg8', 'ljpeg', 'jpegsof3'])
@pytest.mark.parametrize(
    'fname, result',
    [
        ('1px.ljp', ((1, 1), 'uint16', (0, 0), 0)),
        ('2ch.ljp', ((3528, 2640, 2), 'uint16', (1500, 1024, 1), 3195)),
        ('2dht.ljp', ((288, 384, 3), 'uint8', (22, 56), (150, 67, 166))),
        ('3dht.ljp', ((240, 320, 3), 'uint8', (140, 93), (184, 161, 110))),
        ('gray16.ljp', ((535, 800), 'uint16', (418, 478), 54227)),
        ('gray8.ljp', ((535, 800), 'uint8', (418, 478), 211)),
        ('rgb24.ljp', ((535, 800, 3), 'uint8', (418, 478), (226, 209, 190))),
        ('dng0.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng1.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng2.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng3.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng4.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng5.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng6.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dng7.ljp', ((256, 256), 'uint16', (111, 75), 51200)),
        ('dcm1-8bit.ljp', ((512, 512), 'uint8', (256, 256), 51)),
        ('dcm1.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm2.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm3.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm4.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm5.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm6.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        ('dcm7.ljp', ((256, 256), 'uint16', (169, 97), 1192)),
        # tile from Apple DNG
        ('linearraw.ljp', ((378, 504, 3), 'uint16', (20, 30), (114, 212, 88))),
        # https://github.com/cgohlke/imagecodecs/issues/61
        ('pvrg.ljp', ((4608, 2928), 'uint16', (823, 2166), 3050)),
    ],
)
def test_ljpeg(fname, result, codec):
    """Test Lossless JPEG decoders."""
    kwargs = {}
    if codec == 'jpeg8':
        if not imagecodecs.JPEG8.available:
            pytest.skip('jpeg8 missing')
        if imagecodecs.JPEG.legacy:
            pytest.skip('jpeg8 does not support lossless')
        decode = imagecodecs.jpeg8_decode
        check = imagecodecs.jpeg8_check
        if fname in {'2dht.ljp', 'linearraw.ljp'}:
            kwargs["colorspace"] = "YCBCR"
        elif not imagecodecs.JPEG8.all_precisions:
            # libjpeg-turbo 3.0.x
            if fname in {
                '2ch.ljp',  # Unsupported JPEG data precision 14
                # '2dht.ljp',  # Unsupported color conversion request
                'rgb24.ljp',  # Unsupported color conversion request
                # 'linearraw.ljp',  # Unsupported color conversion request
                'dng0.ljp',  # Invalid progressive/lossless parameters Ss=0 ...
                # 'dng1.ljp',  # Bogus Huffman table definition
                # 'dng2.ljp',  # Bogus Huffman table definition
                # 'dng3.ljp',  # Bogus Huffman table definition
                # 'dng4.ljp',  # Bogus Huffman table definition
                # 'dng5.ljp',  # Bogus Huffman table definition
                # 'dng6.ljp',  # Bogus Huffman table definition
                # 'dng7.ljp',  # Bogus Huffman table definition
            }:
                pytest.xfail('libjpeg-turbo does not support this case')
        elif fname in {'rgb24.ljp', 'dng0.ljp'}:
            pytest.xfail('libjpeg-turbo does not support this case')
        # elif fname in {'2dht.ljp', 'linearraw.ljp'}:
        #     kwargs["colorspace"] = "YCBCR"
    elif codec == 'ljpeg':
        if not imagecodecs.LJPEG.available:
            pytest.skip('ljpeg missing')
        decode = imagecodecs.ljpeg_decode
        check = imagecodecs.ljpeg_check
    else:
        if not imagecodecs.JPEGSOF3.available:
            pytest.skip('jpegsof3 missing')
        if fname in {'dcm6.ljp', 'dcm7.ljp'}:
            return  # jpegsof3 segfault
        if fname == 'dng0.ljp':
            pytest.xfail('jpegsof3 known failure or crash')
        decode = imagecodecs.jpegsof3_decode
        check = imagecodecs.jpegsof3_check
    if fname == 'pvrg.ljp':
        pytest.xfail('corrupted LJPEG produced by PVRG')

    try:
        data = readfile(os.path.join('ljpeg', fname))
    except FileNotFoundError:
        pytest.skip(f'{fname} not found')

    assert check(data) in {None, True}
    decoded = decode(data, **kwargs)

    shape, dtype, index, value = result
    assert decoded.shape == shape
    assert decoded.dtype == dtype
    assert_array_equal(decoded[index], value)


@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('fname', ['gray8.ljp', 'gray16.ljp'])
@pytest.mark.parametrize('codec', ['jpeg8', 'jpegsof3', 'ljpeg'])
def test_jpegsof3(fname, output, codec):
    """Test JPEG SOF3 decoder with 8 and 16-bit images."""
    if codec == 'jpeg8':
        if not imagecodecs.JPEG8.available:
            pytest.skip('jpeg8 missing')
        if imagecodecs.JPEG.legacy:
            pytest.skip('jpeg8 does not support lossless')
        decode = imagecodecs.jpeg8_decode
        check = imagecodecs.jpeg8_check
    elif codec == 'ljpeg':
        if not imagecodecs.LJPEG.available:
            pytest.skip('ljpeg missing')
        decode = imagecodecs.ljpeg_decode
        check = imagecodecs.ljpeg_check
    else:
        if not imagecodecs.JPEGSOF3.available:
            pytest.skip('jpegsof3 missing')
        decode = imagecodecs.jpegsof3_decode
        check = imagecodecs.jpegsof3_check

    shape = 535, 800
    if fname == 'gray8.ljp':
        dtype = 'uint8'
        value = 75
        memmap = True  # test read-only, jpegsof3_decode requires writable
    elif fname == 'gray16.ljp':
        dtype = 'uint16'
        value = 19275
        memmap = False

    data = readfile(os.path.join('ljpeg', fname), memmap=memmap)

    assert check(data) in {None, True}

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(535 * 800 * numpy.dtype(dtype).itemsize)
        decoded = decode(data, out=decoded)

    assert decoded.shape == shape
    assert decoded.dtype == dtype
    assert decoded[500, 600] == value


@pytest.mark.skipif(not imagecodecs.JPEGXL.available, reason='jpegxl missing')
@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'float16', 'float32'])
def test_jpegxl_planar(dtype):
    """Test JPEG XL roundtrip with frames and planar channels."""
    image = image_data('channels', dtype, planar=True, frames=True)
    assert image.shape == (11, 8, 32, 31)
    encoded = imagecodecs.jpegxl_encode(image, photometric='gray', planar=True)
    decoded = imagecodecs.jpegxl_decode(encoded)
    assert_array_equal(image, decoded)


@pytest.mark.skipif(not imagecodecs.JPEGXL.available, reason='jpegxl missing')
def test_jpegxl_bitspersample():
    """Test JPEG XL with 12 bitspersample."""
    image = image_data('rgb', 'uint16')
    image >>= 4
    encoded = imagecodecs.jpegxl_encode(image, bitspersample=12)
    # TODO: verify that encoded is a 12-bit image
    # with open('12bit.jxl', 'wb') as fh: fh.write(encoded)
    # jxlinfo 12bit.jxl -> JPEG XL image, 31x32, lossless, 12-bit RGB
    decoded = imagecodecs.jpegxl_decode(encoded)
    assert_array_equal(image, decoded)


@pytest.mark.skipif(
    not imagecodecs.JPEGXL.available or not imagecodecs.JPEG.available,
    reason='jpegxl or jpeg missing',
)
def test_jpegxl_transcode():
    """Test JPEG XL transcoding to/from JPEG."""
    data = image_data('rgb', 'uint8')
    jpeg1 = imagecodecs.jpeg8_encode(data)
    jpegxl = imagecodecs.jpegxl_encode_jpeg(jpeg1)
    jpeg2 = imagecodecs.jpegxl_decode_jpeg(jpegxl)
    assert isinstance(jpeg2, (bytes, bytearray))
    assert imagecodecs.jpeg8_check(jpeg2)
    assert_array_equal(
        imagecodecs.jpeg8_decode(jpeg1), imagecodecs.jpeg8_decode(jpeg2)
    )
    out = bytearray(len(jpeg2))
    imagecodecs.jpegxl_decode_jpeg(jpegxl, out=out)
    assert_array_equal(
        imagecodecs.jpeg8_decode(jpeg1), imagecodecs.jpeg8_decode(out)
    )


@pytest.mark.skipif(not imagecodecs.JPEGXR.available, reason='jpegxr missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpegxr_decode(output):
    """Test JPEG XR decoder with RGBA32 image."""
    decode = imagecodecs.jpegxr_decode
    image = readfile('rgba32.jxr.bin')
    image = numpy.frombuffer(image, dtype='uint8').reshape(100, 100, -1)
    data = readfile('rgba32.jxr')

    assert imagecodecs.jpegxr_check(data) in {None, True}

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.zeros_like(image)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(image.size * image.itemsize)
        decoded = decode(data, out=decoded)
    assert_array_equal(image, decoded)


@pytest.mark.skipif(not imagecodecs.JPEGXR.available, reason='jpegxr missing')
@pytest.mark.parametrize('fp2int', [False, True])
def test_jpegxr_fixedpoint(fp2int):
    """Test JPEG XR decoder with Fixed Point 16 image."""
    # test file provided by E. Pojar on 2021.1.27
    data = readfile('fixedpoint.jxr')
    assert imagecodecs.jpegxr_check(data) in {None, True}
    decoded = imagecodecs.jpegxr_decode(data, fp2int=fp2int)
    if fp2int:
        assert decoded.dtype == 'int16'
        assert decoded[0, 0] == -32765
        assert decoded[255, 255] == 32766
    else:
        assert decoded.dtype == 'float32'
        assert abs(decoded[0, 0] + 3.9996338) < 1e-6
        assert abs(decoded[255, 255] - 3.9997559) < 1e-6


@pytest.mark.skipif(not imagecodecs.AVIF.available, reason='avif missing')
def test_avif_strict_disabled():
    """Test AVIF decoder with file created by old version of libavif."""
    data = readfile('rgba.u1.strict_disabled.avif')
    assert imagecodecs.avif_check(data)
    decoded = imagecodecs.avif_decode(data)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (32, 31, 4)
    assert tuple(decoded[16, 16]) == (44, 123, 57, 88)


@pytest.mark.skipif(not IS_CG, reason='avif missing')
@pytest.mark.parametrize(
    'codec', ['auto', 'aom', 'rav1e', 'svt']  # 'libgav1', 'avm'
)
def test_avif_encoder(codec):
    """Test various AVIF encoder codecs."""
    data = numpy.load(datafiles('rgb.u1.npy'))
    if codec == 'svt':
        if IS_ARM64 or not IS_CG:
            pytest.skip('AVIF SVT not supported')
        data = data[:200, :300]
        pixelformat = '420'
    else:
        pixelformat = None
    encoded = imagecodecs.avif_encode(
        data, level=95, codec=codec, pixelformat=pixelformat, numthreads=2
    )
    decoded = imagecodecs.avif_decode(encoded, numthreads=2)
    assert_allclose(decoded, data, atol=6, rtol=0)


@pytest.mark.skipif(not imagecodecs.JPEGLS.available, reason='jpegls missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpegls_decode(output):
    """Test JPEG LS decoder with RGBA32 image."""
    decode = imagecodecs.jpegls_decode
    data = readfile('rgba.u1.jls')
    dtype = 'uint8'
    shape = 32, 31, 4

    assert imagecodecs.jpegls_check(data) in {None, True}

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 97
    assert decoded[-1, -1, -1] == 63


@pytest.mark.skipif(
    not imagecodecs.BRUNSLI.available, reason='brunsli missing'
)
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_brunsli_decode(output):
    """Test Brunsli decoder with RGBA32 image."""
    decode = imagecodecs.brunsli_decode
    data = readfile('rgba.u1.br')
    dtype = 'uint8'
    shape = 32, 31, 4

    assert imagecodecs.brunsli_check(data) in {None, True}

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 100
    assert decoded[-1, -1, -1] == 81


@pytest.mark.skipif(
    not imagecodecs.BRUNSLI.available, reason='brunsli missing'
)
def test_brunsli_encode_jpeg():
    """Test Brunsli encoder with JPEG input."""
    encode = imagecodecs.brunsli_encode
    decode = imagecodecs.brunsli_decode
    jpg = readfile('rgba.u1.jpg')
    jxl = readfile('rgba.u1.br')

    assert imagecodecs.brunsli_check(jpg) in {None, True}
    assert imagecodecs.brunsli_check(jxl) in {None, True}

    encoded = encode(jpg)
    assert encoded == jxl

    decoded = decode(encoded)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (32, 31, 4)
    assert decoded[25, 25, 1] == 100
    assert decoded[-1, -1, -1] == 81


@pytest.mark.skipif(not imagecodecs.WEBP.available, reason='webp missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_webp_decode(output):
    """Test WebP decoder with RGBA32 image."""
    decode = imagecodecs.webp_decode
    data = readfile('rgba.u1.webp')
    dtype = 'uint8'
    shape = 32, 31, 4

    assert imagecodecs.webp_check(data)

    if output == 'new':
        decoded = decode(data)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[25, 25, 1] == 94  # lossy
    assert decoded[-1, -1, -1] == 63


@pytest.mark.skipif(not imagecodecs.WEBP.available, reason='webp missing')
@pytest.mark.parametrize('index', [0, 10, -2])
def test_webp_animated(index):
    """Test WebP decoder with animated image."""
    # gif2webp frames.u1.gif -kmin 1 -kmax 1 -o frames.webp
    decode = imagecodecs.webp_decode
    # TODO: test animation with partial frames
    data = readfile('frames.webp')
    assert imagecodecs.webp_check(data)
    decoded = decode(data, index=index)
    expected = image_data('gray', 'uint8', frames=True)[..., 0]
    assert_array_equal(decoded[:, :, 0], expected[index])


@pytest.mark.skipif(not imagecodecs.WEBP.available, reason='webp missing')
def test_webp_opaque():
    """Test WebP roundtrip with opaque image."""
    # libwebp drops all-opaque alpha channel
    data = image_data('rgba', 'uint8')
    data[..., 3] = 255

    encoded = imagecodecs.webp_encode(data, level=90, lossless=True, method=5)
    decoded = imagecodecs.webp_decode(encoded)
    assert decoded.shape == (data.shape[0], data.shape[1], 3)
    assert_array_equal(decoded, data[..., :3])

    decoded = imagecodecs.webp_decode(encoded, hasalpha=True)
    assert_array_equal(decoded, data)


@pytest.mark.skipif(not imagecodecs.ZFP.available, reason='zfp missing')
@pytest.mark.parametrize('execution', [None, 'omp'])
@pytest.mark.parametrize('mode', [(None, None), ('p', None)])  # ('r', 24)
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])  # 'view',
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgba', 'view', 'gray', 'line'])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_zfp(dtype, itype, enout, deout, mode, execution):
    """Test ZFP codec."""
    kwargs = {}
    if execution == 'omp':
        if os.environ.get('SKIP_OMP', False):
            pytest.skip('omp test skip because of environment variable')
        kwargs['numthreads'] = 2
        kwargs['chunksize'] = None
    decode = imagecodecs.zfp_decode
    encode = imagecodecs.zfp_encode
    mode, level = mode
    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape

    kwargs = dict(mode=mode, level=level, execution=execution, **kwargs)
    encoded = encode(data, **kwargs)

    assert imagecodecs.zfp_check(encoded)

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, out=decoded)
    elif deout == 'view':
        temp = numpy.zeros((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2 : 2 + shape[0], 3 : 3 + shape[1], :]
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


@pytest.mark.skipif(not imagecodecs.SZ3.available, reason='sz3 missing')
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgba', 'view', 'gray', 'line'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_sz3(dtype, itype, enout, deout):
    """Test SZ3 codec."""
    decode = imagecodecs.sz3_decode
    encode = imagecodecs.sz3_encode

    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape
    atol = 1e-3

    kwargs = dict(mode=imagecodecs.SZ3.MODE.ABS, abs=atol)
    encoded = encode(data, **kwargs)

    assert imagecodecs.sz3_check(encoded) is None

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    kwargs = dict(shape=shape, dtype=dtype)
    if deout == 'new':
        decoded = decode(encoded, **kwargs)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, out=decoded, **kwargs)
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded, **kwargs)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    assert_allclose(data, decoded, atol=atol, rtol=0)


@pytest.mark.skipif(
    not imagecodecs.ULTRAHDR.available, reason='ultrahdr missing'
)
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_ultrahdr_decode(output):
    """Test Ultra HDR decoder with image."""
    # TODO: fails on Apple ARM and AArch64, but not win-arm64
    data = readfile('rgba.uhdr')
    dtype = numpy.float16
    shape = 32, 31, 4

    assert imagecodecs.ultrahdr_check(data)

    if output == 'new':
        decoded = imagecodecs.ultrahdr_decode(data)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        imagecodecs.ultrahdr_decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * 2)
        decoded = imagecodecs.ultrahdr_decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert_allclose(decoded[25, 25, 1], 0.4668, atol=0.01)
    assert_allclose(decoded[-1, -1, -1], 1.0, atol=0.01)


@pytest.mark.skipif(
    not imagecodecs.ULTRAHDR.available, reason='ultrahdr missing'
)
def test_ultrahdr():
    """Test Ultra HDR codec."""
    uhdr = readfile('rgba.uhdr')

    assert imagecodecs.ultrahdr_check(uhdr)
    rgba = imagecodecs.ultrahdr_decode(uhdr, dtype=numpy.float16)
    assert rgba.dtype == numpy.float16
    assert rgba.shape == (32, 31, 4)
    assert_allclose(rgba[0, 0], [0.5405, 0.7485, 0.62, 1], atol=0.01)

    rgba = imagecodecs.ultrahdr_decode(uhdr, dtype=numpy.uint8)
    assert rgba.dtype == numpy.uint8
    assert rgba.shape == (32, 31, 4)
    # TODO: SDR is way too bright
    # assert_allclose(rgba[0, 0], [128, 196, 160, 255])
    assert_allclose(rgba[0, 0], [151, 175, 162, 255])

    rgba = imagecodecs.ultrahdr_decode(uhdr, dtype=numpy.uint32, transfer=1)
    assert rgba.dtype == numpy.uint32
    assert rgba.shape == (32, 31)
    assert rgba[0, 0] == 3940235925  # 3894166016

    uhdr = imagecodecs.ultrahdr_encode(rgba, level=100, transfer=1, gamut=1)

    assert imagecodecs.ultrahdr_check(uhdr)
    rgba = imagecodecs.ultrahdr_decode(uhdr, transfer=0)
    # TODO: values not identical
    # assert_allclose(rgba[0, 0], [0.4941, 0.7524, 0.6313, 1.0], atol=0.01)
    assert_allclose(rgba[0, 0], [0.5244, 0.734, 0.5903, 1.0], atol=0.01)


@pytest.mark.skipif(
    not imagecodecs.ULTRAHDR.available, reason='ultrahdr missing'
)
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgba'])
def test_ultrahdr_roundtrip(itype, enout, deout):
    """Test Ultra HDR codec."""
    decode = imagecodecs.ultrahdr_decode
    encode = imagecodecs.ultrahdr_encode

    dtype = numpy.dtype(numpy.float16)
    data = image_data(itype, numpy.float32)
    data /= data.max()
    data[..., -1] = 1.0  # TODO: decode always return alpha=1.0?
    data = data.astype(dtype)
    shape = data.shape

    kwargs = {'level': 100}
    encoded = encode(data, **kwargs)

    assert imagecodecs.ultrahdr_check(encoded)

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    kwargs = {'dtype': dtype}
    if deout == 'new':
        decoded = decode(encoded, **kwargs)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, out=decoded, **kwargs)
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * 2)
        decoded = decode(encoded, out=decoded, **kwargs)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    assert_allclose(data, decoded, atol=0.1)


@pytest.mark.skipif(
    not imagecodecs.SPERR.available or SKIP_DEBUG, reason='sperr missing'
)
@pytest.mark.parametrize('mode', ['bpp', 'psnr', 'pwe'])
@pytest.mark.parametrize('deout', ['new', 'out'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['gray', 'stack'])
@pytest.mark.parametrize('header', [True, False])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_sperr(dtype, itype, enout, deout, mode, header):
    """Test SPERR codec."""
    decode = imagecodecs.sperr_decode
    encode = imagecodecs.sperr_encode

    dtype = numpy.dtype(dtype)
    data = numpy.squeeze(image_data(itype, dtype))
    shape = data.shape

    if mode == 'bpp':
        level = 16.0
        atol = 1e-6
    elif mode == 'psnr':
        level = 100.0
        atol = 1e-4
    elif mode == 'pwe':
        level = 100.0
        atol = 0.5

    kwargs = dict(header=header, level=level, mode=mode)
    encoded = encode(data, **kwargs)
    encoded_len = len(encoded)

    assert imagecodecs.sperr_check(encoded) is None

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(encoded_len, 'uint8')
        encode(data, out=encoded, **kwargs)
        encoded = encoded[:encoded_len]
    elif enout == 'bytearray':
        encoded = bytearray(encoded_len)
        encode(data, out=encoded, **kwargs)

    assert len(encoded) == encoded_len

    if deout == 'new':
        if not header:
            kwargs = dict(shape=shape, dtype=dtype)
        else:
            kwargs = {}
        decoded = decode(encoded, header=header, **kwargs)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, header=header, out=decoded)

    assert_allclose(data, decoded, atol=atol, rtol=0)


@pytest.mark.skipif(not imagecodecs.LERC.available, reason='lerc missing')
# @pytest.mark.parametrize('version', [None, 3])
@pytest.mark.parametrize('level', [None, 0.02])
@pytest.mark.parametrize('planar', [None, True])
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['gray', 'rgb', 'rgba', 'channels', 'stack'])
@pytest.mark.parametrize(
    'dtype', ['uint8', 'int8', 'uint16', 'int32', 'float32', 'float64']
)
def test_lerc(dtype, itype, enout, deout, planar, level, version=None):
    """Test LERC codec."""
    if version is not None and version < 4 and itype != 'gray':
        pytest.xfail('lerc version does not support this case')
    decode = imagecodecs.lerc_decode
    encode = imagecodecs.lerc_encode
    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape
    if level is not None and dtype.kind != 'f':
        level = level * 256

    kwargs = dict(level=level, version=version, planar=planar)
    encoded = encode(data, **kwargs)

    assert imagecodecs.lerc_check(encoded)

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        out = decoded if planar else numpy.squeeze(decoded)
        decode(encoded, out=out)
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    if itype == 'gray':
        decoded = decoded.reshape(shape)

    if level is None:
        level = 0.00001 if dtype.kind == 'f' else 0
    assert_allclose(data, decoded, atol=level, rtol=0)


@pytest.mark.skipif(not imagecodecs.LERC.available, reason='lerc missing')
@pytest.mark.parametrize(
    'file',
    [
        'world.lerc1',
        'california_400_400_1_float.lerc2',
        'bluemarble_256_256_3_byte.lerc2',
        'zstd.lerc2',
    ],
)
def test_lerc_files(file):
    """Test LERC decoder with lerc testData files."""
    with open(datafiles(f'lerc/{file}'), 'rb') as fh:
        encoded = fh.read()

    decoded = imagecodecs.lerc_decode(encoded, masks=False)
    decoded1, masks = imagecodecs.lerc_decode(encoded, masks=True)

    assert_array_equal(decoded, decoded1)

    if file[:4] != 'zstd':
        out = numpy.zeros_like(masks)
        decoded1, _ = imagecodecs.lerc_decode(encoded, masks=out)
        assert_array_equal(masks, out)

    if file[:5] == 'world':
        assert decoded.dtype == numpy.float32
        assert decoded.shape == (257, 257)
        assert int(decoded[146, 144]) == 1131
        assert masks.dtype == bool
        assert masks.shape == (257, 257)
        assert masks[146, 144] == bool(1)
    elif file[:4] == 'cali':
        assert decoded.dtype == numpy.float32
        assert decoded.shape == (400, 400)
        assert int(decoded[200, 200]) == 1554
        assert masks.dtype == bool
        assert masks.shape == (400, 400)
        assert masks[200, 200] == bool(1)
    elif file[:4] == 'blue':
        assert decoded.dtype == numpy.uint8
        assert decoded.shape == (3, 256, 256)
        assert tuple(decoded[:, 128, 128]) == (2, 5, 20)
        assert masks.dtype == bool
        assert masks.shape == (256, 256)
        assert masks[128, 128] == bool(1)
    elif file[:4] == 'zstd':
        assert decoded.dtype == numpy.uint8
        assert decoded.shape == (512, 512, 3)
        assert tuple(decoded[128, 128]) == (85, 89, 38)
        assert masks is None


@pytest.mark.skipif(not imagecodecs.LERC.available, reason='lerc missing')
@pytest.mark.parametrize('compression', [None, 'zstd', 'deflate'])
def test_lerc_compression(compression):
    """Test LERC with compression."""
    data = image_data('rgb', 'uint16')
    compressionargs = {
        None: None,
        'zstd': {'level': 10},
        'deflate': {'level': 7},
    }[compression]
    compressed = imagecodecs.lerc_encode(
        data, compression=compression, compressionargs=compressionargs
    )
    decompressed = imagecodecs.lerc_decode(compressed)
    assert_array_equal(data, decompressed)


@pytest.mark.skipif(
    not imagecodecs.LERC.available or IS_PYPY, reason='lerc missing'
)
def test_lerc_masks():
    """Test LERC codec with masks."""

    stack = image_data('stack', numpy.float32)
    masks = image_data('stack', bool)

    # 1 band, no mask
    data = stack[0]
    encoded = imagecodecs.lerc_encode(data)
    decoded, masks1 = imagecodecs.lerc_decode(encoded, masks=True)
    assert masks1 is None
    assert_allclose(data, decoded, atol=0.00001, rtol=0)

    # 1 band, 1 mask
    data = stack[0]
    encoded = imagecodecs.lerc_encode(data, masks=masks[0])
    decoded, masks1 = imagecodecs.lerc_decode(encoded, masks=True)
    assert_array_equal(masks[0], masks1)

    # 1 band, 3 masks
    data = stack[:3]
    encoded = imagecodecs.lerc_encode(data, masks=masks[0], planar=True)
    decoded, masks1 = imagecodecs.lerc_decode(encoded, masks=True)
    assert_array_equal(masks[0], masks1)

    # 3 bands, 3 masks
    data = stack[:3]
    encoded = imagecodecs.lerc_encode(
        data,
        masks=masks[:3],
        planar=True,
    )
    decoded, masks1 = imagecodecs.lerc_decode(encoded, masks=True)
    assert_array_equal(masks[:3], masks1)

    # out
    out = numpy.zeros_like(masks[:3])
    decoded, _ = imagecodecs.lerc_decode(encoded, masks=out)
    assert_array_equal(masks[:3], out)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('output', ['new', 'out', 'bytearray'])
def test_jpeg2k_int8_4bit(output):
    """Test JPEG 2000 decoder with int8, 4-bit image."""
    decode = imagecodecs.jpeg2k_decode
    data = readfile('int8_4bit.j2k')
    dtype = 'int8'
    shape = 256, 256

    assert imagecodecs.jpeg2k_check(data)

    if output == 'new':
        decoded = decode(data, verbose=2)
    elif output == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(data, out=decoded)
    elif output == 'bytearray':
        decoded = bytearray(shape[0] * shape[1])
        decoded = decode(data, out=decoded)

    assert decoded.dtype == dtype
    assert decoded.shape == shape
    assert decoded[0, 0] == -6
    assert decoded[-1, -1] == 2


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
def test_jpeg2k_ycbc():
    """Test JPEG 2000 decoder with subsampling."""
    decode = imagecodecs.jpeg2k_decode
    data = readfile('ycbc.j2k')

    assert imagecodecs.jpeg2k_check(data)

    decoded = decode(data, verbose=2)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (256, 256, 3)
    assert tuple(decoded[0, 0]) == (243, 243, 240)
    assert tuple(decoded[-1, -1]) == (0, 0, 0)

    decoded = decode(data, verbose=2, planar=True)
    assert decoded.dtype == 'uint8'
    assert decoded.shape == (3, 256, 256)
    assert tuple(decoded[:, 0, 0]) == (243, 243, 240)
    assert tuple(decoded[:, -1, -1]) == (0, 0, 0)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('codecformat', [0, 2])
def test_jpeg2k_codecformat(codecformat):
    """Test JPEG 2000 codecformats."""
    data = image_data('rgb', 'uint16')
    encoded = imagecodecs.jpeg2k_encode(
        data, codecformat=codecformat, verbose=2
    )
    assert imagecodecs.jpeg2k_check(encoded) in {None, True}
    decoded = imagecodecs.jpeg2k_decode(encoded, verbose=2)
    assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('numthreads', [1, 2])
def test_jpeg2k_numthreads(numthreads):
    """Test JPEG 2000 numthreads."""
    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.jpeg2k_encode(data, numthreads=numthreads, verbose=2)
    assert imagecodecs.jpeg2k_check(encoded) in {None, True}
    decoded = imagecodecs.jpeg2k_decode(
        encoded, numthreads=numthreads, verbose=2
    )
    assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('reversible', [False, True])
def test_jpeg2k_reversible(reversible):
    """Test JPEG 2000 reversible."""
    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.jpeg2k_encode(
        data, level=50, reversible=reversible, verbose=2
    )
    assert imagecodecs.jpeg2k_check(encoded) in {None, True}
    decoded = imagecodecs.jpeg2k_decode(encoded, verbose=2)
    assert_allclose(data, decoded, atol=8)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('mct', [False, True])
def test_jpeg2k_mct(mct):
    """Test JPEG 2000 mct."""
    data = image_data('rgb', 'uint8')
    encoded = imagecodecs.jpeg2k_encode(data, level=50, mct=mct, verbose=2)
    assert imagecodecs.jpeg2k_check(encoded) in {None, True}
    decoded = imagecodecs.jpeg2k_decode(encoded, verbose=2)
    assert_allclose(data, decoded, atol=8)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
def test_jpeg2k_samples():
    """Test JPEG 2000 decoder with many samples."""
    data = image_data('channels', 'uint16')
    encoded = imagecodecs.jpeg2k_encode(data, verbose=2)
    assert imagecodecs.jpeg2k_check(encoded)
    decoded = imagecodecs.jpeg2k_decode(encoded, verbose=2)
    assert_array_equal(data, decoded)

    data = numpy.moveaxis(data, -1, 0)
    encoded = imagecodecs.jpeg2k_encode(data, planar=True, verbose=2)
    decoded = imagecodecs.jpeg2k_decode(encoded, planar=True, verbose=2)
    assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
def test_jpeg2k_realloc(caplog):
    """Test JPEG 2000 decoder with output larger than input."""
    # https://github.com/cgohlke/imagecodecs/issues/101
    data = numpy.random.randint(0, 255, (256, 256), numpy.uint8)
    encoded = imagecodecs.jpeg2k_encode(data, level=0, verbose=3)
    assert imagecodecs.jpeg2k_check(encoded)
    decoded = imagecodecs.jpeg2k_decode(encoded)
    assert_array_equal(data, decoded)

    encoded2 = imagecodecs.jpeg2k_encode(
        data, level=0, out=len(encoded), verbose=3
    )
    assert encoded2 == encoded

    with pytest.raises(imagecodecs.Jpeg2kError):
        imagecodecs.jpeg2k_encode(
            data, level=0, out=len(encoded) - 1, verbose=3
        )
        assert 'Error on writing stream' in caplog.text


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
def test_jpeg2k_trailing_bytes(caplog):
    """Test JPEG 2000 encoder is not leaving trailing bytes."""
    # https://github.com/cgohlke/imagecodecs/issues/104
    data = numpy.ones((146, 2960), dtype=numpy.uint16)
    jpeg2k = imagecodecs.jpeg2k_encode(data, level=0, verbose=3)
    assert len(jpeg2k) < 386
    for _ in range(10):
        encoded = imagecodecs.jpeg2k_encode(data, level=0, verbose=3)
        assert not encoded.endswith(b'\x00\x00\x00\x00')
        assert encoded == jpeg2k
        decoded = imagecodecs.jpeg2k_decode(encoded, verbose=3)
        assert 'stream error' not in caplog.text
        assert_array_equal(decoded, data)


@pytest.mark.skipif(not imagecodecs.JPEG2K.available, reason='jpeg2k missing')
@pytest.mark.parametrize('bitspersample', [None, True])
@pytest.mark.parametrize('dtype', ['u1', 'u2', 'u4', 'i1', 'i2', 'i4'])
@pytest.mark.parametrize('planar', [False, True])
def test_jpeg2k(dtype, planar, bitspersample):
    """Test JPEG 2000 codec."""
    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data('rgb', dtype)

    if bitspersample:
        if itemsize == 1:
            bitspersample = 7
            data //= 2
        elif itemsize == 2:
            bitspersample = 12
            if dtype != 'uint16':
                data //= 16
        elif itemsize == 4:
            bitspersample = 26  # max ~26 bits
            data //= 64
    elif itemsize == 4:
        data //= 128  # max 26 bits

    if planar:
        data = numpy.moveaxis(data, -1, 0)

    encoded = imagecodecs.jpeg2k_encode(
        data, planar=planar, bitspersample=bitspersample, verbose=2
    )
    assert imagecodecs.jpeg2k_check(encoded) in {None, True}
    decoded = imagecodecs.jpeg2k_decode(encoded, planar=planar, verbose=2)
    assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.JPEGXR.available, reason='jpegxr missing')
@pytest.mark.parametrize('level', [None, 90, 0.4])
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])  # 'view',
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize(
    'itype',
    [
        'gray uint8',
        'gray uint16',
        'gray float16',
        'gray float32',
        'rgb uint8',
        'rgb uint16',
        'rgb float16',
        'rgb float32',
        'rgba uint8',
        'rgba uint16',
        'rgba float16',
        'rgba float32',
        'channels uint8',
        'channelsa uint8',
        'channels uint16',
        'channelsa uint16',
        'cmyk uint8',
        'cmyka uint8',
    ],
)
def test_jpegxr(itype, enout, deout, level):
    """Test JPEG XR codec."""
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

    encoded = encode(data, **kwargs)

    assert imagecodecs.jpegxr_check(encoded) in {None, True}

    if enout == 'new':
        pass
    elif enout == 'out':
        encoded = numpy.zeros(len(encoded), 'uint8')
        encode(data, out=encoded, **kwargs)
    elif enout == 'bytearray':
        encoded = bytearray(len(encoded))
        encode(data, out=encoded, **kwargs)

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'view':
        temp = numpy.zeros((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2 : 2 + shape[0], 3 : 3 + shape[1], :]
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


@pytest.mark.skipif(not imagecodecs.JPEGXS.available, reason='jpegxs missing')
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('config', [None, 'p=MLS.12;nlx=2;nly=2'])
def test_jpegxs(dtype, config):
    """Test JPEGXS codec."""
    bitspersample = 8 if dtype == 'uint8' else 12
    data = image_data('rgb', dtype).squeeze()
    encoded = imagecodecs.jpegxs_encode(
        data, config=config, bitspersample=bitspersample, verbose=2
    )
    assert imagecodecs.jpegxs_check(encoded)
    decoded = imagecodecs.jpegxs_decode(encoded)
    assert_array_equal(data, decoded, verbose=True)


@pytest.mark.skipif(not imagecodecs.JPEGXS.available, reason='jpegxs missing')
def test_jpegxs_fail(capsys):
    """Test JPEGXS codec with empty config fails."""
    data = image_data('rgb', numpy.uint8).squeeze()
    with pytest.raises(imagecodecs.JpegxsError):
        imagecodecs.jpegxs_encode(data, config='', verbose=2)
    # captured = capsys.readouterr()
    # assert 'Error: Image dimensions are not supported' in captured.out


@pytest.mark.skipif(not imagecodecs.PNG.available, reason='png missing')
def test_png_encode_fast():
    """Test PNG encoder with fast settings."""
    data = image_data('rgb', numpy.uint8).squeeze()
    encoded = imagecodecs.png_encode(
        data,
        level=imagecodecs.PNG.COMPRESSION.SPEED,
        strategy=imagecodecs.PNG.STRATEGY.RLE,
        filter=imagecodecs.PNG.FILTER.SUB,
    )
    decoded = imagecodecs.png_decode(encoded)
    assert_array_equal(data, decoded, verbose=True)


@pytest.mark.skipif(not imagecodecs.APNG.available, reason='apng missing')
def test_apng_encode_fast():
    """Test APNG encoder with fast settings."""
    data = image_data('rgb', numpy.uint8).squeeze()
    encoded = imagecodecs.apng_encode(
        data,
        level=imagecodecs.APNG.COMPRESSION.SPEED,
        strategy=imagecodecs.APNG.STRATEGY.RLE,
        filter=imagecodecs.APNG.FILTER.SUB,
    )
    decoded = imagecodecs.apng_decode(encoded)
    assert_array_equal(data, decoded, verbose=True)


@pytest.mark.skipif(not imagecodecs.PNG.available, reason='png missing')
def test_png_error():
    """Test PNG exceptions."""
    data = image_data('rgb', numpy.uint8).squeeze()
    encoded = imagecodecs.png_encode(data)

    with pytest.raises(imagecodecs.PngError):
        imagecodecs.png_encode(data, out=bytearray(len(encoded) // 2))

    with pytest.raises(imagecodecs.PngError):
        imagecodecs.png_decode(encoded[: len(encoded) // 2])


@pytest.mark.skipif(not imagecodecs.APNG.available, reason='apng missing')
def test_apng_error():
    """Test APNG exceptions."""
    data = image_data('rgb', numpy.uint8).squeeze()
    encoded = imagecodecs.apng_encode(data)

    with pytest.raises(imagecodecs.ApngError):
        imagecodecs.apng_encode(data, out=bytearray(len(encoded) // 2))

    with pytest.raises(imagecodecs.ApngError):
        imagecodecs.apng_decode(encoded[: len(encoded) // 2])


@pytest.mark.skipif(not imagecodecs.APNG.available, reason='apng missing')
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('samples', [1, 2, 3, 4])
def test_apng(samples, dtype):
    """Test APNG codec."""
    shape = (9, 32, 31, samples) if samples > 1 else (9, 32, 31)
    data = numpy.random.randint(
        numpy.iinfo(dtype).max, size=9 * 32 * 31 * samples, dtype=dtype
    ).reshape(shape)
    encoded = imagecodecs.apng_encode(data, delay=100)
    decoded = imagecodecs.apng_decode(encoded)
    assert_array_equal(data, decoded, verbose=True)
    decoded = imagecodecs.apng_decode(encoded, index=0)
    assert_array_equal(data[0], decoded, verbose=True)
    for index in (0, 5, 8):
        decoded = imagecodecs.apng_decode(encoded, index=index)
        assert_array_equal(data[index], decoded, verbose=True)
    if imagecodecs.PNG.available:
        assert_array_equal(
            imagecodecs.png_decode(encoded), data[0], verbose=True
        )


@pytest.mark.parametrize(
    'codec', ['jpeg', 'png', 'webp', 'jpegxl', 'jpeg2k', 'jpegxr', 'avif']
)
def test_image_strided(codec):
    """Test decoding into buffer with some dimensions of length 1."""
    # https://github.com/cgohlke/imagecodecs/issues/98
    data = image_data('rgb', 'u1')
    out_shape = (1, data.shape[0], 1, data.shape[1], 1, data.shape[2], 1)
    out = numpy.empty(out_shape, data.dtype)
    if not getattr(imagecodecs, codec.upper()).available:
        pytest.skip(f'{codec} not found')
    encode = getattr(imagecodecs, f'{codec}_encode')
    decode = getattr(imagecodecs, f'{codec}_decode')
    encoded = encode(data)
    decoded = decode(encoded, out=out)
    assert decoded.shape == data.shape
    if codec != 'jpeg':
        assert_array_equal(data, decoded)
        assert_array_equal(data, out.squeeze())


@pytest.mark.skipif(not imagecodecs.SPNG.available, reason='spng missing')
@pytest.mark.parametrize('itype', ['rgb', 'rgba', 'gray', 'graya'])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('level', [None, 5, -1])
def test_spng_encode(itype, dtype, level):
    """Test SPNG encoder."""
    data = image_data(itype, numpy.dtype(dtype)).squeeze()
    encoded = imagecodecs.spng_encode(data, level=level)
    decoded = imagecodecs.png_decode(encoded)
    assert_array_equal(data, decoded, verbose=True)


@pytest.mark.parametrize('level', [None, 5, -1])
@pytest.mark.parametrize('deout', ['new', 'out', 'view', 'bytearray'])
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('itype', ['rgb', 'rgba', 'view', 'gray', 'graya'])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize(
    'codec',
    [
        'apng',
        'avif',
        'bmp',
        'brunsli',
        'heif',
        'jpeg_lossless',
        'jpeg2k',
        'jpeg8',
        'jpegls',
        'jpegxl',
        'jpegxr',
        'jpegxs',
        'ljpeg',
        'mozjpeg',
        'png',
        'qoi',
        'spng',
        'webp',
    ],
)
def test_image_roundtrips(codec, dtype, itype, enout, deout, level):
    """Test various image codecs."""
    if codec == 'jpeg8':
        if not imagecodecs.JPEG8.available:
            pytest.skip(f'{codec} missing')
        if imagecodecs.JPEG.legacy and dtype == 'uint16':
            pytest.skip('JPEG12 not supported')
        if itype in {'view', 'graya'} or deout == 'view':
            pytest.xfail('jpeg8 does not support this case')
        decode = imagecodecs.jpeg8_decode
        encode = imagecodecs.jpeg8_encode
        check = imagecodecs.jpeg8_check
        atol = 24 if dtype == 'uint8' else (24 * 16)
        if level:
            level += 95
    elif codec == 'jpeg_lossless':
        if not imagecodecs.JPEG.available or imagecodecs.JPEG.legacy:
            pytest.skip('LJPEG not supported')
        if itype in {'view', 'graya'} or deout == 'view':
            pytest.xfail('jpeg8 does not support this case')
        decode = imagecodecs.jpeg_decode
        check = imagecodecs.jpeg_check
        if level is not None:
            # duplicate test
            # pytest.skip(f'{codec} does not support level')
            level = None

        def encode(
            data, bitspersample=12 if dtype == 'uint16' else 8, **kwargs
        ):
            return imagecodecs.jpeg_encode(
                data, bitspersample=bitspersample, lossless=True, **kwargs
            )

    elif codec == 'mozjpeg':
        if not imagecodecs.MOZJPEG.available:
            pytest.skip(f'{codec} missing')
        if itype in {'view', 'graya'} or deout == 'view' or dtype == 'uint16':
            pytest.xfail('mozjpeg does not support this case')
        decode = imagecodecs.mozjpeg_decode
        encode = imagecodecs.mozjpeg_encode
        check = imagecodecs.mozjpeg_check
        atol = 24
        if level:
            level += 95
    elif codec == 'ljpeg':
        if not imagecodecs.LJPEG.available:
            pytest.skip(f'{codec} missing')
        if itype in {'rgb', 'rgba', 'view', 'graya'} or deout == 'view':
            pytest.xfail('ljpeg does not support this case')
        decode = imagecodecs.ljpeg_decode
        encode = imagecodecs.ljpeg_encode
        check = imagecodecs.ljpeg_check
        if level is not None:
            # duplicate test
            # pytest.skip(f'{codec} does not support level')
            level = None
        if dtype == 'uint16':

            def encode(data, *args, **kwargs):  # noqa
                return imagecodecs.ljpeg_encode(
                    data, bitspersample=12, *args, **kwargs
                )

    elif codec == 'jpegls':
        if not imagecodecs.JPEGLS.available:
            pytest.skip(f'{codec} missing')
        if itype in {'view', 'graya'} or deout == 'view':
            pytest.xfail('jpegls does not support this case')
        decode = imagecodecs.jpegls_decode
        encode = imagecodecs.jpegls_encode
        check = imagecodecs.jpegls_check
    elif codec == 'webp':
        if not imagecodecs.WEBP.available:
            pytest.skip(f'{codec} missing')
        decode = imagecodecs.webp_decode
        encode = imagecodecs.webp_encode
        check = imagecodecs.webp_check
        if dtype != 'uint8' or itype.startswith('gray'):
            pytest.xfail('webp does not support this case')
        if itype == 'rgba':

            def decode(data, out=None):  # noqa
                return imagecodecs.webp_decode(data, hasalpha=True, out=out)

        if level:
            level += 95
    elif codec == 'png':
        if not imagecodecs.PNG.available:
            pytest.skip(f'{codec} missing')
        decode = imagecodecs.png_decode
        encode = imagecodecs.png_encode
        check = imagecodecs.png_check
    elif codec == 'apng':
        if not imagecodecs.APNG.available:
            pytest.skip(f'{codec} missing')
        if itype == 'view' or deout == 'view':
            pytest.xfail('apng does not support this case')
        decode = imagecodecs.apng_decode
        encode = imagecodecs.apng_encode
        check = imagecodecs.apng_check
    elif codec == 'bmp':
        if not imagecodecs.BMP.available:
            pytest.skip(f'{codec} missing')
        if (
            itype in {'rgba', 'view', 'graya'}
            or deout == 'view'
            or dtype != 'uint8'
        ):
            pytest.xfail('bmp does not support this case')
        decode = imagecodecs.bmp_decode
        encode = imagecodecs.bmp_encode
        check = imagecodecs.bmp_check
        if level is not None:
            # duplicate test
            # pytest.skip(f'{codec} does not support level')
            level = None
    elif codec == 'spng':
        if not imagecodecs.SPNG.available:
            pytest.skip(f'{codec} missing')
        if itype == 'view' or deout == 'view':
            pytest.xfail('spng does not support this case')
        if itype == 'graya' or (
            dtype == 'uint16' and itype in {'gray', 'rgb'}
        ):
            pytest.xfail('spng does not support this case')
        decode = imagecodecs.spng_decode
        encode = imagecodecs.spng_encode
        check = imagecodecs.spng_check
    elif codec == 'qoi':
        if not imagecodecs.QOI.available:
            pytest.skip(f'{codec} missing')
        decode = imagecodecs.qoi_decode
        encode = imagecodecs.qoi_encode
        check = imagecodecs.qoi_check
        level = None
        if (
            itype in {'view', 'gray', 'graya'}
            or deout == 'view'
            or dtype == 'uint16'
        ):
            pytest.xfail('qoi does not support this case')
    elif codec == 'jpeg2k':
        if not imagecodecs.JPEG2K.available:
            pytest.skip(f'{codec} missing')
        if itype == 'view' or deout == 'view':
            pytest.xfail('jpeg2k does not support this case')
        check = imagecodecs.jpeg2k_check
        if level and level > 0:
            level = 100 - level  # psnr

        # enable verbose mode for rare failures
        def encode(data, *args, **kwargs):
            return imagecodecs.jpeg2k_encode(data, verbose=3, *args, **kwargs)

        def decode(data, *args, **kwargs):
            return imagecodecs.jpeg2k_decode(data, verbose=3, *args, **kwargs)

    elif codec == 'jpegxl':
        if not imagecodecs.JPEGXL.available:
            pytest.skip(f'{codec} missing')
        if itype == 'view' or deout == 'view':
            pytest.xfail('jpegxl does not support this case')
        if level:
            level += 95
        decode = imagecodecs.jpegxl_decode
        encode = imagecodecs.jpegxl_encode
        check = imagecodecs.jpegxl_check
    elif codec == 'brunsli':
        if not imagecodecs.BRUNSLI.available:
            pytest.skip(f'{codec} missing')
        if itype in {'view', 'graya'} or deout == 'view' or dtype == 'uint16':
            pytest.xfail('brunsli does not support this case')
        decode = imagecodecs.brunsli_decode
        encode = imagecodecs.brunsli_encode
        check = imagecodecs.brunsli_check
        atol = 24
        if level:
            level += 95
    elif codec == 'jpegxr':
        if not imagecodecs.JPEGXR.available:
            pytest.skip(f'{codec} missing')
        if itype == 'graya' or deout == 'view':
            pytest.xfail('jpegxr does not support this case')
        decode = imagecodecs.jpegxr_decode
        encode = imagecodecs.jpegxr_encode
        check = imagecodecs.jpegxr_check
        atol = 10
        if level:
            level = (level + 95) / 100
    elif codec == 'jpegxs':
        if not imagecodecs.JPEGXS.available:
            pytest.skip(f'{codec} missing')
        if itype != 'rgb' or deout == 'view':
            pytest.xfail('jpegxs does not support this case')
        level = None
        decode = imagecodecs.jpegxs_decode
        encode = imagecodecs.jpegxs_encode
        check = imagecodecs.jpegxs_check
    elif codec == 'avif':
        if not imagecodecs.AVIF.available:
            pytest.skip(f'{codec} missing')
        if itype in {'view'} or deout == 'view':
            pytest.xfail('avif does not support this case')
        decode = imagecodecs.avif_decode
        encode = imagecodecs.avif_encode
        check = imagecodecs.avif_check
        if dtype == 'uint16':

            def encode(data, *args, **kwargs):  # noqa
                return imagecodecs.avif_encode(
                    data, bitspersample=12, *args, **kwargs
                )

        if level:
            level += 95
        atol = 10
    elif codec == 'heif':
        if not imagecodecs.HEIF.available:
            pytest.skip(f'{codec} missing')
        if (
            itype in {'gray', 'graya', 'view'}
            or deout == 'view'
            or dtype == 'uint16'
        ):
            pytest.xfail('heif does not support this case')
        decode = imagecodecs.heif_decode
        encode = imagecodecs.heif_encode
        check = imagecodecs.heif_check
        atol = 10
        if level:
            level += 95
        if int(imagecodecs.heif_version().split('.')[1]) < 12:
            pytest.xfail('libheif < 1.12 cannot encode small images')
    else:
        raise ValueError(codec)

    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    data = image_data(itype, dtype)
    shape = data.shape
    kwargs = {} if level is None else {'level': level}

    if enout == 'new':
        encoded = encode(data, **kwargs)
    elif enout == 'out':
        encoded = numpy.zeros(
            3 * shape[0] * shape[1] * shape[2] * itemsize, 'uint8'
        )
        ret = encode(data, out=encoded, **kwargs)
        if codec in {'brunsli'}:
            # Brunsli decoder doesn't like extra bytes
            encoded = encoded[: len(ret)]
    elif enout == 'bytearray':
        encoded = bytearray(3 * shape[0] * shape[1] * shape[2] * itemsize)
        ret = encode(data, out=encoded, **kwargs)
        if codec in {'brunsli'}:
            # Brunsli decoder doesn't like extra bytes
            encoded = encoded[: len(ret)]

    if enout != 'out':
        assert check(encoded) in {None, True}

    if deout == 'new':
        decoded = decode(encoded)
    elif deout == 'out':
        decoded = numpy.zeros(shape, dtype)
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'view':
        temp = numpy.zeros((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        decoded = temp[2 : 2 + shape[0], 3 : 3 + shape[1], :]
        decode(encoded, out=numpy.squeeze(decoded))
    elif deout == 'bytearray':
        decoded = bytearray(shape[0] * shape[1] * shape[2] * itemsize)
        decoded = decode(encoded, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shape)

    if itype == 'gray':
        decoded = decoded.reshape(shape)

    if codec == 'webp' and level is not None:  # or itype == 'rgba'
        assert_allclose(data, decoded, atol=32)
    elif codec in {'jpeg8', 'jpegxr', 'brunsli', 'mozjpeg', 'heif'}:
        assert_allclose(data, decoded, atol=atol)
    elif codec == 'jpegls' and level == 5:
        assert_allclose(data, decoded, atol=6)
    elif codec == 'jpeg2k' and level == 95:
        assert_allclose(data, decoded, atol=7)
    elif codec == 'jpegxl' and level is not None:
        atol = 256 if dtype.itemsize > 1 else 8
        if level < 100:
            atol *= 4
        assert_allclose(data, decoded, atol=atol)
    elif codec == 'avif' and level == 94:
        atol = 38 if dtype.itemsize > 1 else 6
        assert_allclose(data, decoded, atol=atol)
    else:
        assert_array_equal(data, decoded, verbose=True)


@pytest.mark.skipif(not imagecodecs.GIF.available, reason='GIF missing')
@pytest.mark.parametrize('deout', ['new', 'out', 'bytearray'])  # 'view'
@pytest.mark.parametrize('enout', ['new', 'out', 'bytearray'])
@pytest.mark.parametrize('frames', [False, True])
@pytest.mark.parametrize('index', [None, 0])
def test_gif_roundtrips(index, frames, enout, deout):
    """Test GIF codec."""
    decode = imagecodecs.gif_decode
    encode = imagecodecs.gif_encode

    dtype = numpy.dtype('uint8')
    data = numpy.squeeze(image_data('gray', dtype, frames=frames))
    if index == 0 and frames:
        shaped = data.shape[1:] + (3,)
    else:
        shaped = data.shape + (3,)
    sized = data.size * 3

    if enout == 'new':
        encoded = encode(data)
    elif enout == 'out':
        encoded = numpy.zeros(2 * data.size, 'uint8')
        encode(data, out=encoded)
    elif enout == 'bytearray':
        encoded = bytearray(2 * data.size)
        encode(data, out=encoded)

    assert imagecodecs.gif_check(encoded)

    if deout == 'new':
        decoded = decode(encoded, index=index)
    elif deout == 'out':
        decoded = numpy.zeros(shaped, dtype)
        decode(encoded, index=index, out=numpy.squeeze(decoded))
    elif deout == 'bytearray':
        decoded = bytearray(sized)
        decoded = decode(encoded, index=index, out=decoded)
        decoded = numpy.asarray(decoded, dtype=dtype).reshape(shaped)

    if index == 0 and frames:
        data = data[index]
    assert_array_equal(data, decoded[..., 1], verbose=True)


@pytest.mark.skipif(not imagecodecs.PNG.available, reason='png missing')
def test_png_rgba_palette():
    """Test decoding indexed PNG with transparency."""
    png = readfile('rgba.u1.pal.png')
    image = imagecodecs.png_decode(png)
    assert tuple(image[6, 15]) == (255, 255, 255, 0)
    assert tuple(image[6, 16]) == (141, 37, 52, 255)

    if imagecodecs.APNG.available:
        image = imagecodecs.apng_decode(png)
        assert tuple(image[6, 15]) == (255, 255, 255, 0)
        assert tuple(image[6, 16]) == (141, 37, 52, 255)


TIFF_PATH = DATA_PATH / 'tiff/'
TIFF_FILES = [
    os.path.split(f)[-1][:-4] for f in glob.glob(str(TIFF_PATH / '*.tif'))
]


@pytest.mark.skipif(not imagecodecs.TIFF.available, reason='tiff missing')
@pytest.mark.skipif(tifffile is None, reason='tifffile missing')
@pytest.mark.parametrize('asrgb', [False, True])
@pytest.mark.parametrize('name', TIFF_FILES)
def test_tiff_files(name, asrgb):
    """Test TIFF decode with existing files against tifffile."""
    decode = imagecodecs.tiff_decode
    if (
        'depth' in name
        or 'png' in name
        or 'jpeg2000' in name
        or 'jpegxl' in name
        or 'jpegxr' in name
        or 'jpeg.u2' in name
        or (not IS_CG and ('webp' in name or 'zstd' in name or 'lzma' in name))
    ):
        pytest.xfail('not supported by libtiff or tiff_decode')

    filename = TIFF_PATH / f'{name}.tif'
    with open(filename, 'rb') as fh:
        encoded = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    assert imagecodecs.tiff_check(encoded)

    if asrgb:
        if (
            'b1' in name
            or 'u1' in name
            or 'u2' in name
            or 'i1' in name
            or 'i2' in name
        ):
            decoded = decode(encoded, index=0, asrgb=1, verbose=1)
        else:
            with pytest.raises(imagecodecs.TiffError):
                decoded = decode(encoded, index=0, asrgb=1, verbose=1)
        return

    if 'b1' in name:
        pytest.xfail('not supported yet')
    data = tifffile.imread(filename)
    decoded = decode(encoded, index=None, verbose=1)
    if 'jpeg' in name:
        # tiff_decode returns RGBA for jpeg, tifffile returns RGB
        decoded = decoded[..., :3]
    assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.TIFF.available, reason='tiff missing')
@pytest.mark.skipif(tifffile is None, reason='tifffile missing')
@pytest.mark.parametrize('index', [0, 3, 10, 1048576, None, list, slice])
def test_tiff_index(index):
    """Test TIFF decoder index arguments."""
    filename = TIFF_PATH / 'gray.series.u1.tif'
    with open(filename, 'rb') as fh:
        encoded = fh.read()
    if index in {10, 1048576}:
        with pytest.raises((IndexError, OverflowError)):
            decoded = imagecodecs.tiff_decode(encoded, index=index)
    elif index is list:
        data = tifffile.imread(filename, series=1)
        decoded = imagecodecs.tiff_decode(encoded, index=[1, 3, 5, 7])
        assert_array_equal(data, decoded)
    elif index is slice:
        for index in (slice(None), slice(1, None, None), slice(1, 3, None)):
            with pytest.raises((IndexError, ValueError)):
                decoded = imagecodecs.tiff_decode(encoded, index=index)
        data = tifffile.imread(filename, series=1)
        for index in (slice(1, None, 2), slice(1, 8, 2)):
            decoded = imagecodecs.tiff_decode(encoded, index=index)
            assert_array_equal(data, decoded)
    elif index is None:
        data = tifffile.imread(filename)
        decoded = imagecodecs.tiff_decode(encoded, index=None)
        assert_array_equal(data, decoded)
    else:
        data = tifffile.imread(filename, key=index)
        decoded = imagecodecs.tiff_decode(encoded, index=index)
        assert_array_equal(data, decoded)


@pytest.mark.skipif(not imagecodecs.TIFF.available, reason='')
@pytest.mark.skipif(tifffile is None, reason='tifffile missing')
def test_tiff_asrgb():
    """Test TIFF decoder asrgb arguments."""
    filename = TIFF_PATH / 'gray.series.u1.tif'
    with open(filename, 'rb') as fh:
        encoded = fh.read()

    data = tifffile.imread(filename, series=0)
    decoded = imagecodecs.tiff_decode(encoded, index=None, asrgb=True)
    assert decoded.shape[-1] == 4
    assert_array_equal(data, decoded[..., 0])

    data = tifffile.imread(filename, series=1)
    decoded = imagecodecs.tiff_decode(encoded, index=[1, 3, 5, 7], asrgb=True)
    assert decoded.shape[-1] == 4
    assert_array_equal(data, decoded[..., :3])


@pytest.mark.skipif(tifffile is None, reason='tifffile module missing')
@pytest.mark.parametrize('byteorder', ['<', '>'])
@pytest.mark.parametrize('dtype', ['u1', 'u2', 'f2', 'f4'])
@pytest.mark.parametrize('predictor', [False, True])
@pytest.mark.parametrize(
    'codec',
    [
        'deflate',
        'lzw',
        'lzma',
        'zstd',
        'packbits',
        'webp',
        'jpeg',
        'lerc',
        'lerc_zstd',
        'lerc_deflate',
    ],
)
def test_tifffile(byteorder, dtype, codec, predictor):
    """Test tifffile compression."""
    compressionargs = None
    if codec == 'deflate' and not imagecodecs.ZLIB.available:
        # TODO: this should pass in tifffile >= 2020
        pytest.xfail('zlib missing')
    elif codec == 'lzma' and not imagecodecs.LZMA.available:
        pytest.xfail('lzma missing')
    elif codec == 'zstd' and not imagecodecs.ZSTD.available:
        pytest.xfail('zstd missing')
    elif codec == 'packbits' and not imagecodecs.PACKBITS.available:
        pytest.xfail('packbits missing')
    elif codec == 'jpeg':
        if not imagecodecs.JPEG.available:
            pytest.xfail('jpeg missing')
        if predictor or dtype != 'u1':
            pytest.xfail('tiff/jpeg do not support this case')
    elif codec == 'jpegxl':
        if not imagecodecs.JPEGXL.available:
            pytest.xfail('jpegxl missing')
        if predictor:
            pytest.xfail('jpegxl does not support predictor')
    elif codec[:4] == 'lerc':
        if not imagecodecs.LERC.available:
            pytest.xfail('lerc missing')
        elif dtype == 'f2' or byteorder == '>':
            pytest.xfail('dtype not supported by lerc')
        elif dtype == 'f4' and predictor:
            pytest.xfail('lerc does not work with float predictor')
        if codec == 'lerc_zstd':
            if not imagecodecs.ZSTD.available:
                pytest.xfail('zstd codec missing')
            compressionargs = {'compression': 'zstd'}
        elif codec == 'lerc_deflate':
            if not imagecodecs.ZLIB.available:
                pytest.xfail('zlib codec missing')
            compressionargs = {'compression': 'deflate'}
        codec = 'lerc'
    elif codec == 'webp':
        if not imagecodecs.WEBP.available:
            pytest.xfail('webp missing')
        elif dtype != 'u1':
            pytest.xfail('dtype not supported')
        elif predictor:
            pytest.xfail('webp does not support predictor')

    data = image_data('rgb', dtype)
    if byteorder == '>':
        data = data.byteswap()
        data = data.view(data.dtype.newbyteorder())

    with io.BytesIO() as fh:
        tifffile.imwrite(
            fh,
            data,
            photometric='rgb',
            compression=codec,
            compressionargs=compressionargs,
            predictor=predictor,
            byteorder=byteorder,
        )
        # with open(f'{codec}_{dtype}.tif', 'wb') as f:
        #     fh.seek(0)
        #     f.write(fh.read())
        fh.seek(0)
        with tifffile.TiffFile(fh) as tif:
            assert tif.byteorder == byteorder
            image = tif.asarray()
        if byteorder == '>':
            image = image.byteswap()
            image = image.view(image.dtype.newbyteorder())
        if codec != 'jpeg':
            assert_array_equal(data, image, verbose=True)

        if imagecodecs.TIFF.available:
            if not (predictor and codec in {'packbits', 'lerc'}):
                # libtiff does not support {codec} with predictor
                fh.seek(0)
                image2 = imagecodecs.tiff_decode(fh.read())
                assert_array_equal(image2, image, verbose=True)


@pytest.mark.skipif(
    imagecodecs.JPEG.legacy
    or not imagecodecs.LJPEG.available
    or tifffile is None,
    reason='tifffile module or LJPEG missing',
)
@pytest.mark.parametrize('dtype', ['u1'])
def test_tifffile_ljpeg(dtype):
    """Test tifffile with ljpeg compression."""
    data = numpy.squeeze(image_data('gray', dtype))
    with io.BytesIO() as fh:
        tifffile.imwrite(
            fh,
            data,
            photometric='minisblack',
            # compression=(
            #     'jpeg', None, {'lossless': True, 'bitspersample': 8}
            # ),
            compression='jpeg',
            compressionargs={'lossless': True, 'bitspersample': 8},
        )
        fh.seek(0)
        image = tifffile.imread(fh)
    assert_array_equal(data, image, verbose=True)


@pytest.mark.skipif(czifile is None, reason='czifile missing')
def test_czifile():
    """Test JpegXR compressed CZI file."""
    fname = datafiles('jpegxr.czi')
    if not os.path.exists(fname):
        pytest.skip('large file not included with source distribution')
    if not imagecodecs.JPEGXR.available:
        pytest.xfail('jpegxr missing')

    with czifile.CziFile(fname) as czi:
        assert czi.shape == (1, 1, 15, 404, 356, 1)
        assert czi.axes == 'BCZYX0'
        # verify data
        data = czi.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1, 1, 15, 404, 356, 1)
        assert data.dtype == 'uint16'
        assert data[0, 0, 14, 256, 146, 0] == 38086


@pytest.mark.skipif(liffile is None, reason='liffile missing')
def test_liffile():
    """Test reading TIFF-chunked XLIF dataset with liffile."""
    filename = DATA_PATH / 'lif' / 'Metadata' / 'ImageXYZ10C2.xlif'

    with liffile.LifFile(filename, mode='r', squeeze=True) as xlif:
        assert xlif.type == liffile.LifFileType.XLIF
        str(xlif)
        assert xlif.name == 'ImageXYZ10C2'
        assert xlif.version == 2
        assert len(xlif.children) == 0

        series = xlif.images
        str(series)
        assert len(series) == 1
        im = series[0]
        str(im)
        assert im.dtype == numpy.uint8
        assert im.itemsize == 1
        assert im.shape == (10, 2, 512, 512)
        assert im.dims == ('Z', 'C', 'Y', 'X')
        assert im.sizes == {'Z': 10, 'C': 2, 'Y': 512, 'X': 512}
        assert 'C' not in im.coords
        assert_allclose(
            im.coords['Z'][[0, -1]], [-2.345302e-05, 1.786591e-05], atol=1e-4
        )
        assert len(im.timestamps) == 20
        assert im.timestamps[0] == numpy.datetime64('2015-01-27T10:14:30.304')
        assert im.size == 5242880
        assert im.nbytes == 5242880
        assert im.ndim == 4

        data = im.asarray(mode='r', out='memmap')
        assert isinstance(data, numpy.memmap), type(data)
        assert data.sum(dtype=numpy.uint64) == 80177798


@pytest.mark.skipif(SKIP_NUMCODECS, reason='numcodecs missing')
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


@pytest.mark.skipif(SKIP_NUMCODECS, reason='zarr or numcodecs missing')
@pytest.mark.parametrize('photometric', ['gray', 'rgb', 'stack'])
@pytest.mark.parametrize(
    'codec',
    [
        'aec',
        'apng',
        'avif',
        'bitorder',
        'bitshuffle',
        'blosc',
        'blosc2',
        'bmp',
        # 'brotli',  # failing
        'byteshuffle',
        'bz2',
        # 'cms',
        'deflate',
        'delta',
        'float24',
        'floatpred',
        'gif',
        'lz4h5',
        'heif',
        # 'jetraw',  # encoder requires a license
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
        'lzf',
        'lzfse',
        'lzham',
        'lzma',
        'lzw',
        'packbits',
        'pcodec',
        'pglz',
        'png',
        'qoi',
        'rgbe',
        'rcomp',
        'snappy',
        'sperr',
        'spng',
        'sz3',
        'szip',
        'tiff',  # no encoder
        'ultrahdr',
        'webp',
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
    if codec == 'aec':
        if not imagecodecs.AEC.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Aec(
            bitspersample=None, flags=None, blocksize=None, rsi=None
        )
    elif codec == 'apng':
        if not imagecodecs.APNG.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Apng(photometric=photometric, delay=100)
    elif codec == 'avif':
        if not imagecodecs.AVIF.available:
            pytest.skip(f'{codec} not found')
        if photometric != 'rgb':
            pytest.xfail('AVIF does not support grayscale')
        compressor = numcodecs.Avif(
            level=100,
            speed=None,
            tilelog2=None,
            bitspersample=None,
            pixelformat=None,
            numthreads=2,
        )  # lossless
    elif codec == 'bitorder':
        if not imagecodecs.BITORDER.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Bitorder()
    elif codec == 'bitshuffle':
        if not imagecodecs.BITSHUFFLE.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Bitshuffle(
            itemsize=data.dtype.itemsize, blocksize=0
        )
    elif codec == 'blosc':
        if not imagecodecs.BLOSC.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Blosc(
            level=9,
            compressor='blosclz',
            typesize=data.dtype.itemsize * 8,
            blocksize=None,
            shuffle=None,
            numthreads=2,
        )
    elif codec == 'blosc2':
        if not imagecodecs.BLOSC2.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Blosc2(
            level=9,
            compressor='blosclz',
            typesize=data.dtype.itemsize * 8,
            blocksize=None,
            shuffle=None,
            numthreads=2,
        )
    elif codec == 'bmp':
        if not imagecodecs.BMP.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Bmp(asrgb=None)
    elif codec == 'brotli':
        if not imagecodecs.BROTLI.available:
            pytest.skip(f'{codec} not found')
            # TODO: why are these failing?
        pytest.xfail('not sure why this is failing')
        compressor = numcodecs.Brotli(level=11, mode=None, lgwin=None)
    elif codec == 'byteshuffle':
        if not imagecodecs.BYTESHUFFLE.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('int16')
        compressor = numcodecs.Byteshuffle(
            shape=chunks, dtype=data.dtype, axis=axis
        )
    elif codec == 'bz2':
        if not imagecodecs.BZ2.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Bz2(level=9)
    elif codec == 'deflate':
        if not imagecodecs.DEFLATE.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Deflate(level=8)
    elif codec == 'delta':
        if not imagecodecs.DELTA.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Delta(shape=chunks, dtype=data.dtype, axis=axis)
    elif codec == 'float24':
        if not imagecodecs.FLOAT24.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('float32')
        compressor = numcodecs.Float24()
    elif codec == 'floatpred':
        if not imagecodecs.FLOATPRED.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('float32')
        compressor = numcodecs.Floatpred(
            shape=chunks, dtype=data.dtype, axis=axis
        )
    elif codec == 'gif':
        if not imagecodecs.GIF.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Gif()
    elif codec == 'lz4h5':
        if not imagecodecs.LZ4H5.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lz4h5(level=10, blocksize=100)
    elif codec == 'heif':
        if not imagecodecs.HEIF.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Heif(photometric=photometric)
        lossless = False  # TODO: lossless not working
        atol = 1
    elif codec == 'jetraw':
        if not imagecodecs.JETRAW.available:
            pytest.skip(f'{codec} not found')
        if photometric == 'rgb':
            pytest.xfail('Jetraw does not support RGB')
        compressor = numcodecs.Jetraw(
            shape=chunks, identifier='500202_fast_bin1x'
        )
        data = data.astype('uint16')
        lossless = False
        atol = 32
    elif codec == 'jpeg':
        if not imagecodecs.JPEG.available:
            pytest.skip(f'{codec} not found')
        lossless = False
        atol = 4
        compressor = numcodecs.Jpeg(level=99)
    elif codec == 'jpeg12':
        if not imagecodecs.JPEG.available or imagecodecs.JPEG.legacy:
            pytest.skip(f'{codec} not found')
        lossless = False
        atol = 4 << 4
        data = data.astype('uint16') << 4
        compressor = numcodecs.Jpeg(level=99, bitspersample=12)
    elif codec == 'jpeg2k':
        if not imagecodecs.JPEG2K.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Jpeg2k(level=0)  # lossless
    elif codec == 'jpegls':
        if not imagecodecs.JPEGLS.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Jpegls(level=0)  # lossless
    elif codec == 'jpegxl':
        if not imagecodecs.JPEGXL.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Jpegxl(level=101)  # lossless
    elif codec == 'jpegxr':
        if not imagecodecs.JPEGXR.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Jpegxr(
            level=1.0, photometric='RGB' if photometric == 'rgb' else None
        )  # lossless
    elif codec == 'jpegxs':
        if not imagecodecs.JPEGXS.available:
            pytest.skip(f'{codec} not found')
        if photometric != 'rgb':
            pytest.xfail(f'JPEGXS does not support {photometric=}')
        compressor = numcodecs.Jpegxs(config='p=MLS.12', verbose=1)
    elif codec == 'ultrahdr':
        if not imagecodecs.ULTRAHDR.available:
            pytest.skip(f'{codec} not found')
        if photometric != 'rgb':
            pytest.xfail(f'ULTRAHDR does not support {photometric=}')
        else:
            pytest.xfail('ultrahdr_encode not working')
        data = data.astype('float32')
        data /= data.max()
        data = data.astype('float16')
        compressor = numcodecs.Ultrahdr()
        lossless = False
        atol = 1e-2
    elif codec == 'lerc':
        if not imagecodecs.LERC.available:
            pytest.skip(f'{codec} not found')
        if imagecodecs.ZSTD.available:
            compression = 'zstd'
            compressionargs = {'level': 10}
        else:
            compression = None
            compressionargs = None
        compressor = numcodecs.Lerc(
            level=0.0, compression=compression, compressionargs=compressionargs
        )
    elif codec == 'ljpeg':
        if not imagecodecs.LJPEG.available:
            pytest.skip(f'{codec} not found')
        if photometric == 'rgb':
            pytest.xfail('LJPEG does not support rgb')
        data = data.astype('uint16') << 2
        compressor = numcodecs.Ljpeg(bitspersample=10)
    elif codec == 'lz4':
        if not imagecodecs.LZ4.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lz4(level=10, hc=True, header=True)
    elif codec == 'lz4f':
        if not imagecodecs.LZ4F.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lz4f(
            level=12,
            blocksizeid=False,
            contentchecksum=True,
            blockchecksum=True,
        )
    elif codec == 'lzf':
        if not imagecodecs.LZF.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lzf(header=True)
    elif codec == 'lzfse':
        if not imagecodecs.LZFSE.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lzfse()
    elif codec == 'lzham':
        if not imagecodecs.LZHAM.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lzham(level=6)
    elif codec == 'lzma':
        if not imagecodecs.LZMA.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lzma(
            level=6, check=imagecodecs.LZMA.CHECK.CRC32
        )
    elif codec == 'lzw':
        if not imagecodecs.LZW.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Lzw()
    elif codec == 'packbits':
        if not imagecodecs.PACKBITS.available:
            pytest.skip(f'{codec} not found')
        if photometric == 'rgb':
            compressor = numcodecs.Packbits(axis=-2)
        else:
            compressor = numcodecs.Packbits()
    elif codec == 'pcodec':
        if not imagecodecs.PCODEC.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('float32')
        compressor = numcodecs.Pcodec(
            level=8,
            shape=chunks[1:],
            dtype=data.dtype,
        )
        lossless = False
        atol = 1e-2
    elif codec == 'pglz':
        if not imagecodecs.PGLZ.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Pglz(strategy=None)
    elif codec == 'png':
        if not imagecodecs.PNG.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Png()
    elif codec == 'qoi':
        if not imagecodecs.QOI.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Qoi()
        if photometric != 'rgb':
            pytest.xfail('QOI does not support grayscale')
    elif codec == 'rgbe':
        if not imagecodecs.RGBE.available:
            pytest.skip(f'{codec} not found')
        if photometric != 'rgb':
            pytest.xfail('RGBE does not support grayscale')
        data = data.astype('float32')
        # lossless = False
        compressor = numcodecs.Rgbe(shape=chunks[-3:], header=False, rle=True)
    elif codec == 'rcomp':
        if not imagecodecs.RCOMP.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Rcomp(shape=chunks, dtype=data.dtype.str)
    elif codec == 'snappy':
        if not imagecodecs.SNAPPY.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Snappy()
    elif codec == 'sperr':
        if not imagecodecs.SPERR.available:
            pytest.skip(f'{codec} not found')
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
    elif codec == 'spng':
        if not imagecodecs.SPNG.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Spng()
    elif codec == 'sz3':
        if not imagecodecs.SZ3.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('float32')
        atol = 1e-3
        compressor = numcodecs.Sz3(
            mode='abs',
            abs=atol,
            shape=chunks[1:],
            dtype=data.dtype,
        )
        lossless = False
    elif codec == 'szip':
        if not imagecodecs.SZIP.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Szip(
            header=True, **imagecodecs.szip_params(data)
        )
    elif codec == 'tiff':
        if not imagecodecs.TIFF.available:
            pytest.skip(f'{codec} not found')
        pytest.xfail('TIFF encode not implemented')
        compressor = numcodecs.Tiff()
    elif codec == 'webp':
        if not imagecodecs.WEBP.available:
            pytest.skip(f'{codec} not found')
        if photometric != 'rgb':
            pytest.xfail('WebP does not support grayscale')
        compressor = numcodecs.Webp(level=-1)
    elif codec == 'xor':
        if not imagecodecs.XOR.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Xor(shape=chunks, dtype=data.dtype, axis=axis)
    elif codec == 'zfp':
        if not imagecodecs.ZFP.available:
            pytest.skip(f'{codec} not found')
        data = data.astype('float32')
        compressor = numcodecs.Zfp(
            shape=chunks, dtype=data.dtype, header=False
        )
    elif codec == 'zlib':
        if not imagecodecs.ZLIB.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Zlib(level=6)
    elif codec == 'zlibng':
        if not imagecodecs.ZLIBNG.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Zlibng(level=6)
    elif codec == 'zopfli':
        if not imagecodecs.ZOPFLI.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Zopfli()
    elif codec == 'zstd':
        if not imagecodecs.ZSTD.available:
            pytest.skip(f'{codec} not found')
        compressor = numcodecs.Zstd(level=10)
    else:
        raise RuntimeError()

    if 0:
        # use ZIP file on disk
        fname = f'test_{codec}.{photometric}.{data.dtype.str[1:]}.zarr.zip'
        store = zarr.ZipStore(fname, mode='w')
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
        dtype=data.dtype.str,
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
    try:
        store.close()
    except Exception:
        pass


@pytest.mark.skipif(
    not imagecodecs.JPEG8.available or imagecodecs.JPEG8.legacy,
    reason='jpeg8 missing',
)
@pytest.mark.skipif(IS_32BIT, reason='data too large for 32-bit')
def test_jpeg8_large():
    """Test JPEG 8-bit decoder with dimensions > 65000."""
    decode = imagecodecs.jpeg8_decode
    try:
        data = readfile('33792x79872.jpg', memmap=True)
    except OSError:
        pytest.skip('large file not included with source distribution')

    assert imagecodecs.jpeg8_check(data)

    decoded = decode(data, shape=(33792, 79872))
    assert decoded.shape == (33792, 79872, 3)
    assert decoded.dtype == 'uint8'
    assert tuple(decoded[33791, 79871]) == (204, 195, 180)


###############################################################################


class TempFileName:
    """Temporary file name context manager."""

    def __init__(self, name=None, suffix='', remove=True):
        self.remove = bool(remove)
        if not name:
            with tempfile.NamedTemporaryFile(
                prefix='test_', suffix=suffix
            ) as temp:
                self.name = temp.name
        else:
            self.name = os.path.join(
                tempfile.gettempdir(), f'test_{name}{suffix}'
            )

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            try:
                os.remove(self.name)
            except Exception:
                pass


def datafiles(pathname: str, base: str | None = None) -> Any:
    """Return path to data file(s)."""
    if base is None:
        base = str(DATA_PATH)
    path = os.path.join(base, *pathname.split('/'))
    if any(i in path for i in '*?'):
        return glob.glob(path)
    return path


def readfile(fname: str, memmap: bool = False) -> Any:
    """Return content of data file."""
    data: Any
    with open(datafiles(fname), 'rb') as fh:
        if memmap:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            data = fh.read()
    return data


def image_data(
    itype: str,
    dtype: numpy.typing.DTypeLike,
    planar: bool = False,
    frames: bool = False,
) -> numpy.ndarray[Any, Any]:
    """Return test image array."""
    data = TEST_DATA if frames else TEST_DATA[0]

    if itype in {'rgb', 'view'}:
        data = data[..., [0, 2, 4]]
    elif itype == 'rgba':
        data = data[..., [0, 2, 4, -1]]
    elif itype == 'cmyk':
        data = data[..., [0, 2, 4, 6]]
    elif itype == 'cmyka':
        data = data[..., [0, 2, 4, 6, -1]]
    elif itype == 'gray':
        data = data[..., 0:1]
    elif itype == 'graya':
        data = data[..., [0, -1]]
    elif itype == 'channels':
        data = data[..., :-1]
    elif itype == 'channelsa':
        data = data[..., :]
    elif itype == 'line':
        data = data[0:1, :, 0:1]
    elif itype == 'stack':
        # TODO: remove this
        assert not frames
        assert not planar
        data = numpy.moveaxis(data, -1, 0)
    else:
        raise ValueError('itype not found')

    if planar:
        data = numpy.moveaxis(data, -1, -3)

    data = data.copy()

    dtype = numpy.dtype(dtype)
    if dtype.char == '?':
        data = data > data.mean()
    elif dtype.kind in 'iu':
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
        assert not frames
        assert not planar
        shape = data.shape
        temp = numpy.zeros((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        temp[2 : 2 + shape[0], 3 : 3 + shape[1], :] = data
        data = temp[2 : 2 + shape[0], 3 : 3 + shape[1], :]

    return data


# (32, 31, 9) float64
TEST_DATA: numpy.ndarray[Any, Any] = numpy.load(datafiles('testdata.npy'))
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
    sys.exit(pytest.main(argv))
