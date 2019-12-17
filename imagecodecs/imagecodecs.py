# -*- coding: utf-8 -*-
# imagecodecs.py

# Copyright (c) 2008-2019, Christoph Gohlke
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

"""Image transformation, compression, and decompression codecs.

This module implements limited functionality of the imagecodecs Cython
extension modules using pure Python and 3rd party packages.
The module is intended for testing and reference, not production code.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.16

Revisions
---------
2019.12.16
    Rename jxr and j2k codecs to jpegxr and jpeg2k.
    Add zopfli codec via zopflipy package.
    Add snappy codec via python-snappy package.
2019.12.10
    Add brotli codec via Brotli package.
    Add dummy dtype and jpegxl codecs.
2019.11.28
    Add dummy AEC codec.
2019.11.18
    Add Bitshuffle codec.
    Fix doctests.
2019.11.5
    Update dependencies.
2019.5.22
    Add ZFP codec via zfpy package.
2019.4.20
    Include with imagecodecs-lite.
2019.1.14
    Add dummy ZFP codec.
    Add numpy NPY and NPZ codec.
2019.1.1
    Update copyright year.
2018.12.1
    Use logging.warning instead of warnings.warn.
2018.10.30
    Check signatures before calling pil_decode.
2018.10.28
    Add dummy JpegLS codec.
    Rename jpeg0xc3 to jpegsof3.
2018.10.22
    Add Blosc codec via blosc package.
    Fix FutureWarning with numpy 1.15.
2018.10.18
    Use Pillow for decoding jpeg, png, j2k, and webp.
2018.10.17
    Add dummy jpeg0xc3 codec.
2018.10.10
    Add dummy PNG codec.
    Improve Delta codecs.
2018.9.30
    Add LZF codec via python-lzf package.
2018.9.22
    Add dummy webp codec.
2018.8.29
    Add FloatPred, LZW, PackBits, PackInts decoders from tifffile.py module.
    Use zlib, bz2, lzma, zstd, and lz4 modules for codecs.

"""

from __future__ import division, print_function

__version__ = '2019.12.16.py'

import sys
import struct
import logging
import functools
import zlib
import io

import numpy

try:
    import lzma
except ImportError:
    try:
        from backports import lzma
    except ImportError:
        lzma = None

try:
    import bz2
except ImportError:
    bz2 = None

try:
    import zstd
except ImportError:
    zstd = None

try:
    import lz4
    import lz4.block
except ImportError:
    lz4 = None

try:
    import lzf
except ImportError:
    lzf = None

try:
    import zfpy as zfp
except ImportError:
    zfp = None

try:
    import blosc
except ImportError:
    blosc = None

try:
    import brotli
except ImportError:
    brotli = None

try:
    import bitshuffle
except ImportError:
    bitshuffle = None

try:
    import snappy
except ImportError:
    snappy = None

try:
    import zopfli
except ImportError:
    zopfli = None

try:
    import PIL
except ImportError:
    PIL = None


def notimplemented(arg=False):
    """Return function decorator that raises NotImplementedError if not arg.

    >>> @notimplemented
    ... def test(): pass
    >>> test()
    Traceback (most recent call last):
    ...
    NotImplementedError: test not implemented

    >>> @notimplemented(True)
    ... def test(): pass
    >>> test()

    """
    def wrapper(func):
        @functools.wraps(func)
        def notimplemented(*args, **kwargs):
            raise NotImplementedError('%s not implemented' % func.__name__)
        return notimplemented

    if callable(arg):
        return wrapper(arg)
    if not arg:
        return wrapper

    def nop(func):
        return func

    return nop


def version(astype=None):
    """Return detailed version information."""
    versions = (
        ('imagecodecs.py', __version__),
        ('numpy', numpy.__version__),
        ('zlib', zlib.ZLIB_VERSION),
        ('bz2', 'stdlib' if bz2 else 'n/a'),
        ('lzma', getattr(lzma, '__version__', 'stdlib') if lzma else 'n/a'),
        ('blosc', blosc.__version__ if blosc else 'n/a'),
        ('zstd', zstd.version() if zstd else 'n/a'),
        ('lz4', lz4.VERSION if lz4 else 'n/a'),
        ('lzf', 'unknown' if lzf else 'n/a'),
        ('snappy', 'unknown' if snappy else 'n/a'),
        ('zopflipy', zopfli.__version__ if zopfli else 'n/a'),
        ('zfpy', zfp.__version__ if zfp else 'n/a'),
        ('bitshuffle', bitshuffle.__version__ if bitshuffle else 'n/a'),
        ('pillow', PIL.PILLOW_VERSION if PIL else 'n/a'),
    )
    if astype is str or astype is None:
        return ', '.join('%s-%s' % (k, v) for k, v in versions)
    if astype is dict:
        return dict(versions)
    return versions


def none_decode(data, *args, **kwargs):
    """Decode NOP."""
    return data


def none_encode(data, *args, **kwargs):
    """Encode NOP."""
    return data


def numpy_decode(data, index=0, out=None, **kwargs):
    """Decode NPY and NPZ."""
    with io.BytesIO(data) as fh:
        out = numpy.load(fh, **kwargs)
        if hasattr(out, 'files'):
            try:
                index = out.files[index]
            except Exception:
                pass
            out = out[index]
    return out


def numpy_encode(data, level=None, out=None, **kwargs):
    """Encode NPY and NPZ."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data, **kwargs)
        else:
            numpy.save(fh, data, **kwargs)
        fh.seek(0)
        out = fh.read()
    return out


def delta_encode(data, axis=-1, out=None):
    """Encode Delta."""
    if isinstance(data, (bytes, bytearray)):
        data = numpy.frombuffer(data, dtype='u1')
        diff = numpy.diff(data, axis=0)
        return numpy.insert(diff, 0, data[0]).tobytes()

    dtype = data.dtype
    if dtype.kind == 'f':
        data = data.view('u%i' % dtype.itemsize)

    diff = numpy.diff(data, axis=axis)
    key = [slice(None)] * data.ndim
    key[axis] = 0
    diff = numpy.insert(diff, 0, data[tuple(key)], axis=axis)

    if dtype.kind == 'f':
        return diff.view(dtype)
    return diff


def delta_decode(data, axis=-1, out=None):
    """Decode Delta."""
    if out is not None and not out.flags.writeable:
        out = None

    if isinstance(data, (bytes, bytearray)):
        data = numpy.frombuffer(data, dtype='u1')
        return numpy.cumsum(data, axis=0, dtype='u1', out=out).tobytes()

    if data.dtype.kind == 'f':
        view = data.view('u%i' % data.dtype.itemsize)
        view = numpy.cumsum(view, axis=axis, dtype=view.dtype)
        return view.view(data.dtype)
    return numpy.cumsum(data, axis=axis, dtype=data.dtype, out=out)


def xor_encode(data, axis=-1, out=None):
    """Encode XOR delta."""
    if isinstance(data, (bytes, bytearray)):
        data = numpy.frombuffer(data, dtype='u1')
        xor = numpy.bitwise_xor(data[1:], data[:-1])
        return numpy.insert(xor, 0, data[0]).tobytes()

    dtype = data.dtype
    if dtype.kind == 'f':
        data = data.view('u%i' % dtype.itemsize)

    key = [slice(None)] * data.ndim
    key[axis] = 0
    key0 = [slice(None)] * data.ndim
    key0[axis] = slice(1, None, None)
    key1 = [slice(None)] * data.ndim
    key1[axis] = slice(0, -1, None)

    key = tuple(key)
    key0 = tuple(key0)
    key1 = tuple(key1)

    xor = numpy.bitwise_xor(data[key0], data[key1])
    xor = numpy.insert(xor, 0, data[key], axis=axis)

    if dtype.kind == 'f':
        return xor.view(dtype)
    return xor


def xor_decode(data, axis=-1, out=None):
    """Decode XOR delta."""
    if isinstance(data, (bytes, bytearray)):
        prev = data[0]
        b = [chr(prev)]
        for c in data[1:]:
            prev = c ^ prev
            b.append(chr(prev))
        return ''.join(b).encode('latin1')
    raise NotImplementedError()


def floatpred_decode(data, axis=-2, out=None):
    """Decode floating point horizontal differencing.

    The TIFF predictor type 3 reorders the bytes of the image values and
    applies horizontal byte differencing to improve compression of floating
    point images. The ordering of interleaved color channels is preserved.

    Parameters
    ----------
    data : numpy.ndarray
        The image to be decoded. The dtype must be a floating point.
        The shape must include the number of contiguous samples per pixel
        even if 1.

    """
    # logging.warning('using numpy FloatPred decoder')
    if axis != -2:
        raise NotImplementedError('axis %i != -2' % axis)
    shape = data.shape
    dtype = data.dtype
    if len(shape) < 3:
        raise ValueError('invalid data shape')
    if dtype.char not in 'dfe':
        raise ValueError('not a floating point image')
    littleendian = data.dtype.byteorder == '<' or (
        sys.byteorder == 'little' and data.dtype.byteorder == '=')
    # undo horizontal byte differencing
    data = data.view('uint8')
    data.shape = shape[:-2] + (-1,) + shape[-1:]
    numpy.cumsum(data, axis=-2, dtype='uint8', out=data)
    # reorder bytes
    if littleendian:
        data.shape = shape[:-2] + (-1,) + shape[-2:]
    data = numpy.swapaxes(data, -3, -2)
    data = numpy.swapaxes(data, -2, -1)
    data = data[..., ::-1]
    # back to float
    data = numpy.ascontiguousarray(data)
    data = data.view(dtype)
    data.shape = shape
    return data


@notimplemented
def floatpred_encode(data, axis=-1, out=None):
    """Encode Floating Point Predictor."""


def bitorder_decode(data, out=None, _bitorder=[]):
    """Reverse bits in each byte of byte string or numpy array.

    Decode data where pixels with lower column values are stored in the
    lower-order bits of the bytes (TIFF FillOrder is LSB2MSB).

    Parameters
    ----------
    data : byte string or ndarray
        The data to be bit reversed. If byte string, a new bit-reversed byte
        string is returned. Numpy arrays are bit-reversed in-place.

    Examples
    --------
    >>> bitorder_decode(b'\\x01\\x64')
    b'\\x80&'
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
            b'\xdf?\xbf\x7f\xff')
        _bitorder.append(numpy.frombuffer(_bitorder[0], dtype='uint8'))
    try:
        view = data.view('uint8')
        numpy.take(_bitorder[1], view, out=view)
        return data
    except AttributeError:
        return data.translate(_bitorder[0])
    except ValueError:
        raise NotImplementedError('slices of arrays not supported')
    return None


bitorder_encode = bitorder_decode


@notimplemented
def packbits_encode(data, level=None, out=None):
    """Compress PackBits."""


def packbits_decode(encoded, out=None):
    r"""Decompress PackBits encoded byte string.

    >>> packbits_decode(b'\x80\x80')  # NOP
    b''
    >>> packbits_decode(b'\x02123')
    b'123'
    >>> packbits_decode(
    ...   b'\xfe\xaa\x02\x80\x00\x2a\xfd\xaa\x03\x80\x00\x2a\x22\xf7\xaa')[:-4]
    b'\xaa\xaa\xaa\x80\x00*\xaa\xaa\xaa\xaa\x80\x00*"\xaa\xaa\xaa\xaa\xaa\xaa'

    """
    # logging.warning('using pure Python PackBits decoder')
    out = []
    out_extend = out.extend
    i = 0
    try:
        while True:
            n = ord(encoded[i:i+1]) + 1
            i += 1
            if n > 129:
                # replicate
                out_extend(encoded[i:i+1] * (258 - n))
                i += 1
            elif n < 129:
                # literal
                out_extend(encoded[i:i+n])
                i += n
    except TypeError:
        pass
    return b''.join(out) if sys.version[0] == '2' else bytes(out)


@notimplemented
def lzw_encode(data, level=None, out=None):
    """Compress LZW."""


def lzw_decode(encoded, buffersize=0, out=None):
    r"""Decompress LZW (Lempel-Ziv-Welch) encoded TIFF strip (byte string).

    The strip must begin with a CLEAR code and end with an EOI code.

    This implementation of the LZW decoding algorithm is described in TIFF v6
    and is not compatible with old style LZW compressed files like
    quad-lzw.tif.

    >>> lzw_decode(b'\x80\x1c\xcc\'\x91\x01\xa0\xc2m6\x99NB\x03\xc9\xbe\x0b'
    ...            b'\x07\x84\xc2\xcd\xa68|"\x14 3\xc3\xa0\xd1c\x94\x02\x02')
    b'say hammer yo hammer mc hammer go hammer'

    """
    # logging.warning('using pure Python LZW decoder')
    len_encoded = len(encoded)
    bitcount_max = len_encoded * 8
    unpack = struct.unpack

    if sys.version[0] == '2':
        newtable = [chr(i) for i in range(256)]
    else:
        newtable = [bytes([i]) for i in range(256)]
    newtable.extend((0, 0))

    def next_code():
        """Return integer of 'bitw' bits at 'bitcount' position in encoded."""
        start = bitcount // 8
        s = encoded[start:start+4]
        try:
            code = unpack('>I', s)[0]
        except Exception:
            code = unpack('>I', s + b'\x00'*(4-len(s)))[0]
        code <<= bitcount % 8
        code &= mask
        return code >> shr

    switchbits = {  # code: bit-width, shr-bits, bit-mask
        255: (9, 23, int(9*'1'+'0'*23, 2)),
        511: (10, 22, int(10*'1'+'0'*22, 2)),
        1023: (11, 21, int(11*'1'+'0'*21, 2)),
        2047: (12, 20, int(12*'1'+'0'*20, 2)), }
    bitw, shr, mask = switchbits[255]
    bitcount = 0

    if len_encoded < 4:
        raise ValueError('strip must be at least 4 characters long')

    if next_code() != 256:
        raise ValueError('strip must begin with CLEAR code')

    code = 0
    oldcode = 0
    result = []
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
        logging.warning('unexpected end of LZW stream (code %i)', code)

    return b''.join(result)


@notimplemented
def packints_encode(*args, **kwargs):
    """Pack integers."""


def packints_decode(data, dtype, numbits, runlen=0, out=None):
    """Decompress byte string to array of integers of any bit size <= 32.

    This Python implementation is slow and only handles itemsizes 1, 2, 4, 8,
    16, 32, and 64.

    Parameters
    ----------
    data : byte str
        Data to decompress.
    dtype : numpy.dtype or str
        A numpy boolean or integer type.
    numbits : int
        Number of bits per integer.
    runlen : int
        Number of consecutive integers, after which to start at next byte.

    Examples
    --------
    >>> packints_decode(b'a', 'B', 1)
    array([0, 1, 1, 0, 0, 0, 0, 1], dtype=uint8)
    >>> packints_decode(b'ab', 'B', 2)
    array([1, 2, 0, 1, 1, 2, 0, 2], dtype=uint8)

    """
    # logging.warning('using pure Python PackInts decoder')
    if numbits == 1:  # bitarray
        data = numpy.frombuffer(data, '|B')
        data = numpy.unpackbits(data)
        if runlen % 8:
            data = data.reshape(-1, runlen + (8 - runlen % 8))
            data = data[:, :runlen].reshape(-1)
        return data.astype(dtype)

    dtype = numpy.dtype(dtype)
    if numbits in (8, 16, 32, 64):
        return numpy.frombuffer(data, dtype)
    if numbits not in (1, 2, 4, 8, 16, 32):
        raise ValueError('itemsize not supported: %i' % numbits)
    if dtype.kind not in 'biu':
        raise ValueError('invalid dtype')

    itembytes = next(i for i in (1, 2, 4, 8) if 8 * i >= numbits)
    if itembytes != dtype.itemsize:
        raise ValueError('dtype.itemsize too small')
    if runlen == 0:
        runlen = (8 * len(data)) // numbits
    skipbits = runlen * numbits % 8
    if skipbits:
        skipbits = 8 - skipbits
    shrbits = itembytes*8 - numbits
    bitmask = int(numbits*'1'+'0'*shrbits, 2)
    dtypestr = '>' + dtype.char  # dtype always big-endian?

    unpack = struct.unpack
    size = runlen * (len(data)*8 // (runlen*numbits + skipbits))
    result = numpy.empty((size,), dtype)
    bitcount = 0
    for i in range(size):
        start = bitcount // 8
        s = data[start:start+itembytes]
        try:
            code = unpack(dtypestr, s)[0]
        except Exception:
            code = unpack(dtypestr, s + b'\x00'*(itembytes-len(s)))[0]
        code <<= bitcount % 8
        code &= bitmask
        result[i] = code >> shrbits
        bitcount += numbits
        if (i+1) % runlen == 0:
            bitcount += skipbits
    return result


@notimplemented(bitshuffle)
def bitshuffle_encode(data, level=1, itemsize=1, blocksize=0, out=None):
    """Bitshuffle."""
    if isinstance(data, numpy.ndarray):
        return bitshuffle.bitshuffle(data, blocksize)
    data = numpy.frombuffer(data, dtype='uint%i' % (itemsize * 8))
    data = bitshuffle.bitshuffle(data, blocksize)
    return data.tobytes()


@notimplemented(bitshuffle)
def bitshuffle_decode(data, itemsize=1, blocksize=0, out=None):
    """Bitunshuffle."""
    if isinstance(data, numpy.ndarray):
        return bitshuffle.bitunshuffle(data, blocksize)
    data = numpy.frombuffer(data, dtype='uint%i' % (itemsize * 8))
    data = bitshuffle.bitunshuffle(data, blocksize)
    return data.tobytes()


def zlib_encode(data, level=6, out=None):
    """Compress Zlib DEFLATE."""
    return zlib.compress(data, level)


def zlib_decode(data, out=None):
    """Decompress Zlib DEFLATE."""
    return zlib.decompress(data)


@notimplemented(bz2)
def bz2_encode(data, level=9, out=None):
    """Compress BZ2."""
    return bz2.compress(data, level)


@notimplemented(bz2)
def bz2_decode(data, out=None):
    """Decompress BZ2."""
    return bz2.decompress(data)


@notimplemented(blosc)
def blosc_encode(data, level=None, compressor='blosclz', numthreads=1,
                 typesize=8, blocksize=0, shuffle=None, out=None):
    """Compress Blosc."""
    if shuffle is None:
        shuffle = blosc.SHUFFLE
    if level is None:
        level = 9
    return blosc.compress(data, typesize=typesize, clevel=level,
                          shuffle=shuffle, cname=compressor)


@notimplemented(blosc)
def blosc_decode(data, out=None):
    """Decompress Blosc."""
    return blosc.decompress(data)


@notimplemented(lzma)
def lzma_encode(data, level=5, out=None):
    """Compress LZMA."""
    return lzma.compress(data, level)


@notimplemented(lzma)
def lzma_decode(data, out=None):
    """Decompress LZMA."""
    return lzma.decompress(data)


@notimplemented(zstd)
def zstd_encode(data, level=5, out=None):
    """Compress ZStandard."""
    return zstd.compress(data, level)


@notimplemented(zstd)
def zstd_decode(data, out=None):
    """Decompress ZStandard."""
    return zstd.decompress(data)


@notimplemented(brotli)
def brotli_encode(data, level=11, mode=0, lgwin=22, out=None):
    """Compress Brotli."""
    return brotli.compress(data, quality=level, mode=mode, lgwin=lgwin)


@notimplemented(brotli)
def brotli_decode(data, out=None):
    """Decompress Brotli."""
    return brotli.decompress(data)


@notimplemented(snappy)
def snappy_encode(data, level=None, out=None):
    """Compress Snappy."""
    return snappy.compress(data)


@notimplemented(snappy)
def snappy_decode(data, out=None):
    """Decompress Snappy."""
    return snappy.decompress(data)


@notimplemented(zopfli)
def zopfli_encode(data, level=None, out=None):
    """Compress Zopfli."""
    c = zopfli.ZopfliCompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
    return c.compress(data) + c.flush()


@notimplemented(zopfli)
def zopfli_decode(data, out=None):
    """Compress Zopfli."""
    d = zopfli.ZopfliDecompressor(zopfli.ZOPFLI_FORMAT_ZLIB)
    return d.decompress(data) + d.flush()


@notimplemented(lzf)
def lzf_encode(data, level=None, header=False, out=None):
    """Compress LZF."""
    return lzf.compress(data)


@notimplemented(lzf)
def lzf_decode(data, header=False, out=None):
    """Decompress LZF."""
    return lzf.decompress(data)


@notimplemented(zfp)
def zfp_encode(data, level=None, mode=None, execution=None, header=True,
               out=None):
    kwargs = {'write_header': header}
    if mode in (None, zfp.mode_null, 'R', 'reversible'):  # zfp.mode_reversible
        pass
    elif mode in (zfp.mode_fixed_precision, 'p', 'precision'):
        kwargs['precision'] = -1 if level is None else level
    elif mode in (zfp.mode_fixed_rate, 'r', 'rate'):
        kwargs['rate'] = -1 if level is None else level
    elif mode in (zfp.mode_fixed_accuracy, 'a', 'accuracy'):
        kwargs['tolerance'] = -1 if level is None else level
    elif mode in (zfp.mode_expert, 'c', 'expert'):
        minbits, maxbits, maxprec, minexp = level
        raise NotImplementedError()
    return zfp.compress_numpy(data, **kwargs)


@notimplemented(zfp)
def zfp_decode(data, shape=None, dtype=None, out=None):
    """Decompress ZFP."""
    return zfp.decompress_numpy(data)


@notimplemented(bitshuffle)
def bitshuffle_lz4_encode(data, level=1, blocksize=0, out=None):
    """Compress LZ4 with Bitshuffle."""
    return bitshuffle.compress_lz4(data, blocksize)


@notimplemented(bitshuffle)
def bitshuffle_lz4_decode(data, shape, dtype, blocksize=0, out=None):
    """Decompress LZ4 with Bitshuffle."""
    return bitshuffle.decompress_lz4(data, shape, dtype, blocksize)


@notimplemented(lz4)
def lz4_encode(data, level=1, header=False, out=None):
    """Compress LZ4."""
    return lz4.block.compress(data, store_size=header)


@notimplemented(lz4)
def lz4_decode(data, header=False, out=None):
    """Decompress LZ4."""
    if header:
        return lz4.block.decompress(data)
    if isinstance(out, int):
        return lz4.block.decompress(data, uncompressed_size=out)
    outsize = max(24, 24 + 255 * (len(data) - 10))  # ugh
    return lz4.block.decompress(data, uncompressed_size=outsize)


@notimplemented(PIL)
def pil_decode(data, out=None):
    """Decode image data using PIL."""
    return numpy.asarray(PIL.Image.open(io.BytesIO(data)))


@notimplemented(PIL)
def jpeg_decode(data, bitspersample=None, tables=None, colorspace=None,
                outcolorspace=None, out=None):
    """Decode JPEG."""
    jpeg8_decode(data, tables=tables, colorspace=colorspace,
                 outcolorspace=outcolorspace, out=out)


@notimplemented
def jpeg_encode(*args, **kwargs):
    """Encode JPEG."""


@notimplemented(PIL)
def jpeg8_decode(data, tables=None, colorspace=None, outcolorspace=None,
                 out=None):
    """Decode JPEG 8-bit."""
    if data[:3] != b'\xff\xd8\xff':
        raise ValueError('not a JPEG image')
    if tables or colorspace or outcolorspace:
        raise NotImplementedError(
            'JPEG tables, colorspace, and outcolorspace otions not supported')
    return pil_decode(data)


@notimplemented
def jpeg8_encode(*args, **kwargs):
    """Encode JPEG 8-bit."""


@notimplemented
def jpeg12_decode(*args, **kwargs):
    """Decode JPEG 12-bit."""


@notimplemented
def jpeg12_encode(*args, **kwargs):
    """Encode JPEG 12-bit."""


@notimplemented
def jpegls_decode(*args, **kwargs):
    """Decode JPEG LS."""


@notimplemented
def jpegls_encode(*args, **kwargs):
    """Encode JPEG LS."""


@notimplemented
def jpegxp_decode(*args, **kwargs):
    """Decode JPEG XP."""


@notimplemented
def jpegxp_encode(*args, **kwargs):
    """Encode JPEG XP."""


@notimplemented
def jpegsof3_decode(*args, **kwargs):
    """Decode JPEG SOF3."""


@notimplemented
def jpegsof3_encode(*args, **kwargs):
    """Encode JPEG SOF3."""


@notimplemented(PIL)
def jpeg2k_decode(data, verbose=0, out=None):
    """Decode JPEG 2000."""
    if (
        data[:4] != b'\xff\x4f\xff\x51' and data[:4] != b'\x0d\x0a\x87\x0a' and
        data[:12] != b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a'
    ):
        raise ValueError('not a JPEG 2000 image')
    return pil_decode(data)


@notimplemented
def jpeg2k_encode(*args, **kwargs):
    """Encode JPEG 2000."""


@notimplemented
def jpegxr_decode(*args, **kwargs):
    """Decode JPEG XR."""


@notimplemented
def jpegxr_encode(*args, **kwargs):
    """Encode JPEG XR."""


@notimplemented(PIL)
def webp_decode(data, out=None):
    """Decode WebP."""
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        raise ValueError('not a WebP image')
    return pil_decode(data)


@notimplemented
def webp_encode(*args, **kwargs):
    """Encode WebP."""


@notimplemented(PIL)
def png_decode(data, out=None):
    """Decode PNG."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG image')
    return pil_decode(data)


@notimplemented
def png_encode(*args, **kwargs):
    """Encode PNG."""


@notimplemented
def aec_decode(*args, **kwargs):
    """Decode AEC."""


@notimplemented
def aec_encode(*args, **kwargs):
    """Encode AEC."""


@notimplemented
def dtype_decode(*args, **kwargs):
    """Convert dtype."""


@notimplemented
def dtype_encode(*args, **kwargs):
    """Convert dtype."""


if __name__ == '__main__':
    import doctest
    print(version())
    numpy.set_printoptions(suppress=True, precision=2)
    doctest.testmod()
