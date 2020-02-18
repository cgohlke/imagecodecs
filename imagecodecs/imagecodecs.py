# imagecodecs.py

# Copyright (c) 2008-2020, Christoph Gohlke
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

Imagecodecs is a Python library that provides block-oriented, in-memory buffer
transformation, compression, and decompression functions for use in the
tifffile, czifile, and other scientific imaging modules.

Decode and/or encode functions are implemented for Zlib (DEFLATE),
ZStandard (ZSTD), Blosc, Brotli, Snappy, LZMA, BZ2, LZ4, LZW, LZF, ZFP, AEC,
NPY, PNG, GIF, TIFF, WebP, JPEG 8-bit, JPEG 12-bit, JPEG SOF3, JPEG 2000,
JPEG LS, JPEG XR, JPEG XL, PackBits, Packed Integers, Delta, XOR Delta,
Floating Point Predictor, Bitorder reversal, and Bitshuffle.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.2.18

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.6.8, 3.7.6, 3.8.1 64-bit <https://www.python.org>`_
* `Numpy 1.16.6 <https://www.numpy.org>`_
* `Cython 0.29.15 <https://cython.org>`_
* `zlib 1.2.11 <https://github.com/madler/zlib>`_
* `lz4 1.9.2 <https://github.com/lz4/lz4>`_
* `zstd 1.4.4 <https://github.com/facebook/zstd>`_
* `blosc 1.17.1 <https://github.com/Blosc/c-blosc>`_
* `bzip2 1.0.8 <https://sourceware.org/bzip2>`_
* `liblzma 5.2.4 <https://github.com/xz-mirror/xz>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libpng 1.6.37 <https://github.com/glennrp/libpng>`_
* `libwebp 1.0.3 <https://github.com/webmproject/libwebp>`_
* `libtiff 4.1.0 <https://gitlab.com/libtiff/libtiff>`_
* `libjpeg-turbo 2.0.4 <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
  (8 and 12-bit)
* `charls 2.1.0 <https://github.com/team-charls/charls>`_
* `openjpeg 2.3.1 <https://github.com/uclouvain/openjpeg>`_
* `jxrlib 1.1 <https://packages.debian.org/source/sid/jxrlib>`_
* `zfp 0.5.5 <https://github.com/LLNL/zfp>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_
* `libaec 1.0.4 <https://gitlab.dkrz.de/k202009/libaec>`_
* `snappy 1.1.8 <https://github.com/google/snappy>`_
* `zopfli-1.0.3 <https://github.com/google/zopfli>`_
* `brotli 1.0.7 <https://github.com/google/brotli>`_
* `brunsli 0.1 <https://github.com/google/brunsli>`_
* `giflib 5.2.1 <http://giflib.sourceforge.net/>`_
* `lcms 2.9 <https://github.com/mm2/Little-CMS>`_

Required Python packages for testing (other versions may work):

* `tifffile 2020.2.16 <https://pypi.org/project/tifffile/>`_
* `czifile 2019.7.2 <https://pypi.org/project/czifile/>`_
* `python-blosc 1.8.3 <https://github.com/Blosc/python-blosc>`_
* `python-lz4 3.0.2 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd 1.4.4 <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzf 0.2.4 <https://github.com/teepark/python-lzf>`_
* `python-brotli 1.0.7 <https://github.com/google/brotli/tree/master/python>`_
* `python-snappy 0.5.4 <https://github.com/andrix/python-snappy>`_
* `zopflipy 1.3 <https://github.com/hattya/zopflipy>`_
* `bitshuffle 0.3.5 <https://github.com/kiyo-masui/bitshuffle>`_

Notes
-----
The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 32-bit versions are deprecated. Python 2.7 and 3.5 are no longer
supported.

The latest `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017
and 2019 <https://support.microsoft.com/en-us/help/2977003/
the-latest-supported-visual-c-downloads>`_ is required on Windows.

Refer to the imagecodecs/licenses folder for 3rd party library licenses.

This software is based in part on the work of the Independent JPEG Group.

This software includes modified versions of `dcm2niix's jpg_0XC3.cpp
<https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp>`_
and `OpenJPEG's color.c
<https://github.com/uclouvain/openjpeg/blob/master/src/bin/common/color.c>`_.

Build instructions and wheels for manylinux and macOS courtesy of
`Grzegorz Bokota <https://github.com/Czaki/imagecodecs>`_.

To install the requirements for building imagecodecs from source code on
latest Ubuntu Linux distributions, run:

    ``sudo apt-get install build-essential python3-dev cython3
    python3-setuptools python3-pip python3-wheel python3-numpy
    python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4
    libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev
    libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-turbo8-dev libjxr-dev
    liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev
    libzopfli-dev libgif-dev libtiff-dev``

Use the ``--skip-extension`` build options to skip building specific
extensions. Use the ``--lite`` build option to only build extensions without
3rd-party dependencies. Edit ``setup.py`` to modify other build options.

Other Python packages and C libraries providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_
* `backports.lzma <https://github.com/peterjc/backports.lzma>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `packbits <https://github.com/psd-tools/packbits>`_
* `fpzip <https://github.com/seung-lab/fpzip>`_
* `libmng <https://sourceforge.net/projects/libmng/>`_
* `APNG patch for libpng <https://sourceforge.net/projects/libpng-apng/>`_
* `OpenEXR <https://github.com/AcademySoftwareFoundation/openexr>`_
* `LERC <https://github.com/Esri/lerc>`_

Revisions
---------
2020.2.18
    Pass 3469 tests.
    Fix segfault when decoding corrupted LZW segments.
    Work around Cython raises AttributeError when using incompatible numpy.
    Raise ValueError if in-place decoding is not possible (except floatpred).
2020.1.31
    Add GIF codec via giflib.
    Add TIFF decoder via libtiff (WIP).
    Add codec_check functions (WIP).
    Fix formatting libjpeg error messages.
    Use xfail in tests.
    Load extensions on demand on Python >= 3.7.
    Add build options to skip building specific extensions.
    Split imagecodecs extension into individual extensions.
    Move shared code into shared extension.
    Rename imagecodecs_lite extension and imagecodecs C library to 'imcd'.
    Remove support for Python 2.7 and 3.5.
2019.12.31
    Fix decoding of indexed PNG with transparency.
    Last version to support Python 2.7 and 3.5.
2019.12.16
    Add Zopfli codec.
    Add Snappy codec.
    Rename j2k codec to jpeg2k.
    Rename jxr codec to jpegxr.
    Use Debian's jxrlib.
    Support pathlib and binary streams in imread and imwrite.
    Move external C declarations to pxd files.
    Move shared code to pxi file.
    Update copyright notices.
2019.12.10
    Add version functions.
    Add Brotli codec (WIP).
    Add optional JPEG XL codec via Brunsli repacker (WIP).
2019.12.3
    Sync with imagecodecs-lite.
2019.11.28
    Add AEC codec via libaec (WIP).
    Do not require scikit-image for testing.
    Require CharLS 2.1.
2019.11.18
    Add bitshuffle codec.
    Fix formatting of unknown error numbers.
    Fix test failures with official python-lzf.
2019.11.5
    Rebuild with updated dependencies.
2019.5.22
    Add optional YCbCr chroma subsampling to JPEG encoder.
    Add default reversible mode to ZFP encoder.
    Add imread and imwrite helper functions.
2019.4.20
    Fix setup requirements.
2019.2.22
    Move codecs without 3rd-party C library dependencies to imagecodecs_lite.
2019.2.20
    Rebuild with updated dependencies.
2019.1.20
    Add more pixel formats to JPEG XR codec.
    Add JPEG XR encoder.
2019.1.14
    Add optional ZFP codec via zfp library (WIP).
    Add numpy NPY and NPZ codecs.
    Fix some static codechecker errors.
2019.1.1
    ...

Refer to the CHANGES file for older revisions.

"""

__version__ = '2020.2.18'

import os
import sys
import io
import importlib

import numpy

# names of public attributes by module
# will be updated with standard codec attributes
_API = {
    None: [
        'imread',
        'imwrite',
        'version',
        'DelayedImportError',
        (
            'none',
            'numpy',
            'jpeg',
        )
        ],
    'imcd': [
        'imcd_version',
        'numpy_abi_version',
        'cython_version',
        (
            'bitorder',
            'delta',
            'floatpred',
            'lzw',
            'packbits',
            'packints',
            'xor',
        )
        ],
    'aec': [],
    'bitshuffle': [],
    'blosc': [],
    'brotli': [],
    'bz2': [],
    'gif': [],
    'jpeg2k': [],
    'jpeg8': ['jpeg8_turbo_version'],
    'jpeg12': ['jpeg12_turbo_version'],
    'jpegls': [],
    'jpegsof3': [],
    'jpegxl': [],
    'jpegxr': [],
    'lz4': [],
    'lzf': [],
    'lzma': [],
    'png': [],
    'snappy': [],
    # 'szip': [],
    'tiff': [],
    'webp': [],
    'zfp': [],
    'zlib': ['zlib_crc32'],
    'zopfli': [],
    'zstd': [],
    # 'module': ['attribute1', 'attribute2', ('codec1', 'code2', )]
}

# map extra to existing attributes
# e.g. keep deprecated names for older versions of tifffile and czifile
_COMPATIBILITY = {
    'JPEG': 'JPEG8',
    'jpeg_check': 'jpeg8_check',
    'jpeg_version': 'jpeg8_version',
    'zopfli_check': 'zlib_check',
    'zopfli_decode': 'zlib_decode',
    'j2k_encode': 'jpeg2k_encode',
    'j2k_decode': 'jpeg2k_decode',
    'jxr_encode': 'jpegxr_encode',
    'jxr_decode': 'jpegxr_decode',
}

# map attribute names to module names
_ATTRIBUTES = {}

# map of codec names to module names
_CODECS = {}


def _add_codec(module, codec=None, attributes=None):
    """Register codec in global _API, _ATTRIBUTES, and _CODECS."""
    if codec is None:
        codec = module
    if attributes is None:
        attributes = (
            f'{codec}_encode',
            f'{codec}_decode',
            f'{codec}_check',
            f'{codec}_version',
            f'{codec.capitalize()}Error',
            f'{codec.upper()}',
        )
    if module in _API:
        _API[module].extend(attributes)
    else:
        _API[module] = attributes
    _ATTRIBUTES.update({attr: module for attr in _API[module]})
    _CODECS[codec] = module


def _register_codecs():
    """Parse _API and register all codecs."""
    for module, attributes in _API.items():
        for attr in attributes.copy():
            if isinstance(attr, tuple):
                attributes.remove(attr)
                for codec in attr:
                    _add_codec(module, codec)
                break
        else:
            _add_codec(module)


def _load_all():
    """Add all registered attributes to package namespace."""
    for name in __dir__():
        __getattr__(name)


def __dir__():
    """Module __dir__."""
    return sorted(list(_ATTRIBUTES) + list(_COMPATIBILITY))


def __getattr__(name):
    """Load attribute's extension and add its attributes to package namespace.

    """
    name_ = name
    name = _COMPATIBILITY.get(name, name)

    if name not in _ATTRIBUTES:
        raise AttributeError(f"module 'imagecodecs' has no attribute {name!r}")

    module_ = _ATTRIBUTES[name]
    if module_ is None:
        return None

    try:
        module = importlib.import_module('._' + module_, 'imagecodecs')
    except ImportError:
        module = None
    except AttributeError:
        # AttributeError: type object 'imagecodecs._module.array' has no
        # attribute '__reduce_cython__'
        # work around Cython raises AttributeError e.g. when the _shared
        # module failed to import due to an incompatible numpy version
        from . import _shared  # noqa
        module = None

    for n in _API[module_]:
        if n in _COMPATIBILITY:
            continue
        attr = getattr(module, n, None)
        if attr is None:
            attr = _stub(n, module)
        setattr(imagecodecs, n, attr)

    attr = getattr(imagecodecs, name)
    if name != name_:
        setattr(imagecodecs, name_, attr)
    return attr


class DelayedImportError(ImportError):
    def __init__(self, name):
        msg = f"could not import name {name!r} from 'imagecodecs'"
        super().__init__(msg)


def _stub(name, module):
    """Return stub function or class."""

    if name.endswith('_version'):
        if module is None:
            def stub_version():
                f"""Stub for imagecodecs.{name}."""
                return f"{name[:-8]} n/a"
        else:
            def stub_version():
                f"""Stub for imagecodecs.{name}."""
                return f"{name[:-8]} unknow"

        return stub_version

    if name.endswith('_check'):
        if module is None:
            def stub_check(arg):
                f"""Stub for imagecodecs.{name}."""
                return False
        else:
            def stub_check(arg):
                f"""Stub for imagecodecs.{name}."""
                return None

        return stub_check

    if name.endswith('_decode'):
        def stub_decode(*args, **kwargs):
            f"""Stub for imagecodecs.{name}."""
            raise DelayedImportError(name)

        return stub_decode

    if name.endswith('_encode'):
        def stub_encode(*args, **kwargs):
            f"""Stub for imagecodecs.{name}."""
            raise DelayedImportError(name)

        return stub_encode

    if name.islower():
        def stub_function(*args, **kwargs):
            f"""Stub for imagecodecs.{name}."""
            raise DelayedImportError(name)

        return stub_function

    if name.endswith('Error'):
        class StubError(RuntimeError):
            f"""Stub for imagecodecs.{name}."""

            def __init__(self, *args, **kwargs):
                raise DelayedImportError(name)

        return StubError

    class StubType(type):
        def __getattr__(cls, arg):
            raise DelayedImportError(name)

        if module is None:
            def __bool__(cls):
                return False

    if name.isupper():
        class STUB(metaclass=StubType):
            f"""Stub for imagecodecs.{name}."""

        return STUB

    class Stub(metaclass=StubType):
        f"""Stub for imagecodecs.{name}."""

    return Stub


def _extensions():
    """Return sorted list of extension names."""
    return sorted(e for e in _API if e is not None)


def version(astype=None, _versions_=[]):
    """Return version information about all codecs and dependencies."""
    if not _versions_:
        _versions_.extend((
            f'imagecodecs {__version__}',
            imagecodecs.cython_version(),
            imagecodecs.numpy_version(),
            imagecodecs.numpy_abi_version(),
            imagecodecs.imcd_version(),
        ))
        _versions_.extend(
            sorted(set(
                getattr(imagecodecs, v)()
                for v in _ATTRIBUTES
                if v.endswith('_version') and v not in (
                    'imcd_version', 'numpy_abi_version', 'numpy_version',
                    'cython_version', 'none_version')
            )))

    if astype is None or astype is str:
        return ', '.join(ver.replace(' ', '-') for ver in _versions_)
    if astype is dict:
        return dict(ver.split(' ') for ver in _versions_)
    return tuple(_versions_)


def imread(fileobj, codec=None, memmap=True, return_codec=False, **kwargs):
    """Return image data from file as numpy array."""
    import pathlib
    import mmap

    codecs = []
    if codec is None:
        # find codec based on file extension
        if isinstance(fileobj, (str, pathlib.Path)):
            ext = os.path.splitext(str(fileobj))[-1][1:].lower()
        else:
            ext = None
        if ext in _imcodecs():
            codec = _imcodecs()[ext]
            if codec == 'jpeg':
                codecs.extend(('jpeg8', 'jpeg12', 'jpegls', 'jpegsof3'))
            else:
                codecs.append(codec)
        # try other imaging codecs
        codecs.extend(
            c for c in (
                'tiff', 'png', 'gif', 'webp', 'jpeg8', 'jpeg12', 'jpegsof3',
                'jpeg2k', 'jpegls', 'jpegxr', 'jpegxl', 'zfp', 'numpy'
            ) if c not in codecs
        )
    else:
        # use provided codecs
        if not isinstance(codec, (list, tuple)):
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                c = c.lower()
                c = _imcodecs().get(c, c)
            codecs.append(c)

    offset = None
    close = False
    if isinstance(fileobj, mmap.mmap):
        data = fileobj
        offset = data.tell()
    elif hasattr(fileobj, 'read'):
        # binary stream: open file, BytesIO
        data = fileobj.read()
    elif isinstance(fileobj, (str, pathlib.Path)):
        # TODO: support urllib.request.urlopen ?
        # file name
        with open(str(fileobj), 'rb') as fh:
            if memmap:
                offset = 0
                close = True
                data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                data = fh.read()
    else:
        # binary data
        data = fileobj

    exceptions = []
    image = None
    for codec in codecs:
        if callable(codec):
            func = codec
        else:
            try:
                func = getattr(imagecodecs, codec + '_decode')
            except Exception as exc:
                exceptions.append(f'{repr(codec).upper()}: {exc}')
                continue
        try:
            image = func(data, **kwargs)
            if image.dtype == 'object':
                image = None
                raise ValueError('failed')
            break
        except DelayedImportError:
            pass
        except Exception as exc:
            # raise
            exceptions.append(f'{func.__name__.upper()}: {exc}')
        if offset is not None:
            data.seek(offset)

    if close:
        data.close()

    if image is None:
        raise ValueError('\n'.join(exceptions))

    if return_codec:
        return image, func
    return image


def imwrite(fileobj, data, codec=None, **kwargs):
    """Write numpy array to image file."""
    if codec is None:
        # find codec based on file extension
        import pathlib

        if isinstance(fileobj, (str, pathlib.Path)):
            ext = os.path.splitext(str(fileobj))[-1].lower()[1:]
        else:
            raise ValueError(f'no codec specified')

        codec = _imcodecs().get(ext, ext)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            raise ValueError(f'invalid codec {codec!r}') from exc

    elif isinstance(codec, str):
        codec = codec.lower()
        codec = _imcodecs().get(codec, codec)
        try:
            codec = getattr(imagecodecs, codec + '_encode')
        except AttributeError as exc:
            raise ValueError(f'invalid codec {codec!r}') from exc

    elif not callable(codec):
        raise ValueError(f'invalid codec {codec!r}')

    data = codec(data, **kwargs)
    if hasattr(fileobj, 'write'):
        # binary stream: open file, BytesIO
        fileobj.write(data)
    else:
        # file name
        with open(str(fileobj), 'wb') as fh:
            fh.write(data)


def _imcodecs(_codecs_={}):
    """Return map of image file extensions to codec names."""
    if not _codecs_:
        codecs = {
            'numpy': ('npy', 'npz'),
            'zfp': ('zfp', ),
            'gif': ('gif', ),
            'png': ('png', ),
            'webp': ('webp', ),
            'tiff': ('tif', 'tiff', 'tf8', 'tf2', 'btf'),
            'jpeg': ('jpg', 'jpeg', 'jpe', 'jfif', 'jif'),
            'jpegls': ('jls', ),
            'jpegxl': ('jxl', 'brn'),
            'jpegxr': ('jxr', 'hdp', 'wdp'),
            'jpeg2k': ('j2k', 'jp2', 'j2c', 'jpc', 'jpx', 'jpf'),
            # 'jpeg8': ('jpg8', 'jpeg8'),
            # 'jpeg12': ('jpg12', 'jpeg12'),
            # 'jpegsof3': ('jsof3', 'jpegsof3', 'jpeg0xc3')
        }
        _codecs_.update(
            (ext, codec) for codec, exts in codecs.items() for ext in exts
        )
    return _codecs_


NONE = True
NoneError = RuntimeError


def none_version():
    """Return empty version string."""
    return ''


def none_check(data):
    """Return True if data likely contains Template data."""


def none_decode(data, *args, **kwargs):
    """Decode NOP."""
    return data


def none_encode(data, *args, **kwargs):
    """Encode NOP."""
    return data


NUMPY = True
NumpyError = RuntimeError


def numpy_version():
    """Return numpy version string."""
    return f'numpy {numpy.__version__}'


def numpy_check(data):
    """Return True if data likely contains NPY or NPZ data."""
    with io.BytesIO(data) as fh:
        data = fh.read(64)
    magic = b'\x93NUMPY'
    return data.startswith(magic) or (data.startswith(b'PK') and magic in data)


def numpy_decode(data, index=0, out=None, **kwargs):
    """Decode NPY and NPZ."""
    with io.BytesIO(data) as fh:
        try:
            out = numpy.load(fh, **kwargs)
        except ValueError as exc:
            raise ValueError('not a numpy array') from exc
        if hasattr(out, 'files'):
            try:
                index = out.files[index]
            except Exception:
                pass
            out = out[index]
    return out


def numpy_encode(data, level=None, out=None):
    """Encode NPY and NPZ."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data)
        else:
            numpy.save(fh, data)
        fh.seek(0)
        out = fh.read()
    return out


JpegError = RuntimeError


def jpeg_decode(data, bitspersample=None, tables=None, colorspace=None,
                outcolorspace=None, shape=None, out=None):
    """Decode JPEG 8-bit, 12-bit, SOF3, LS, or XL.

    """
    if bitspersample is None:
        try:
            return imagecodecs.jpeg8_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        except Exception as exc:
            msg = str(exc)
            if 'Empty JPEG image' in msg:
                # TODO: handle Hamamatsu NDPI slides with dimensions > 65500
                raise exc
            if 'Unsupported JPEG data precision' in msg:
                return imagecodecs.jpeg12_decode(
                    data, tables=tables, colorspace=colorspace,
                    outcolorspace=outcolorspace, shape=shape, out=out)
            if 'SOF type' in msg:
                return imagecodecs.jpegsof3_decode(data, out=out)
            # Unsupported marker type
            try:
                return imagecodecs.jpegls_decode(data, out=out)
            except Exception:
                return imagecodecs.jpegxl_decode(data, out=out)
    try:
        if bitspersample == 8:
            return imagecodecs.jpeg8_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        if bitspersample == 12:
            return imagecodecs.jpeg12_decode(
                data, tables=tables, colorspace=colorspace,
                outcolorspace=outcolorspace, shape=shape, out=out)
        try:
            return imagecodecs.jpegls_decode(data, out=out)
        except Exception:
            return imagecodecs.jpegsof3_decode(data, out=out)
    except Exception as exc:
        msg = str(exc)
        if 'Empty JPEG image' in msg:
            raise exc
        if 'SOF type' in msg:
            return imagecodecs.jpegsof3_decode(data, out=out)
        try:
            return imagecodecs.jpegls_decode(data, out=out)
        except Exception:
            return imagecodecs.jpegxl_decode(data, out=out)


def jpeg_encode(data, level=None, colorspace=None, outcolorspace=None,
                subsampling=None, optimize=None, smoothing=None, out=None):
    """Encode JPEG 8-bit or 12-bit.

    """
    if data.dtype == numpy.uint8:
        func = imagecodecs.jpeg8_encode
    elif data.dtype == numpy.uint16:
        func = imagecodecs.jpeg12_encode
    else:
        raise ValueError(f'invalid data type {data.dtype}')
    return func(data, level=level, colorspace=colorspace,
                outcolorspace=outcolorspace, subsampling=subsampling,
                optimize=optimize, smoothing=smoothing, out=out)


# initialize package

imagecodecs = sys.modules['imagecodecs']

_register_codecs()

if sys.version_info < (3, 7):
    _load_all()
