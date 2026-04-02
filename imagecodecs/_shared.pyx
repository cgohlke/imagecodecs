# imagecodecs/_shared.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2018-2026, Christoph Gohlke
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

"""Private, shared functions for imagecodecs extension modules."""

import dataclasses
import enum

import numpy

cimport numpy
from cpython.bytearray cimport (
    PyByteArray_Check,
    PyByteArray_FromStringAndSize,
    PyByteArray_Resize,
)
from cpython.bytes cimport (
    PyBytes_AsString,
    PyBytes_Check,
    PyBytes_FromStringAndSize,
)
from libc.stdint cimport uint8_t

numpy.import_array()


cdef tuple _parse_output(
    out, ssize_t outsize=-1, bint outgiven=False, outtype=bytes
):
    """Return out, outsize, outgiven, outtype from output argument."""
    if out is None:
        # create new bytes output
        return out, outsize, outgiven, outtype
    if out is bytes:
        # create new bytes output
        out = None
        outtype = bytes
    elif out is bytearray:
        # create new bytearray output
        out = None
        outtype = bytearray
    elif isinstance(out, int):
        # create new bytes output of expected length
        outsize = out
        out = None
    elif isinstance(out, bytes):
        raise TypeError("'bytes' object does not support item assignment")
    else:
        # use provided output buffer
        # outsize = len(out)
        # outtype = type(out)
        outgiven = True
    return out, outsize, outgiven, outtype


cdef _create_output(out, ssize_t size, const char* string=NULL):
    """Return new bytes or bytesarray of length size.

    Copy content of 'string', if provided, to new object.
    Return NULL on failure.

    """
    if out is None or out is bytes or PyBytes_Check(out):
        obj = PyBytes_FromStringAndSize(string, size)
        if obj is None:
            raise MemoryError('PyBytes_FromStringAndSize failed')
        return obj
    obj = PyByteArray_FromStringAndSize(string, size)
    if obj is None:
        raise MemoryError('PyByteArray_FromStringAndSize failed')
    return obj


cdef _return_output(out, ssize_t size, ssize_t used, bint outgiven):
    """Return a memoryview, slice, or copy of 'out' of length 'used'.

    If 'used >= size', return 'out' unchanged.

    If 'out' was provided by the user, return a memoryview or a ndarray slice
    of length 'used' of 'out'. The memoryview will be read-only if 'out' is
    bytes.

    Else if 'out' is a bytesarray, return 'out' resized to length 'used'.

    Else, 'out' is bytes, return a copy of 'out' of length 'used'.

    """
    if used >= size:
        return out
    if not outgiven:
        if PyByteArray_Check(out):
            # resize bytearray
            if PyByteArray_Resize(out, used) < 0:
                raise MemoryError('PyByteArray_Resize failed')
        else:
            # TODO: avoid copy bytes
            # _PyBytes_Resize is not part of the stable ABI.
            # One could get access to _PyBytes_Resize from the versioned DLL
            # via dlsym/GetProcAddress. However, on free-threaded builds
            # the refcount bias also makes _PyBytes_Resize non-callable
            # from Cython cdef context.
            # Revisit when PyBytesWriter becomes available in CPython 3.15.
            # In the meantime, users can avoid copy by using bytearray
            out = PyBytes_FromStringAndSize(PyBytes_AsString(out), used)
    elif numpy.PyArray_Check(out):
        # slice of user numpy array
        out = out[:used]
    else:
        # memoryview of user bytes or bytearray
        out = memoryview(out)[:used]
    return out


cdef numpy.ndarray _create_array(
    out,
    tuple shape,
    dtype,
    tuple strides=None,
    bint zero=False,
    bint contig=True
):
    """Return numpy array of shape and dtype from output argument."""
    cdef:
        ssize_t dstsize

    if out is None or isinstance(out, int):  # numbers.Integral
        if zero:
            return numpy.zeros(shape, dtype)
        return numpy.empty(shape, dtype)

    if isinstance(out, numpy.ndarray):
        if out.itemsize != numpy.dtype(dtype).itemsize:
            raise ValueError(f'invalid {out.dtype=}, {dtype=}')
        if out.shape != shape:
            # check that shapes match, allowing length-one dimensions.
            # arrays with incompatible strides will fail to reshape
            if (
                tuple(int(i) for i in out.shape if i != 1)
                != tuple(int(i) for i in shape if i != 1)
            ):
                raise ValueError(f'invalid {out.shape=!r}, {shape=!r}')
            out = out.reshape(shape)
        if strides is not None:
            if len(strides) != len(out.strides):
                raise ValueError(f'{out.strides=!r} do not match {strides=!r}')
            for i, j in zip(strides, out.strides):
                if i is not None and i != j:
                    raise ValueError(
                        f'invalid {out.strides=!r}, {strides=!r}'
                    )
        elif contig and not numpy.PyArray_ISCONTIGUOUS(out):
            raise ValueError(
                f'output is not contiguous {out.shape=!r}, {out.strides=!r}'
            )
    else:
        dstsize = 1
        for i in shape:
            dstsize *= i
        out = numpy.frombuffer(out, dtype, dstsize).reshape(shape)

    if zero:
        out.fill(0)
    return out


cdef const uint8_t[::1] _readable_input(data):
    """Return readable, contiguous 1D bytes memoryview of data.

    Make copy if necessary.

    """
    cdef:
        const uint8_t[::1] src

    try:
        src = data
    except Exception:
        # not contiguous
        try:
            # numpy array
            # src = numpy.ravel(data, 'K').view(numpy.uint8)
            src = data.reshape(-1).view(numpy.uint8)
        except Exception:
            # buffer protocol
            src = data.tobytes()
    return src


cdef const uint8_t[::1] _writable_input(data):
    """Return writable, contiguous 1D bytes memoryview to data.

    Make copy if necessary.

    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] writable

    if isinstance(data, bytes):
        src = data
        return src

    try:
        writable = data
        src = writable
    except Exception:
        # not writable or not contiguous
        if hasattr(data, 'read'):
            # mmap
            src = data.read()
        else:
            try:
                # numpy array
                writable = data.reshape(-1).view(numpy.uint8)
                src = writable
            except Exception:
                # buffer protocol
                src = data.tobytes()

    return src


cdef const uint8_t[::1] _inplace_input(data):
    """Return writable, contiguous 1D bytes memoryview to data.

    Fail if input is not writable and contiguous.

    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] writable

    if isinstance(data, bytes):
        raise TypeError("'bytes' object does not support item assignment")

    try:
        writable = data
        src = writable
    except Exception:
        # not writable or not contiguous
        view = memoryview(data)
        if not view.contiguous:
            raise ValueError('input data is not writable and contiguous')
        writable = view.cast('B')
        src = writable

    return src


cdef tuple _squeeze_shape(tuple shape, ssize_t ndim):
    """Return shape with leading length-one dimensions removed."""
    cdef:
        ssize_t i = 0
        ssize_t length = len(shape)

    if length <= ndim:
        return shape
    length -= ndim
    for size in shape:
        if size != 1 or i == length:
            break
        i += 1
    return shape[i:]


cdef _default_value(value, default, smallest, largest):
    """Return default value or value in range [smallest, largest]."""
    if value is None:
        return default
    if largest is not None and value >= largest:
        return largest
    if smallest is not None and value <= smallest:
        return smallest
    return value


cdef _enum_value(value, enum_class, default=None):
    """Return enum member for value, or default if value is None."""
    if value is None:
        return default
    if isinstance(value, str):
        return enum_class[value.upper()]
    return value


cdef uint8_t _default_threads(numthreads):
    """Return default number of threads or value in range [0, 32]."""
    if numthreads is None:
        return 1
    if numthreads <= 0:
        return 0
    if numthreads >= 32:
        return 32
    return <uint8_t> numthreads


def _log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger('imagecodecs').warning(msg, *args, **kwargs)


# Image layout ################################################################


class Photometric(enum.IntEnum):
    """Photometric interpretation."""

    UNSPECIFIED = IC_PHOTO_UNSPECIFIED
    GRAY = IC_PHOTO_GRAY
    RGB = IC_PHOTO_RGB
    PALETTE = IC_PHOTO_PALETTE
    CMYK = IC_PHOTO_CMYK
    YCBCR = IC_PHOTO_YCBCR
    CIELAB = IC_PHOTO_CIELAB
    ICCLAB = IC_PHOTO_ICCLAB


class SampleFormat(enum.IntEnum):
    """Sample format."""

    UINT = IC_SF_UINT
    SINT = IC_SF_SINT
    FLOAT = IC_SF_FLOAT
    COMPLEX = IC_SF_COMPLEX
    BOOL = IC_SF_BOOL


class ExtraSample(enum.IntEnum):
    """Extra sample type."""

    UNSPECIFIED = IC_EXTRA_UNSPECIFIED
    ASSOCALPHA = IC_EXTRA_ASSOCALPHA
    UNASSALPHA = IC_EXTRA_UNASSALPHA


class IC(enum.IntFlag):
    """Image codec capability flags."""

    GRAY = IC_GRAY
    RGB = IC_RGB
    PALETTE = IC_PALETTE
    CMYK = IC_CMYK
    YCBCR = IC_YCBCR
    CIELAB = IC_CIELAB
    ICCLAB = IC_ICCLAB
    FRAMES = IC_FRAMES
    PLANAR = IC_PLANAR
    DEPTH = IC_DEPTH
    ALPHA = IC_ALPHA
    EXTRA = IC_EXTRA
    UINT = IC_UINT
    SINT = IC_SINT
    FLOAT = IC_FLOAT
    COMPLEX = IC_COMPLEX
    BOOL = IC_BOOL
    SZ1 = IC_SZ1
    SZ2 = IC_SZ2
    SZ4 = IC_SZ4
    SZ8 = IC_SZ8
    SZ16 = IC_SZ16
    BPS = IC_BPS


@dataclasses.dataclass(slots=True, frozen=True)
class ImageLayout:
    """Resolved image layout for codec encode/decode."""

    frames: int
    # Number of frames (product of all leading batch dimensions).

    depth: int
    # Number of slices along the volumetric axis; 1 if 2-D.

    height: int
    # Number of rows; 0 for 1-D arrays.

    width: int
    # Number of columns.

    samples: int
    # Total samples per pixel (color channels + alpha + extra).

    sampleformat: SampleFormat
    # Numeric sample format (uint, sint, float, complex, bool).

    bitspersample: int
    # Effective bits per sample; may differ from itemsize*8 if packed.

    itemsize: int
    # dtype size in bytes.

    photometric: Photometric
    # Photometric interpretation (GRAY, RGB, CMYK, ...).

    planar: bool
    # True if samples axis precedes spatial axes.

    extraindex: int
    # 0-based sample index of first extra channel; 0 means none.

    extratype: ExtraSample
    # Type of first extra channel (UNSPECIFIED, ASSOCALPHA, UNASSALPHA).

    extracount: int
    # Number of extra samples beyond photometric channels.


cdef int _photo_samples(int photometric) noexcept nogil:
    """Return number of color channels for photometric."""
    if (
        photometric == 2  # RGB
        or photometric == 6  # YCBCR
        or photometric == 8  # CIELAB
        or photometric == 9  # ICCLAB
    ):
        return 3
    if photometric == 5:
        return 4  # CMYK
    return 1  # GRAY, PALETTE, UNSPECIFIED


cdef imagecaps_t _photo_cap(int photometric) noexcept nogil:
    """Return capability flag bit for photometric."""
    if photometric == 1:  # GRAY
        return <imagecaps_t> (1 << 0)
    if photometric == 2:  # RGB
        return <imagecaps_t> (1 << 1)
    if photometric == 3:  # PALETTE
        return <imagecaps_t> (1 << 2)
    if photometric == 5:  # CMYK
        return <imagecaps_t> (1 << 3)
    if photometric == 6:  # YCBCR
        return <imagecaps_t> (1 << 4)
    if photometric == 8:  # CIELAB
        return <imagecaps_t> (1 << 5)
    if photometric == 9:  # ICCLAB
        return <imagecaps_t> (1 << 6)
    return <imagecaps_t> 0


cdef int _image_layout(
    imagecaps_t caps,
    int ndim,
    const numpy.npy_intp* shape,
    numpy.dtype dtype,
    object photometric,
    object bitspersample,
    object planar,
    object frames,
    object volumetric,
    object extrasample,
    imagelayout_t* layout,
) except -1:
    """Resolve imagelayout_t from array shape, dtype, and encoder hints.

    Write resolved fields into *layout. Return 0 on success.
    caps == 0 disables capability validation (for testing).

    Axis order conventions:
        Contig:  [frames...] [depth] height width [samples]
        Planar:  [frames...] [samples] [depth] height width

    """
    cdef:
        imagecaps_t sz_cap = IC_SZ1
        imagecaps_t sf_cap = IC_UINT
        int sf = IC_SF_UINT
        int itemsize_ = 1
        int bitspersample_ = 8
        int photo_samples_ = 1
        int photo_hint = IC_PHOTO_UNSPECIFIED
        int photo_ = IC_PHOTO_UNSPECIFIED
        int extra_type_ = IC_EXTRA_UNSPECIFIED
        int extra = 0
        int core_axes
        int min_channel_ndim
        bint planar_ = False
        bint volumetric_ = False
        bint has_channels = False
        uint8_t extra_index_ = 0
        ssize_t frames_ = 1
        ssize_t depth_ = 1
        ssize_t height_ = 0
        ssize_t width_ = 0
        ssize_t samples_ = 1
        ssize_t i
        str s

    # dtype -> sampleformat + itemsize
    kind = dtype.kind
    if kind == 'u':
        sf = IC_SF_UINT
        sf_cap = IC_UINT
    elif kind == 'i':
        sf = IC_SF_SINT
        sf_cap = IC_SINT
    elif kind == 'f':
        sf = IC_SF_FLOAT
        sf_cap = IC_FLOAT
    elif kind == 'c':
        sf = IC_SF_COMPLEX
        sf_cap = IC_COMPLEX
    elif kind == 'b':
        sf = IC_SF_BOOL
        sf_cap = IC_BOOL
    else:
        raise ValueError(f'unsupported dtype {dtype!r}')

    itemsize_ = dtype.itemsize
    if itemsize_ == 1:
        sz_cap = IC_SZ1
    elif itemsize_ == 2:
        sz_cap = IC_SZ2
    elif itemsize_ == 4:
        sz_cap = IC_SZ4
    elif itemsize_ == 8:
        sz_cap = IC_SZ8
    elif itemsize_ == 16:
        sz_cap = IC_SZ16
    else:
        raise ValueError(f'unsupported dtype itemsize {itemsize_}')

    # bitspersample hint
    bitspersample_ = itemsize_ * 8
    if bitspersample is not None:
        bitspersample_ = int(bitspersample)
        if bitspersample_ != itemsize_ * 8:
            if caps != 0 and not (caps & IC_BPS):
                raise ValueError(
                    f'bitspersample={bitspersample_} differs from dtype '
                    f'({itemsize_ * 8} bits); codec does not accept '
                    f'custom bitspersample'
                )

    # planar hint
    if planar is None:
        planar_ = False
    elif planar is True or planar is False:
        planar_ = <bint> planar
    else:
        s = str(planar).upper()
        if s == 'SEPARATE':
            planar_ = True
        elif s == 'CONTIG':
            planar_ = False
        else:
            raise ValueError(f'planar={planar!r} not recognised')

    if planar_ and ndim < 3:
        raise ValueError('planar requires at least 3 dimensions')

    if planar_ and caps != 0 and not (caps & IC_PLANAR):
        raise ValueError('planar layout not supported by codec')

    # volumetric hint
    volumetric_ = volumetric is True

    if volumetric_ and ndim < 3:
        raise ValueError('volumetric requires at least 3 dimensions')

    if volumetric_ and caps != 0 and not (caps & IC_DEPTH):
        raise ValueError('volumetric/depth not supported by codec')

    # photometric hint
    if photometric is None:
        photo_hint = IC_PHOTO_UNSPECIFIED
    elif isinstance(photometric, int):
        photo_hint = photometric
    else:
        s = str(photometric).upper()
        if s not in _IC_PHOTO_NAMES:
            raise ValueError(
                f'photometric={photometric!r} not recognised'
            )
        photo_hint = _IC_PHOTO_NAMES[s]

    # extrasample hint
    if extrasample is None:
        extra_type_ = IC_EXTRA_UNSPECIFIED
    elif isinstance(extrasample, int):
        extra_type_ = extrasample
    else:
        s = str(extrasample).upper()
        if s not in _IC_EXTRA_NAMES:
            raise ValueError(
                f'extrasample={extrasample!r} not recognised'
            )
        extra_type_ = _IC_EXTRA_NAMES[s]

    # shape dispatch
    if ndim == 0:
        raise ValueError('cannot encode 0-dimensional array')

    elif ndim == 1:
        width_ = shape[0]
        if photo_hint != IC_PHOTO_UNSPECIFIED:
            photo_ = photo_hint
        else:
            photo_ = IC_PHOTO_GRAY

    elif ndim == 2:
        height_ = shape[0]
        width_ = shape[1]
        if photo_hint != IC_PHOTO_UNSPECIFIED:
            photo_ = photo_hint
        else:
            photo_ = IC_PHOTO_GRAY

    else:
        # decide does a channel axis exist?
        #
        # min_channel_ndim: minimum ndim to have a channel axis
        #   non-volumetric: 3  (H, W, S) or (S, H, W)
        #   volumetric:     4  (D, H, W, S) or (S, D, H, W)
        #
        # planar RGB is never auto-detected

        min_channel_ndim = 3 + (1 if volumetric_ else 0)

        if planar_:
            has_channels = True
        elif ndim > min_channel_ndim:
            if (
                photo_hint != IC_PHOTO_UNSPECIFIED
                and _photo_samples(photo_hint) == 1
                and extrasample is None
                and shape[ndim - 1] != 1
            ):
                # single-sample photometric and non-trivial trailing dim:
                # extra leading dims are all frames, no channel axis
                has_channels = False
            else:
                has_channels = True
        elif ndim < min_channel_ndim:
            has_channels = False
        # ndim == min_channel_ndim, resolve with hints
        elif (
            photo_hint != IC_PHOTO_UNSPECIFIED
            and _photo_samples(photo_hint) == 1
            and extrasample is None
        ):
            # single-sample photometric (gray, palette) with no extrasample
            # hint: all extra dims are frames, never channels
            has_channels = False
        elif frames is True:
            has_channels = False
        elif frames is False:
            has_channels = True
        elif photo_hint != IC_PHOTO_UNSPECIFIED or extrasample is not None:
            has_channels = True
        elif caps != 0 and not (caps & IC_FRAMES):
            has_channels = True
        else:
            # heuristic: trailing dim <= 4 -> likely channel samples
            has_channels = (shape[ndim - 1] <= 4)

        # unpack dimensions
        core_axes = (
            2
            + (1 if has_channels else 0)
            + (1 if volumetric_ else 0)
        )

        if ndim < core_axes:
            raise ValueError(
                f'ndim={ndim} too small for '
                f'{"volumetric " if volumetric_ else ""}'
                f'{"planar " if planar_ else ""}'
                f'image with {"" if has_channels else "no "}channels'
            )

        if has_channels and not planar_:
            # contig: ... [D] H W S
            samples_ = shape[ndim - 1]
            width_ = shape[ndim - 2]
            height_ = shape[ndim - 3]
            if volumetric_:
                depth_ = shape[ndim - 4]
        elif has_channels and planar_:
            # planar: ... S [D] H W
            width_ = shape[ndim - 1]
            height_ = shape[ndim - 2]
            if volumetric_:
                depth_ = shape[ndim - 3]
                samples_ = shape[ndim - 4]
            else:
                samples_ = shape[ndim - 3]
        elif volumetric_:
            # volumetric, no channels: ... D H W
            width_ = shape[ndim - 1]
            height_ = shape[ndim - 2]
            depth_ = shape[ndim - 3]
        else:
            # no channels, no depth: ... H W
            width_ = shape[ndim - 1]
            height_ = shape[ndim - 2]

        for i in range(ndim - core_axes):
            frames_ *= shape[i]

        # infer photometric from sample count
        # auto-detect RGB only for unambiguous cases:
        #   uint8/uint16 or float16/32/64, and 3 or 4 samples
        #   (or >4 samples when extrasample hint is given)
        if photo_hint != IC_PHOTO_UNSPECIFIED:
            photo_ = photo_hint
        elif (
            (
                samples_ == 3
                or samples_ == 4
                or (samples_ > 4 and extrasample is not None)
            )
            and (
                (sf == IC_SF_UINT and itemsize_ <= 2)
                or (sf == IC_SF_FLOAT and itemsize_ <= 8)
            )
        ):
            photo_ = IC_PHOTO_RGB
        else:
            photo_ = IC_PHOTO_GRAY

    # alpha / extra accounting
    photo_samples_ = _photo_samples(photo_)

    if samples_ < photo_samples_:
        raise ValueError(
            f'samples={samples_} less than expected '
            f'{photo_samples_} for photometric {photo_}'
        )

    extra = <int> (samples_ - photo_samples_)

    if extra > 0:
        if extra_type_ != IC_EXTRA_UNSPECIFIED:
            # explicit extrasample hint -> first extra is alpha
            extra_index_ = <uint8_t>photo_samples_
        elif extra == 1:
            # single unspecified extra -> assume unassociated alpha
            extra_index_ = <uint8_t>photo_samples_
            extra_type_ = IC_EXTRA_UNASSALPHA

    # validate against caps
    if caps != 0:
        if not (caps & sf_cap):
            raise ValueError(
                f'{dtype!r} sample format not supported by codec'
            )
        if not (caps & sz_cap):
            raise ValueError(
                f'{dtype!r} item size not supported by codec'
            )
        if photo_ != IC_PHOTO_UNSPECIFIED and not (caps & _photo_cap(photo_)):
            raise ValueError(
                f'photometric {photo_} not supported by codec'
            )
        if frames_ > 1 and not (caps & IC_FRAMES):
            raise ValueError(
                f'frames={frames_} not supported by codec'
            )
        if extra_index_ and not (caps & IC_ALPHA):
            raise ValueError(
                'alpha channel not supported by codec'
            )
        if extra > 1 and not (caps & IC_EXTRA):
            raise ValueError(
                f'{extra} extra samples not supported by codec'
            )

    # write output
    layout.frames = frames_
    layout.depth = depth_
    layout.height = height_
    layout.width = width_
    layout.samples = samples_
    layout.sampleformat = sf
    layout.bitspersample = bitspersample_
    layout.itemsize = itemsize_
    layout.photometric = photo_
    layout.planar = planar_
    layout.extraindex = extra_index_
    layout.extratype = <uint8_t>extra_type_
    layout.extracount = extra

    return 0


def image_layout(
    tuple shape,
    dtype,
    imagecaps_t caps=0,
    photometric=None,
    bitspersample=None,
    planar=None,
    frames=None,
    volumetric=None,
    extrasample=None,
):
    """Return resolved image layout as ImageLayout instance."""
    cdef:
        imagelayout_t layout
        numpy.npy_intp[32] shape_buf
        int ndim_

    ndim_ = <int> len(shape)
    for i in range(ndim_):
        shape_buf[i] = shape[i]
    dtype = numpy.dtype(dtype)
    _image_layout(
        caps,
        ndim_,
        shape_buf,
        dtype,
        photometric,
        bitspersample,
        planar,
        frames,
        volumetric,
        extrasample,
        &layout,
    )
    return ImageLayout(
        frames=layout.frames,
        depth=layout.depth,
        height=layout.height,
        width=layout.width,
        samples=layout.samples,
        sampleformat=SampleFormat(layout.sampleformat),
        bitspersample=layout.bitspersample,
        itemsize=layout.itemsize,
        photometric=Photometric(layout.photometric),
        planar=bool(layout.planar),
        extraindex=layout.extraindex,
        extratype=ExtraSample(layout.extratype),
        extracount=layout.extracount,
    )


# string -> int lookup (keys are upper-cased)
_IC_PHOTO_NAMES = {
    'GRAY': IC_PHOTO_GRAY,
    'GRAYSCALE': IC_PHOTO_GRAY,
    'MINISBLACK': IC_PHOTO_GRAY,
    'MINISWHITE': IC_PHOTO_GRAY,
    'GRAYA': IC_PHOTO_GRAY,
    'RGB': IC_PHOTO_RGB,
    'RGBA': IC_PHOTO_RGB,
    'PALETTE': IC_PHOTO_PALETTE,
    'CMYK': IC_PHOTO_CMYK,
    'CMYKA': IC_PHOTO_CMYK,
    'SEPARATED': IC_PHOTO_CMYK,
    'YCBCR': IC_PHOTO_YCBCR,
    'LAB': IC_PHOTO_CIELAB,
    'CIELAB': IC_PHOTO_CIELAB,
    'ICCLAB': IC_PHOTO_ICCLAB,
}

_IC_EXTRA_NAMES = {
    'UNSPECIFIED': IC_EXTRA_UNSPECIFIED,
    'ASSOCALPHA': IC_EXTRA_ASSOCALPHA,
    'ASSOCIATED': IC_EXTRA_ASSOCALPHA,
    'PREMULTIPLIED': IC_EXTRA_ASSOCALPHA,
    'UNASSALPHA': IC_EXTRA_UNASSALPHA,
    'UNASSOCIATED': IC_EXTRA_UNASSALPHA,
    'STRAIGHT': IC_EXTRA_UNASSALPHA,
}
