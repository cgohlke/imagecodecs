# imagecodecs/_tiff.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2019-2026, Christoph Gohlke
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
# SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""TIFF (Tagged Image File Format) codec for the imagecodecs package."""

include '_shared.pxi'

import cython

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from imcd cimport imcd_packints_decode, imcd_packints_encode
from libc.stdio cimport SEEK_CUR, SEEK_END, SEEK_SET
from libtiff cimport *


cdef extern from '<stdio.h>':
    int vsnprintf(char* s, size_t n, const char* format, va_list arg) nogil


cdef:
    const tdir_t TIFF_MAX_DIR_COUNT = 1048576  # private def in tiffiop.h


ctypedef struct page_t:
    ssize_t planes  # 1 for CONTIG, samplesperpixel for SEPARATE
    ssize_t depth  # imagedepth, usually 1
    ssize_t length  # imagelength (rows)
    ssize_t width  # imagewidth  (columns)
    ssize_t samples  # samplesperpixel for CONTIG, 1 for SEPARATE
    ssize_t truesamples  # non-zero when asrgb forces RGBA
    ssize_t itemsize  # output type itemsize in bytes; bitspersample on failure
    int bitspersample  # non-standard bps for packints unpacking; else 0
    int compression  # compression scheme
    int status  # 1 on success; unsupported SAMPLEFORMAT_* value on failure
    bint asrgb  # force RGBA using TIFFReadRGBAImageOriented
    bint istiled
    char[2] dtype


class _TIFF:
    """TIFF codec constants."""

    available = True

    class VERSION(enum.IntEnum):
        """TIFF codec file types."""

        CLASSIC = TIFF_VERSION_CLASSIC
        BIG = TIFF_VERSION_BIG

    class ENDIAN(enum.IntEnum):
        """TIFF codec endian values."""

        BIG = TIFF_BIGENDIAN
        LITTLE = TIFF_LITTLEENDIAN

    class COMPRESSION(enum.IntEnum):
        """TIFF codec compression schemes."""

        NONE = COMPRESSION_NONE
        CCITTRLE = COMPRESSION_CCITTRLE
        CCITTFAX3 = COMPRESSION_CCITTFAX3
        CCITTFAX4 = COMPRESSION_CCITTFAX4
        LZW = COMPRESSION_LZW
        JPEG = COMPRESSION_JPEG
        PACKBITS = COMPRESSION_PACKBITS
        DEFLATE = COMPRESSION_DEFLATE  # maps to COMPRESSION_ADOBE_DEFLATE
        ADOBE_DEFLATE = COMPRESSION_ADOBE_DEFLATE
        LZMA = COMPRESSION_LZMA
        ZSTD = COMPRESSION_ZSTD
        WEBP = COMPRESSION_WEBP
        LERC = COMPRESSION_LERC
        PIXARLOG = COMPRESSION_PIXARLOG
        # JXL = COMPRESSION_JXL

    class PHOTOMETRIC(enum.IntEnum):
        """TIFF codec photometric interpretations."""

        MINISWHITE = PHOTOMETRIC_MINISWHITE
        MINISBLACK = PHOTOMETRIC_MINISBLACK
        RGB = PHOTOMETRIC_RGB
        PALETTE = PHOTOMETRIC_PALETTE
        MASK = PHOTOMETRIC_MASK
        SEPARATED = PHOTOMETRIC_SEPARATED
        YCBCR = PHOTOMETRIC_YCBCR

    class PLANARCONFIG(enum.IntEnum):
        """TIFF codec planar configurations."""

        CONTIG = PLANARCONFIG_CONTIG
        SEPARATE = PLANARCONFIG_SEPARATE

    class PREDICTOR(enum.IntEnum):
        """TIFF codec predictor schemes."""

        NONE = PREDICTOR_NONE
        HORIZONTAL = PREDICTOR_HORIZONTAL
        FLOATINGPOINT = PREDICTOR_FLOATINGPOINT

    class EXTRASAMPLE(enum.IntEnum):
        """TIFF codec extrasample types."""

        UNSPECIFIED = EXTRASAMPLE_UNSPECIFIED
        ASSOCALPHA = EXTRASAMPLE_ASSOCALPHA
        UNASSALPHA = EXTRASAMPLE_UNASSALPHA

    class FILETYPE(enum.IntFlag):
        """TIFF subfile types."""

        REDUCEDIMAGE = FILETYPE_REDUCEDIMAGE
        PAGE = FILETYPE_PAGE
        MASK = FILETYPE_MASK

    class RESUNIT(enum.IntEnum):
        """TIFF codec resolution unit types."""

        NONE = RESUNIT_NONE
        INCH = RESUNIT_INCH
        CENTIMETER = RESUNIT_CENTIMETER

    class SUBCODEC(enum.IntEnum):
        """TIFF codec LERC additional compression schemes."""

        NONE = LERC_ADD_COMPRESSION_NONE
        DEFLATE = LERC_ADD_COMPRESSION_DEFLATE
        ZSTD = LERC_ADD_COMPRESSION_ZSTD


class TiffError(RuntimeError):
    """TIFF codec exceptions."""

    def __init__(self, arg=None, msg=''):
        """Initialize Exception from string or memtif capsule."""
        cdef:
            memtif_t* memtif

        if arg is None:
            pass
        elif isinstance(arg, str):
            msg += arg
        else:
            memtif = <memtif_t*> PyCapsule_GetPointer(arg, NULL)
            msg += memtif.errmsg.decode()
        super().__init__(msg)


@cython.wraparound(True)
def tiff_version():
    """Return libtiff library version string."""
    cdef:
        const char* ver = TIFFGetVersion()

    return 'libtiff ' + ver.decode().split('\n')[0].split()[-1]


def tiff_check(const uint8_t[::1] data, /):
    """Return whether data is TIFF encoded image or None if unknown."""
    cdef:
        bytes sig = bytes(data[:4])

    return (
        # Classic
        sig == b'II\x2A\x00'
        or sig == b'MM\x00\x2A'
        # BigTiff
        or sig == b'II\x2B\x00'
        or sig == b'MM\x00\x2B'
        # MDI
        or sig == b'EP\x2A\x00'
        or sig == b'PE\x00\x2A'
    )


def tiff_encode(
    data,
    /,
    level=None,  # -1 uses libtiff compression defaults
    *,
    bigtiff=None,
    byteorder=None,
    subfiletype=None,
    photometric=None,
    planarconfig=None,
    extrasample=None,
    # volumetric=False,
    tile=None,
    rowsperstrip=None,
    bitspersample=None,
    compression=None,
    subcodec=None,  # for lerc
    predictor=None,
    colormap=None,
    iccprofile=None,
    resolution=None,
    resolutionunit=None,
    description=None,
    datetime=None,
    software=None,
    verbose=None,
    appendto=None,
    out=None,
):
    """Return TIFF encoded image."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        numpy.ndarray pal
        const uint8_t[::1] buf  # must be const to write to bytes
        uint8_t* srcptr = <uint8_t*> src.data
        uint8_t* tile_ = NULL
        uint8_t* tile_packed_ = NULL
        uint8_t* rowbuf = NULL
        uint16_t* palptr = NULL
        TIFF* tif = NULL
        TIFFOpenOptions* openoptions = NULL
        memtif_t* memtif = NULL
        uint32_t planarconfig_ = PLANARCONFIG_CONTIG
        uint32_t photometric_ = PHOTOMETRIC_MINISBLACK
        uint32_t compression_ = COMPRESSION_NONE
        uint32_t subcodec_ = LERC_ADD_COMPRESSION_NONE
        uint32_t sampleformat_ = SAMPLEFORMAT_UINT
        uint32_t predictor_ = PREDICTOR_NONE
        uint32_t resolutionunit_ = RESUNIT_NONE
        uint32_t pixarlogdatafmt_ = PIXARLOGDATAFMT_8BIT
        uint16_t extrasample_ = EXTRASAMPLE_UNSPECIFIED
        uint16_t* extrasamples_ = NULL
        int32_t level_ = -1
        uint32_t subfiletype_ = 0
        uint32_t rowsperstrip_ = 0
        uint16_t samplesperpixel_ = 1
        uint16_t bitspersample_ = src.dtype.itemsize * 8
        uint16_t subsample_ = 1
        ssize_t itemsize = src.dtype.itemsize
        ssize_t ndim = src.ndim
        ssize_t dstsize, incsize, rowsize, framesize, tilesize, memtif_len, i
        ssize_t packedrowsize = 0
        ssize_t tile_packedrowsize = 0
        int bps_encode = 0
        ssize_t planes = 1  # planar samples
        ssize_t length = 1
        ssize_t samples = 1  # contig samples
        ssize_t extrasamples = 0
        ssize_t photometric_samples = 1
        ssize_t palsize = 0
        ssize_t append_size = 0
        uint32_t iccprofile_size = 0
        uint32_t tile_width = 0
        uint32_t tile_length = 0
        double maxzerror = 0.0
        float xresolution = 1.0
        float yresolution = 1.0
        bytes mode
        char* mode_ = NULL
        char* description_ = NULL
        char* software_ = NULL
        char* datetime_ = NULL
        char* iccprofile_ = NULL
        int ret
        bint bigendian = False
        imagelayout_t layout

    if src.dtype.kind == 'u':
        sampleformat_ = SAMPLEFORMAT_UINT
    elif src.dtype.kind == 'f':
        sampleformat_ = SAMPLEFORMAT_IEEEFP
    elif src.dtype.kind == 'i':
        sampleformat_ = SAMPLEFORMAT_INT
    elif src.dtype.kind == 'c':
        sampleformat_ = SAMPLEFORMAT_COMPLEXIEEEFP
    elif src.dtype.kind == 'b':
        sampleformat_ = SAMPLEFORMAT_UINT
    else:
        raise ValueError(f'{src.dtype.kind=!r} not supported')

    if appendto is None or len(appendto) == 0:
        if bigtiff is None:
            mode = b'w8' if src.nbytes > INT32_MAX else b'w4'
        elif bigtiff:
            mode = b'w8'
        else:
            mode = b'w4'

        if byteorder is None or byteorder == '=':
            pass
        elif byteorder in {TIFF_BIGENDIAN, '>', 'big'}:
            mode += b'b'
            bigendian = True
        elif byteorder in {TIFF_LITTLEENDIAN, '<', 'little'}:
            mode += b'l'
        else:
            raise ValueError(f'{byteorder=!r} not supported')
    else:
        mode = b'a'
        append_size = len(appendto)
    mode_ = mode

    if subfiletype is not None:
        subfiletype_ = _enum_value(subfiletype, _TIFF.FILETYPE)

    if compression is None:
        if level is None:
            compression_ = COMPRESSION_NONE
        else:
            compression_ = COMPRESSION_ADOBE_DEFLATE
            level_ = _default_value(level, 6, 0, 12)
    else:
        compression_ = _enum_value(compression, _TIFF.COMPRESSION)
        if compression_ == COMPRESSION_DEFLATE:
            compression_ = COMPRESSION_ADOBE_DEFLATE  # normalize alias
        # not elif: catches alias
        if compression_ == COMPRESSION_ADOBE_DEFLATE:
            level_ = _default_value(level, 6, -1, 12)
        elif compression_ == COMPRESSION_ZSTD:
            level_ = _default_value(level, 3, -1, 22)  # ZSTD_CLEVEL_DEFAULT=3
        elif compression_ == COMPRESSION_JPEG:
            level_ = _default_value(level, 95, -1, 100)
        elif compression_ == COMPRESSION_WEBP:
            level_ = _default_value(level, 100, -1, 100)
        elif compression_ == COMPRESSION_LZMA:
            level_ = _default_value(level, 6, -1, 9)
        elif compression_ == COMPRESSION_LERC:
            maxzerror = _default_value(level, 0.0, 0.0, None)
            subcodec_ = _enum_value(
                subcodec, _TIFF.SUBCODEC, LERC_ADD_COMPRESSION_NONE
            )
            if subcodec_ == LERC_ADD_COMPRESSION_ZSTD:
                level_ = 3  # ZSTD_CLEVEL_DEFAULT
            elif subcodec_ == LERC_ADD_COMPRESSION_DEFLATE:
                level_ = 6  # Z_DEFAULT_COMPRESSION
        elif compression_ == COMPRESSION_PIXARLOG:
            level_ = _default_value(level, 6, -1, 12)
        # elif compression_ == COMPRESSION_JXL:
        #     pass

    if predictor is None:
        pass
    elif isinstance(predictor, bool):
        if predictor:
            if sampleformat_ in {SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT}:
                predictor_ = PREDICTOR_HORIZONTAL
            else:
                predictor_ = PREDICTOR_FLOATINGPOINT
    else:
        predictor_ = _enum_value(predictor, _TIFF.PREDICTOR)

    if resolution is not None:
        xresolution, yresolution = resolution
        resolutionunit_ = RESUNIT_INCH

    resolutionunit_ = _enum_value(
        resolutionunit, _TIFF.RESUNIT, resolutionunit_
    )

    planarconfig_ = _enum_value(
        planarconfig, _TIFF.PLANARCONFIG, planarconfig_
    )

    extrasample_ = _enum_value(extrasample, _TIFF.EXTRASAMPLE, extrasample_)

    if photometric is None:
        if colormap is not None:
            photometric_ = PHOTOMETRIC_PALETTE
    else:
        photometric_ = _enum_value(photometric, _TIFF.PHOTOMETRIC)

    if photometric_ == PHOTOMETRIC_PALETTE:
        if extrasample is not None:
            raise ValueError('palette image with extrasamples not supported')
        if colormap is None:
            raise ValueError('palette image requires colormap')
        if src.dtype.kind != 'u':
            raise ValueError('palette image requires unsigned image')
        pal = _array_input(colormap)
        if pal.dtype.kind != 'u' or pal.dtype.itemsize != 2:
            raise ValueError(f'invalid colormap dtype={pal.dtype}')
        if (
            pal.ndim != 2
            or pal.shape[0] != 3
            or pal.shape[1] != 2**bitspersample_
        ):
            raise ValueError('invalid colormap shape')
        palptr = <uint16_t*> pal.data
        palsize = 2**bitspersample_

    if iccprofile is not None:
        iccprofile_ = iccprofile
        iccprofile_size = <uint32_t> len(iccprofile)

    if description is not None:
        if not isinstance(description, bytes):
            description = description.encode('ascii')
        description_ = description

    if software is not None:
        software = software.encode('ascii')
        software_ = software

    if datetime is not None:
        # if len(datetime) != 19:
        #     raise ValueError('invalid datetime != YYYY:MM:DD HH:MM:SS')
        datetime = datetime.encode('ascii')
        datetime_ = datetime

    # while ndim > 1 and src.shape[ndim - 1] == 1:
    #     # remove trailing length-1 dimensions
    #     ndim -= 1

    # pre-compute photometric hint for _image_layout
    # _image_layout handles string/int mapping for explicit photometric,
    # but TIFF's RGB auto-detect is stricter (uint + bps<=16 only)
    if photometric is not None:
        photo_for_layout = photometric
    elif colormap is not None:
        photo_for_layout = IC_PHOTO_PALETTE
    elif (
        sampleformat_ == SAMPLEFORMAT_UINT
        and bitspersample_ <= 16
        and ndim >= 3
        and (
            (
                src.shape[ndim - 1] in {3, 4}
                or (extrasample is not None and src.shape[ndim - 1] > 4)
            )
            or (
                planarconfig_ == PLANARCONFIG_SEPARATE
                and (
                    src.shape[ndim - 3] in {3, 4}
                    or (extrasample is not None and src.shape[ndim - 3] > 4)
                )
            )
        )
    ):
        photo_for_layout = IC_PHOTO_RGB
    else:
        photo_for_layout = IC_PHOTO_GRAY

    _image_layout(
        IC_UINT
        | IC_SINT
        | IC_FLOAT
        | IC_COMPLEX
        | IC_BOOL
        | IC_SZ1
        | IC_SZ2
        | IC_SZ4
        | IC_SZ8
        | IC_SZ16
        | IC_GRAY
        | IC_RGB
        | IC_PALETTE
        | IC_CMYK
        | IC_YCBCR
        | IC_FRAMES
        | IC_PLANAR
        | IC_ALPHA
        | IC_EXTRA
        | IC_BPS
        | IC_BE,
        src.ndim,
        src.shape,
        src.dtype,
        photo_for_layout,
        bitspersample,
        planarconfig_ == PLANARCONFIG_SEPARATE,
        None,  # layout.frames
        None,  # volumetric
        <int> extrasample_ if extrasample is not None else None,
        &layout,
    )

    length = max(1, layout.height)
    if layout.planar:
        planes = layout.samples
        samples = 1
        planarconfig_ = PLANARCONFIG_SEPARATE
    else:
        samples = layout.samples
        planes = 1

    if photometric is None and colormap is None:
        if layout.photometric == IC_PHOTO_RGB:
            photometric_ = PHOTOMETRIC_RGB
    # else: keep photometric_ from user parameter parsing

    if layout.samples > UINT16_MAX:
        raise ValueError(f'too many samples={layout.samples}')

    samplesperpixel_ = <uint16_t> layout.samples

    photometric_samples = _photo_samples(layout.photometric)
    extrasamples = samplesperpixel_ - photometric_samples
    if extrasamples < 0:
        raise ValueError(f'{samplesperpixel_=} < {photometric_samples=}')
    if extrasamples > 0:
        if extrasamples >= UINT16_MAX:
            raise ValueError(f'{extrasamples=} > {UINT16_MAX}')
        extrasamples_ = <uint16_t*> calloc(extrasamples, 2)
        if extrasamples_ == NULL:
            raise MemoryError('failed to allocate extrasamples array')
        if extrasample is None and photometric_ == PHOTOMETRIC_RGB:
            extrasample_ = EXTRASAMPLE_UNASSALPHA
        extrasamples_[0] = extrasample_

    framesize = planes * length * layout.width * samples * itemsize
    rowsize = layout.width * samples * itemsize
    if tile is None:
        if rowsperstrip is None:
            rowsperstrip = 262144 // rowsize
        rowsperstrip_ = max(1, min(rowsperstrip, length))
        tilesize = 0
    else:
        tile_length, tile_width = tile
        tilesize = tile_length * tile_width * samples * itemsize
        rowsperstrip_ = 0

    # determine bps_encode: explicit bitspersample param, or bool implies bps=1
    if bitspersample is None:
        if src.dtype.kind == 'b':
            bps_encode = 1
    else:
        bps_encode = int(bitspersample)

    if bps_encode != 0:
        if bps_encode < 1 or bps_encode > 32:
            raise ValueError(f'{bitspersample=} out of range 1-32')
        if sampleformat_ not in {SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT}:
            raise ValueError('bitspersample requires uint or int data')
        bitspersample_ = <uint16_t> bps_encode
        if bps_encode == itemsize * 8:
            # standard bit depth matches dtype. skip packints path
            bps_encode = 0
        else:
            packedrowsize = (layout.width * samples * bps_encode + 7) // 8
            if tile is not None:
                tile_packedrowsize = (
                    tile_width * samples * bps_encode + 7
                ) // 8

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        buf = out
        dstsize = buf.shape[0]
        memtif = memtif_open(<unsigned char*> buf._data, dstsize, 0)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        buf = out
        dstsize = buf.shape[0]
        memtif = memtif_open(<unsigned char*> buf._data, dstsize, 0)
    else:
        out = None
        if compression_ == COMPRESSION_NONE:
            dstsize = src.nbytes + layout.frames * 512
            incsize = layout.frames * 512
        else:
            dstsize = src.nbytes // 3 + layout.frames * 512
            incsize = src.nbytes // 3
        if description:
            dstsize += len(description)
        if appendto is not None:
            dstsize += len(appendto)
        memtif = memtif_new(_align_ssize_t(dstsize), _align_ssize_t(incsize))

    if memtif == NULL:
        raise MemoryError('memtif allocation failed')
    memtif.warn = 1 if verbose else 0
    memtifobj = PyCapsule_New(<void*> memtif, NULL, NULL)

    if appendto is not None:
        buf = appendto
        if memtif.size < <toff_t> append_size:
            raise ValueError(f'{len(appendto)=} > {memtif.size}')

    try:
        with nogil:

            if rowsperstrip_ == 0:
                # allocate tile buffers
                tile_ = <uint8_t*> malloc(tilesize)
                if tile_ == NULL:
                    raise MemoryError('failed to allocate tile buffer')
                if bps_encode != 0:
                    tile_packed_ = <uint8_t*> malloc(
                        tile_packedrowsize * tile_length
                    )
                    if tile_packed_ == NULL:
                        raise MemoryError('failed to allocate tile_packed')
            elif bps_encode != 0:
                # allocate strip row buffer for packed integers
                rowbuf = <uint8_t*> malloc(packedrowsize)
                if rowbuf == NULL:
                    raise MemoryError('failed to allocate rowbuf')
            elif bigendian or predictor_ > PREDICTOR_NONE:
                # allocate strip row buffer for bigendian or predictor
                rowbuf = <uint8_t*> malloc(rowsize)
                if rowbuf == NULL:
                    raise MemoryError('failed to allocate rowbuf')

            if append_size > 0:
                memcpy(
                    <void*> memtif.data,
                    <const void*> buf._data,
                    <size_t> append_size
                )
                memtif.flen = <toff_t> append_size

            openoptions = TIFFOpenOptionsAlloc()
            if openoptions == NULL:
                raise MemoryError('TIFFOpenOptionsAlloc failed')

            TIFFOpenOptionsSetErrorHandlerExtR(
                openoptions, tif_error_handler, <void*> memtif
            )

            TIFFOpenOptionsSetWarningHandlerExtR(
                openoptions, tif_warning_handler, <void*> memtif
            )

            tif = TIFFClientOpenExt(
                'memtif',
                mode_,
                <thandle_t> memtif,
                memtif_TIFFReadProc,
                memtif_TIFFWriteProc,
                memtif_TIFFSeekProc,
                memtif_TIFFCloseProc,
                memtif_TIFFSizeProc,
                memtif_TIFFMapFileProc,
                memtif_TIFFUnmapFileProc,
                openoptions
            )
            if tif == NULL:
                raise TiffError(memtifobj)

            TIFFOpenOptionsFree(openoptions)
            openoptions = NULL

            for i in range(layout.frames):

                if subfiletype_ != 0:
                    ret = TIFFSetField(tif, TIFFTAG_SUBFILETYPE, subfiletype_)
                    if ret == 0:
                        raise TiffError(memtifobj)
                if sampleformat_ != SAMPLEFORMAT_UINT:
                    ret = TIFFSetField(
                        tif, TIFFTAG_SAMPLEFORMAT, sampleformat_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                ret = TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitspersample_)
                if ret == 0:
                    raise TiffError(memtifobj)
                ret = TIFFSetField(
                    tif, TIFFTAG_IMAGEWIDTH, <uint32_t> layout.width
                )
                if ret == 0:
                    raise TiffError(memtifobj)
                ret = TIFFSetField(tif, TIFFTAG_IMAGELENGTH, <uint32_t> length)
                if ret == 0:
                    raise TiffError(memtifobj)
                ret = TIFFSetField(
                    tif, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel_
                )
                if ret == 0:
                    raise TiffError(memtifobj)
                if samplesperpixel_ > 1:
                    ret = TIFFSetField(
                        tif, TIFFTAG_PLANARCONFIG, planarconfig_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                ret = TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometric_)
                if ret == 0:
                    raise TiffError(memtifobj)

                if photometric_ == PHOTOMETRIC_YCBCR:
                    ret = TIFFSetField(
                        tif, TIFFTAG_YCBCRSUBSAMPLING, subsample_, subsample_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                    # TIFFSetField(tif, TIFFTAG_REFERENCEBLACKWHITE, refbw)

                if extrasamples > 0:
                    ret = TIFFSetField(
                        tif, TIFFTAG_EXTRASAMPLES, extrasamples, extrasamples_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)

                if palptr != NULL:
                    ret = TIFFSetField(
                        tif,
                        TIFFTAG_COLORMAP,
                        palptr,
                        palptr + palsize,
                        palptr + palsize + palsize
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)

                ret = TIFFSetField(tif, TIFFTAG_COMPRESSION, compression_)
                if ret == 0:
                    raise TiffError(memtifobj)

                if compression_ > 1:
                    if predictor_ > PREDICTOR_NONE:
                        ret = TIFFSetField(tif, TIFFTAG_PREDICTOR, predictor_)
                        if ret == 0:
                            raise TiffError(memtifobj)

                    if compression_ == COMPRESSION_JPEG:
                        ret = TIFFSetField(
                            tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                        ret = TIFFSetField(tif, TIFFTAG_JPEGTABLESMODE, 0)
                        if ret == 0:
                            raise TiffError(memtifobj)

                    elif compression_ == COMPRESSION_PIXARLOG:
                        if sampleformat_ == SAMPLEFORMAT_IEEEFP:
                            pixarlogdatafmt_ = PIXARLOGDATAFMT_FLOAT
                        elif bitspersample_ == 16:
                            pixarlogdatafmt_ = PIXARLOGDATAFMT_16BIT
                        else:
                            pixarlogdatafmt_ = PIXARLOGDATAFMT_8BIT
                        ret = TIFFSetField(
                            tif, TIFFTAG_PIXARLOGDATAFMT, pixarlogdatafmt_
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)

                    if level_ < 0:
                        pass
                    elif compression_ == COMPRESSION_ADOBE_DEFLATE:
                        ret = TIFFSetField(tif, TIFFTAG_ZIPQUALITY, level_)
                        if ret == 0:
                            raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_ZSTD:
                        ret = TIFFSetField(tif, TIFFTAG_ZSTD_LEVEL, level_)
                        if ret == 0:
                            raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_LZMA:
                        ret = TIFFSetField(tif, TIFFTAG_LZMAPRESET, level_)
                        if ret == 0:
                            raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_LERC:
                        if maxzerror > 0.0:
                            ret = TIFFSetField(
                                tif, TIFFTAG_LERC_MAXZERROR, maxzerror
                            )
                            if ret == 0:
                                raise TiffError(memtifobj)
                        if level_ > 0:
                            ret = TIFFSetField(
                                tif, TIFFTAG_LERC_ADD_COMPRESSION, subcodec_
                            )
                            if ret == 0:
                                raise TiffError(memtifobj)
                            if subcodec_ == LERC_ADD_COMPRESSION_DEFLATE:
                                ret = TIFFSetField(
                                    tif, TIFFTAG_ZIPQUALITY, level_
                                )
                            elif subcodec_ == LERC_ADD_COMPRESSION_ZSTD:
                                ret = TIFFSetField(
                                    tif, TIFFTAG_ZSTD_LEVEL, level_
                                )
                            if ret == 0:
                                raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_JPEG:
                        ret = TIFFSetField(tif, TIFFTAG_JPEGQUALITY, level_)
                        if ret == 0:
                            raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_WEBP:
                        if level_ == 100:
                            ret = TIFFSetField(tif, TIFFTAG_WEBP_LOSSLESS, 1)
                            if ret == 0:
                                raise TiffError(memtifobj)
                        else:
                            ret = TIFFSetField(tif, TIFFTAG_WEBP_LEVEL, level_)
                            if ret == 0:
                                raise TiffError(memtifobj)
                    elif compression_ == COMPRESSION_PIXARLOG:
                        ret = TIFFSetField(
                            tif, TIFFTAG_PIXARLOGQUALITY, level_
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)

                if rowsperstrip_ > 0:
                    ret = TIFFSetField(
                        tif, TIFFTAG_ROWSPERSTRIP, rowsperstrip_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                else:
                    ret = TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width)
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_length)
                    if ret == 0:
                        raise TiffError(memtifobj)

                if resolutionunit_ != RESUNIT_INCH:
                    ret = TIFFSetField(
                        tif, TIFFTAG_RESOLUTIONUNIT, resolutionunit_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                ret = TIFFSetField(tif, TIFFTAG_XRESOLUTION, xresolution)
                if ret == 0:
                    raise TiffError(memtifobj)
                ret = TIFFSetField(tif, TIFFTAG_YRESOLUTION, yresolution)
                if ret == 0:
                    raise TiffError(memtifobj)

                if iccprofile_ != NULL:
                    ret = TIFFSetField(
                        tif, TIFFTAG_ICCPROFILE, iccprofile_size, iccprofile_
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)

                if i == 0:
                    if description_ != NULL:
                        ret = TIFFSetField(
                            tif, TIFFTAG_IMAGEDESCRIPTION, description_
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                    if software_ != NULL:
                        ret = TIFFSetField(tif, TIFFTAG_SOFTWARE, software_)
                        if ret == 0:
                            raise TiffError(memtifobj)
                    if datetime_ != NULL:
                        ret = TIFFSetField(tif, TIFFTAG_DATETIME, datetime_)
                        if ret == 0:
                            raise TiffError(memtifobj)

                if rowsperstrip_ > 0:
                    # write striped

                    if bps_encode == 0:
                        # write striped, except packed integers
                        if <ssize_t> TIFFScanlineSize64(tif) != rowsize:
                            raise ValueError(
                                f'{TIFFScanlineSize64(tif)=} != {rowsize=}'
                            )
                        ret = _tif_encode_striped(
                            tif,
                            srcptr + i * framesize,
                            rowbuf,
                            planes,
                            length,
                            rowsize
                        )
                        if ret < 0:
                            raise TiffError(memtifobj)

                    else:
                        # write striped packed integers
                        ret = _tif_encode_striped_packints(
                            tif,
                            srcptr + i * framesize,
                            rowbuf,
                            planes,
                            length,
                            layout.width * samples,
                            itemsize,
                            packedrowsize,
                            bps_encode
                        )
                        if ret < 0:
                            raise TiffError(memtifobj)

                elif bps_encode == 0:
                    # write tiled, except packed integers
                    if <ssize_t> TIFFTileSize(tif) != tilesize:
                        raise ValueError(
                            f'{TIFFTileSize(tif)=} != {tilesize=}'
                        )
                    ret = _tif_encode_tiled(
                        tif,
                        srcptr + i * framesize,
                        tile_,
                        planes,
                        length,
                        layout.width,
                        tile_length,
                        tile_width,
                        tilesize,
                        rowsize,
                        samples * itemsize
                    )
                    if ret < 0:
                        raise TiffError(memtifobj)

                else:
                    # write tiled packed integers
                    ret = _tif_encode_tiled_packints(
                        tif,
                        srcptr + i * framesize,
                        tile_,
                        tile_packed_,
                        planes,
                        length,
                        layout.width,
                        tile_length,
                        tile_width,
                        tilesize,
                        rowsize,
                        samples * itemsize,
                        tile_packedrowsize,
                        tile_width * samples,
                        itemsize,
                        bps_encode
                    )
                    if ret < 0:
                        raise TiffError(memtifobj)

                ret = TIFFWriteDirectory(tif)
                if ret == 0:
                    raise TiffError(memtifobj)

            memtif_len = <ssize_t> memtif.flen

        if out is None:
            dstsize = memtif_len
            out = _create_output(
                outtype, memtif_len, <const char *> memtif.data
            )

    finally:
        free(tile_)
        free(tile_packed_)
        free(rowbuf)
        free(extrasamples_)
        if tif != NULL:
            TIFFClose(tif)
        if openoptions != NULL:
            TIFFOpenOptionsFree(openoptions)
        memtif_del(memtif)

    return _return_output(out, dstsize, memtif_len, outgiven)


def tiff_decode(
    data,
    /,
    index=0,
    *,
    asrgb=False,
    verbose=None,
    out=None,
):
    """Return decoded TIFF image.

    By default, the image from the first directory/page is returned.
    If index is None, all images in the file with matching shape and
    dtype are returned in one array.

    If asrgb is True, return decoded image as RGBA32.
    Return JPEG compressed images as 8-bit Grayscale or RGB24.
    Return images stored in CMYK colorspace as RGB24.

    The libtiff library does not correctly handle truncated ImageJ hyperstacks,
    SGI depth, STK, LSM, and many other bio-TIFF files.

    """
    cdef:
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        uint8_t* outptr
        uint8_t* tile = NULL
        uint8_t* scanbuf = NULL
        uint8_t* tile_unpacked = NULL
        numpy.npy_intp* strides
        memtif_t* memtif = NULL
        TIFF* tif = NULL
        TIFFOpenOptions* openoptions = NULL
        dirlist_t* dirlist = NULL
        page_t page, page2
        int dirraise = 0
        tdir_t dirnum, dirstart, dirstop, dirstep
        int ret
        uint32_t strip
        ssize_t i, size, sizeleft, outindex, imagesize, images
        ssize_t items_per_row, scansize
        bint rgb = asrgb

    if data is out:
        raise ValueError('cannot transform in-place')

    # TODO: special case STK, ImageJ hyperstacks, and shaped TIFF

    dirnum = dirstart = dirstop = dirstep = 0
    if index is None:
        dirstart = 0
        dirstop = TIFF_MAX_DIR_COUNT
        dirstep = 1
        dirlist = dirlist_new(64)
        dirlist_append(dirlist, dirstart)
    elif index == 0 or isinstance(index, (int, numpy.integer)):
        dirnum = index
        dirlist = dirlist_new(1)
        dirlist_append(dirlist, dirnum)
    elif isinstance(index, (list, tuple, numpy.ndarray)):
        if not 0 < len(index) < <ssize_t> TIFF_MAX_DIR_COUNT:
            raise ValueError('invalid index')
        try:
            dirnum = index[0]  # validate index[0] is non-negative integer
            dirnum = <tdir_t> len(index)
        except Exception as exc:
            raise ValueError('invalid index') from exc
        dirlist = dirlist_new(dirnum)
        dirlist_extend(dirlist, index)
    elif isinstance(index, slice):
        if index.step is not None and index.step < 1:
            raise NotImplementedError('negative steps not implemented')  # TODO
        dirstart = 0 if index.start is None else index.start
        dirstop = TIFF_MAX_DIR_COUNT if index.stop is None else index.stop
        dirstep = 1 if index.step is None else index.step
        dirraise = 1  # raise error when incompatible IFD
        dirlist = dirlist_new(64)
        dirlist_append(dirlist, dirstart)
    else:
        raise ValueError('invalid index')

    if dirlist == NULL:
        raise MemoryError('dirlist_new failed')

    memtif = memtif_open(<unsigned char*> src._data, srcsize, srcsize)
    if memtif == NULL:
        raise MemoryError('memtif_open failed')
    memtif.warn = 1 if verbose else 0
    memtifobj = PyCapsule_New(<void*> memtif, NULL, NULL)

    try:
        with nogil:
            memset(&page, 0, sizeof(page_t))
            memset(&page2, 0, sizeof(page_t))

            openoptions = TIFFOpenOptionsAlloc()
            if openoptions == NULL:
                raise MemoryError('TIFFOpenOptionsAlloc failed')

            TIFFOpenOptionsSetErrorHandlerExtR(
                openoptions, tif_error_handler, <void*> memtif
            )

            TIFFOpenOptionsSetWarningHandlerExtR(
                openoptions, tif_warning_handler, <void*> memtif
            )

            tif = TIFFClientOpenExt(
                'memtif',
                'rh',  # do not load first frame
                <thandle_t> memtif,
                memtif_TIFFReadProc,
                memtif_TIFFWriteProc,
                memtif_TIFFSeekProc,
                memtif_TIFFCloseProc,
                memtif_TIFFSizeProc,
                memtif_TIFFMapFileProc,
                memtif_TIFFUnmapFileProc,
                openoptions
            )
            if tif == NULL:
                raise TiffError(memtifobj)

            TIFFOpenOptionsFree(openoptions)
            openoptions = NULL

            dirnum = dirlist.data[0]
            ret = _tiff_set_directory(tif, dirnum)
            if ret == 0:
                raise IndexError('directory out of range')

            page.asrgb = rgb
            ret = _tiff_decode_ifd(tif, &page)
            if ret == 0:
                raise TiffError(memtifobj)
            if ret == -1:
                raise ValueError(
                    f'sampleformat {int(page.status)} and '
                    f'bitspersample {int(page.itemsize)} not supported'
                )

            # if page.depth > 1:
            #     raise NotImplementedError(f'libtiff does not support depth')

            if dirlist.size > 1 and dirlist.index == 1:
                # index is None or slice
                while 1:
                    if (
                        <ssize_t> dirnum + <ssize_t> dirstep
                        >= <ssize_t> dirstop
                    ):
                        break
                    dirnum += dirstep

                    ret = _tiff_set_directory(tif, dirnum)
                    if ret == 0:
                        break
                    page2.asrgb = rgb
                    ret = _tiff_decode_ifd(tif, &page2)
                    if ret == 0:
                        if dirraise:
                            raise TiffError(memtifobj)
                        if memtif.warn > 0:
                            with gil:
                                _log_warning(memtif.errmsg.decode())
                        continue

                    if (
                        ret < 0
                        or page.planes != page2.planes
                        or page.depth != page2.depth
                        or page.length != page2.length
                        or page.width != page2.width
                        or page.samples != page2.samples
                        or page.itemsize != page2.itemsize
                        or page.bitspersample != page2.bitspersample
                        or page.compression != page2.compression
                        or page.istiled != page2.istiled
                        or page.asrgb != page2.asrgb
                        or page.dtype[0] != page2.dtype[0]
                    ):
                        if dirraise:
                            raise ValueError(
                                f'incompatible directory {dirnum}'
                            )
                        continue

                    ret = dirlist_append(dirlist, dirnum)
                    if ret < 0:
                        raise RuntimeError('dirlist_append failed')

                ret = TIFFSetDirectory(tif, dirlist.data[0])
                if ret == 0:
                    raise TiffError(memtifobj)

            images = dirlist.index
            if images == 0:
                raise ValueError('no matching directories found')

            # ssize_t overflow detected during _create_array() call below
            imagesize = (
                page.planes
                * page.depth
                * page.length
                * page.width
                * page.samples
                * page.itemsize
            )

        shape = (
            images,
            int(page.planes),
            int(page.depth),
            int(page.length),
            int(page.width),
            int(page.samples)
        )
        shapeout = tuple(
            s for i, s in enumerate(shape) if s > 1 or i in {3, 4}
        )

        out = _create_array(
            out, shapeout, f'{page.dtype.decode()}{int(page.itemsize)}'
        )
        out = out.reshape(shape)
        outptr = <uint8_t*> numpy.PyArray_DATA(out)
        strides = numpy.PyArray_STRIDES(out)
        # out[:] = 0

        with nogil:
            if page.asrgb:
                # read as RGB
                for i in range(images):
                    ret = _tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = TIFFReadRGBAImageOriented(
                        tif,
                        <uint32_t> page.width,
                        <uint32_t> page.length,
                        <uint32_t*> &outptr[i * imagesize],
                        ORIENTATION_TOPLEFT,
                        0
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)

            elif page.istiled:
                # read tiled
                if (
                    page.compression == COMPRESSION_JPEG
                    and TIFFSetField(
                        tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB
                    ) == 0
                ):
                    raise TiffError(memtifobj)

                size = TIFFTileSize(tif)
                if size == 0:
                    raise ValueError('TIFFTileSize returned 0')
                tile = <uint8_t*> malloc(size)
                if tile == NULL:
                    raise MemoryError('failed to allocate tile buffer')

                if page.bitspersample == 0:
                    # read tiled, except packed integers
                    for i in range(images):
                        ret = _tiff_set_directory(tif, dirlist.data[i])
                        if ret == 0:
                            raise TiffError(memtifobj)
                        ret = _tiff_decode_tiled(
                            tif,
                            &page,
                            &outptr[i * imagesize],
                            strides,
                            tile,
                            size
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                        if ret < 0:
                            # TODO: libtiff does not seem to handle
                            # tiledepth > 1
                            raise TiffError(
                                f'_tiff_decode_tiled returned {ret}'
                            )

                else:
                    # read tiled packed integers
                    tile_unpacked = <uint8_t*> malloc(
                        (size * 8 // page.bitspersample) * page.itemsize
                    )
                    if tile_unpacked == NULL:
                        raise MemoryError(
                            'failed to allocate tile_unpacked buffer'
                        )
                    for i in range(images):
                        ret = _tiff_set_directory(tif, dirlist.data[i])
                        if ret == 0:
                            raise TiffError(memtifobj)
                        ret = _tiff_decode_tiled_packints(
                            tif,
                            &page,
                            &outptr[i * imagesize],
                            strides,
                            tile,
                            tile_unpacked,
                            size
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                        if ret < 0:
                            raise TiffError(
                                f'_tiff_decode_tiled_packints returned {ret}'
                            )

            elif page.bitspersample == 0:
                # read striped, except packed integers
                for i in range(images):
                    ret = _tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    if TIFFIsTiled(tif) != 0:
                        raise RuntimeError('not a strip image')
                    if (
                        page.compression == COMPRESSION_JPEG
                        and TIFFSetField(
                            tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB
                        ) == 0
                    ):
                        raise TiffError(memtifobj)
                    outindex = i * imagesize
                    sizeleft = imagesize
                    for strip in range(TIFFNumberOfStrips(tif)):
                        size = TIFFReadEncodedStrip(
                            tif,
                            strip,
                            <void*> &outptr[outindex],
                            sizeleft
                        )
                        if size < 0:
                            raise TiffError(memtifobj)
                        outindex += size
                        sizeleft -= size
                        if sizeleft <= 0:
                            break

            else:
                # read striped packed integers
                if (
                    page.compression == COMPRESSION_JPEG
                    and TIFFSetField(
                        tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB
                    ) == 0
                ):
                    raise TiffError(memtifobj)

                items_per_row = page.width * page.samples
                scansize = (items_per_row * page.bitspersample + 7) // 8
                size = TIFFStripSize(tif)
                if size == 0:
                    raise ValueError('TIFFTileSize returned 0')
                scanbuf = <uint8_t*> calloc(size + 4, 1)
                if scanbuf == NULL:
                    raise MemoryError('failed to allocate strip buffer')
                for i in range(images):
                    ret = _tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = _tiff_decode_strips_packints(
                        tif,
                        &page,
                        &outptr[i * imagesize],
                        scanbuf,
                        size + 4,
                        scansize,
                        items_per_row,
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)
                    if ret < 0:
                        raise TiffError(
                            f'_tiff_decode_strips_packints returned {ret}'
                        )

    finally:
        free(tile)
        free(scanbuf)
        free(tile_unpacked)
        dirlist_del(dirlist)
        if tif != NULL:
            TIFFClose(tif)
        if openoptions != NULL:
            TIFFOpenOptionsFree(openoptions)
        memtif_del(memtif)

    if not rgb and page.asrgb and page.truesamples > 0:
        # discard Alpha channel if JPEG compression, YCBCR...
        out = out[..., : page.truesamples]
        shape = (
            images,
            int(page.planes),
            int(page.depth),
            int(page.length),
            int(page.width),
            int(page.truesamples)
        )
        out = out.reshape(
            tuple(s for i, s in enumerate(shape) if s > 1 or i in {3, 4})
        )
        # ? out = numpy.ascontiguousarray(out)
    else:
        out = out.reshape(shapeout)

    return out


cdef inline int _tiff_set_directory(
    TIFF* tif,
    tdir_t dirnum,
) noexcept nogil:
    """Set current directory, avoiding TIFFSetDirectory if possible."""
    cdef:
        ssize_t diff = <ssize_t> dirnum - <ssize_t> TIFFCurrentDirectory(tif)

    if diff == 1:
        return TIFFReadDirectory(tif)
    if diff == 0:
        return 1
    return TIFFSetDirectory(tif, dirnum)


cdef int _tif_encode_striped(
    TIFF* tif,
    uint8_t* srcptr,
    uint8_t* rowbuf,
    const ssize_t planes,
    const ssize_t length,
    const ssize_t rowstride,
) noexcept nogil:
    """Encode stripes."""
    cdef:
        ssize_t p, y
        int ret

    if rowbuf == NULL:
        for p in range(planes):
            for y in range(length):
                ret = TIFFWriteScanline(
                    tif,
                    <void*> srcptr,
                    <uint32_t> y,
                    <uint16_t> p
                )
                if ret < 0:
                    return -1
                srcptr += rowstride
        return 1

    for p in range(planes):
        for y in range(length):
            memcpy(rowbuf, srcptr, rowstride)
            ret = TIFFWriteScanline(
                tif,
                <void*> rowbuf,
                <uint32_t> y,
                <uint16_t> p
            )
            if ret < 0:
                return -1
            srcptr += rowstride
    return 1


cdef int _tif_encode_tiled(
    TIFF* tif,
    uint8_t* srcptr,
    uint8_t* tile,
    const ssize_t planes,
    const ssize_t length,
    const ssize_t width,
    const ssize_t tile_length,
    const ssize_t tile_width,
    const ssize_t tilesize,
    const ssize_t rowstride,
    const ssize_t colstride,
) noexcept nogil:
    """Encode tiles."""
    cdef:
        ssize_t i, p, y, x, size
        tmsize_t ret

    for p in range(planes):
        for y from 0 <= y < length by tile_length:
            for x from 0 <= x < width by tile_width:
                if width - x < tile_width or length - y < tile_length:
                    memset(<void*> tile, 0, tilesize)
                size = min(tile_width, width - x) * colstride
                for i in range(min(tile_length, length - y)):
                    memcpy(
                        tile + i * tile_width * colstride,
                        srcptr + ((y + i) * rowstride + x * colstride),
                        size
                    )
                ret = TIFFWriteTile(
                    tif,
                    <void*> tile,
                    <uint32_t> x,
                    <uint32_t> y,
                    <uint32_t> 0,  # z, depth
                    <uint16_t> p
                )
                if ret < 0:
                    return -1
        srcptr += length * rowstride
    return 1


cdef int _tif_encode_striped_packints(
    TIFF* tif,
    uint8_t* srcptr,
    uint8_t* rowbuf,
    const ssize_t planes,
    const ssize_t length,
    const ssize_t items_per_row,
    const ssize_t itemsize,
    const ssize_t packedrowsize,
    const int bps,
) noexcept nogil:
    """Encode stripes with non-standard bits-per-sample."""
    cdef:
        ssize_t p, y
        ssize_t ret
        int ret2

    for p in range(planes):
        for y in range(length):
            ret = imcd_packints_encode(
                srcptr,
                items_per_row * itemsize,
                rowbuf,
                items_per_row,
                bps
            )
            if ret < 0:
                return <int> ret
            ret2 = TIFFWriteScanline(
                tif,
                <void*> rowbuf,
                <uint32_t> y,
                <uint16_t> p
            )
            if ret2 < 0:
                return -1
            srcptr += items_per_row * itemsize
    return 1


cdef int _tif_encode_tiled_packints(
    TIFF* tif,
    uint8_t* srcptr,
    uint8_t* tile,
    uint8_t* tile_packed,
    const ssize_t planes,
    const ssize_t length,
    const ssize_t width,
    const ssize_t tile_length,
    const ssize_t tile_width,
    const ssize_t tilesize,
    const ssize_t rowstride,
    const ssize_t colstride,
    const ssize_t tile_packedrowsize,
    const ssize_t items_per_tilerow,
    const ssize_t itemsize,
    const int bps,
) noexcept nogil:
    """Encode tiles with non-standard bits-per-sample."""
    cdef:
        ssize_t p, y, x, i
        ssize_t copy_bytes
        ssize_t packed_tile_bytes = tile_packedrowsize * tile_length
        ssize_t ret
        tmsize_t ret2

    for p in range(planes):
        for y from 0 <= y < length by tile_length:
            for x from 0 <= x < width by tile_width:
                # assemble unpacked tile pixels
                if width - x < tile_width or length - y < tile_length:
                    memset(<void*> tile, 0, tilesize)
                copy_bytes = min(tile_width, width - x) * colstride
                for i in range(min(tile_length, length - y)):
                    memcpy(
                        tile + i * tile_width * colstride,
                        srcptr + ((y + i) * rowstride + x * colstride),
                        copy_bytes
                    )
                # pack tile row by row into tile_packed
                memset(<void*> tile_packed, 0, packed_tile_bytes)
                for i in range(tile_length):
                    ret = imcd_packints_encode(
                        tile + i * items_per_tilerow * itemsize,
                        items_per_tilerow * itemsize,
                        tile_packed + i * tile_packedrowsize,
                        items_per_tilerow,
                        bps
                    )
                    if ret < 0:
                        return -1
                # write packed tile
                ret2 = TIFFWriteTile(
                    tif,
                    <void*> tile_packed,
                    <uint32_t> x,
                    <uint32_t> y,
                    <uint32_t> 0,  # z, depth
                    <uint16_t> p
                )
                if ret2 < 0:
                    return -1
        srcptr += length * rowstride
    return 1


cdef int _tiff_decode_ifd(
    TIFF* tif,
    page_t* page
) noexcept nogil:
    """Get normalized image shape and dtype from current IFD tags."""
    cdef:
        uint32_t imagewidth, imagelength, imagedepth
        uint16_t planarconfig, photometric, bitspersample, sampleformat
        uint16_t samplesperpixel, compression
        int ret

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planarconfig)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photometric)
    if ret == 0:
        # this is ambiguous because PHOTOMETRIC_MINISWHITE == 0
        photometric = PHOTOMETRIC_MINISWHITE

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGEWIDTH, &imagewidth)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGELENGTH, &imagelength)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGEDEPTH, &imagedepth)
    if ret == 0 or imagedepth < 1:
        imagedepth = 1

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bitspersample)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compression)
    if ret == 0:
        return 0

    page.compression = <ssize_t> compression

    if compression == COMPRESSION_JPEG:
        page.asrgb = 0
        page.truesamples = 0
        ret = TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB)
        if ret == 0:
            return 0
    elif compression == COMPRESSION_OJPEG or photometric == PHOTOMETRIC_YCBCR:
        # let libtiff handle OJPEG and YCbCr -> RGB conversion
        page.asrgb = 1
        page.truesamples = <ssize_t> samplesperpixel
    elif photometric == PHOTOMETRIC_SEPARATED:
        # let libtiff handle CMYK -> RGB conversion
        page.asrgb = 1
        page.truesamples = 3
    else:
        page.truesamples = 0

    if page.asrgb != 0:
        page.istiled = 0  # don't care
    else:
        page.istiled = TIFFIsTiled(tif)

    page.status = 1
    page.length = <ssize_t> imagelength
    page.width = <ssize_t> imagewidth
    if page.asrgb:
        page.planes = 1
        page.depth = 1
        page.samples = 4
    elif planarconfig == PLANARCONFIG_CONTIG:
        page.planes = 1
        page.depth = <ssize_t> imagedepth
        page.samples = <ssize_t> samplesperpixel
    else:
        page.planes = <ssize_t> samplesperpixel
        page.depth = <ssize_t> imagedepth
        page.samples = 1

    page.dtype[1] = 0
    if page.asrgb:
        page.dtype[0] = b'u'
    elif photometric == PHOTOMETRIC_LOGLUV:
        # return LogLuv as float32
        page.dtype[0] = b'f'
        page.status = <int> SAMPLEFORMAT_IEEEFP
        bitspersample = 32
        ret = TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT)
        if ret == 0:
            return 0
    elif sampleformat == SAMPLEFORMAT_UINT:
        page.dtype[0] = b'u'
    elif sampleformat == SAMPLEFORMAT_INT:
        page.dtype[0] = b'i'
    elif sampleformat == SAMPLEFORMAT_IEEEFP:
        page.dtype[0] = b'f'
        if (
            bitspersample != 16
            # and bitspersample != 24
            and bitspersample != 32
            and bitspersample != 64
        ):
            page.status = <int> sampleformat
            page.itemsize = <ssize_t> bitspersample
            return -1
    elif sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP:
        page.dtype[0] = b'c'
        if (
            bitspersample != 32
            and bitspersample != 64
            and bitspersample != 128
        ):
            page.status = <int> sampleformat
            page.itemsize = <ssize_t> bitspersample
            return -1
    else:
        # sampleformat == SAMPLEFORMAT_VOID
        # sampleformat == SAMPLEFORMAT_COMPLEXINT
        page.status = <int> sampleformat
        page.itemsize = <ssize_t> bitspersample
        return -1

    page.bitspersample = 0
    if page.asrgb:
        page.itemsize = 1
    elif bitspersample == 8:
        page.itemsize = 1
    elif bitspersample == 16:
        page.itemsize = 2
    elif bitspersample == 32:
        page.itemsize = 4
    elif bitspersample == 64:
        page.itemsize = 8
    elif bitspersample == 128:
        page.itemsize = 16
    elif page.dtype[0] == b'u' or page.dtype[0] == b'i':
        # non-standard bit depths: 1-7, 9-15, 17-31
        if bitspersample == 1 and page.dtype[0] == b'u':
            page.dtype[0] = b'b'  # return 1-bit uint as bool
            page.itemsize = 1
        elif bitspersample <= 8:
            page.itemsize = 1
        elif bitspersample <= 16:
            page.itemsize = 2
        elif bitspersample <= 32:
            page.itemsize = 4
        else:
            page.status = <int> sampleformat
            page.itemsize = <ssize_t> bitspersample
            return -1
        page.bitspersample = <ssize_t> bitspersample
    # elif page.dtype[0] == b'f' and bitspersample == 24:
    #     page.itemsize = 4
    #     page.bitspersample = 24
    else:
        page.status = <int> sampleformat
        page.itemsize = <ssize_t> bitspersample
        return -1

    return 1


cdef int _tiff_decode_tiled(
    TIFF* tif,
    page_t* page,
    uint8_t* dst,
    numpy.npy_intp* strides,
    uint8_t* tile,
    ssize_t size,
) noexcept nogil:
    """Decode tiled image. Return 1 on success."""
    cdef:
        ssize_t i, j, h, d, samplesize
        ssize_t tiledepth, tilelength, tilewidth, tilesize, tileindex
        ssize_t tileddepth, tiledlength, tiledwidth
        ssize_t sizeleft
        ssize_t sp, sd, sl, sw
        ssize_t tp, td, tl, tw
        uint32_t value
        int ret

    if (
        page.compression == COMPRESSION_JPEG
        and TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB) == 0
    ):
        return 0

    if TIFFTileSize(tif) != size:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILEWIDTH, &value)
    if ret == 0:
        return 0
    tilewidth = <ssize_t> value

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILELENGTH, &value)
    if ret == 0:
        return 0
    tilelength = <ssize_t> value

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILEDEPTH, &value)
    if ret == 0 or value == 0:
        tiledepth = 1
    else:
        tiledepth = <ssize_t> value
    samplesize = page.samples * page.itemsize
    sizeleft = page.planes * page.depth * page.length * page.width * samplesize
    tilesize = tiledepth * tilelength * tilewidth * samplesize
    tiledwidth = (page.width + tilewidth - 1) // tilewidth
    tiledlength = (page.length + tilelength - 1) // tilelength
    tileddepth = (page.depth + tiledepth - 1) // tiledepth
    sp = strides[1]
    sd = strides[2]
    sl = strides[3]
    sw = strides[4]

    if size != tilesize:
        # raise TiffError(f'TIFFTileSize {size} != {tilesize}')
        return -1

    for tileindex in range(<ssize_t> TIFFNumberOfTiles(tif)):
        size = TIFFReadEncodedTile(
            tif, <uint32_t> tileindex, <void*> tile, tilesize
        )
        if size < 0:
            return 0
        if size != tilesize:
            # raise TiffError(f'TIFFReadEncodedTile {size} != {tilesize}')
            return -2
        tp = tileindex // (tiledwidth * tiledlength * tileddepth)
        td = (tileindex // (tiledwidth * tiledlength)) % tileddepth * tiledepth
        tl = (tileindex // tiledwidth) % tiledlength * tilelength
        tw = tileindex % tiledwidth * tilewidth
        size = min(tilewidth, page.width - tw) * samplesize

        for d in range(min(tiledepth, page.depth - td)):
            for h in range(min(tilelength, page.length - tl)):
                sizeleft -= size
                if sizeleft < 0:
                    return -2
                i = tp * sp + (td + d) * sd + (tl + h) * sl + tw * sw
                j = (d * tilelength + h) * tilewidth * samplesize
                # TODO: check out of bounds writes?
                memcpy(<void*> &dst[i], <const void*> &tile[j], size)
    return 1


cdef int _tiff_decode_tiled_packints(
    TIFF* tif,
    page_t* page,
    uint8_t* dst,
    numpy.npy_intp* strides,
    uint8_t* tile_packed,
    uint8_t* tile_unpacked,
    const ssize_t tile_packed_size,
) noexcept nogil:
    """Decode tiled non-standard bitspersample image."""
    cdef:
        ssize_t imagedepth, imagelength, imagewidth, samplesize, itemsize
        ssize_t tiledepth, tilelength, tilewidth
        ssize_t tileddepth, tiledlength, tiledwidth
        ssize_t tp, td, tl, tw, tileindex
        ssize_t d, h, i, row_packed_bytes, copy_bytes
        ssize_t sp, sd, sl, sw
        uint32_t value
        tmsize_t size
        ssize_t ret

    if (
        page.compression == COMPRESSION_JPEG
        and TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB) == 0
    ):
        return 0

    if TIFFTileSize(tif) != tile_packed_size:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILEWIDTH, &value)
    if ret == 0:
        return 0
    tilewidth = <ssize_t> value

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILELENGTH, &value)
    if ret == 0:
        return 0
    tilelength = <ssize_t> value

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILEDEPTH, &value)
    if ret == 0 or value == 0:
        tiledepth = 1
    else:
        tiledepth = <ssize_t> value

    imagedepth = page.depth
    imagelength = page.length
    imagewidth = page.width
    itemsize = page.itemsize
    samplesize = page.samples * itemsize
    row_packed_bytes = (tilewidth * page.samples * page.bitspersample + 7) // 8
    tiledwidth = (imagewidth + tilewidth - 1) // tilewidth
    tiledlength = (imagelength + tilelength - 1) // tilelength
    tileddepth = (imagedepth + tiledepth - 1) // tiledepth
    sp = strides[1]
    sd = strides[2]
    sl = strides[3]
    sw = strides[4]

    if tile_packed_size != row_packed_bytes * tiledepth * tilelength:
        return -1

    for tileindex in range(<ssize_t> TIFFNumberOfTiles(tif)):
        size = TIFFReadEncodedTile(
            tif, <uint32_t> tileindex, <void*> tile_packed, tile_packed_size
        )
        if size < 0:
            return 0
        tp = tileindex // (tiledwidth * tiledlength * tileddepth)
        td = (tileindex // (tiledwidth * tiledlength)) % tileddepth * tiledepth
        tl = (tileindex // tiledwidth) % tiledlength * tilelength
        tw = tileindex % tiledwidth * tilewidth
        copy_bytes = min(tilewidth, imagewidth - tw) * samplesize

        for d in range(min(tiledepth, imagedepth - td)):
            for h in range(min(tilelength, imagelength - tl)):
                ret = imcd_packints_decode(
                    tile_packed + (d * tilelength + h) * row_packed_bytes,
                    row_packed_bytes,
                    tile_unpacked,
                    tilewidth * page.samples,
                    page.bitspersample
                )
                if ret < 0:
                    return <int> ret
                i = tp * sp + (td + d) * sd + (tl + h) * sl + tw * sw
                memcpy(
                    <void*> &dst[i],
                    <const void*> tile_unpacked,
                    copy_bytes
                )
    return 1


cdef int _tiff_decode_strips_packints(
    TIFF* tif,
    page_t* page,
    uint8_t* dst,
    uint8_t* strip_buf,
    const ssize_t strip_buf_size,
    const ssize_t packedrowsize,
    const ssize_t items_per_row,
) noexcept nogil:
    """Decode non-standard bitspersample strip image."""
    cdef:
        ssize_t strip, nstrips, size, actual_rows, r, ret
        ssize_t rows_remaining = page.planes * page.depth * page.length

    if (
        page.compression == COMPRESSION_JPEG
        and TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB) == 0
    ):
        return 0

    if TIFFStripSize(tif) + 4 != strip_buf_size:
        return 0

    nstrips = TIFFNumberOfStrips(tif)
    for strip in range(nstrips):
        if rows_remaining <= 0:
            break
        size = TIFFReadEncodedStrip(
            tif, <uint32_t> strip, <void*> strip_buf, strip_buf_size
        )
        if size < 0:
            return 0
        actual_rows = size // packedrowsize
        if actual_rows > rows_remaining:
            actual_rows = rows_remaining
        for r in range(actual_rows):
            ret = imcd_packints_decode(
                strip_buf + r * packedrowsize,
                packedrowsize,
                dst,
                items_per_row,
                page.bitspersample
            )
            if ret < 0:
                return <int> ret
            dst += items_per_row * page.itemsize
        rows_remaining -= actual_rows
    return 1


ctypedef struct dirlist_t:
    tdir_t* data
    tdir_t size
    tdir_t index


cdef dirlist_t* dirlist_new(tdir_t size) noexcept nogil:
    """Return new dirlist."""
    cdef:
        dirlist_t* dirlist = <dirlist_t*> calloc(1, sizeof(dirlist_t))

    if dirlist == NULL:
        return NULL
    if size < 1:
        size = 1
    dirlist.index = 0
    dirlist.size = size
    dirlist.data = <tdir_t*> calloc(size, sizeof(tdir_t))
    if dirlist.data == NULL:
        free(dirlist)
        return NULL
    return dirlist


cdef void dirlist_del(dirlist_t* dirlist) noexcept nogil:
    """Free memory."""
    if dirlist != NULL:
        free(dirlist.data)
        free(dirlist)


cdef int dirlist_append(dirlist_t* dirlist, tdir_t ifd) noexcept nogil:
    """Append IFD to list."""
    cdef:
        tdir_t* tmp = NULL
        ssize_t newsize = 0

    if dirlist == NULL:
        return -1
    if dirlist.index == TIFF_MAX_DIR_COUNT:
        return -1  # list full
    if dirlist.index == dirlist.size:
        newsize = max(16, <ssize_t> dirlist.size * 2)
        if newsize > <ssize_t> TIFF_MAX_DIR_COUNT:
            newsize = TIFF_MAX_DIR_COUNT
        tmp = <tdir_t*> realloc(dirlist.data, newsize * sizeof(tdir_t))
        if tmp == NULL:
            return -2  # memory error
        dirlist.data = tmp
        dirlist.size = <tdir_t> newsize
    dirlist.data[dirlist.index] = ifd
    dirlist.index += 1
    return 0


cdef int dirlist_extend(dirlist_t* dirlist, values):
    """Append list of IFD to list."""
    cdef:
        tdir_t ifd
        int ret = 0

    for ifd in values:
        ret = dirlist_append(dirlist, ifd)
        if ret != 0:
            break
    return ret


cdef const ssize_t MEMTIF_CHECK = 1234567890


ctypedef struct memtif_t:
    ssize_t check
    unsigned char* data
    toff_t size
    toff_t inc
    toff_t flen
    toff_t fpos
    int owner
    int warn
    char[80] errmsg


cdef memtif_t* memtif_open(
    unsigned char* data,
    toff_t size,
    toff_t flen,
) noexcept nogil:
    """Return new memtif from existing buffer for reading."""
    cdef:
        memtif_t* memtif = <memtif_t*> calloc(1, sizeof(memtif_t))

    if memtif == NULL or flen > size:
        return NULL
    if data == NULL:
        free(memtif)
        return NULL
    memtif.check = MEMTIF_CHECK
    memtif.data = data
    memtif.size = size
    memtif.inc = 0
    memtif.flen = flen
    memtif.fpos = 0
    memtif.owner = 0
    memtif.warn = 1
    memtif.errmsg[0] = b'\0'
    return memtif


cdef memtif_t* memtif_new(
    toff_t size,
    toff_t inc,
) noexcept nogil:
    """Return new memtif with new buffer for writing."""
    cdef:
        memtif_t* memtif = <memtif_t*> calloc(1, sizeof(memtif_t))

    if memtif == NULL:
        return NULL
    memtif.data = <unsigned char*> malloc(<size_t> size)
    if memtif.data == NULL:
        free(memtif)
        return NULL
    memtif.check = MEMTIF_CHECK
    memtif.size = size
    memtif.inc = inc
    memtif.flen = 0
    memtif.fpos = 0
    memtif.owner = 1
    memtif.warn = 1
    memtif.errmsg[0] = b'\0'
    return memtif


cdef void memtif_del(
    memtif_t* memtif,
) noexcept nogil:
    """Delete memtif."""
    if memtif != NULL:
        if memtif.owner:
            free(memtif.data)
        free(memtif)


cdef tsize_t memtif_TIFFReadProc(
    thandle_t handle,
    void* buf,
    tmsize_t size,
) noexcept nogil:
    """Callback function to read from memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle

    if memtif.flen < memtif.fpos + size:
        size = <tmsize_t> (memtif.flen - memtif.fpos)
    memcpy(buf, <const void*> &memtif.data[memtif.fpos], size)
    memtif.fpos += size
    return size


cdef tmsize_t memtif_TIFFWriteProc(
    thandle_t handle,
    void* buf,
    tmsize_t size,
) noexcept nogil:
    """Callback function to write to memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle
        unsigned char* tmp
        toff_t newsize

    if memtif.size < memtif.fpos + size:
        if memtif.owner == 0:
            return -1
        newsize = memtif.fpos + memtif.inc + size
        tmp = <unsigned char*> realloc(&memtif.data[0], <size_t> newsize)
        if tmp == NULL:
            return -1
        memtif.data = tmp
        memtif.size = newsize
    memcpy(<void*> &memtif.data[memtif.fpos], <const void*> buf, size)
    memtif.fpos += size
    if memtif.fpos > memtif.flen:
        memtif.flen = memtif.fpos
    return size


cdef toff_t memtif_TIFFSeekProc(
    thandle_t handle,
    toff_t off,
    int whence,
) noexcept nogil:
    """Callback function to seek in memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle
        unsigned char* tmp
        toff_t newsize

    if whence == SEEK_SET:
        if memtif.size < off:
            if memtif.owner == 0:
                return -1
            newsize = memtif.size + memtif.inc + off
            tmp = <unsigned char*> realloc(&memtif.data[0], <size_t> newsize)
            if tmp == NULL:
                return -1
            memtif.data = tmp
            memtif.size = newsize
        memtif.fpos = off

    elif whence == SEEK_CUR:
        if memtif.size < memtif.fpos + off:
            if memtif.owner == 0:
                return -1
            newsize = memtif.fpos + memtif.inc + off
            tmp = <unsigned char*> realloc(&memtif.data[0], <size_t> newsize)
            if tmp == NULL:
                return -1
            memtif.data = tmp
            memtif.size = newsize
        memtif.fpos += off

    elif whence == SEEK_END:
        if memtif.size < memtif.flen + off:
            if memtif.owner == 0:
                return -1
            newsize = memtif.flen + memtif.inc + off
            tmp = <unsigned char*> realloc(&memtif.data[0], <size_t> newsize)
            if tmp == NULL:
                return -1
            memtif.data = tmp
            memtif.size = newsize
        memtif.fpos = memtif.flen + off

    if memtif.fpos > memtif.flen:
        memtif.flen = memtif.fpos

    return memtif.fpos


cdef int memtif_TIFFCloseProc(
    thandle_t handle,
) noexcept nogil:
    """Callback function to close memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle

    memtif.fpos = 0
    return 0


cdef toff_t memtif_TIFFSizeProc(
    thandle_t handle
) noexcept nogil:
    """Callback function to return size of memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle

    return memtif.flen


cdef int memtif_TIFFMapFileProc(
    thandle_t handle,
    void** base,
    toff_t* size,
) noexcept nogil:
    """Callback function to map memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle

    base[0] = memtif.data
    size[0] = memtif.flen
    return 1


cdef void memtif_TIFFUnmapFileProc(
    thandle_t handle,
    void* base,
    toff_t size,
) noexcept nogil:
    """Callback function to unmap memtif."""
    return


cdef int tif_error_handler(
    TIFF* tif,
    void* user_data,
    const char* module,
    const char* fmt,
    va_list args,
) noexcept nogil:
    """Callback function to write libtiff error message to memtif."""
    cdef:
        memtif_t* memtif
        int i

    if user_data == NULL or tif == NULL:
        return 0  # call global error handler
    memtif = <memtif_t*> user_data
    if memtif.check != MEMTIF_CHECK:
        return 0  # call global error handler
    i = vsnprintf(&memtif.errmsg[0], 80, fmt, args)
    memtif.errmsg[0 if i < 0 else 79] = 0
    return 1


cdef int tif_warning_handler(
    TIFF* tif,
    void* user_data,
    const char* module,
    const char* fmt,
    va_list args,
) noexcept with gil:
    """Callback function to output libtiff warning message to logging."""
    cdef:
        char[80] msg
        memtif_t* memtif
        int i

    # TODO: is this freethreading compatible?
    if user_data == NULL or tif == NULL:
        return 0  # call global warning handler
    memtif = <memtif_t*> user_data
    if memtif.check != MEMTIF_CHECK:
        return 0  # call global warning handler
    if memtif.warn == 0:
        return 1  # done
    i = vsnprintf(&msg[0], 80, fmt, args)
    if i > 0:
        msg[79] = 0
        try:
            _log_warning(msg.decode('utf-8', errors='replace').strip())
        except Exception:
            pass
    return 1


# work around TIFF name conflict
globals().update({'TIFF': _TIFF})
