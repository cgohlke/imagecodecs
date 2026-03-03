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

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from imcd cimport imcd_packints_decode, imcd_packints_encode
from libc.stdio cimport SEEK_CUR, SEEK_END, SEEK_SET
from libtiff cimport *


cdef extern from '<stdio.h>':
    int vsnprintf(char* s, size_t n, const char* format, va_list arg) nogil


cdef:
    const tdir_t TIFF_MAX_DIR_COUNT = 1048576  # private def in tiffiop.h

    # index constants for the sizes[] geometry array
    const int SZ_SAMPLEFORMAT = 0  # error payload on failure, 1 on success
    const int SZ_PLANES = 1  # 1 for CONTIG, samplesperpixel for SEPARATE
    const int SZ_DEPTH = 2  # imagedepth, usually 1
    const int SZ_LENGTH = 3  # imagelength (rows)
    const int SZ_WIDTH = 4  # imagewidth  (columns)
    const int SZ_SAMPLES = 5  # samplesperpixel for CONTIG, 1 for SEPARATE
    const int SZ_ITEMSIZE = 6  # itemsize of output container dtype in bytes
    const int SZ_TRUESAMPLES = 7  # non-zero when asrgb forces RGBA
    const int SZ_BPS = 8  # non-standard bps for packints unpacking; else 0


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
        DEFLATE = COMPRESSION_DEFLATE
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
        numpy.ndarray src = numpy.ascontiguousarray(data)
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
        ssize_t frames = 1
        ssize_t planes = 1  # planar samples
        ssize_t length = 1
        ssize_t width = 1
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

    if data is out:
        raise ValueError('cannot encode in-place')

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
        elif byteorder in {TIFF_LITTLEENDIAN, '<', 'little'}:
            mode += b'l'
        else:
            raise ValueError(f'{byteorder=!r} not supported')
    else:
        mode = b'a'
        append_size = len(appendto)
    mode_ = mode

    if subfiletype is not None:
        subfiletype_ = subfiletype

    if compression is None:
        if level is None:
            compression_ = COMPRESSION_NONE
        else:
            compression_ = COMPRESSION_ADOBE_DEFLATE
            level_ = _default_value(level, 6, 0, 12)
    elif compression in {
        COMPRESSION_DEFLATE, COMPRESSION_ADOBE_DEFLATE, 'deflate',
    }:
        compression_ = COMPRESSION_ADOBE_DEFLATE
        level_ = _default_value(level, 6, -1, 12)
    elif compression in {COMPRESSION_ZSTD, 'zstd'}:
        compression_ = COMPRESSION_ZSTD
        level_ = _default_value(level, 3, -1, 22)  # ZSTD_CLEVEL_DEFAULT = 3
    elif compression in {COMPRESSION_LZW, 'lzw'}:
        compression_ = COMPRESSION_LZW
    elif compression in {COMPRESSION_JPEG, 'jpeg'}:
        compression_ = COMPRESSION_JPEG
        level_ = _default_value(level, 95, -1, 100)
    elif compression in {COMPRESSION_WEBP, 'webp'}:
        compression_ = COMPRESSION_WEBP
        level_ = _default_value(level, 100, -1, 100)
    elif compression in {COMPRESSION_LZMA, 'lzma'}:
        compression_ = COMPRESSION_LZMA
        level_ = _default_value(level, 6, -1, 9)
    elif compression in {COMPRESSION_PACKBITS, 'packbits'}:
        compression_ = COMPRESSION_PACKBITS
    elif compression in {COMPRESSION_LERC, 'lerc'}:
        compression_ = COMPRESSION_LERC
        maxzerror = _default_value(level, 0.0, 0.0, None)
        if subcodec is None:
            subcodec_ = LERC_ADD_COMPRESSION_NONE
        elif subcodec in {LERC_ADD_COMPRESSION_ZSTD, 'zstd'}:
            subcodec_ = LERC_ADD_COMPRESSION_ZSTD
            level_ = 3  # ZSTD_CLEVEL_DEFAULT
        elif subcodec in {LERC_ADD_COMPRESSION_DEFLATE, 'deflate'}:
            subcodec_ = LERC_ADD_COMPRESSION_DEFLATE
            level_ = 6  # Z_DEFAULT_COMPRESSION
        else:
            raise ValueError(f'{subcodec=} not supported')
    elif compression in {COMPRESSION_CCITTRLE, 'ccittrle'}:
        compression_ = COMPRESSION_CCITTRLE
    elif compression in {COMPRESSION_CCITTFAX3, 'ccittfax3'}:
        compression_ = COMPRESSION_CCITTFAX3
    elif compression in {COMPRESSION_CCITTFAX4, 'ccittfax4'}:
        compression_ = COMPRESSION_CCITTFAX4
    elif compression in {COMPRESSION_PIXARLOG, 'pixarlog'}:
        compression_ = COMPRESSION_PIXARLOG
        level_ = _default_value(level, 6, -1, 12)
    # elif compression in {COMPRESSION_JXL, 'jxl'}:
    #     compression_ = COMPRESSION_JXL

    elif compression in {COMPRESSION_NONE, 'none'}:
        compression_ = COMPRESSION_NONE
    else:
        raise ValueError(f'{compression=} not supported')

    if predictor is None:
        pass
    elif isinstance(predictor, bool):
        if predictor:
            if sampleformat_ in {SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT}:
                predictor_ = PREDICTOR_HORIZONTAL
            else:
                predictor_ = PREDICTOR_FLOATINGPOINT
    elif predictor in {PREDICTOR_HORIZONTAL, 'horizontal'}:
        predictor_ = PREDICTOR_HORIZONTAL
    elif predictor in {PREDICTOR_FLOATINGPOINT, 'floatingpoint'}:
        predictor_ = PREDICTOR_FLOATINGPOINT
    else:
        raise ValueError(f'{predictor=} not supported')

    if resolution is not None:
        xresolution, yresolution = resolution
        resolutionunit_ = RESUNIT_INCH

    if resolutionunit is None:
        pass
    elif resolutionunit in {RESUNIT_INCH, 'inch'}:
        resolutionunit_ = RESUNIT_INCH
    elif resolutionunit in {RESUNIT_CENTIMETER, 'cm'}:
        resolutionunit_ = RESUNIT_CENTIMETER
    elif resolutionunit in {RESUNIT_NONE, 'none'}:
        resolutionunit_ = RESUNIT_NONE
    else:
        raise ValueError(f'{resolutionunit=} not supported')

    if extrasample is None:
        pass
    elif extrasample in {EXTRASAMPLE_ASSOCALPHA, 'assocalpha'}:
        extrasample_ = EXTRASAMPLE_ASSOCALPHA
    elif extrasample in {EXTRASAMPLE_UNASSALPHA, 'unassalpha'}:
        extrasample_ = EXTRASAMPLE_UNASSALPHA
    elif extrasample in {EXTRASAMPLE_UNSPECIFIED, 'unspecified'}:
        extrasample_ = EXTRASAMPLE_UNSPECIFIED
    else:
        raise ValueError(f'{extrasample=!r} not supported')

    if planarconfig is None:
        pass
    elif planarconfig in {PLANARCONFIG_SEPARATE, 'separate'}:
        planarconfig_ = PLANARCONFIG_SEPARATE
    elif planarconfig in {PLANARCONFIG_CONTIG, 'contig'}:
        planarconfig_ = PLANARCONFIG_CONTIG
    else:
        raise ValueError(f'{planarconfig=!r} not supported')

    if photometric is None:
        if colormap is not None:
            photometric_ = PHOTOMETRIC_PALETTE
    elif photometric in {PHOTOMETRIC_RGB, 'rgb'}:
        photometric_ = PHOTOMETRIC_RGB
        photometric_samples = 3
    elif photometric in {PHOTOMETRIC_MINISBLACK, 'minisblack'}:
        photometric_ = PHOTOMETRIC_MINISBLACK
    elif photometric in {PHOTOMETRIC_MINISWHITE, 'miniswhite'}:
        photometric_ = PHOTOMETRIC_MINISWHITE
    elif photometric in {PHOTOMETRIC_SEPARATED, 'separated'}:
        photometric_ = PHOTOMETRIC_SEPARATED
        photometric_samples = 4
    elif photometric in {PHOTOMETRIC_YCBCR, 'ycbcr'}:
        photometric_ = PHOTOMETRIC_YCBCR
        photometric_samples = 3
    elif photometric in {PHOTOMETRIC_PALETTE, 'palette'}:
        photometric_ = PHOTOMETRIC_PALETTE
        if extrasample is not None:
            raise ValueError('palette image with extrasamples not supported')
    else:
        raise ValueError(f'{photometric=!r} not supported')

    if photometric_ == PHOTOMETRIC_PALETTE:
        if colormap is None:
            raise ValueError('palette image requires colormap')
        if src.dtype.kind != 'u':
            raise ValueError('palette image requires unsigned image')
        pal = numpy.ascontiguousarray(colormap)
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

    if ndim == 0:
        pass
    elif ndim == 1:
        width = src.shape[0]
    elif ndim == 2:
        length = src.shape[0]
        width = src.shape[1]
    elif (
        # autodetect RGB(A)
        photometric is None
        and sampleformat_ == SAMPLEFORMAT_UINT
        and bitspersample_ <= 16
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
        photometric_ = PHOTOMETRIC_RGB
        photometric_samples = 3
        if planarconfig_ == PLANARCONFIG_CONTIG:
            length = src.shape[ndim - 3]
            width = src.shape[ndim - 2]
            samples = src.shape[ndim - 1]
        else:
            planes = src.shape[ndim - 3]
            length = src.shape[ndim - 2]
            width = src.shape[ndim - 1]
        for i in range(ndim - 3):
            frames *= src.shape[i]
    elif photometric_samples == 1 and extrasample is None:
        length = src.shape[ndim - 2]
        width = src.shape[ndim - 1]
        for i in range(ndim - 2):
            frames *= src.shape[i]
    else:
        if planarconfig_ == PLANARCONFIG_CONTIG:
            length = src.shape[ndim - 3]
            width = src.shape[ndim - 2]
            samples = src.shape[ndim - 1]
        else:
            planes = src.shape[ndim - 3]
            length = src.shape[ndim - 2]
            width = src.shape[ndim - 1]
        for i in range(ndim - 3):
            frames *= src.shape[i]

    if samples * planes > UINT16_MAX:
        raise ValueError(f'too many samples={samples * planes}')

    samplesperpixel_ = <uint16_t> (samples * planes)

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

    framesize = planes * length * width * samples * itemsize
    rowsize = width * samples * itemsize
    if tile is None:
        if rowsperstrip is None:
            rowsperstrip = 262144 // rowsize
        rowsperstrip_ = max(1, min(rowsperstrip, length))
        tilesize = 0
    else:
        tile_length, tile_width = tile
        tilesize = tile_length * tile_width * samples * itemsize
        tile_ = <uint8_t*> malloc(tilesize)
        if tile_ == NULL:
            raise MemoryError('failed to allocate tile')
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
            packedrowsize = (width * samples * bps_encode + 7) // 8
            rowbuf = <uint8_t*> malloc(packedrowsize)
            if rowbuf == NULL:
                raise MemoryError('failed to allocate rowbuf')
            if tile is not None:
                tile_packedrowsize = (
                    tile_width * samples * bps_encode + 7
                ) // 8
                tile_packed_ = <uint8_t*> malloc(
                    tile_packedrowsize * tile_length
                )
                if tile_packed_ == NULL:
                    raise MemoryError('failed to allocate tile_packed')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        buf = out
        dstsize = buf.nbytes
        memtif = memtif_open(<unsigned char*> &buf[0], dstsize, 0)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        buf = out
        dstsize = buf.nbytes
        memtif = memtif_open(<unsigned char*> &buf[0], dstsize, 0)
    else:
        out = None
        if compression_ == COMPRESSION_NONE:
            dstsize = src.nbytes + frames * 512
            incsize = frames * 512
        else:
            dstsize = src.nbytes // 3 + frames * 512
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
            if append_size > 0:
                memcpy(
                    <void*> memtif.data,
                    <const void*> &buf[0],
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

            for i in range(frames):

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
                ret = TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, <uint32_t> width)
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
                    # write strips
                    if bps_encode != 0:
                        ret = _tif_encode_striped_packints(
                            tif,
                            srcptr + i * framesize,
                            rowbuf,
                            planes,
                            length,
                            width * samples,
                            itemsize,
                            packedrowsize,
                            bps_encode
                        )
                        if ret < 0:
                            raise TiffError(memtifobj)
                    else:
                        if <ssize_t> TIFFScanlineSize64(tif) != rowsize:
                            raise ValueError(
                                f'{TIFFScanlineSize64(tif)=} != {rowsize=}'
                            )
                        ret = _tif_encode_striped(
                            tif,
                            srcptr + i * framesize,
                            planes,
                            length,
                            rowsize
                        )
                        if ret < 0:
                            raise TiffError(memtifobj)
                else:
                    # write tiles
                    if bps_encode != 0:
                        ret = _tif_encode_tiled_packints(
                            tif,
                            srcptr + i * framesize,
                            tile_,
                            tile_packed_,
                            planes,
                            length,
                            width,
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
                    else:
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
                            width,
                            tile_length,
                            tile_width,
                            tilesize,
                            rowsize,
                            samples * itemsize
                        )
                        if ret < 0:
                            raise TiffError(memtifobj)

                ret = TIFFWriteDirectory(tif)
                if ret == 0:
                    raise TiffError(memtifobj)

            memtif_len = memtif.flen

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
        ssize_t srcsize = src.nbytes
        uint8_t* outptr
        uint8_t* tile = NULL
        uint8_t* scanbuf = NULL
        uint8_t* tile_unpacked = NULL
        numpy.npy_intp* strides
        memtif_t* memtif = NULL
        TIFF* tif = NULL
        TIFFOpenOptions* openoptions = NULL
        dirlist_t* dirlist = NULL
        int dirraise = 0
        tdir_t dirnum, dirstart, dirstop, dirstep
        int ret
        int bps_
        uint32_t strip
        ssize_t i, size, sizeleft, outindex, imagesize, images
        ssize_t items_per_row, scansize
        ssize_t[9] sizes
        ssize_t[9] sizes2
        char[2] dtype
        char[2] dtype2
        bint rgb = asrgb
        int isrgb, isrgb2, istiled, istiled2
        uint16_t compression_ = 0

    if data is out:
        raise ValueError('cannot decode in-place')

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
        if not 0 < len(index) < TIFF_MAX_DIR_COUNT:
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

    memtif = memtif_open(<unsigned char*> &src[0], srcsize, srcsize)
    if memtif == NULL:
        raise MemoryError('memtif_open failed')
    memtif.warn = 1 if verbose else 0
    memtifobj = PyCapsule_New(<void*> memtif, NULL, NULL)

    try:
        with nogil:
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

            isrgb = rgb
            ret = _tiff_decode_ifd(tif, &sizes[0], &dtype[0], &isrgb, &istiled)
            TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compression_)
            if ret == 0:
                raise TiffError(memtifobj)
            if ret == -1:
                raise ValueError(
                    f'sampleformat {int(sizes[SZ_SAMPLEFORMAT])} and '
                    f'bitspersample {int(sizes[SZ_ITEMSIZE])} not supported'
                )

            # if sizes[SZ_DEPTH] > 1:
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
                    isrgb2 = rgb
                    ret = _tiff_decode_ifd(
                        tif, &sizes2[0], &dtype2[0], &isrgb2, &istiled2
                    )
                    if ret == 0:
                        if dirraise:
                            raise TiffError(memtifobj)
                        if memtif.warn > 0:
                            with gil:
                                _log_warning(memtif.errmsg.decode())
                        continue

                    if (
                        ret < 0
                        or sizes[SZ_PLANES] != sizes2[SZ_PLANES]
                        or sizes[SZ_DEPTH] != sizes2[SZ_DEPTH]
                        or sizes[SZ_LENGTH] != sizes2[SZ_LENGTH]
                        or sizes[SZ_WIDTH] != sizes2[SZ_WIDTH]
                        or sizes[SZ_SAMPLES] != sizes2[SZ_SAMPLES]
                        or sizes[SZ_ITEMSIZE] != sizes2[SZ_ITEMSIZE]
                        or sizes[SZ_BPS] != sizes2[SZ_BPS]
                        or dtype[0] != dtype2[0]
                        or istiled != istiled2
                        or isrgb != isrgb2
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
                sizes[SZ_PLANES]
                * sizes[SZ_DEPTH]
                * sizes[SZ_LENGTH]
                * sizes[SZ_WIDTH]
                * sizes[SZ_SAMPLES]
                * sizes[SZ_ITEMSIZE]
            )

        shape = (
            images,
            int(sizes[SZ_PLANES]),
            int(sizes[SZ_DEPTH]),
            int(sizes[SZ_LENGTH]),
            int(sizes[SZ_WIDTH]),
            int(sizes[SZ_SAMPLES])
        )
        shapeout = tuple(
            s for i, s in enumerate(shape) if s > 1 or i in {3, 4}
        )

        out = _create_array(
            out, shapeout, f'{dtype.decode()}{int(sizes[SZ_ITEMSIZE])}'
        )
        out = out.reshape(shape)
        outptr = <uint8_t*> numpy.PyArray_DATA(out)
        strides = numpy.PyArray_STRIDES(out)
        # out[:] = 0

        with nogil:
            if isrgb:
                for i in range(images):
                    ret = _tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = TIFFReadRGBAImageOriented(
                        tif,
                        <uint32_t> sizes[SZ_WIDTH],
                        <uint32_t> sizes[SZ_LENGTH],
                        <uint32_t*> &outptr[i * imagesize],
                        ORIENTATION_TOPLEFT,
                        0
                    )
                    if ret == 0:
                        raise TiffError(memtifobj)

            elif istiled:
                size = TIFFTileSize(tif)
                tile = <uint8_t*> malloc(size)
                if tile == NULL:
                    raise MemoryError('failed to allocate tile buffer')
                if sizes[SZ_BPS] != 0:
                    bps_ = <int> sizes[SZ_BPS]
                    tile_unpacked = <uint8_t*> malloc(
                        (size * 8 // bps_) * sizes[SZ_ITEMSIZE]
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
                            &outptr[i * imagesize],
                            sizes,
                            strides,
                            tile,
                            tile_unpacked,
                            size,
                            bps_
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                        if ret < 0:
                            raise TiffError(
                                f'_tiff_decode_tiled_packints returned {ret}'
                            )
                else:
                    for i in range(images):
                        ret = _tiff_set_directory(tif, dirlist.data[i])
                        if ret == 0:
                            raise TiffError(memtifobj)
                        ret = _tiff_decode_tiled(
                            tif,
                            &outptr[i * imagesize],
                            sizes,
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
                if sizes[SZ_BPS] != 0:
                    bps_ = <int> sizes[SZ_BPS]
                    items_per_row = sizes[SZ_WIDTH] * sizes[SZ_SAMPLES]
                    scansize = TIFFScanlineSize(tif)
                    scanbuf = <uint8_t*> malloc(scansize)
                    if scanbuf == NULL:
                        raise MemoryError(
                            'failed to allocate scanline buffer'
                        )
                    for i in range(images):
                        ret = _tiff_set_directory(tif, dirlist.data[i])
                        if ret == 0:
                            raise TiffError(memtifobj)
                        ret = _tiff_decode_scanlines_packints(
                            tif,
                            &outptr[i * imagesize],
                            scanbuf,
                            scansize,
                            sizes[SZ_PLANES],
                            sizes[SZ_LENGTH],
                            items_per_row,
                            sizes[SZ_ITEMSIZE],
                            bps_
                        )
                        if ret == 0:
                            raise TiffError(memtifobj)
                        if ret < 0:
                            raise TiffError(
                                f'imcd_packints_decode returned {ret}'
                            )
                else:
                    for i in range(images):
                        ret = _tiff_set_directory(tif, dirlist.data[i])
                        if ret == 0:
                            raise TiffError(memtifobj)
                        if TIFFIsTiled(tif) != 0:
                            raise RuntimeError('not a strip image')
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

    if not rgb and isrgb and sizes[SZ_TRUESAMPLES] > 0:
        # discard Alpha channel if JPEG compression, YCBCR...
        out = out[..., : sizes[SZ_TRUESAMPLES]]
        shape = (
            images,
            int(sizes[SZ_PLANES]),
            int(sizes[SZ_DEPTH]),
            int(sizes[SZ_LENGTH]),
            int(sizes[SZ_WIDTH]),
            int(sizes[SZ_TRUESAMPLES])
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
    const ssize_t planes,
    const ssize_t length,
    const ssize_t rowstride,
) noexcept nogil:
    """Encode stripes."""
    cdef:
        ssize_t p, y
        int ret

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
    ssize_t* sizes,
    char* dtype,
    int* asrgb,
    int* istiled,
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

    if compression == COMPRESSION_JPEG:
        asrgb[0] = 1
        sizes[SZ_TRUESAMPLES] = <ssize_t> samplesperpixel
        ret = TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB)
        if ret == 0:
            return 0
    elif compression == COMPRESSION_OJPEG or photometric == PHOTOMETRIC_YCBCR:
        asrgb[0] = 1
        sizes[SZ_TRUESAMPLES] = <ssize_t> samplesperpixel
    elif photometric == PHOTOMETRIC_SEPARATED:
        asrgb[0] = 1
        sizes[SZ_TRUESAMPLES] = 3  # CMYK -> RGB
    else:
        sizes[SZ_TRUESAMPLES] = 0

    if asrgb[0] != 0:
        istiled[0] = 0  # don't care
    else:
        istiled[0] = TIFFIsTiled(tif)

    sizes[SZ_SAMPLEFORMAT] = 1
    sizes[SZ_LENGTH] = <ssize_t> imagelength
    sizes[SZ_WIDTH] = <ssize_t> imagewidth
    if asrgb[0]:
        sizes[SZ_PLANES] = 1
        sizes[SZ_DEPTH] = 1
        sizes[SZ_SAMPLES] = 4
    elif planarconfig == PLANARCONFIG_CONTIG:
        sizes[SZ_PLANES] = 1
        sizes[SZ_DEPTH] = <ssize_t> imagedepth
        sizes[SZ_SAMPLES] = <ssize_t> samplesperpixel
    else:
        sizes[SZ_PLANES] = <ssize_t> samplesperpixel
        sizes[SZ_DEPTH] = <ssize_t> imagedepth
        sizes[SZ_SAMPLES] = 1

    dtype[1] = 0
    if asrgb[0]:
        dtype[0] = b'u'
    elif photometric == PHOTOMETRIC_LOGLUV:
        # return LogLuv as float32
        dtype[0] = b'f'
        sizes[SZ_SAMPLEFORMAT] = <ssize_t> SAMPLEFORMAT_IEEEFP
        bitspersample = 32
        ret = TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT)
        if ret == 0:
            return 0
    elif sampleformat == SAMPLEFORMAT_UINT:
        dtype[0] = b'u'
    elif sampleformat == SAMPLEFORMAT_INT:
        dtype[0] = b'i'
    elif sampleformat == SAMPLEFORMAT_IEEEFP:
        dtype[0] = b'f'
        if (
            bitspersample != 16
            and bitspersample != 32
            and bitspersample != 64
        ):
            sizes[SZ_SAMPLEFORMAT] = <ssize_t> sampleformat
            sizes[SZ_ITEMSIZE] = <ssize_t> bitspersample
            return -1
    elif sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP:
        dtype[0] = b'c'
        if (
            bitspersample != 32
            and bitspersample != 64
            and bitspersample != 128
        ):
            sizes[SZ_SAMPLEFORMAT] = <ssize_t> sampleformat
            sizes[SZ_ITEMSIZE] = <ssize_t> bitspersample
            return -1
    else:
        # sampleformat == SAMPLEFORMAT_VOID
        # sampleformat == SAMPLEFORMAT_COMPLEXINT
        sizes[SZ_SAMPLEFORMAT] = <ssize_t> sampleformat
        sizes[SZ_ITEMSIZE] = <ssize_t> bitspersample
        return -1

    sizes[SZ_BPS] = 0
    if asrgb[0]:
        sizes[SZ_ITEMSIZE] = 1
    elif bitspersample == 8:
        sizes[SZ_ITEMSIZE] = 1
    elif bitspersample == 16:
        sizes[SZ_ITEMSIZE] = 2
    elif bitspersample == 32:
        sizes[SZ_ITEMSIZE] = 4
    elif bitspersample == 64:
        sizes[SZ_ITEMSIZE] = 8
    elif bitspersample == 128:
        sizes[SZ_ITEMSIZE] = 16
    elif dtype[0] == b'u' or dtype[0] == b'i':
        # non-standard bit depths: 1-7, 9-15, 17-31
        if bitspersample == 1 and dtype[0] == b'u':
            dtype[0] = b'b'  # return 1-bit uint as bool
            sizes[SZ_ITEMSIZE] = 1
        elif bitspersample <= 8:
            sizes[SZ_ITEMSIZE] = 1
        elif bitspersample <= 16:
            sizes[SZ_ITEMSIZE] = 2
        elif bitspersample <= 32:
            sizes[SZ_ITEMSIZE] = 4
        else:
            sizes[SZ_SAMPLEFORMAT] = <ssize_t> sampleformat
            sizes[SZ_ITEMSIZE] = <ssize_t> bitspersample
            return -1
        sizes[SZ_BPS] = <ssize_t> bitspersample
    else:
        sizes[SZ_SAMPLEFORMAT] = <ssize_t> sampleformat
        sizes[SZ_ITEMSIZE] = <ssize_t> bitspersample
        return -1

    return 1


cdef int _tiff_decode_tiled(
    TIFF* tif,
    uint8_t* dst,
    ssize_t* sizes,
    numpy.npy_intp* strides,
    uint8_t* tile,
    ssize_t size,
) noexcept nogil:
    """Decode tiled image. Return 1 on success."""
    cdef:
        ssize_t i, j, h, d
        ssize_t imageplane, imagedepth, imagelength, imagewidth, samplesize
        ssize_t tiledepth, tilelength, tilewidth, tilesize, tileindex
        ssize_t tileddepth, tiledlength, tiledwidth
        ssize_t sizeleft
        ssize_t sp, sd, sl, sw
        ssize_t tp, td, tl, tw
        uint32_t value
        int ret

    if TIFFIsTiled(tif) == 0:
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

    imageplane = sizes[SZ_PLANES]
    imagedepth = sizes[SZ_DEPTH]
    imagelength = sizes[SZ_LENGTH]
    imagewidth = sizes[SZ_WIDTH]
    samplesize = sizes[SZ_SAMPLES] * sizes[SZ_ITEMSIZE]
    sizeleft = imageplane * imagedepth * imagelength * imagewidth * samplesize
    sp = strides[1]
    sd = strides[2]
    sl = strides[3]
    sw = strides[4]
    tilesize = tiledepth * tilelength * tilewidth * samplesize
    tiledwidth = (imagewidth + tilewidth - 1) // tilewidth
    tiledlength = (imagelength + tilelength - 1) // tilelength
    tileddepth = (imagedepth + tiledepth - 1) // tiledepth

    if size != tilesize:
        # raise TiffError(f'TIFFTileSize {size} != {tilesize}')
        return -1

    for tileindex in range(TIFFNumberOfTiles(tif)):
        size = TIFFReadEncodedTile(
            tif, <uint32_t> tileindex, <void*> tile, tilesize
        )
        if size < 0:
            return 0
        if size != tilesize:
            # raise TiffError(f'TIFFReadEncodedTile {size} != {tilesize}')
            return -1
        tp = tileindex // (tiledwidth * tiledlength * tileddepth)
        td = (tileindex // (tiledwidth * tiledlength)) % tileddepth * tiledepth
        tl = (tileindex // tiledwidth) % tiledlength * tilelength
        tw = tileindex % tiledwidth * tilewidth
        size = min(tilewidth, imagewidth - tw) * samplesize

        for d in range(min(tiledepth, imagedepth - td)):
            for h in range(min(tilelength, imagelength - tl)):
                sizeleft -= size
                if sizeleft < 0:
                    return -2
                i = tp * sp + (td + d) * sd + (tl + h) * sl + tw * sw
                j = h * tilewidth * samplesize
                # TODO: check out of bounds writes?
                memcpy(<void*> &dst[i], <const void*> &tile[j], size)
    return 1


cdef int _tiff_decode_scanlines_packints(
    TIFF* tif,
    uint8_t* dst,
    uint8_t* scanbuf,
    const ssize_t scansize,
    const ssize_t planes,
    const ssize_t length,
    const ssize_t items_per_row,
    const ssize_t itemsize,
    const int bps,
) noexcept nogil:
    """Decode non-standard bitspersample strip image row by row."""
    cdef:
        ssize_t p, y
        ssize_t ret

    for p in range(planes):
        for y in range(length):
            if TIFFReadScanline(
                tif, <void*> scanbuf, <uint32_t> y, <uint16_t> p
            ) < 0:
                return -1
            ret = imcd_packints_decode(
                scanbuf, scansize, dst, items_per_row, bps
            )
            if ret < 0:
                return <int> ret
            dst += items_per_row * itemsize
    return 1


cdef int _tiff_decode_tiled_packints(
    TIFF* tif,
    uint8_t* dst,
    ssize_t* sizes,
    numpy.npy_intp* strides,
    uint8_t* tile_packed,
    uint8_t* tile_unpacked,
    const ssize_t tile_packed_size,
    const int bps,
) noexcept nogil:
    """Decode tiled non-standard bitspersample image."""
    cdef:
        ssize_t imagelength, imagewidth, samplesize, itemsize
        ssize_t tilelength, tilewidth
        ssize_t tiledlength, tiledwidth
        ssize_t tp, tl, tw, tileindex
        ssize_t h, i, row_packed_bytes, copy_bytes
        ssize_t sp, sl, sw
        uint32_t value
        tmsize_t size
        ssize_t ret

    if TIFFIsTiled(tif) == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILEWIDTH, &value)
    if ret == 0:
        return 0
    tilewidth = <ssize_t> value

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_TILELENGTH, &value)
    if ret == 0:
        return 0
    tilelength = <ssize_t> value

    imagelength = sizes[SZ_LENGTH]
    imagewidth = sizes[SZ_WIDTH]
    itemsize = sizes[SZ_ITEMSIZE]
    samplesize = sizes[SZ_SAMPLES] * itemsize
    row_packed_bytes = (tilewidth * sizes[SZ_SAMPLES] * bps + 7) // 8
    tiledwidth = (imagewidth + tilewidth - 1) // tilewidth
    tiledlength = (imagelength + tilelength - 1) // tilelength
    sp = strides[1]
    sl = strides[3]
    sw = strides[4]

    for tileindex in range(TIFFNumberOfTiles(tif)):
        size = TIFFReadEncodedTile(
            tif, <uint32_t> tileindex, <void*> tile_packed, tile_packed_size
        )
        if size < 0:
            return 0
        tp = tileindex // (tiledwidth * tiledlength)
        tl = (tileindex // tiledwidth) % tiledlength * tilelength
        tw = tileindex % tiledwidth * tilewidth
        copy_bytes = min(tilewidth, imagewidth - tw) * samplesize

        for h in range(min(tilelength, imagelength - tl)):
            ret = imcd_packints_decode(
                tile_packed + h * row_packed_bytes,
                row_packed_bytes,
                tile_unpacked,
                tilewidth * sizes[SZ_SAMPLES],
                bps
            )
            if ret < 0:
                return <int> ret
            i = tp * sp + (tl + h) * sl + tw * sw
            memcpy(
                <void*> &dst[i],
                <const void*> tile_unpacked,
                copy_bytes
            )
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
        if newsize > TIFF_MAX_DIR_COUNT:
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
