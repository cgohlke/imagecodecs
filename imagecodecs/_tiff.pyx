# imagecodecs/_tiff.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2025, Christoph Gohlke
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

"""TIFF codec for the imagecodecs package."""

include '_shared.pxi'

from libtiff cimport *

from libc.stdio cimport SEEK_SET, SEEK_CUR, SEEK_END

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

cdef extern from '<stdio.h>':
    int vsnprintf(char* s, size_t n, const char* format, va_list arg) nogil

# private definition in tiffiop.h
DEF TIFF_MAX_DIR_COUNT = 1048576


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
        LZW = COMPRESSION_LZW
        JPEG = COMPRESSION_JPEG
        PACKBITS = COMPRESSION_PACKBITS
        DEFLATE = COMPRESSION_DEFLATE
        ADOBE_DEFLATE = COMPRESSION_ADOBE_DEFLATE
        LZMA = COMPRESSION_LZMA
        ZSTD = COMPRESSION_ZSTD
        WEBP = COMPRESSION_WEBP
        # LERC = COMPRESSION_LERC
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


class TiffError(RuntimeError):
    """TIFF codec exceptions."""

    def __init__(self, arg=None, msg=''):
        """Initialize Exception from string or memtif capsule."""
        cdef:
            memtif_t* memtif

        if arg is None or isinstance(arg, str):
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


def tiff_check(const uint8_t[::1] data):
    """Return whether data is TIFF encoded image."""
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
    level=None,
    bigtiff=None,
    append=None,
    photometric=None,
    planarconfig=None,
    extrasamples=None,
    # volumetric=False,
    tile=None,
    rowsperstrip=None,
    bitspersample=None,
    compression=None,
    predictor=None,
    # colormap=None,
    description=None,
    datetime=None,
    resolution=None,
    subfiletype=0,
    software=None,
    verbose=None,
    out=None
):
    """Return TIFF encoded image (not implemented)."""
    raise NotImplementedError('tiff_encode')  # TODO


def tiff_decode(
    data, index=0, asrgb=False, verbose=None, out=None
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
        ssize_t srcsize = src.size
        uint8_t* outptr
        uint8_t* tile = NULL
        numpy.npy_intp* strides
        memtif_t* memtif = NULL
        TIFF* tif = NULL
        TIFFOpenOptions* openoptions = NULL
        dirlist_t* dirlist = NULL
        int dirraise = 0
        tdir_t dirnum, dirstart, dirstop, dirstep
        int ret
        uint32_t strip
        ssize_t i, size, sizeleft, outindex, imagesize, images
        ssize_t[8] sizes
        ssize_t[8] sizes2
        char[2] dtype
        char[2] dtype2
        bint rgb = asrgb
        int isrgb, isrgb2, istiled, istiled2

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
            dirnum = index[0]
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
                raise TiffError(memtifobj)

            TIFFOpenOptionsSetErrorHandlerExtR(
                openoptions, tif_error_handler, <void*> memtif
            )

            TIFFOpenOptionsSetWarningHandlerExtR(
                openoptions, tif_warning_handler, <void*> memtif
            )

            tif = TIFFClientOpenExt(
                'memtif',
                'r',
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
            if dirnum != 0:
                ret = tiff_set_directory(tif, dirnum)
                if ret == 0:
                    raise IndexError('directory out of range')

            isrgb = rgb
            ret = tiff_read_ifd(tif, &sizes[0], &dtype[0], &isrgb, &istiled)
            if ret == 0:
                raise TiffError(memtifobj)
            if ret == -1:
                raise ValueError(
                    f'sampleformat {int(sizes[0])} and '
                    f'bitspersample {int(sizes[6])} not supported'
                )

            # if sizes[2] > 1:
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

                    ret = tiff_set_directory(tif, dirnum)
                    if ret == 0:
                        break
                    isrgb2 = rgb
                    ret = tiff_read_ifd(
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
                        or sizes[1] != sizes2[1]
                        or sizes[2] != sizes2[2]
                        or sizes[3] != sizes2[3]
                        or sizes[4] != sizes2[4]
                        or sizes[5] != sizes2[5]
                        or sizes[6] != sizes2[6]
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
            imagesize = (
                sizes[1] * sizes[2] * sizes[3] * sizes[4] * sizes[5] * sizes[6]
            )

        shape = (
            images,
            int(sizes[1]),
            int(sizes[2]),
            int(sizes[3]),
            int(sizes[4]),
            int(sizes[5])
        )
        shapeout = tuple(
            s for i, s in enumerate(shape) if s > 1 or i in {3, 4}
        )

        out = _create_array(out, shapeout, f'{dtype.decode()}{int(sizes[6])}')
        out.shape = shape
        outptr = <uint8_t*> numpy.PyArray_DATA(out)
        strides = numpy.PyArray_STRIDES(out)
        # out[:] = 0

        with nogil:
            if isrgb:
                for i in range(images):
                    ret = tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = TIFFReadRGBAImageOriented(
                        tif,
                        <uint32_t> sizes[4],
                        <uint32_t> sizes[3],
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
                for i in range(images):
                    ret = tiff_set_directory(tif, dirlist.data[i])
                    if ret == 0:
                        raise TiffError(memtifobj)
                    ret = tiff_decode_tiled(
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
                        # TODO: libtiff does not seem to handle tiledepth > 1
                        raise TiffError(f'tile size != {size}')

            else:
                for i in range(images):
                    ret = tiff_set_directory(tif, dirlist.data[i])
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
        dirlist_del(dirlist)
        if tif != NULL:
            TIFFClose(tif)
        if openoptions != NULL:
            TIFFOpenOptionsFree(openoptions)
        memtif_del(memtif)

    if not rgb and isrgb and sizes[7] > 0:
        # discard Alpha channel if JPEG compression, YCBCR...
        out = out[..., : sizes[7]]
        shape = (
            images,
            int(sizes[1]),
            int(sizes[2]),
            int(sizes[3]),
            int(sizes[4]),
            int(sizes[7])
        )
        out.shape = tuple(
            s for i, s in enumerate(shape) if s > 1 or i in {3, 4}
        )
        # ? out = numpy.ascontiguousarray(out)
    else:
        out.shape = shapeout

    return out


cdef int tiff_read_ifd(
    TIFF* tif,
    ssize_t* sizes,
    char* dtype,
    int* asrgb,
    int* istiled
) nogil:
    """Get normalized image shape and dtype from current IFD tags.

    'sizes' contains images, planes, depth, height, width, samples, itemsize,
    true_samples.

    """
    cdef:
        uint32_t imagewidth, imageheight, imagedepth
        uint16_t planarconfig, photometric, bitspersample, sampleformat
        uint16_t samplesperpixel, compression
        int ret

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planarconfig)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photometric)
    if ret == 0:
        photometric = PHOTOMETRIC_MINISWHITE

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGEWIDTH, &imagewidth)
    if ret == 0:
        return 0

    ret = TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGELENGTH, &imageheight)
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

    if (
        compression == COMPRESSION_JPEG
        or compression == COMPRESSION_OJPEG
        or photometric == PHOTOMETRIC_YCBCR
    ):
        asrgb[0] = 1
        sizes[7] = <ssize_t> samplesperpixel
    elif photometric == PHOTOMETRIC_SEPARATED:
        asrgb[0] = 1
        sizes[7] = 3  # CMYK -> RGB
    else:
        sizes[7] = 0

    if asrgb[0] != 0:
        istiled[0] = 0  # don't care
    else:
        istiled[0] = TIFFIsTiled(tif)

    sizes[0] = 1
    sizes[3] = <ssize_t> imageheight
    sizes[4] = <ssize_t> imagewidth
    if asrgb[0]:
        sizes[1] = 1
        sizes[2] = 1
        sizes[5] = 4
    elif planarconfig == PLANARCONFIG_CONTIG:
        sizes[1] = 1
        sizes[2] = <ssize_t> imagedepth
        sizes[5] = <ssize_t> samplesperpixel
    else:
        sizes[1] = <ssize_t> samplesperpixel
        sizes[2] = <ssize_t> imagedepth
        sizes[5] = 1

    dtype[1] = 0
    if asrgb[0]:
        dtype[0] = b'u'
    elif photometric == PHOTOMETRIC_LOGLUV:
        # return LogLuv as float32
        dtype[0] = b'f'
        sizes[0] = <ssize_t> SAMPLEFORMAT_IEEEFP
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
            sizes[0] = <ssize_t> sampleformat
            sizes[6] = <ssize_t> bitspersample
            return -1
    elif sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP:
        dtype[0] = b'c'
        if (
            bitspersample != 32
            and bitspersample != 64
            and bitspersample != 128
        ):
            sizes[0] = <ssize_t> sampleformat
            sizes[6] = <ssize_t> bitspersample
            return -1
    else:
        # sampleformat == SAMPLEFORMAT_VOID
        # sampleformat == SAMPLEFORMAT_COMPLEXINT
        sizes[0] = <ssize_t> sampleformat
        sizes[6] = <ssize_t> bitspersample
        return -1

    if asrgb[0]:
        sizes[6] = 1
    elif bitspersample == 8:
        sizes[6] = 1
    elif bitspersample == 16:
        sizes[6] = 2
    elif bitspersample == 32:
        sizes[6] = 4
    elif bitspersample == 64:
        sizes[6] = 8
    elif bitspersample == 128:
        sizes[6] = 16
    # TODO: support 1, 2, and 4 bit integers
    # elif bitspersample == 1:
    #     dtype[0] = b'b'
    #     sizes[6] = 1
    # elif bitspersample < 8:
    #     sizes[6] = 1
    else:
        sizes[0] = <ssize_t> sampleformat
        sizes[6] = <ssize_t> bitspersample
        return -1

    return 1


cdef int tiff_decode_tiled(
    TIFF* tif,
    uint8_t* dst,
    ssize_t* sizes,
    numpy.npy_intp* strides,
    uint8_t* tile,
    ssize_t size
) nogil:
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

    imageplane = sizes[1]
    imagedepth = sizes[2]
    imagelength = sizes[3]
    imagewidth = sizes[4]
    samplesize = sizes[5] * sizes[6]
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
                if sizeleft >= 0:
                    i = tp * sp + (td + d) * sd + (tl + h) * sl + tw * sw
                    j = h * tilewidth * samplesize
                    memcpy(&dst[i], &tile[j], size)
    return 1


cdef inline int tiff_set_directory(TIFF* tif, tdir_t dirnum) nogil:
    """Set current directory, avoiding TIFFSetDirectory if possible."""
    cdef:
        ssize_t diff = <ssize_t> dirnum - <ssize_t> TIFFCurrentDirectory(tif)

    if diff == 1:
        return TIFFReadDirectory(tif)
    if diff == 0:
        return 1
    return TIFFSetDirectory(tif, dirnum)


ctypedef struct dirlist_t:
    tdir_t* data
    tdir_t size
    tdir_t index


cdef dirlist_t* dirlist_new(tdir_t size) nogil:
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


ctypedef struct memtif_t:
    ssize_t check
    unsigned char* data
    toff_t size
    toff_t inc
    toff_t flen
    toff_t fpos
    int owner
    int warn
    char errmsg[80]


cdef memtif_t* memtif_open(
    unsigned char* data,
    toff_t size,
    toff_t flen
) nogil:
    """Return new memtif from existing buffer for reading."""
    cdef:
        memtif_t* memtif = <memtif_t*> calloc(1, sizeof(memtif_t))

    if memtif == NULL:
        return NULL
    if data == NULL:
        free(memtif)
        return NULL
    memtif.check = 1234567890
    memtif.data = data
    memtif.size = size
    memtif.inc = 0
    memtif.flen = flen
    memtif.fpos = 0
    memtif.owner = 0
    memtif.warn = 1
    memtif.errmsg[0] = b'\0'
    return memtif


cdef memtif_t* memtif_new(toff_t size, toff_t inc) noexcept nogil:
    """Return new memtif with new buffer for writing."""
    cdef:
        memtif_t* memtif = <memtif_t*> calloc(1, sizeof(memtif_t))

    if memtif == NULL:
        return NULL
    memtif.data = <unsigned char*> malloc(<size_t> size)
    if memtif.data == NULL:
        free(memtif)
        return NULL
    memtif.check = 1234567890
    memtif.size = size
    memtif.inc = inc
    memtif.flen = 0
    memtif.fpos = 0
    memtif.owner = 1
    memtif.warn = 1
    memtif.errmsg[0] = b'\0'
    return memtif


cdef void memtif_del(memtif_t* memtif) noexcept nogil:
    """Delete memtif."""
    if memtif != NULL:
        if memtif.owner:
            free(memtif.data)
        free(memtif)


cdef tsize_t memtif_TIFFReadProc(
    thandle_t handle,
    void* buf,
    tmsize_t size
) noexcept nogil:
    """Callback function to read from memtif."""
    cdef:
        memtif_t* memtif = <memtif_t*> handle

    if memtif.flen < memtif.fpos + size:
        size = <tmsize_t> (memtif.flen - memtif.fpos)
    memcpy(buf, &memtif.data[memtif.fpos], size)
    memtif.fpos += size
    return size


cdef tmsize_t memtif_TIFFWriteProc(
    thandle_t handle,
    void* buf,
    tmsize_t size
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
    memcpy(&memtif.data[memtif.fpos], buf, size)
    memtif.fpos += size
    if memtif.fpos > memtif.flen:
        memtif.flen = memtif.fpos
    return size


cdef toff_t memtif_TIFFSeekProc(
    thandle_t handle,
    toff_t off,
    int whence
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
        if memtif.size < memtif.size + off:
            if memtif.owner == 0:
                return -1
            newsize = memtif.size + memtif.inc + off
            tmp = <unsigned char*> realloc(&memtif.data[0], <size_t> newsize)
            if tmp == NULL:
                return -1
            memtif.data = tmp
            memtif.size = newsize
        memtif.fpos = memtif.size + off

    if memtif.fpos > memtif.flen:
        memtif.flen = memtif.fpos

    return memtif.fpos


cdef int memtif_TIFFCloseProc(
    thandle_t handle
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
    toff_t* size
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
    toff_t size
) noexcept nogil:
    """Callback function to unmap memtif."""
    return


cdef int tif_error_handler(
    TIFF *tif,
    void* user_data,
    const char* module,
    const char* fmt,
    va_list args
) noexcept nogil:
    """Callback function to write libtiff error message to memtif."""
    cdef:
        memtif_t* memtif
        int i

    if user_data == NULL or tif == NULL:
        return 0  # call global error handler
    memtif = <memtif_t*> user_data
    if memtif.check != 1234567890:
        return 0  # call global error handler
    i = vsnprintf(&memtif.errmsg[0], 80, fmt, args)
    memtif.errmsg[0 if i < 0 else 79] = 0
    return 1


cdef int tif_warning_handler(
    TIFF *tif,
    void* user_data,
    const char* module,
    const char* fmt,
    va_list args
) noexcept with gil:
    """Callback function to output libtiff warning message to logging."""
    cdef:
        char msg[80]
        memtif_t* memtif
        int i

    if user_data == NULL or tif == NULL:
        return 0  # call global warning handler
    memtif = <memtif_t*> user_data
    if memtif.check != 1234567890:
        return 0  # call global warning handler
    if memtif.warn == 0:
        return 1  # done
    i = vsnprintf(&msg[0], 80, fmt, args)
    if i > 0:
        msg[79] = 0
        _log_warning(msg.decode().strip())
    return 1


# work around TIFF name conflict
globals().update({'TIFF': _TIFF})
