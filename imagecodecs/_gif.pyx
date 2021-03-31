# imagecodecs/_gif.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""GIF codec for the imagecodecs package."""

__version__ = '2021.1.28'

include '_shared.pxi'

from giflib cimport *


class GIF:
    """Gif Constants."""


class GifError(RuntimeError):
    """GIF Exceptions."""

    def __init__(self, func, err):
        cdef:
            char* errormessage
            int errorcode

        try:
            errorcode = int(err)
            errormessage = <char*> GifErrorString(errorcode)
            if errormessage == NULL:
                raise RuntimeError('GifErrorString returned NULL')
            msg = errormessage.decode().strip()
        except Exception:
            msg = 'NULL' if err is None else f'unknown error {err!r}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def gif_version():
    """Return giflib library version string."""
    return f'giflib {GIFLIB_MAJOR}.{GIFLIB_MINOR}.{GIFLIB_RELEASE}'


def gif_check(const uint8_t[::1] data):
    """Return True if data likely contains a GIF image."""
    cdef:
        bytes sig = bytes(data[:6])

    return sig == b'GIF87a' or sig == b'GIF89a'


def gif_encode(data, level=None, colormap=None, out=None):
    """Return GIF image from numpy array."""
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        memgif_t memgif
        GifFileType* gif = NULL
        ColorMapObject* colormapobj = NULL
        GifWord width, height
        uint8_t[:, ::1] cmap
        uint8_t* srcptr = <uint8_t*> src.data
        ssize_t i, j, imagesize
        int imagecount = 1
        int ret, err = 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        src.dtype == numpy.uint8
        and src.ndim in (2, 3)
        and src.shape[0] < 2 ** 16
        and src.shape[1] < 2 ** 16
        and numpy.PyArray_ISCONTIGUOUS(src)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    if src.ndim > 2:
        imagecount = <int> src.shape[0]
        height = <GifWord> src.shape[1]
        width = <GifWord> src.shape[2]
        imagesize = src.shape[1] * src.shape[2]
    else:
        imagecount = 1
        height = <GifWord> src.shape[0]
        width = <GifWord> src.shape[1]
        imagesize = src.shape[0] * src.shape[1]

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use dynamic memgif?
            dstsize = 2048 + srcsize + 256 * imagecount
        out = _create_output(outtype, dstsize)
        memgif.owner = 0
    else:
        memgif.owner = 0

    dst = out
    dstsize = dst.size

    cmap = _create_array(colormap, (256, 3), numpy.uint8)
    if colormap is None:
        for i in range(256):
            cmap[i, 0] = <uint8_t> i
            cmap[i, 1] = <uint8_t> i
            cmap[i, 2] = <uint8_t> i

    try:
        with nogil:

            memgif.data = <GifByteType*> &dst[0]
            memgif.size = dstsize
            memgif.offset = 0

            gif = EGifOpen(<void*> &memgif, gif_output_func, &err)
            if gif == NULL:
                raise GifError('EGifOpen', err)

            colormapobj = GifMakeMapObject(256, <GifColorType*> &cmap[0, 0])
            if colormapobj == NULL:
                raise MemoryError('failed to allocate ColorMapObject')

            ret = EGifPutScreenDesc(gif, width, height, 256, 0, colormapobj)
            if ret != GIF_OK:
                raise GifError('EGifPutScreenDesc', gif.Error)

            # TODO: save DelayTime and TransparentColor to GifExtension

            # ret = EGifPutComment(gif, b'imagecodecs.py')
            # if ret != GIF_OK:
            #     raise GifError('EGifPutComment', gif.Error)

            for i in range(imagecount):
                ret = EGifPutImageDesc(gif, 0, 0, width, height, 0, NULL)
                if ret != GIF_OK:
                    raise GifError('EGifPutImageDesc', gif.Error)

                for j in range(height):
                    ret = EGifPutLine(
                        gif,
                        <GifByteType*> &srcptr[i * imagesize + j * width],
                        width
                    )
                    if ret != GIF_OK:
                        raise GifError('EGifPutImageDesc', gif.Error)
    finally:
        free(colormapobj)
        ret = EGifCloseFile(gif, &err)
        if ret != GIF_OK:
            raise GifError('EGifCloseFile', err)  # gif.Error ?

    del dst
    return _return_output(out, dstsize, memgif.offset, outgiven)


def gif_decode(data, index=None, asrgb=True, out=None):
    """Decode GIF image to numpy array.

    By default all images in the file are returned in one array.
    If an image index is specified, ignore the disposal mode and return the
    image data on black background.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        GifFileType* gif
        SavedImage* image
        GifImageDesc* descr
        ColorMapObject* colormap
        ExtensionBlock* extblock
        memgif_t memgif
        int ret, err = 0
        int colorcount, transparent, disposal
        ssize_t i, j, k, m, w, h, previous
        ssize_t imagesize, rowsize, imagecount, width, height, samples
        uint8_t background[4]
        uint8_t* palptr
        uint8_t* srcptr
        uint8_t* dstptr
        uint8_t* extptr
        uint8_t color
        bint rgb = asrgb

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            # open image and determie size

            memgif.data = <GifByteType*> &src[0]
            memgif.size = srcsize
            memgif.offset = 0
            memgif.owner = 0

            gif = DGifOpen(<void*> &memgif, gif_input_func, &err)
            if gif == NULL:
                raise GifError('DGifOpen', err)

            ret = DGifSlurp(gif)
            if ret != GIF_OK or gif.SavedImages == NULL or gif.ImageCount <= 0:
                raise GifError('DGifSlurp', gif.Error)

            width = <ssize_t> gif.SWidth
            height = <ssize_t> gif.SHeight
            imagecount = <ssize_t> gif.ImageCount
            image = &gif.SavedImages[0]
            previous = 0

            for i in range(imagecount):
                descr = &gif.SavedImages[i].ImageDesc
                if width < <ssize_t> descr.Width + <ssize_t> descr.Left:
                    width = <ssize_t> descr.Width + <ssize_t> descr.Left
                if height < <ssize_t> descr.Height + <ssize_t> descr.Top:
                    height = <ssize_t> descr.Height + <ssize_t> descr.Top

            if rgb:
                # add alpha channel if first image has transparent pixels
                samples = 3
                transparent = -1
                for j in range(image.ExtensionBlockCount):
                    extblock = &image.ExtensionBlocks[j]
                    if (
                        extblock.Function == 0xf9
                        and extblock.ByteCount > 3
                        and extblock.Bytes[0] & 0x01
                    ):
                        transparent = <int> extblock.Bytes[3]
                        break
                if transparent != -1:
                    srcptr = <uint8_t*> image.RasterBits
                    color = <uint8_t> transparent
                    for j in range(
                        image.ImageDesc.Width * image.ImageDesc.Height
                    ):
                        if srcptr[j] == color:
                            samples = 4
                            break
            else:
                samples = 1

            imagesize = height * width * samples
            rowsize = width * samples

        # create output array

        if index is not None:
            imagesize = 0
            imagecount = index + 1

        if imagecount == 1 or imagesize == 0:
            if samples > 1:
                shape = int(height), int(width), int(samples)
            else:
                shape = int(height), int(width)
        else:
            if samples > 1:
                shape = int(imagecount), int(height), int(width), int(samples)
            else:
                shape = int(imagecount), int(height), int(width)

        out = _create_array(out, shape, numpy.uint8, strides=None, zero=rgb==0)
        dst = out
        dstsize = dst.size
        dstptr = <uint8_t*> &dst.data[0]

        with nogil:

            if rgb:
                # compose image sequence
                previous = 0
                disposal = DISPOSAL_UNSPECIFIED

                background[0] = (gif.SBackGroundColor) & <GifWord> 0xff
                background[1] = (gif.SBackGroundColor >> 8) & <GifWord> 0xff
                background[2] = (gif.SBackGroundColor >> 16) & <GifWord> 0xff
                background[3] = (gif.SBackGroundColor >> 24) & <GifWord> 0xff

                for i in range(imagecount):

                    if i == 0 or imagesize == 0:
                        # initialize frame to background
                        k = i * imagesize
                        for j in range(height * width):
                            dstptr[k] = background[0]
                            k += 1
                            dstptr[k] = background[1]
                            k += 1
                            dstptr[k] = background[2]
                            k += 1
                            if samples == 4:
                                dstptr[k] = background[3]
                                k += 1
                        if disposal == DISPOSAL_UNSPECIFIED:
                            previous = i

                    elif disposal == DISPOSE_BACKGROUND:
                        # restore area of previous frame to background
                        previous = i
                        memcpy(
                            &dstptr[imagesize * i],
                            &dstptr[imagesize * (i - 1)],
                            imagesize
                        )
                        for h in range(descr.Height):
                            k = (
                                imagesize * i +
                                rowsize * (descr.Top + h) +
                                samples * descr.Left
                            )
                            for w in range(descr.Width):
                                dstptr[k] = background[0]
                                k += 1
                                dstptr[k] = background[1]
                                k += 1
                                dstptr[k] = background[2]
                                k += 1
                                if samples == 4:
                                    dstptr[k] = background[3]
                                    k += 1

                    elif (
                        disposal == DISPOSE_DO_NOT or
                        disposal == DISPOSAL_UNSPECIFIED
                    ):
                        # use previous frame as background
                        previous = i - 1
                        memcpy(
                            &dstptr[imagesize * i],
                            &dstptr[imagesize * previous],
                            imagesize
                        )

                    elif disposal == DISPOSE_PREVIOUS:
                        # restore to previous, undisposed frame
                        memcpy(
                            &dstptr[imagesize * i],
                            &dstptr[imagesize * previous],
                            imagesize
                        )

                    else:
                        memset(&dstptr[imagesize * i], 0, imagesize)

                    image = &gif.SavedImages[i]
                    descr = &image.ImageDesc
                    srcptr = <uint8_t*> image.RasterBits

                    # find colormap
                    colormap = descr.ColorMap
                    if colormap == NULL:
                        colormap = gif.SColorMap
                        if colormap == NULL:
                            raise RuntimeError('no colormap')
                    colorcount = colormap.ColorCount
                    palptr = <uint8_t*> colormap.Colors

                    # find TransparentColor and DisposalMode
                    transparent = NO_TRANSPARENT_COLOR
                    disposal = DISPOSAL_UNSPECIFIED
                    if image.ExtensionBlocks != NULL:
                        for j in range(image.ExtensionBlockCount):
                            extblock = &image.ExtensionBlocks[j]
                            if (
                                extblock.Function != 0xf9
                                or extblock.ByteCount != 4
                            ):
                                continue
                            if extblock.Bytes[0] & 0x01:
                                transparent = <int> extblock.Bytes[3]
                            disposal = <int> ((extblock.Bytes[0] >> 2) & 0x07)

                    # paste new image
                    j = 0
                    for h in range(descr.Height):
                        k = (
                            imagesize * i +
                            rowsize * (descr.Top + h) +
                            samples * descr.Left
                        )
                        for w in range(descr.Width):
                            m = <int> srcptr[j]
                            j += 1
                            if m >= colorcount:
                                k += samples
                            elif m != transparent:
                                m *= 3
                                dstptr[k] = palptr[m]
                                k += 1
                                dstptr[k] = palptr[m + 1]
                                k += 1
                                dstptr[k] = palptr[m + 2]
                                k += 1
                                if samples == 4:
                                    dstptr[k] = 255
                                    k += 1
                            else:
                                k += samples

            elif imagesize == 0:
                # copy indices from selected image
                image = &gif.SavedImages[imagecount - 1]
                descr = &image.ImageDesc
                srcptr = <uint8_t*> image.RasterBits
                j = 0
                for h in range(descr.Height):
                    k = rowsize * (descr.Top + h) + samples * descr.Left
                    for w in range(descr.Width):
                        dstptr[k] = srcptr[j]
                        k += 1
                        j += 1

            else:
                # copy indices from all images
                for i in range(imagecount):
                    image = &gif.SavedImages[i]
                    descr = &image.ImageDesc
                    srcptr = <uint8_t*> image.RasterBits
                    j = 0
                    for h in range(descr.Height):
                        k = (
                            imagesize * i
                            + rowsize * (descr.Top + h)
                            + samples * descr.Left
                        )
                        for w in range(descr.Width):
                            dstptr[k] = srcptr[j]
                            k += 1
                            j += 1

    finally:
        ret = DGifCloseFile(gif, &err)
        if ret != GIF_OK:
            raise GifError('DGifCloseFile', err)  # gif.Error?

    return out


ctypedef struct memgif_t:
    GifByteType* data
    ssize_t size
    ssize_t offset
    int owner


cdef int gif_input_func(
    GifFileType* gif,
    GifByteType* dst,
    int size
) nogil:
    """GIF read callback function."""
    cdef:
        memgif_t* memgif = <memgif_t*> gif.UserData

    if memgif == NULL:
        return 0
    if memgif.offset >= memgif.size:
        return 0
    if <ssize_t> size > memgif.size - memgif.offset:
        size = <int> (memgif.size - memgif.offset)
        # raise RuntimeError(f'GIF input stream too small {memgif.size}')
    memcpy(
        <void*> dst,
        <const void*> &memgif.data[memgif.offset],
        size
    )
    memgif.offset += size
    return size


cdef int gif_output_func(
    GifFileType* gif,
    const GifByteType* src,
    int size
) nogil:
    """GIF write callback function."""
    cdef:
        memgif_t* memgif = <memgif_t*> gif.UserData
        ssize_t newsize
        uint8_t* tmp

    if memgif == NULL:
        return 0
    if memgif.offset >= memgif.size:
        return 0
    if <ssize_t> size > memgif.size - memgif.offset:
        # output stream too small; realloc if owner
        if not memgif.owner:
            # raise GifError('OutputFunc', E_GIF_ERR_WRITE_FAILED)
            return 0
        newsize = memgif.offset + size
        if newsize <= <ssize_t> (<double> memgif.size * 1.25):
            # moderate upsize: overallocate
            newsize = newsize + newsize // 4
            newsize = (((newsize - 1) // 4096) + 1) * 4096
        tmp = <uint8_t*> realloc(<void*> memgif.data, newsize)
        if tmp == NULL:
            # raise MemoryError('OutputFunc realloc failed')
            return 0
        memgif.data = tmp
        memgif.size = newsize
    memcpy(
        <void*> &memgif.data[memgif.offset],
        <const void*> src,
        size
    )
    memgif.offset += size
    return size
