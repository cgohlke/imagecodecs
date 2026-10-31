# imagecodecs/_gif.pyx
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
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""GIF (Graphics Interchange Format) codec for the imagecodecs package."""

include '_shared.pxi'

from giflib cimport *


class GIF:
    """GIF codec constants."""

    available = True


class GifError(RuntimeError):
    """GIF codec exceptions."""

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


def gif_check(const uint8_t[::1] data not None, /):
    """Return whether data is GIF encoded image or None if unknown."""
    return _startswith(data, b'GIF87a') or _startswith(data, b'GIF89a')


def gif_encode(
    data,
    /,
    *,
    colormap=None,
    out=None,
):
    """Return GIF encoded image."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t[:, ::1] cmap
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        memgif_t memgif
        GifFileType* gif = NULL
        ColorMapObject* colormapobj = NULL
        GifWord width, height
        uint8_t* srcptr = <uint8_t*> src.data
        ssize_t i, j, imagesize
        int imagecount = 1
        int ret, err = 0
        bint success = False
        imagelayout_t layout

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_FRAMES,
        src.ndim,
        src.shape,
        src.dtype,
        IC_PHOTO_GRAY,  # photometric (paletted, single channel)
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (
        layout.height > 0
        and layout.width > 0
        and layout.height < 65536
        and layout.width < 65536
    ):
        raise ValueError('invalid data shape or dtype')

    imagecount = <int> layout.frames
    height = <GifWord> layout.height
    width = <GifWord> layout.width
    imagesize = layout.height * layout.width

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # LZW worst case: 12-bit code per pixel plus sub-block overhead
            dstsize = (
                2048 + 256 * imagecount + srcsize + srcsize // 2
                + (srcsize // 64)
            )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    if colormap is None:
        colormap = numpy.repeat(
            numpy.arange(256, dtype=numpy.uint8).reshape(256, 1), 3, axis=1
        )
    cmap = numpy.ascontiguousarray(colormap, numpy.uint8).reshape(256, 3)

    try:
        with nogil:
            memgif.data = <GifByteType*> dst._data
            memgif.size = dstsize
            memgif.offset = 0
            memgif.owner = 0

            gif = EGifOpen(<void*> &memgif, gif_output_func, &err)
            if gif == NULL:
                raise GifError('EGifOpen', err)

            colormapobj = GifMakeMapObject(
                256, <const GifColorType*> &cmap[0, 0]
            )
            if colormapobj == NULL:
                raise MemoryError('failed to allocate ColorMapObject')

            ret = EGifPutScreenDesc(gif, width, height, 8, 0, colormapobj)
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
                        raise GifError('EGifPutLine', gif.Error)
        success = True
    finally:
        GifFreeMapObject(colormapobj)
        if gif != NULL:
            ret = EGifCloseFile(gif, &err)
            if ret != GIF_OK and success:
                raise GifError('EGifCloseFile', err)

    del dst
    return _return_output(out, dstsize, memgif.offset, outgiven)


def gif_decode(
    data,
    /,
    index=None,
    *,
    asrgb=True,
    out=None,
    bint _info=False,
):
    """Return decoded GIF image.

    By default, all images in the file are returned in one array.
    If an image index is specified, ignore the disposal mode and return the
    image data on background.
    An alpha channel is added only if the first image has transparent pixels.
    Otherwise, transparent and disposed areas are filled with the opaque
    background color.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        GifFileType* gif = NULL
        SavedImage* image
        GifImageDesc* descr
        ColorMapObject* colormap
        ExtensionBlock* extblock
        GifColorType* bgcolor
        memgif_t memgif
        int ret, err = 0
        int colorcount, transparent, disposal
        ssize_t i, j, k, m, h, imageindex, istart
        ssize_t imagesize, rowsize, imagecount, width, height, samples
        uint8_t[4] background
        uint8_t* palptr
        uint8_t* srcptr
        uint8_t* dstptr
        uint8_t* saved = NULL  # canvas to restore for DISPOSE_PREVIOUS
        uint8_t color
        bint rgb = asrgb
        bint success = False

    if data is out:
        raise ValueError('cannot transform in-place')

    try:
        with nogil:
            # open image and determine size
            memgif.data = <GifByteType*> src._data
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
                        and extblock.ByteCount == 4
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
            imageindex = index
            if imageindex < 0 or imageindex >= imagecount:
                raise IndexError(
                    f'frame index={imageindex} out of range [0, {imagecount})'
                )
            imagesize = 0
            imagecount = imageindex + 1

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

        if _info:
            colormap_array = None
            if gif.SColorMap != NULL and gif.SColorMap.ColorCount > 0:
                # slicing copies to bytes; Colors is freed by DGifCloseFile
                colormap_array = numpy.frombuffer(
                    (<char*> gif.SColorMap.Colors)[
                        :gif.SColorMap.ColorCount * 3
                    ],
                    numpy.uint8,
                ).reshape(-1, 3).copy()
            success = True
            return {
                '_codec': 'gif',
                'shape': shape,
                'dtype': numpy.dtype(numpy.uint8),
                'frames': imagecount,
                'colormap': colormap_array,
            }

        out = _create_array(out, shape, numpy.uint8, strides=None, zero=rgb==0)
        dst = out
        dstptr = <uint8_t*> dst.data

        with nogil:
            if rgb:
                # compose image sequence
                disposal = DISPOSAL_UNSPECIFIED

                if (
                    gif.SColorMap != NULL
                    and gif.SBackGroundColor < gif.SColorMap.ColorCount
                ):
                    bgcolor = &gif.SColorMap.Colors[gif.SBackGroundColor]
                    background[0] = bgcolor.Red
                    background[1] = bgcolor.Green
                    background[2] = bgcolor.Blue
                    background[3] = 0 if samples == 4 else 255
                else:
                    background[0] = 0
                    background[1] = 0
                    background[2] = 0
                    background[3] = 0 if samples == 4 else 255

                # with index, only the selected image is drawn on background
                istart = imagecount - 1 if imagesize == 0 else 0
                for i in range(istart, imagecount):

                    if i == 0 or imagesize == 0:
                        # initialize frame to background
                        k = i * imagesize
                        for _j in range(height * width):
                            dstptr[k] = background[0]
                            k += 1
                            dstptr[k] = background[1]
                            k += 1
                            dstptr[k] = background[2]
                            k += 1
                            if samples == 4:
                                dstptr[k] = background[3]
                                k += 1

                    elif disposal == DISPOSE_BACKGROUND:
                        # restore area of previous frame to background
                        memcpy(
                            <void*> &dstptr[imagesize * i],
                            <const void*> &dstptr[imagesize * (i - 1)],
                            imagesize
                        )
                        for h in range(descr.Height):
                            k = (
                                imagesize * i +
                                rowsize * (descr.Top + h) +
                                samples * descr.Left
                            )
                            for _w in range(descr.Width):
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
                        memcpy(
                            <void*> &dstptr[imagesize * i],
                            <const void*> &dstptr[imagesize * (i - 1)],
                            imagesize
                        )

                    elif disposal == DISPOSE_PREVIOUS:
                        # restore canvas from before previous frame was drawn
                        memcpy(
                            <void*> &dstptr[imagesize * i],
                            <const void*> saved,
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
                            raise GifError('gif.SColorMap', None)
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
                    if disposal == DISPOSE_PREVIOUS and imagesize > 0:
                        if saved == NULL:
                            saved = <uint8_t*> malloc(imagesize)
                            if saved == NULL:
                                raise MemoryError('failed to allocate buffer')
                        memcpy(
                            <void*> saved,
                            <const void*> &dstptr[imagesize * i],
                            imagesize
                        )

                    # paste new image
                    j = 0
                    for h in range(descr.Height):
                        k = (
                            imagesize * i +
                            rowsize * (descr.Top + h) +
                            samples * descr.Left
                        )
                        for _w in range(descr.Width):
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
                    for _w in range(descr.Width):
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
                        for _w in range(descr.Width):
                            dstptr[k] = srcptr[j]
                            k += 1
                            j += 1
        success = True
    finally:
        free(saved)
        if gif != NULL:
            ret = DGifCloseFile(gif, &err)
            if ret != GIF_OK and success:
                raise GifError('DGifCloseFile', err)

    return out


def gif_info(
    data,
    /,
    index=None,
    *,
    asrgb=True,
):
    """Return information about GIF image."""
    return gif_decode(data, index=index, asrgb=asrgb, _info=True)


ctypedef struct memgif_t:
    GifByteType* data
    ssize_t size
    ssize_t offset
    int owner  # data was allocated with malloc and may be reallocated


cdef int gif_input_func(
    GifFileType* gif,
    GifByteType* dst,
    int size,
) noexcept nogil:
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
    int size,
) noexcept nogil:
    """GIF write callback function."""
    cdef:
        memgif_t* memgif = <memgif_t*> gif.UserData
        GifByteType* tmp
        ssize_t newsize

    if memgif == NULL or size < 0:
        return 0
    if <ssize_t> size > memgif.size - memgif.offset:
        if not memgif.owner:
            # raise GifError('OutputFunc', E_GIF_ERR_WRITE_FAILED)
            return 0
        # over-allocate to avoid repeated reallocations
        newsize = memgif.offset + size
        newsize = _align_ssize_t(newsize + newsize // 4)
        tmp = <GifByteType*> realloc(<void*> memgif.data, newsize)
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
