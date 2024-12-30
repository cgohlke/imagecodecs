# imagecodecs/_avif.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2020-2025, Christoph Gohlke
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

"""AVIF codec for the imagecodecs package."""

include '_shared.pxi'

from libavif cimport *


class AVIF:
    """AVIF codec constants."""

    available = True

    class PIXEL_FORMAT(enum.IntEnum):
        """AVIF codec pixel formats."""

        NONE = AVIF_PIXEL_FORMAT_NONE
        YUV444 = AVIF_PIXEL_FORMAT_YUV444
        YUV422 = AVIF_PIXEL_FORMAT_YUV422
        YUV420 = AVIF_PIXEL_FORMAT_YUV420
        YUV400 = AVIF_PIXEL_FORMAT_YUV400
        COUNT = AVIF_PIXEL_FORMAT_COUNT

    class QUALITY(enum.IntEnum):
        """AVIF codec quality."""

        DEFAULT = AVIF_QUALITY_DEFAULT  # -1
        LOSSLESS = AVIF_QUALITY_LOSSLESS  # 100
        WORST = AVIF_QUALITY_WORST  # 0
        BEST = AVIF_QUALITY_BEST  # 100

    class SPEED(enum.IntEnum):
        """AVIF codec speeds."""

        DEFAULT = AVIF_SPEED_DEFAULT
        SLOWEST = AVIF_SPEED_SLOWEST
        FASTEST = AVIF_SPEED_FASTEST

    class CHROMA_UPSAMPLING(enum.IntEnum):
        """AVIF codec chroma upsampling types."""

        AUTOMATIC = AVIF_CHROMA_UPSAMPLING_AUTOMATIC
        FASTEST = AVIF_CHROMA_UPSAMPLING_FASTEST
        BEST_QUALITY = AVIF_CHROMA_UPSAMPLING_BEST_QUALITY
        NEAREST = AVIF_CHROMA_UPSAMPLING_NEAREST
        BILINEAR = AVIF_CHROMA_UPSAMPLING_BILINEAR

    class CODEC_CHOICE(enum.IntEnum):
        """AVIF codec choices."""

        AUTO = AVIF_CODEC_CHOICE_AUTO
        AOM = AVIF_CODEC_CHOICE_AOM
        DAV1D = AVIF_CODEC_CHOICE_DAV1D
        LIBGAV1 = AVIF_CODEC_CHOICE_LIBGAV1
        RAV1E = AVIF_CODEC_CHOICE_RAV1E
        SVT = AVIF_CODEC_CHOICE_SVT
        AVM = AVIF_CODEC_CHOICE_AVM


class AvifError(RuntimeError):
    """AVIF codec exceptions."""

    def __init__(self, func, err):
        cdef:
            char* errormessage
            int errorcode

        try:
            errorcode = int(err)
            errormessage = <char*> avifResultToString(<avifResult> errorcode)
            if errormessage == NULL:
                raise RuntimeError('avifResultToString returned NULL')
            msg = errormessage.decode()
        except Exception:
            msg = 'NULL' if err is None else f'unknown error {err!r}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def avif_version():
    """Return libavif library version string."""
    return 'libavif {}.{}.{}'.format(
        AVIF_VERSION_MAJOR, AVIF_VERSION_MINOR, AVIF_VERSION_PATCH
    )


def avif_check(const uint8_t[::1] data):
    """Return whether data is AVIF encoded image."""
    cdef:
        bytes sig = bytes(data[4:12])

    if sig == b'ftypavif':
        return True
    if sig[:4] == b'ftyp':
        return None
    return False


def avif_encode(
    data,
    level=None,
    speed=None,
    tilelog2=None,
    bitspersample=None,
    pixelformat=None,
    codec=None,
    numthreads=None,
    out=None
):
    """Return AVIF encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, size
        ssize_t itemsize = data.dtype.itemsize
        int quality = AVIF_QUALITY_LOSSLESS
        int speed_ = AVIF_SPEED_DEFAULT
        int tilerowslog2 = 0
        int tilecolslog2 = 0
        # int duration = 1
        int timescale = 1
        int keyframeinterval = 0
        int maxthreads = <int> _default_threads(numthreads)
        uint32_t imagecount, width, height, samples, depth
        ssize_t i, j, k, srcindex
        size_t rawsize
        bint monochrome = 0  # must be initialized
        bint hasalpha = 0
        uint8_t temp
        uint8_t* dstptr = NULL
        uint8_t* srcptr = NULL
        avifEncoder* encoder = NULL
        avifImage* image = NULL
        avifRGBImage rgb
        avifRWData raw  # = AVIF_DATA_EMPTY
        avifPixelFormat yuvformat = AVIF_PIXEL_FORMAT_YUV444
        avifAddImageFlags flags = AVIF_ADD_IMAGE_FLAG_NONE
        avifCodecChoice codecchoice = AVIF_CODEC_CHOICE_AUTO
        avifResult res

    if not (
        src.dtype in {numpy.uint8, numpy.uint16}
        # and numpy.PyArray_ISCONTIGUOUS(src)
        and src.ndim in {2, 3, 4}
        and src.shape[0] <= 2147483647
        and src.shape[1] <= 2147483647
        and src.shape[src.ndim - 1] <= 2147483647
        and src.shape[src.ndim - 2] <= 2147483647
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    if src.ndim == 2:
        samples = 1
        imagecount = 1
        height = <uint32_t> src.shape[0]
        width = <uint32_t> src.shape[1]
    elif src.ndim == 4:
        imagecount = <uint32_t> src.shape[0]
        height = <uint32_t> src.shape[1]
        width = <uint32_t> src.shape[2]
        samples = <uint32_t> src.shape[3]
        if samples > 4:
            raise ValueError('invalid number of samples')
    elif src.shape[src.ndim - 1] < 5:
        imagecount = 1
        height = <uint32_t> src.shape[0]
        width = <uint32_t> src.shape[1]
        samples = <uint32_t> src.shape[2]
    else:
        samples = 1
        imagecount = <uint32_t> src.shape[0]
        height = <uint32_t> src.shape[1]
        width = <uint32_t> src.shape[2]

    monochrome = samples < 3
    hasalpha = samples == 2 or samples == 4

    if bitspersample is None:
        depth = <uint32_t> itemsize * 8
    else:
        depth = bitspersample
        if depth not in {8, 10, 12, 16} or (depth == 8 and itemsize == 2):
            raise ValueError('invalid bitspersample')

    if tilelog2 is not None:
        tilecolslog2, tilerowslog2 = tilelog2
        if 0 <= tilerowslog2 <= 6:
            raise ValueError('invalid tileRowsLog2')
        if 0 <= tilecolslog2 <= 6:
            raise ValueError('invalid tileColsLog2')

    quality = _default_value(
        level,
        AVIF_QUALITY_LOSSLESS,  # 100
        AVIF_QUALITY_DEFAULT,  # -1
        AVIF_QUALITY_BEST  # 100
    )

    speed_ = _default_value(
        speed, AVIF_SPEED_DEFAULT, AVIF_SPEED_SLOWEST, AVIF_SPEED_FASTEST
    )

    if codec is not None:
        codecchoice = _avif_codecchoice(codec)

    if monochrome:
        yuvformat = AVIF_PIXEL_FORMAT_YUV400
        quality = AVIF_QUALITY_LOSSLESS
    elif quality == AVIF_QUALITY_LOSSLESS:
        yuvformat = AVIF_PIXEL_FORMAT_YUV444
    elif pixelformat is not None:
        yuvformat = _avif_pixelformat(pixelformat)

    try:
        with nogil:
            raw.data = NULL
            raw.size = 0

            encoder = avifEncoderCreate()
            if encoder == NULL:
                raise AvifError('avifEncoderCreate', 'NULL')

            encoder.quality = quality
            encoder.qualityAlpha = AVIF_QUALITY_LOSSLESS
            encoder.maxThreads = maxthreads
            encoder.tileRowsLog2 = tilerowslog2
            encoder.tileColsLog2 = tilecolslog2
            encoder.speed = speed_
            encoder.keyframeInterval = keyframeinterval
            encoder.timescale = timescale

            image = avifImageCreate(width, height, depth, yuvformat)
            if image == NULL:
                raise AvifError('avifImageCreate', 'NULL')

            if monochrome or quality == AVIF_QUALITY_LOSSLESS:
                if codecchoice == AVIF_CODEC_CHOICE_AUTO:
                    encoder.codecChoice = AVIF_CODEC_CHOICE_AOM
                else:
                    encoder.codecChoice = codecchoice
                image.matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_IDENTITY
            else:
                encoder.codecChoice = codecchoice
                image.matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT601

            image.yuvRange = AVIF_RANGE_FULL
            # image.colorPrimaries = AVIF_COLOR_PRIMARIES_UNSPECIFIED
            # image.transferCharacteristics = (
            #     AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED
            # )

            avifRGBImageSetDefaults(&rgb, image)
            if monochrome:
                res = avifRGBImageAllocatePixels(&rgb)
                if res != AVIF_RESULT_OK:
                    raise AvifError('avifRGBImageAllocatePixels', res)
                if rgb.format != AVIF_RGB_FORMAT_RGBA:
                    raise RuntimeError('rgb.format != AVIF_RGB_FORMAT_RGBA')
                srcptr = <uint8_t *> src.data
                size = 0
            else:
                rgb.format = (
                    AVIF_RGB_FORMAT_RGBA if hasalpha else AVIF_RGB_FORMAT_RGB
                )
                rgb.depth = depth
                rgb.pixels = <uint8_t *> src.data
                rgb.rowBytes = <uint32_t> (width * samples * itemsize)
                size = height * width * samples * itemsize

            if imagecount == 1:
                flags = AVIF_ADD_IMAGE_FLAG_SINGLE

            srcindex = 0
            for i in range(imagecount):

                if monochrome:
                    # TODO: do not copy array to avifRGBImage first
                    dstptr = <uint8_t *> rgb.pixels
                    if itemsize == 1:
                        # uint8
                        if hasalpha:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    temp = srcptr[srcindex]
                                    dstptr[k] = temp
                                    dstptr[k + 1] = temp
                                    dstptr[k + 2] = temp
                                    dstptr[k + 3] = srcptr[srcindex + 1]
                                    k += 4
                                    srcindex += 2
                        else:
                            rgb.ignoreAlpha = AVIF_TRUE
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    temp = srcptr[srcindex]
                                    dstptr[k] = temp
                                    dstptr[k + 1] = temp
                                    dstptr[k + 2] = temp
                                    dstptr[k + 3] = 0
                                    k += 4
                                    srcindex += 1
                    else:
                        # uint16
                        if hasalpha:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    temp = srcptr[srcindex]
                                    dstptr[k] = temp
                                    dstptr[k + 2] = temp
                                    dstptr[k + 4] = temp
                                    temp = srcptr[srcindex + 1]
                                    dstptr[k + 1] = temp
                                    dstptr[k + 3] = temp
                                    dstptr[k + 5] = temp
                                    dstptr[k + 6] = srcptr[srcindex + 2]
                                    dstptr[k + 7] = srcptr[srcindex + 3]
                                    k += 8
                                    srcindex += 4
                        else:
                            rgb.ignoreAlpha = AVIF_TRUE
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    temp = srcptr[srcindex]
                                    dstptr[k] = temp
                                    dstptr[k + 2] = temp
                                    dstptr[k + 4] = temp
                                    temp = srcptr[srcindex + 1]
                                    dstptr[k + 1] = temp
                                    dstptr[k + 3] = temp
                                    dstptr[k + 5] = temp
                                    dstptr[k + 6] = 0
                                    dstptr[k + 7] = 0
                                    k += 8
                                    srcindex += 2

                res = avifImageRGBToYUV(image, &rgb)
                if res != AVIF_RESULT_OK:
                    raise AvifError('avifImageRGBToYUV', res)

                res = avifEncoderAddImage(encoder, image, 1, flags)
                if res != AVIF_RESULT_OK:
                    raise AvifError('avifEncoderAddImage', res)

                if not monochrome:
                    rgb.pixels += size

            res = avifEncoderFinish(encoder, &raw)
            if res != AVIF_RESULT_OK:
                raise AvifError('avifEncoderFinish', res)

    except Exception:
        avifRWDataFree(&raw)
        raise

    finally:
        if encoder != NULL:
            avifEncoderDestroy(encoder)
        if image != NULL:
            avifImageDestroy(image)
        if monochrome:
            avifRGBImageFreePixels(&rgb)

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = <ssize_t> raw.size
        elif <size_t> dstsize < raw.size:
            raise ValueError('output too small')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if <size_t> dstsize < raw.size:
        raise ValueError('output too small')
    memcpy(<void*> &dst[0], <void*> raw.data, raw.size)

    rawsize = raw.size
    avifRWDataFree(&raw)

    del dst
    return _return_output(out, dstsize, rawsize, outgiven)


def avif_decode(data, index=None, numthreads=None, out=None):
    """Return decoded AVIF image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t frameindex = -1 if index is None else index
        ssize_t samples, size, itemsize, i, j, k, dstindex, imagecount
        bint monochrome = 0  # must be initialized
        bint hasalpha = 0
        int maxthreads = <int> _default_threads(numthreads)
        uint8_t* dstptr = NULL
        uint8_t* srcptr = NULL
        avifDecoder* decoder = NULL
        avifImage* image = NULL
        avifRGBImage rgb
        avifResult res

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            decoder = avifDecoderCreate()
            if decoder == NULL:
                raise AvifError('avifDecoderCreate', 'NULL')

            # required to read AVIF files created by ImageMagick
            decoder.strictFlags = AVIF_STRICT_DISABLED

            decoder.maxThreads = maxthreads

            res = avifDecoderSetSource(decoder, AVIF_DECODER_SOURCE_AUTO)
            if res != AVIF_RESULT_OK:
                raise AvifError('avifDecoderSetSource', res)

            res = avifDecoderSetIOMemory(
                decoder,
                <const uint8_t*> &src[0],
                <size_t> srcsize
            )
            if res != AVIF_RESULT_OK:
                raise AvifError('avifDecoderSetIOMemory', res)

            res = avifDecoderParse(decoder)
            if res != AVIF_RESULT_OK:
                raise AvifError('avifDecoderParse', res)

            image = decoder.image

            imagecount = decoder.imageCount

            if frameindex >= imagecount:
                raise IndexError(f'{frameindex=} out of range {imagecount=}')

            if imagecount == 1:
                frameindex = 0

            res = avifDecoderNthImage(
                decoder, <uint32_t> (frameindex if frameindex > 0 else 0)
            )
            if res != AVIF_RESULT_OK:
                raise AvifError('avifDecoderNthImage', res)

            hasalpha = image.alphaPlane != NULL and image.alphaRowBytes > 0
            monochrome = (
                image.yuvFormat == AVIF_PIXEL_FORMAT_YUV400
                and image.yuvPlanes[<ssize_t> AVIF_CHAN_U] == NULL
                and image.yuvPlanes[<ssize_t> AVIF_CHAN_V] == NULL
            )

            avifRGBImageSetDefaults(&rgb, image)

            if monochrome:
                samples = 2 if hasalpha else 1
                avifRGBImageAllocatePixels(&rgb)
                if rgb.format != AVIF_RGB_FORMAT_RGBA:
                    raise RuntimeError('rgb.format != AVIF_RGB_FORMAT_RGBA')
            elif hasalpha:
                samples = 4
                rgb.format = AVIF_RGB_FORMAT_RGBA
            else:
                samples = 3
                rgb.format = AVIF_RGB_FORMAT_RGB

        dtype = numpy.dtype('uint8' if image.depth <= 8 else 'uint16')
        itemsize = dtype.itemsize

        shape = int(image.height), int(image.width)
        if samples > 1:
            shape = shape + (int(samples),)
        if imagecount > 1 and frameindex < 0:
            shape = (int(imagecount),) + shape

        out = _create_array(out, shape, dtype)
        dst = out

        if monochrome:
            dstptr = <uint8_t *> dst.data
        else:
            rgb.pixels = <uint8_t *> dst.data
            rgb.rowBytes = <uint32_t> (rgb.width * samples * itemsize)
            size = image.height * image.width * samples * itemsize

        with nogil:
            dstindex = 0
            for i in range(imagecount):
                # TODO: verify that images have same shape and dtype
                if i > 0:
                    res = avifDecoderNextImage(decoder)
                    if res == AVIF_RESULT_NO_IMAGES_REMAINING:
                        break
                    if res != AVIF_RESULT_OK:
                        raise AvifError('avifDecoderNextImage', res)

                res = avifImageYUVToRGB(decoder.image, &rgb)
                if res != AVIF_RESULT_OK:
                    raise AvifError('avifImageYUVToRGB', res)

                if monochrome:
                    # TODO: do not to decode to avifRGBImage first
                    # Copying Y and A directly is not sufficient
                    srcptr = <uint8_t *> rgb.pixels
                    if itemsize == 1:
                        # uint8
                        if hasalpha:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    # red
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 3
                                    # alpha
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 1
                        else:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 4  # requires RGBA buffer
                    else:
                        # uint16
                        if hasalpha:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    # red
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 1
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 5
                                    # alpha
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 1
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 1
                        else:
                            for j in range(<ssize_t> rgb.height):
                                k = j * <ssize_t> rgb.rowBytes
                                while k < (j + 1) * <ssize_t> rgb.rowBytes:
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 1
                                    dstptr[dstindex] = srcptr[k]
                                    dstindex += 1
                                    k += 7  # requires RGBA buffer
                else:
                    rgb.pixels += size
    finally:
        if decoder != NULL:
            avifDecoderDestroy(decoder)
        if monochrome:
            avifRGBImageFreePixels(&rgb)

    return out


cdef _avif_pixelformat(pixelformat):
    """Return AVIF colorspace value from user input."""
    return {
        AVIF_PIXEL_FORMAT_NONE: AVIF_PIXEL_FORMAT_NONE,
        AVIF_PIXEL_FORMAT_YUV444: AVIF_PIXEL_FORMAT_YUV444,
        AVIF_PIXEL_FORMAT_YUV422: AVIF_PIXEL_FORMAT_YUV422,
        AVIF_PIXEL_FORMAT_YUV420: AVIF_PIXEL_FORMAT_YUV420,
        AVIF_PIXEL_FORMAT_YUV400: AVIF_PIXEL_FORMAT_YUV400,
        'none': AVIF_PIXEL_FORMAT_NONE,
        'yuv444': AVIF_PIXEL_FORMAT_YUV444,
        'yuv422': AVIF_PIXEL_FORMAT_YUV422,
        'yuv420': AVIF_PIXEL_FORMAT_YUV420,
        'yuv400': AVIF_PIXEL_FORMAT_YUV400,
        '444': AVIF_PIXEL_FORMAT_YUV444,
        '422': AVIF_PIXEL_FORMAT_YUV422,
        '420': AVIF_PIXEL_FORMAT_YUV420,
        '400': AVIF_PIXEL_FORMAT_YUV400,
    }[pixelformat]  # .get(pixelformat, AVIF_PIXEL_FORMAT_YUV444)


cdef _avif_codecchoice(codec):
    """Return AVIF codecchoice value from user input."""
    return {
        AVIF_CODEC_CHOICE_AUTO: AVIF_CODEC_CHOICE_AUTO,
        AVIF_CODEC_CHOICE_AOM: AVIF_CODEC_CHOICE_AOM,
        AVIF_CODEC_CHOICE_DAV1D: AVIF_CODEC_CHOICE_DAV1D,
        AVIF_CODEC_CHOICE_LIBGAV1: AVIF_CODEC_CHOICE_LIBGAV1,
        AVIF_CODEC_CHOICE_RAV1E: AVIF_CODEC_CHOICE_RAV1E,
        AVIF_CODEC_CHOICE_SVT: AVIF_CODEC_CHOICE_SVT,
        AVIF_CODEC_CHOICE_AVM: AVIF_CODEC_CHOICE_AVM,
        'auto': AVIF_CODEC_CHOICE_AUTO,
        'aom': AVIF_CODEC_CHOICE_AOM,
        'dav1d': AVIF_CODEC_CHOICE_DAV1D,
        'libgav1': AVIF_CODEC_CHOICE_LIBGAV1,
        'rav1e': AVIF_CODEC_CHOICE_RAV1E,
        'svt': AVIF_CODEC_CHOICE_SVT,
        'avm': AVIF_CODEC_CHOICE_AVM,
    }[codec]
