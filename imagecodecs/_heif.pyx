# imagecodecs/_heif.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2022-2025, Christoph Gohlke
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

"""HEIF codec for the imagecodecs package.

Libheif does not currently support image sequences/tracks, or bursts.
This implementation reads and writes sequences of top level images only.

"""

include '_shared.pxi'

from libheif cimport *


class HEIF:
    """HEIF codec constants."""

    available = True

    class COMPRESSION(enum.IntEnum):
        """HEIF codec compression levels."""

        UNDEFINED = heif_compression_undefined
        HEVC = heif_compression_HEVC  # H.265
        AVC = heif_compression_AVC  # H.264
        JPEG = heif_compression_JPEG
        AV1 = heif_compression_AV1
        VVC = heif_compression_VVC
        EVC = heif_compression_EVC
        JPEG2000 = heif_compression_JPEG2000
        UNCOMPRESSED = heif_compression_uncompressed
        MASK = heif_compression_mask

    class COLORSPACE(enum.IntEnum):
        """HEIF codec color spaces."""

        UNDEFINED = heif_colorspace_undefined
        YCBCR = heif_colorspace_YCbCr
        RGB = heif_colorspace_RGB
        MONOCHROME = heif_colorspace_monochrome


class HeifError(RuntimeError):
    """HEIF codec exceptions."""

    def __init__(self, func, const char* message):
        msg = f'{func} returned {message.decode()!r}'
        super().__init__(msg)


def heif_version():
    """Return libheif library version string."""
    return 'libheif ' + heif_get_version().decode()


def heif_check(const uint8_t[::1] data):
    """Return whether data is HEIF encoded image."""
    cdef:
        heif_filetype_result result

    if data.size < 12:
        return False
    result = heif_check_filetype(<const uint8_t*> &data[0], 12)
    if result == heif_filetype_no:
        return False
    if result == heif_filetype_yes_unsupported:
        return False
    if result == heif_filetype_yes_supported:
        return True
    if result == heif_filetype_maybe:
        return None
    return False


def heif_encode(
    data,
    level=None,
    bitspersample=None,
    photometric=None,
    compression=None,
    out=None
):
    """Return HEIF encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t itemsize = data.dtype.itemsize
        ssize_t dstsize
        ssize_t srcindex
        ssize_t height, width, samples, depth, row, rowsize, imagecount
        output_t* compressed = NULL
        heif_context* context = NULL
        heif_encoder* encoder = NULL
        heif_image* image = NULL
        heif_encoding_options* options = NULL
        heif_color_profile_nclx* nclx = NULL
        heif_compression_format compression_format
        heif_colorspace colorspace
        heif_writer writer
        heif_error err
        int lossless = 1 if (level is None or level > 100) else 0
        int quality = _default_value(level, 90, 0, 100)
        int stride, ystride, astride
        bint monochrome, hasalpha
        uint8_t* srcptr = NULL
        uint8_t* dstptr = NULL
        uint8_t* yptr = NULL
        uint8_t* aptr = NULL
        uint16_t* src2ptr = NULL
        uint16_t* y2ptr = NULL
        uint16_t* a2ptr = NULL

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

    compression_format = _heif_compression(compression)
    colorspace = _heif_photometric(photometric)

    # TODO: encode extrasamples as aux images
    if src.ndim == 2:
        imagecount = 1
        height = src.shape[0]
        width = src.shape[1]
        samples = 1
    elif src.ndim == 3:
        if src.shape[2] > 4 or colorspace == heif_colorspace_monochrome:
            imagecount = src.shape[0]
            height = src.shape[1]
            width = src.shape[2]
            samples = 1
        else:
            imagecount = 1
            height = src.shape[0]
            width = src.shape[1]
            samples = src.shape[2]
    elif src.ndim == 4:
        imagecount = src.shape[0]
        height = src.shape[1]
        width = src.shape[2]
        samples = src.shape[3]
    else:
        raise ValueError(f'{src.ndim} dimensions not supported')

    monochrome = samples < 3
    hasalpha = samples in {2, 4}

    if bitspersample is None:
        depth = 8 if itemsize == 1 else 10
    else:
        depth = bitspersample
        if (
            depth not in {8, 10, 12}
            or (depth == 8 and itemsize == 2)
            or (depth > 8 and itemsize == 1)
        ):
            raise ValueError('invalid bitspersample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        compressed = output_new(<uint8_t*> &dst[0], dstsize)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.nbytes
        compressed = output_new(<uint8_t*> &dst[0], dstsize)
    else:
        compressed = output_new(
            NULL,
            max(
                <ssize_t> 32768,
                src.size // <ssize_t> (4 if lossless != 0 else 16)
            )
        )
    if compressed == NULL:
        raise MemoryError('output_new failed')

    srcptr = <uint8_t*> src.data

    try:
        with nogil:
            rowsize = width * samples * itemsize

            context = heif_context_alloc()
            if context == NULL:
                raise HeifError('heif_context_alloc', b'NULL')

            err = heif_context_get_encoder_for_format(
                context, compression_format, &encoder
            )
            if err.code != 0:
                raise HeifError(
                    'heif_context_get_encoder_for_format', err.message
                )

            # heif_encoder_set_logging_level(encoder, 4)

            # disable color subsampling
            # TODO: encode monochrome in chroma 400?
            if not monochrome:
                err = heif_encoder_set_parameter(encoder, 'chroma', '444')
                if err.code != 0:
                    raise HeifError('heif_encoder_set_parameter', err.message)

            if lossless != 0:
                err = heif_encoder_set_lossless(encoder, lossless)
                if err.code != 0:
                    raise HeifError('heif_encoder_set_lossless', err.message)

                if not monochrome:
                    # encode in RGB color-space
                    nclx = heif_nclx_color_profile_alloc()
                    if nclx == NULL:
                        raise HeifError(
                            'heif_nclx_color_profile_alloc', b'NULL'
                        )
                    nclx.matrix_coefficients = heif_matrix_coefficients_RGB_GBR

                    options = heif_encoding_options_alloc()
                    if options == NULL:
                        raise HeifError('heif_encoding_options_alloc', b'NULL')
                    options.output_nclx_profile = nclx

            else:
                err = heif_encoder_set_lossy_quality(encoder, quality)
                if err.code != 0:
                    raise HeifError(
                        'heif_encoder_set_lossy_quality', err.message
                    )

            if monochrome:
                colorspace = heif_colorspace_monochrome
                chroma = heif_chroma_monochrome
            elif samples == 3:
                colorspace = heif_colorspace_RGB
                if depth == 8:
                    chroma = heif_chroma_interleaved_RGB
                else:
                    chroma = heif_chroma_interleaved_RRGGBB_LE
            elif depth == 8:
                colorspace = heif_colorspace_RGB
                chroma = heif_chroma_interleaved_RGBA
            else:
                colorspace = heif_colorspace_RGB
                chroma = heif_chroma_interleaved_RRGGBBAA_LE

            srcindex = 0
            for imageindex in range(imagecount):

                err = heif_image_create(
                    <int> width,
                    <int> height,
                    colorspace,
                    chroma,
                    &image
                )
                if err.code != 0:
                    raise HeifError('heif_image_create', err.message)

                if monochrome:
                    err = heif_image_add_plane(
                        image,
                        heif_channel_Y,
                        <int> width,
                        <int> height,
                        <int> depth
                    )
                    if err.code != 0:
                        raise HeifError('heif_image_add_plane', err.message)

                    yptr = heif_image_get_plane(
                        image, heif_channel_Y, &ystride
                    )
                    if yptr == NULL:
                        raise HeifError('heif_image_get_plane', b'NULL')

                    if hasalpha:
                        err = heif_image_add_plane(
                            image,
                            heif_channel_Alpha,
                            <int> width,
                            <int> height,
                            <int> depth
                        )
                        if err.code != 0:
                            raise HeifError(
                                'heif_image_add_plane', err.message
                            )

                        aptr = heif_image_get_plane(
                            image, heif_channel_Alpha, &astride
                        )
                        if aptr == NULL:
                            raise HeifError('heif_image_get_plane', b'NULL')

                        # if planar:
                        #    TODO: handle planar input

                        if itemsize == 1:
                            for row in range(height):
                                for col in range(width):
                                    yptr[col] = srcptr[srcindex]
                                    srcindex += 1
                                    aptr[col] = srcptr[srcindex]
                                    srcindex += 1
                                yptr += ystride
                                aptr += astride
                        else:
                            src2ptr = <uint16_t*> srcptr
                            for row in range(height):
                                y2ptr = <uint16_t*> yptr
                                a2ptr = <uint16_t*> aptr
                                for col in range(width):
                                    y2ptr[col] = src2ptr[srcindex]
                                    srcindex += 1
                                    a2ptr[col] = src2ptr[srcindex]
                                    srcindex += 1
                                yptr += ystride
                                aptr += astride

                    else:
                        for row in range(height):
                            memcpy(
                                <void*> yptr,
                                <const void*> srcptr,
                                rowsize
                            )
                            yptr += ystride
                            srcptr += rowsize

                # elif planar:
                #    TODO: handle planar input

                else:
                    # interleaved RGB(A)
                    err = heif_image_add_plane(
                        image,
                        heif_channel_interleaved,
                        <int> width,
                        <int> height,
                        <int> depth
                    )
                    if err.code != 0:
                        raise HeifError('heif_image_add_plane', err.message)

                    dstptr = heif_image_get_plane(
                        image,
                        heif_channel_interleaved,
                        &stride
                    )
                    if dstptr == NULL:
                        raise HeifError('heif_image_get_plane', b'NULL')

                    # assert stride >= <int> rowsize:
                    for row in range(height):
                        memcpy(
                            <void*> dstptr, <const void*> srcptr, rowsize
                        )
                        dstptr += stride
                        srcptr += rowsize

                err = heif_context_encode_image(
                    context, image, encoder, options, NULL  # &handle
                )
                if err.code != 0:
                    raise HeifError(
                        'heif_context_encode_image', err.message
                    )

                heif_image_release(image)
                image = NULL

            # heif_context_write_to_file(context, '_test.heic')

            writer.writer_api_version = 1
            writer.write = heif_write_callback
            err = heif_context_write(
                context,
                &writer,
                <void*> compressed
            )
            if err.code != 0:
                raise HeifError('heif_context_write', err.message)

        if compressed.owner:
            out = _create_output(
                out, compressed.used, <const char *> compressed.data
            )
        else:
            out = _return_output(
                out, compressed.size, compressed.used, outgiven
            )

    finally:
        if encoder != NULL:
            heif_encoder_release(encoder)
        if image != NULL:
            heif_image_release(image)
        if context != NULL:
            heif_context_free(context)
        if nclx != NULL:
            heif_nclx_color_profile_free(nclx)
        if options != NULL:
            heif_encoding_options_free(options)
        if compressed != NULL:
            output_del(compressed)

    return out


def heif_decode(data, index=0, photometric=None, out=None):
    """Return decoded HEIF image.

    By default, the first top level image is returned. If index is None, all
    top level images are returned as one array if possible or a ValueError
    is raised.

    Monochrome images are returned as RGB(A) unless
    photometric==heif_colorspace_monochrome.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t imageindex = -1 if index is None else index
        ssize_t height, width, samples, imagecount
        ssize_t col, row, rowsize, srcindex, dstindex
        int stride, itemsize, bps
        uint8_t* srcptr = NULL
        uint8_t* dstptr = NULL
        uint16_t* src2ptr = NULL
        uint16_t* dst2ptr = NULL
        heif_context* context = NULL
        heif_decoding_options* options = NULL
        heif_image_handle* handle = NULL
        heif_image* image = NULL
        heif_item_id* imageids = NULL
        heif_colorspace colorspace = heif_colorspace_RGB
        heif_chroma chroma = heif_chroma_undefined
        heif_chroma chroma_sequence = heif_chroma_undefined
        heif_error err
        bint hasalpha = False
        bint monochrome = False

    if data is out:
        raise ValueError('cannot decode in-place')

    if photometric is not None:
        monochrome = (
            _heif_photometric(photometric) == heif_colorspace_monochrome
        )

    try:
        with nogil:
            context = heif_context_alloc()
            if context == NULL:
                raise HeifError('heif_context_alloc', b'NULL')

            err = heif_context_read_from_memory_without_copy(
                context,
                <const void*> &src[0],
                <size_t> srcsize,
                <const heif_reading_options*> NULL
            )
            if err.code != 0:
                raise HeifError(
                    'heif_context_read_from_memory_without_copy', err.message
                )

            # options = heif_decoding_options_alloc()
            # if options == NULL:
            #     raise HeifError('heif_decoding_options_alloc', b'NULL')
            # options.convert_hdr_to_8bit = 0

            imagecount = heif_context_get_number_of_top_level_images(context)
            if imagecount <= 0:
                raise HeifError(
                    'heif_context_get_number_of_top_level_images', b'NULL'
                )

            if imageindex >= imagecount:
                raise IndexError(
                    f'index {imageindex} out of range {imagecount}'
                )

            if imagecount == 1:
                err = heif_context_get_primary_image_handle(context, &handle)
                if err.code != 0:
                    raise HeifError(
                        'heif_context_get_primary_image_handle', err.message
                    )
            else:
                imageids = <heif_item_id*> calloc(
                    imagecount, sizeof(heif_item_id)
                )
                if imageids == NULL:
                    raise MemoryError('failed to allocate imageids')

                heif_context_get_list_of_top_level_image_IDs(
                    context, imageids, <int> imagecount
                )

                err = heif_context_get_image_handle(
                    context,
                    imageids[imageindex if imageindex > 0 else 0],
                    &handle
                )
                if err.code != 0:
                    raise HeifError(
                        'heif_context_get_image_handle', err.message
                    )

                if imageindex >= 0:
                    imagecount = 1

            imageindex = 0
            while True:

                bps = heif_image_handle_get_luma_bits_per_pixel(handle)
                if bps <= 0:
                    raise HeifError(
                        'heif_image_handle_get_luma_bits_per_pixel',
                        b'%i' % bps
                    )

                # TODO: detect monochrome images from handle
                # TODO: decode only luma and alpha for monochrome images
                # if heif_image_handle_get_chroma_bits_per_pixel(handle) <= 0:

                if heif_image_handle_has_alpha_channel(handle):
                    samples = 2 if monochrome else 4
                    if bps > 8:
                        chroma = heif_chroma_interleaved_RRGGBBAA_LE
                    else:
                        chroma = heif_chroma_interleaved_RGBA
                else:
                    samples = 1 if monochrome else 3
                    if bps > 8:
                        chroma = heif_chroma_interleaved_RRGGBB_LE
                    else:
                        chroma = heif_chroma_interleaved_RGB
                # TODO: decode aux images as extrasamples

                err = heif_decode_image(
                    handle, &image, colorspace, chroma, options
                )
                if err.code != 0:
                    raise HeifError('heif_decode_image', err.message)

                if dstptr == NULL:
                    # first image
                    width = <ssize_t> heif_image_get_primary_width(image)
                    height = <ssize_t> heif_image_get_primary_height(image)
                    if heif_image_get_bits_per_pixel(
                        image, heif_channel_interleaved
                    ) > 32:
                        itemsize = 2
                    else:
                        itemsize = 1
                    rowsize = width * samples * itemsize
                    chroma_sequence = chroma

                    with gil:
                        dtype = numpy.uint16 if itemsize == 2 else numpy.uint8
                        if imagecount > 1:
                            if samples > 1:
                                shape = (
                                    int(imagecount),
                                    int(height),
                                    int(width),
                                    int(samples)
                                )
                            else:
                                shape = (
                                    int(imagecount),
                                    int(height),
                                    int(width)
                                )
                        elif samples > 1:
                            shape = (int(height), int(width), int(samples))
                        else:
                            shape = (int(height), int(width))

                        out = _create_array(out, shape, dtype)
                        dst = out
                        dstptr = <uint8_t*> dst.data

                elif (
                    width != <ssize_t> heif_image_get_primary_width(image)
                    or height != <ssize_t> heif_image_get_primary_height(image)
                    or chroma_sequence != chroma
                ):
                    raise ValueError('image sequence shape or dtype mismatch')

                srcptr = <uint8_t*> heif_image_get_plane_readonly(
                    image, heif_channel_interleaved, &stride
                )
                if srcptr == NULL:
                    raise HeifError(
                        'heif_image_get_plane_readonly', b'NULL'
                    )

                if monochrome:
                    hasalpha = samples == 2
                    dstindex = 0
                    if itemsize == 1:
                        for row in range(height):
                            srcindex = 0
                            for col in range(width):
                                dstptr[dstindex] = srcptr[srcindex]
                                srcindex += 3
                                dstindex += 1
                                if hasalpha:
                                    dstptr[dstindex] = srcptr[srcindex]
                                    srcindex += 1
                                    dstindex += 1
                            srcptr += stride
                    else:
                        dst2ptr = <uint16_t*> dstptr
                        dstindex = 0
                        for row in range(height):
                            src2ptr = <uint16_t*> srcptr
                            srcindex = 0
                            for col in range(width):
                                dst2ptr[dstindex] = src2ptr[srcindex]
                                srcindex += 3
                                dstindex += 1
                                if hasalpha:
                                    dst2ptr[dstindex] = src2ptr[srcindex]
                                    srcindex += 1
                                    dstindex += 1
                            srcptr += stride
                    dstptr += height * rowsize

                else:
                    # RGBA
                    for row in range(height):
                        memcpy(
                            <void*> dstptr,
                            <const void*> srcptr,
                            rowsize
                        )
                        srcptr += stride
                        dstptr += rowsize

                heif_image_release(image)
                image = NULL
                heif_image_handle_release(handle)
                handle = NULL

                imageindex += 1
                if imageindex == imagecount:
                    break

                err = heif_context_get_image_handle(
                    context, imageids[imageindex], &handle
                )
                if err.code != 0:
                    raise HeifError(
                        'heif_context_get_image_handle', err.message
                    )

    finally:
        if imageids != NULL:
            free(imageids)
        if image != NULL:
            heif_image_release(image)
        if handle != NULL:
            heif_image_handle_release(handle)
        if context != NULL:
            heif_context_free(context)
        if options != NULL:
            heif_decoding_options_free(options)

    return out


cdef _heif_photometric(photometric):
    """Return heif_colorspace value from photometric argument."""
    if photometric is None:
        return heif_colorspace_undefined
    if photometric in {
        heif_colorspace_undefined,
        heif_colorspace_YCbCr,
        heif_colorspace_RGB,
        heif_colorspace_monochrome,
    }:
        return photometric
    if isinstance(photometric, str):
        photometric = photometric.upper()
        if photometric[:3] == 'RGB':
            return heif_colorspace_RGB
        if photometric[:3] == 'YCBCR':
            return heif_colorspace_YCbCr
        if photometric in {
            'GRAY', 'BLACKISZERO', 'MINISBLACK', 'WHITEISZERO', 'MINISWHITE'
        }:
            return heif_colorspace_monochrome
    raise ValueError(f'{photometric=!r} not supported')


cdef _heif_compression(compression):
    """Return heif_compression_format value from compression argument."""
    if compression is None:
        return heif_compression_HEVC
    if compression in {
        heif_compression_undefined,
        heif_compression_HEVC,
        heif_compression_AVC,
        heif_compression_JPEG,
        heif_compression_AV1,
        heif_compression_VVC,
        heif_compression_EVC,
        heif_compression_JPEG2000,
    }:
        return compression
    if isinstance(compression, str):
        compression = compression.upper()
        if compression == 'HEVC':
            return heif_compression_HEVC
        if compression == 'AVC':
            return heif_compression_AVC
        if compression == 'AV1':
            return heif_compression_AV1
        if compression == 'JPEG':
            return heif_compression_JPEG
        if compression == 'VVC':
            return heif_compression_VVC
        if compression == 'EVC':
            return heif_compression_EVC
        if compression == 'JPEG2000':
            return heif_compression_JPEG2000
        if compression == 'UNDEFINED':
            return heif_compression_undefined
    raise ValueError(f'{compression=!r} not supported')


cdef heif_error heif_write_callback(
    heif_context* ctx,
    const void* data,
    size_t size,
    void* userdata
) noexcept nogil:
    """heif_writer callback function."""
    cdef:
        output_t* output = <output_t*> userdata
        heif_error err

    if output_write(output, data, size) == 0:
        err.code = heif_error_Encoding_error
        err.message = b'Error during encoding or writing output file'
    else:
        err.code = heif_error_Ok
        err.message = b'Success'
    err.subcode = heif_suberror_Unspecified
    return err


ctypedef struct output_t:
    uint8_t* data
    size_t size
    size_t pos
    size_t used
    int owner


cdef output_t* output_new(uint8_t* data, size_t size) noexcept nogil:
    """Return new output."""
    cdef:
        output_t* output = <output_t*> calloc(1, sizeof(output_t))

    if output == NULL:
        return NULL
    output.size = size
    output.used = 0
    output.pos = 0
    if data == NULL:
        output.owner = 1
        output.data = <uint8_t*> malloc(size)
    else:
        output.owner = 0
        output.data = data
    if output.data == NULL:
        free(output)
        return NULL
    return output


cdef void output_del(output_t* output) noexcept nogil:
    """Free output."""
    if output != NULL:
        if output.owner != 0:
            free(output.data)
        free(output)


cdef int output_write(
    output_t* output,
    const void* data,
    size_t size
) noexcept nogil:
    """Write data to output."""
    if output == NULL:
        return 0
    if output.pos + size > output.size:
        if output_resize(output, output.pos + size) == 0:
            return 0
    memcpy(<void*> (output.data + output.pos), data, size)
    output.pos += size
    if output.pos > output.used:
        output.used = output.pos
    return 1


cdef int output_seek(output_t* output, size_t pos) noexcept nogil:
    """Seek output to position."""
    if output == NULL or pos > output.size:
        return 0
    output.pos = pos
    if pos > output.used:
        output.used = pos
    return 1


cdef int output_resize(output_t* output, size_t newsize) noexcept nogil:
    """Resize output."""
    cdef:
        uint8_t* tmp

    if (
        output == NULL
        or newsize == 0
        or output.used > output.size
        or output.owner == 0
    ):
        return 0

    if newsize == output.size:
        return 1

    tmp = <uint8_t*> realloc(<void*> output.data, newsize)
    if tmp == NULL:
        return 0
    output.data = tmp
    output.size = newsize
    return 1
