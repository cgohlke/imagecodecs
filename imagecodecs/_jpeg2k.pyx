# imagecodecs/_jpeg2k.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2022, Christoph Gohlke
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

"""JPEG 2000 codec for the imagecodecs package."""

__version__ = '2022.2.22'

include '_shared.pxi'

from openjpeg cimport *

from libc.math cimport log

class JPEG2K:
    """OpenJPEG Constants."""

    class CODEC(enum.IntEnum):
        JP2 = OPJ_CODEC_JP2
        J2K = OPJ_CODEC_J2K
        # JPT = OPJ_CODEC_JPT
        # JPP = OPJ_CODEC_JPP
        # JPX = OPJ_CODEC_JPX

    class CLRSPC(enum.IntEnum):
        UNSPECIFIED = OPJ_CLRSPC_UNSPECIFIED
        SRGB = OPJ_CLRSPC_SRGB
        GRAY = OPJ_CLRSPC_GRAY
        SYCC = OPJ_CLRSPC_SYCC
        EYCC = OPJ_CLRSPC_EYCC
        CMYK = OPJ_CLRSPC_CMYK


class Jpeg2kError(RuntimeError):
    """OpenJPEG Exceptions."""


def jpeg2k_version():
    """Return OpenJPEG library version string."""
    return 'openjpeg ' + opj_version().decode()


def jpeg2k_check(const uint8_t[::1] data):
    """Return True if data likely contains a JPEG 2000 image."""
    cdef:
        bytes sig = bytes(data[:12])

    return (
        sig == b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a'  # JP2
        or sig[:4] == b'\xff\x4f\xff\x51'  # J2K
        or sig[:4] == b'\x0d\x0a\x87\x0a'  # JP2
    )


def jpeg2k_encode(
    data,
    level=None,  # quality, psnr
    codecformat=None,
    colorspace=None,
    planar=None,
    tile=None,  # not implemented
    bitspersample=None,
    reversible=None,
    resolutions=None,
    mct=True,  # multiple component transform: rgb->ycc
    verbose=0,
    numthreads=None,
    out=None
):
    """Return JPEG 2000 image from numpy array.

    WIP: currently only single-tile, single-quality-layer formats are supported

    TODO: (u)int32 must contain <= 26-bits data ?

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t byteswritten
        memopj_t memopj
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_cparameters_t parameters
        opj_image_cmptparm_t cmptparms[4]
        OPJ_CODEC_FORMAT codec_format
        OPJ_BOOL ret = OPJ_TRUE
        OPJ_COLOR_SPACE color_space
        OPJ_UINT32 signed, prec, width, height, samples
        ssize_t i, j
        int verbosity = verbose
        int tile_width = 0
        int tile_height = 0
        float quality = _default_value(level, 0, 0, None)
        int numresolution = _default_value(resolutions, 6, 1, OPJ_J2K_MAXRLVLS)
        int num_threads = <int> _default_threads(numthreads)
        int irreversible = 0 if reversible else 1
        bint tcp_mct = bool(mct)

    if not (src.dtype.char in 'bBhHiIlL' and src.ndim in (2, 3)):
        raise ValueError('invalid data shape or dtype')

    if srcsize >= 2 ** 32:
        raise ValueError('tile size must not exceed 4 GB')

    if quality < 1 or quality > 1000:
        # lossless
        quality = 0.0
        if reversible is None:
            irreversible = 0

    if codecformat is None:
        codec_format = OPJ_CODEC_JP2  # use container format by default
    elif codecformat in (OPJ_CODEC_JP2, 'JP2', 'jp2'):
        codec_format = OPJ_CODEC_JP2
    elif codecformat in (OPJ_CODEC_J2K, 'J2K', 'j2k'):
        codec_format = OPJ_CODEC_J2K
    else:
        raise ValueError('invalid codecformat')

    signed = 1 if src.dtype.kind == 'i' else 0
    prec = <OPJ_UINT32> src.itemsize * 8
    width = <OPJ_UINT32> src.shape[1]
    height = <OPJ_UINT32> src.shape[0]
    samples = 1 if src.ndim == 2 else <OPJ_UINT32> src.shape[2]

    if samples > 1:
        if planar or (planar is None and samples > 4 and height <= 4):
            # separate bands
            samples = <OPJ_UINT32> src.shape[0]
            height = <OPJ_UINT32> src.shape[1]
            width = <OPJ_UINT32> src.shape[2]
        else:
            # contig
            # TODO: avoid full copy
            src = numpy.ascontiguousarray(numpy.moveaxis(src, -1, 0))

    if bitspersample is not None:
        if prec == 8:
            if bitspersample > 0 and bitspersample < 8:
                prec = bitspersample
        elif prec == 16:
            if bitspersample > 8 and bitspersample < 16:
                prec = bitspersample
        elif prec == 32:
            if bitspersample > 16 and bitspersample < 32:
                prec = bitspersample
    if prec == 32:
        # TODO: OpenJPEG currently only supports up to 31, effectively 26-bit?
        prec = 26

    if tile:
        tile_height, tile_width = tile
        # if width % tile_width or height % tile_height:
        #     raise ValueError('invalid tiles')
        raise NotImplementedError('writing tiles not implemented yet')  # TODO
    else:
        # TODO: use one tile for now. Other code path not implemented yet
        tile_height = height
        tile_width = width

    if colorspace is None:
        if samples <= 2:
            color_space = OPJ_CLRSPC_GRAY
        elif samples <= 4:
            color_space = OPJ_CLRSPC_SRGB
        else:
            color_space = OPJ_CLRSPC_UNSPECIFIED
    else:
        color_space = opj_colorspace(colorspace)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + 2048  # TODO: ?
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes

    try:
        with nogil:

            # create memory stream
            memopj.data = <OPJ_UINT8*> &dst[0]
            memopj.size = dstsize
            memopj.offset = 0
            memopj.written = 0

            stream = opj_memstream_create(&memopj, OPJ_FALSE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            # create image
            memset(&cmptparms, 0, sizeof(cmptparms))
            for i in range(<ssize_t> samples):
                cmptparms[i].dx = 1  # subsampling
                cmptparms[i].dy = 1
                cmptparms[i].h = height
                cmptparms[i].w = width
                cmptparms[i].x0 = 0
                cmptparms[i].y0 = 0
                cmptparms[i].prec = prec
                # cmptparms[i].bpp = prec  # redundant, not required
                cmptparms[i].sgnd = signed

            if tile_height > 0:
                image = opj_image_tile_create(samples, cmptparms, color_space)
                if image == NULL:
                    raise Jpeg2kError('opj_image_tile_create failed')
            else:
                image = opj_image_create(samples, cmptparms, color_space)
                if image == NULL:
                    raise Jpeg2kError('opj_image_create failed')

            image.x0 = 0
            image.y0 = 0
            image.x1 = width
            image.y1 = height
            image.color_space = color_space
            image.numcomps = samples

            # Set encoding parameters to default values, that means:
            # Lossless
            # 1 tile
            # Size of precinct : 2^15 x 2^15 (means 1 precinct)
            # Size of code-block : 64 x 64
            # Number of resolutions: 6
            # No SOP marker in the codestream
            # No EPH marker in the codestream
            # No sub-sampling in x or y direction
            # No mode switch activated
            # Progression order: LRCP
            # No index file
            # No ROI upshifted
            # No offset of the origin of the image
            # No offset of the origin of the tiles
            # Reversible DWT 5-3
            opj_set_default_encoder_parameters(&parameters)

            # single quality layer
            parameters.tcp_numlayers = 1

            # number of resolutions depends on tile size
            parameters.numresolution = <int> (
                (log(<double> min(tile_height, tile_width)) / log(2)) - 2
            )
            parameters.numresolution = min(
                max(parameters.numresolution, 1), numresolution
            )

            if quality == 0.0:
                # lossless
                parameters.irreversible = 0
                parameters.cp_disto_alloc = 1
                parameters.tcp_rates[0] = 0
            else:
                # lossy
                parameters.irreversible = irreversible

                # progression order resolution-position-component-layer
                parameters.prog_order = OPJ_RPCL

                # multiple component transform: rgb->ycc
                if samples == 3:
                    parameters.tcp_mct = <char> tcp_mct

                # code block width and height
                # parameters.cblockw_init = 64
                # parameters.cblockh_init = 64

                # fixed quality
                parameters.cp_fixed_quality = 1
                parameters.tcp_distoratio[0] = quality

                # fixed rate
                # parameters.cp_disto_alloc = 1
                # parameters.tcp_rates[0] = quality

                # precinct width and height
                parameters.res_spec = parameters.numresolution
                for i in range(parameters.res_spec):
                    parameters.prcw_init[i] = 256
                    parameters.prch_init[i] = 256
                parameters.csty = 1

            if tile_height > 0:
                parameters.tile_size_on = OPJ_TRUE
                parameters.cp_tx0 = 0
                parameters.cp_ty0 = 0
                parameters.cp_tdy = tile_height
                parameters.cp_tdx = tile_width

            # create and setup encoder
            codec = opj_create_compress(codec_format)
            if codec == NULL:
                raise Jpeg2kError('opj_create_compress failed')

            if verbosity > 0:
                opj_set_error_handler(
                    codec,
                    <opj_msg_callback> j2k_error_callback,
                    NULL
                )
                if verbosity > 1:
                    opj_set_warning_handler(
                        codec,
                        <opj_msg_callback> j2k_warning_callback,
                        NULL
                    )
                    if verbosity > 2:
                        opj_set_info_handler(
                            codec,
                            <opj_msg_callback> j2k_info_callback,
                            NULL
                        )

            ret = opj_setup_encoder(codec, &parameters, image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_encoder failed')

            if num_threads != 1 and opj_has_thread_support():
                if num_threads == 0:
                    num_threads = opj_get_num_cpus() / 2
                ret = opj_codec_set_threads(codec, num_threads)
                if ret == OPJ_FALSE:
                    raise Jpeg2kError('opj_codec_set_threads failed')

            ret = opj_start_compress(codec, image, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_start_compress failed')

            if tile_height > 0:
                # TODO: loop over tiles. Assume one tile for now
                ret = opj_write_tile(
                    codec,
                    0,
                    <OPJ_BYTE*> src.data,
                    <OPJ_UINT32> srcsize,
                    stream
                )
            else:
                raise NotImplementedError
                # TODO: copy data to image.comps[band].data[y, x]
                # ret = opj_encode(codec, stream)

            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_encode or opj_write_tile failed')

            ret = opj_end_compress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_end_compress failed')

            if memopj.written > memopj.size:
                raise Jpeg2kError('output buffer too small')

            byteswritten = <ssize_t> memopj.written

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpeg2k_decode(
    data,
    index=None,
    planar=None,
    verbose=0,
    numthreads=None,
    out=None
):
    """Decode JPEG 2000 J2K or JP2 image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        int32_t* band
        uint32_t* u4
        uint16_t* u2
        uint8_t* u1
        int32_t* i4
        int16_t* i2
        int8_t* i1
        ssize_t itemsize
        memopj_t memopj
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_image_comp_t* comp = NULL
        opj_dparameters_t parameters
        OPJ_BOOL ret = OPJ_FALSE
        OPJ_CODEC_FORMAT codecformat
        OPJ_UINT32 signed, prec, width, height
        ssize_t i, j, k, bandsize, samples
        int num_threads = <int> _default_threads(numthreads)
        int verbosity = verbose
        bytes sig
        bint contig = not planar

    if data is out:
        raise ValueError('cannot decode in-place')

    sig = bytes(src[:12])
    if sig[:4] == b'\xff\x4f\xff\x51':
        codecformat = OPJ_CODEC_J2K
    elif (
        sig == b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a'
        or sig[:4] == b'\x0d\x0a\x87\x0a'
    ):
        codecformat = OPJ_CODEC_JP2
    else:
        raise Jpeg2kError('not a J2K or JP2 data stream')

    try:
        memopj.data = <OPJ_UINT8*> &src[0]
        memopj.size = src.size
        memopj.offset = 0
        memopj.written = 0

        with nogil:
            stream = opj_memstream_create(&memopj, OPJ_TRUE)
            if stream == NULL:
                raise Jpeg2kError('opj_memstream_create failed')

            codec = opj_create_decompress(codecformat)
            if codec == NULL:
                raise Jpeg2kError('opj_create_decompress failed')

            if verbosity > 0:
                opj_set_error_handler(
                    codec, <opj_msg_callback> j2k_error_callback, NULL
                )
                if verbosity > 1:
                    opj_set_warning_handler(
                        codec, <opj_msg_callback> j2k_warning_callback, NULL
                    )
                    if verbosity > 2:
                        opj_set_info_handler(
                            codec, <opj_msg_callback> j2k_info_callback, NULL
                        )

            opj_set_default_decoder_parameters(&parameters)

            ret = opj_setup_decoder(codec, &parameters)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_setup_decoder failed')

            if num_threads != 1 and opj_has_thread_support():
                if num_threads == 0:
                    num_threads = opj_get_num_cpus() / 2
                ret = opj_codec_set_threads(codec, num_threads)
                if ret == OPJ_FALSE:
                    raise Jpeg2kError('opj_codec_set_threads failed')

            ret = opj_read_header(stream, codec, &image)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_read_header failed')

            ret = opj_set_decode_area(
                codec,
                image,
                <OPJ_INT32> parameters.DA_x0,
                <OPJ_INT32> parameters.DA_y0,
                <OPJ_INT32> parameters.DA_x1,
                <OPJ_INT32> parameters.DA_y1
            )
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_set_decode_area failed')

            ret = opj_decode(codec, stream, image)
            if ret != OPJ_FALSE:
                ret = opj_end_decompress(codec, stream)
            if ret == OPJ_FALSE:
                raise Jpeg2kError('opj_decode or opj_end_decompress failed')

            # handle subsampling and color profiles
            if (
                image.color_space != OPJ_CLRSPC_SYCC
                and image.numcomps == 3
                and image.comps[0].dx == image.comps[0].dy
                and image.comps[1].dx != 1
            ):
                image.color_space = OPJ_CLRSPC_SYCC
            elif image.numcomps <= 2:
                image.color_space = OPJ_CLRSPC_GRAY

            if image.color_space == OPJ_CLRSPC_SYCC:
                color_sycc_to_rgb(image)
            if image.icc_profile_buf:
                color_apply_icc_profile(image)
                free(image.icc_profile_buf)
                image.icc_profile_buf = NULL
                image.icc_profile_len = 0

            comp = &image.comps[0]
            signed = comp.sgnd
            prec = comp.prec
            height = comp.h * comp.dy
            width = comp.w * comp.dx
            samples = <ssize_t> image.numcomps
            itemsize = (prec + 7) // 8

            for i in range(samples):
                comp = &image.comps[i]
                if comp.sgnd != signed or comp.prec != prec:
                    raise NotImplementedError('components dtype mismatch')
                    # TODO: support upcast
                    # use scale_component
                if comp.w != width or comp.h != height:
                    raise NotImplementedError('subsampling not supported')
                    # TODO: support upsampling
                    # use upsample_image_components in opj_decompress.c
            if itemsize == 3:
                itemsize = 4
            elif itemsize < 1 or itemsize > 4:
                raise Jpeg2kError(f'unsupported itemsize {itemsize}')

        dtype = '{}{}'.format('i' if signed else 'u', itemsize)
        if samples == 1:
            shape = int(height), int(width)
            contig = 0
        elif contig:
            shape = int(height), int(width), int(samples)
        else:
            shape = int(samples), int(height), int(width)

        out = _create_array(out, shape, dtype)
        dst = out

        with nogil:
            # TODO: use OMP prange?
            # memset(<void*> dst.data, 0, dst.nbytes)
            bandsize = <ssize_t> (height * width)
            if itemsize == 1:
                if signed:
                    if contig:
                        for i in range(samples):
                            i1 = <int8_t*> dst.data + i
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                i1[j * samples] = <int8_t> band[j]
                    else:
                        k = 0
                        i1 = <int8_t*> dst.data
                        for i in range(samples):
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                i1[k] = <int8_t> band[j]
                                k += 1
                else:
                    if contig:
                        for i in range(samples):
                            u1 = <uint8_t*> dst.data + i
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                u1[j * samples] = <uint8_t> band[j]
                    else:
                        k = 0
                        u1 = <uint8_t*> dst.data
                        for i in range(samples):
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                u1[k] = <uint8_t> band[j]
                                k += 1
            elif itemsize == 2:
                if signed:
                    if contig:
                        for i in range(samples):
                            i2 = <int16_t*> dst.data + i
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                i2[j * samples] = <int16_t> band[j]
                    else:
                        k = 0
                        i2 = <int16_t*> dst.data
                        for i in range(samples):
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                i2[k] = <int16_t> band[j]
                                k += 1
                else:
                    if contig:
                        for i in range(samples):
                            u2 = <uint16_t*> dst.data + i
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                u2[j * samples] = <uint16_t> band[j]
                    else:
                        k = 0
                        u2 = <uint16_t*> dst.data
                        for i in range(samples):
                            band = <int32_t*> image.comps[i].data
                            for j in range(bandsize):
                                u2[k] = <uint16_t> band[j]
                                k += 1
            elif not contig:
                for i in range(samples):
                    memcpy(
                        <void *> &dst.data[i * bandsize * 4],
                        <void *> image.comps[i].data,
                        bandsize * 4
                    )
            elif signed:
                for i in range(samples):
                    i4 = <int32_t*> dst.data + i
                    band = <int32_t*> image.comps[i].data
                    for j in range(bandsize):
                        i4[j * samples] = <int32_t> band[j]
            else:
                for i in range(samples):
                    u4 = <uint32_t*> dst.data + i
                    band = <int32_t*> image.comps[i].data
                    for j in range(bandsize):
                        u4[j * samples] = <uint32_t> band[j]

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    return out


ctypedef struct memopj_t:
    OPJ_UINT8* data
    OPJ_UINT64 size
    OPJ_UINT64 offset
    OPJ_UINT64 written


cdef OPJ_SIZE_T opj_mem_read(void* dst, OPJ_SIZE_T size, void* data) nogil:
    """opj_stream_set_read_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data
        OPJ_SIZE_T count = size

    if memopj.offset >= memopj.size:
        return <OPJ_SIZE_T> -1
    if size > memopj.size - memopj.offset:
        count = <OPJ_SIZE_T> (memopj.size - memopj.offset)
    memcpy(
        <void*> dst,
        <const void*> &(memopj.data[memopj.offset]),
        count
    )
    memopj.offset += count
    return count


cdef OPJ_SIZE_T opj_mem_write(void* dst, OPJ_SIZE_T size, void* data) nogil:
    """opj_stream_set_write_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data
        OPJ_SIZE_T count = size

    if memopj.offset >= memopj.size:
        return <OPJ_SIZE_T> -1
    if size > memopj.size - memopj.offset:
        count = <OPJ_SIZE_T> (memopj.size - memopj.offset)
        memopj.written = memopj.size + 1  # indicates error
    memcpy(
        <void*> &(memopj.data[memopj.offset]),
        <const void*> dst,
        count
    )
    memopj.offset += count
    if memopj.written < memopj.offset:
        memopj.written = memopj.offset
    return count


cdef OPJ_BOOL opj_mem_seek(OPJ_OFF_T size, void* data) nogil:
    """opj_stream_set_seek_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data

    if size < 0 or size >= <OPJ_OFF_T> memopj.size:
        return OPJ_FALSE
    memopj.offset = <OPJ_SIZE_T> size
    return OPJ_TRUE


cdef OPJ_OFF_T opj_mem_skip(OPJ_OFF_T size, void* data) nogil:
    """opj_stream_set_skip_function."""
    cdef:
        memopj_t* memopj = <memopj_t*> data
        OPJ_SIZE_T count

    if size < 0:
        return -1
    count = <OPJ_SIZE_T> size
    if count > memopj.size - memopj.offset:
        count = <OPJ_SIZE_T> (memopj.size - memopj.offset)
    memopj.offset += count
    return count


cdef void opj_mem_nop(void* data) nogil:
    """opj_stream_set_user_data."""


cdef opj_stream_t* opj_memstream_create(
    memopj_t* memopj,
    OPJ_BOOL isinput
) nogil:
    """Return OPJ stream using memory as input or output."""
    cdef:
        opj_stream_t* stream = opj_stream_default_create(isinput)

    if stream == NULL:
        return NULL
    if isinput:
        opj_stream_set_read_function(stream, <opj_stream_read_fn> opj_mem_read)
    else:
        opj_stream_set_write_function(
            stream,
            <opj_stream_write_fn> opj_mem_write
        )
    opj_stream_set_seek_function(stream, <opj_stream_seek_fn> opj_mem_seek)
    opj_stream_set_skip_function(stream, <opj_stream_skip_fn> opj_mem_skip)
    opj_stream_set_user_data(
        stream,
        memopj,
        <opj_stream_free_user_data_fn> opj_mem_nop
    )
    opj_stream_set_user_data_length(stream, memopj.size)
    return stream


cdef void j2k_error_callback(char* msg, void* client_data) with gil:
    raise Jpeg2kError(msg.decode().strip())


cdef void j2k_warning_callback(char* msg, void* client_data) with gil:
    _log_warning('JPEG2K warning: %s', msg.decode().strip())


cdef void j2k_info_callback(char* msg, void* client_data) with gil:
    _log_warning('JPEG2K info: %s', msg.decode().strip())


cdef OPJ_COLOR_SPACE opj_colorspace(colorspace):
    """Return OPJ colorspace value from user input."""
    return {
        'GRAY': OPJ_CLRSPC_GRAY,
        'GRAYSCALE': OPJ_CLRSPC_GRAY,
        'MINISWHITE': OPJ_CLRSPC_GRAY,
        'MINISBLACK': OPJ_CLRSPC_GRAY,
        'RGB': OPJ_CLRSPC_SRGB,
        'SRGB': OPJ_CLRSPC_SRGB,
        'RGBA': OPJ_CLRSPC_SRGB,  # ?
        'CMYK': OPJ_CLRSPC_CMYK,
        'SYCC': OPJ_CLRSPC_SYCC,
        'EYCC': OPJ_CLRSPC_EYCC,
        'UNSPECIFIED': OPJ_CLRSPC_UNSPECIFIED,
        'UNKNOWN': OPJ_CLRSPC_UNSPECIFIED,
        None: OPJ_CLRSPC_UNSPECIFIED,
        OPJ_CLRSPC_UNSPECIFIED: OPJ_CLRSPC_UNSPECIFIED,
        OPJ_CLRSPC_SRGB: OPJ_CLRSPC_SRGB,
        OPJ_CLRSPC_GRAY: OPJ_CLRSPC_GRAY,
        OPJ_CLRSPC_SYCC: OPJ_CLRSPC_SYCC,
        OPJ_CLRSPC_EYCC: OPJ_CLRSPC_EYCC,
        OPJ_CLRSPC_CMYK: OPJ_CLRSPC_CMYK,
    }[colorspace]  # .get(colorspace, OPJ_CLRSPC_UNSPECIFIED)
