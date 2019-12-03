# -*- coding: utf-8 -*-
# _jpegls.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2019, Christoph Gohlke
# Copyright (c) 2018-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
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

"""JPEG-LS codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.11.28

"""

__version__ = '2019.11.28'

import numbers
import numpy

cimport cython
cimport numpy

from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, int32_t, uint32_t
from libc.string cimport memset

numpy.import_array()


# JPEG LS #####################################################################

cdef extern from 'charls/charls.h':

    int CHARLS_VERSION_MAJOR
    int CHARLS_VERSION_MINOR
    int CHARLS_VERSION_PATCH

    ctypedef enum charls_jpegls_errc:
        CHARLS_JPEGLS_ERRC_SUCCESS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT
        CHARLS_JPEGLS_ERRC_PARAMETER_VALUE_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_DESTINATION_BUFFER_TOO_SMALL
        CHARLS_JPEGLS_ERRC_SOURCE_BUFFER_TOO_SMALL
        CHARLS_JPEGLS_ERRC_INVALID_ENCODED_DATA
        CHARLS_JPEGLS_ERRC_TOO_MUCH_ENCODED_DATA
        CHARLS_JPEGLS_ERRC_INVALID_OPERATION
        CHARLS_JPEGLS_ERRC_BIT_DEPTH_FOR_TRANSFORM_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_COLOR_TRANSFORM_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_ENCODING_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_UNKNOWN_JPEG_MARKER_FOUND
        CHARLS_JPEGLS_ERRC_JPEG_MARKER_START_BYTE_NOT_FOUND
        CHARLS_JPEGLS_ERRC_NOT_ENOUGH_MEMORY
        CHARLS_JPEGLS_ERRC_UNEXPECTED_FAILURE
        CHARLS_JPEGLS_ERRC_START_OF_IMAGE_MARKER_NOT_FOUND
        CHARLS_JPEGLS_ERRC_START_OF_FRAME_MARKER_NOT_FOUND
        CHARLS_JPEGLS_ERRC_INVALID_MARKER_SEGMENT_SIZE
        CHARLS_JPEGLS_ERRC_DUPLICATE_START_OF_IMAGE_MARKER
        CHARLS_JPEGLS_ERRC_DUPLICATE_START_OF_FRAME_MARKER
        CHARLS_JPEGLS_ERRC_DUPLICATE_COMPONENT_ID_IN_SOF_SEGMENT
        CHARLS_JPEGLS_ERRC_UNEXPECTED_END_OF_IMAGE_MARKER
        CHARLS_JPEGLS_ERRC_INVALID_JPEGLS_PRESET_PARAMETER_TYPE
        CHARLS_JPEGLS_ERRC_JPEGLS_PRESET_EXTENDED_PARAMETER_TYPE_NOT_SUPPORTED
        CHARLS_JPEGLS_ERRC_MISSING_END_OF_SPIFF_DIRECTORY
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_WIDTH
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_HEIGHT
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_COMPONENT_COUNT
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_BITS_PER_SAMPLE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_INTERLEAVE_MODE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_NEAR_LOSSLESS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_PC_PARAMETERS
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_SPIFF_ENTRY_SIZE
        CHARLS_JPEGLS_ERRC_INVALID_ARGUMENT_COLOR_TRANSFORMATION
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_WIDTH
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_HEIGHT
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_COMPONENT_COUNT
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_BITS_PER_SAMPLE
        CHARLS_JPEGLS_ERRC_INVALID_PARAMETER_INTERLEAVE_MODE

    ctypedef enum charls_interleave_mode:
        CHARLS_INTERLEAVE_MODE_NONE
        CHARLS_INTERLEAVE_MODE_LINE
        CHARLS_INTERLEAVE_MODE_SAMPLE

    ctypedef enum charls_color_transformation:
        CHARLS_COLOR_TRANSFORMATION_NONE
        CHARLS_COLOR_TRANSFORMATION_HP1
        CHARLS_COLOR_TRANSFORMATION_HP2
        CHARLS_COLOR_TRANSFORMATION_HP3

    ctypedef enum charls_spiff_profile_id:
        CHARLS_SPIFF_PROFILE_ID_NONE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_BASE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_PROGRESSIVE
        CHARLS_SPIFF_PROFILE_ID_BI_LEVEL_FACSIMILE
        CHARLS_SPIFF_PROFILE_ID_CONTINUOUS_TONE_FACSIMILE

    ctypedef enum charls_spiff_color_space:
        CHARLS_SPIFF_COLOR_SPACE_BI_LEVEL_BLACK
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_709_VIDEO
        CHARLS_SPIFF_COLOR_SPACE_NONE
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_601_1_RGB
        CHARLS_SPIFF_COLOR_SPACE_YCBCR_ITU_BT_601_1_VIDEO
        CHARLS_SPIFF_COLOR_SPACE_GRAYSCALE
        CHARLS_SPIFF_COLOR_SPACE_PHOTO_YCC
        CHARLS_SPIFF_COLOR_SPACE_RGB
        CHARLS_SPIFF_COLOR_SPACE_CMY
        CHARLS_SPIFF_COLOR_SPACE_CMYK
        CHARLS_SPIFF_COLOR_SPACE_YCCK
        CHARLS_SPIFF_COLOR_SPACE_CIE_LAB
        CHARLS_SPIFF_COLOR_SPACE_BI_LEVEL_WHITE

    ctypedef enum charls_spiff_compression_type:
        CHARLS_SPIFF_COMPRESSION_TYPE_UNCOMPRESSED
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_HUFFMAN
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_READ
        CHARLS_SPIFF_COMPRESSION_TYPE_MODIFIED_MODIFIED_READ
        CHARLS_SPIFF_COMPRESSION_TYPE_JBIG
        CHARLS_SPIFF_COMPRESSION_TYPE_JPEG
        CHARLS_SPIFF_COMPRESSION_TYPE_JPEG_LS

    ctypedef enum charls_spiff_resolution_units:
        CHARLS_SPIFF_RESOLUTION_UNITS_ASPECT_RATIO
        CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_INCH
        CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_CENTIMETER

    ctypedef enum charls_spiff_entry_tag:
        CHARLS_SPIFF_ENTRY_TAG_TRANSFER_CHARACTERISTICS
        CHARLS_SPIFF_ENTRY_TAG_COMPONENT_REGISTRATION
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_ORIENTATION
        CHARLS_SPIFF_ENTRY_TAG_THUMBNAIL
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_TITLE
        CHARLS_SPIFF_ENTRY_TAG_IMAGE_DESCRIPTION
        CHARLS_SPIFF_ENTRY_TAG_TIME_STAMP
        CHARLS_SPIFF_ENTRY_TAG_VERSION_IDENTIFIER
        CHARLS_SPIFF_ENTRY_TAG_CREATOR_IDENTIFICATION
        CHARLS_SPIFF_ENTRY_TAG_PROTECTION_INDICATOR
        CHARLS_SPIFF_ENTRY_TAG_COPYRIGHT_INFORMATION
        CHARLS_SPIFF_ENTRY_TAG_CONTACT_INFORMATION
        CHARLS_SPIFF_ENTRY_TAG_TILE_INDEX
        CHARLS_SPIFF_ENTRY_TAG_SCAN_INDEX
        CHARLS_SPIFF_ENTRY_TAG_SET_REFERENCE

    struct charls_jpegls_decoder:
        pass

    struct charls_jpegls_encoder:
        pass

    struct charls_spiff_header:
        charls_spiff_profile_id profile_id
        int32_t component_count
        uint32_t height
        uint32_t width
        charls_spiff_color_space color_space
        int32_t bits_per_sample
        charls_spiff_compression_type compression_type
        charls_spiff_resolution_units resolution_units
        uint32_t vertical_resolution
        uint32_t horizontal_resolution

    struct charls_jpegls_pc_parameters:
        int32_t maximum_sample_value
        int32_t threshold1
        int32_t threshold2
        int32_t threshold3
        int32_t reset_value

    struct charls_frame_info:
        uint32_t width
        uint32_t height
        int32_t bits_per_sample
        int32_t component_count

    const void* charls_get_jpegls_category() nogil

    const char* charls_get_error_message(int32_t error_value) nogil

    charls_jpegls_decoder* charls_jpegls_decoder_create() nogil

    void charls_jpegls_decoder_destroy(
        const charls_jpegls_decoder* decoder) nogil

    charls_jpegls_errc charls_jpegls_decoder_set_source_buffer(
        charls_jpegls_decoder* decoder,
        const void* source_buffer,
        size_t source_size_bytes) nogil

    charls_jpegls_errc charls_jpegls_decoder_read_spiff_header(
        charls_jpegls_decoder* decoder,
        charls_spiff_header* spiff_header,
        int32_t* header_found) nogil

    charls_jpegls_errc charls_jpegls_decoder_read_header(
        charls_jpegls_decoder* decoder) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_frame_info(
        const charls_jpegls_decoder* decoder,
        charls_frame_info* frame_info) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_near_lossless(
        const charls_jpegls_decoder* decoder,
        int32_t component,
        int32_t* near_lossless) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_interleave_mode(
        const charls_jpegls_decoder* decoder,
        charls_interleave_mode* interleave_mode) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_preset_coding_parameters(
        const charls_jpegls_decoder* decoder,
        int32_t reserved,
        charls_jpegls_pc_parameters* preset_coding_parameters) nogil

    charls_jpegls_errc charls_jpegls_decoder_get_destination_size(
        const charls_jpegls_decoder* decoder,
        size_t* destination_size_bytes) nogil

    charls_jpegls_errc charls_jpegls_decoder_decode_to_buffer(
        const charls_jpegls_decoder* decoder,
        void* destination_buffer,
        size_t destination_size_bytes,
        uint32_t stride) nogil

    charls_jpegls_encoder* charls_jpegls_encoder_create() nogil

    void charls_jpegls_encoder_destroy(
        const charls_jpegls_encoder* encoder) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_frame_info(
        charls_jpegls_encoder* encoder,
        const charls_frame_info* frame_info) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_near_lossless(
        charls_jpegls_encoder* encoder,
        int32_t near_lossless) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_interleave_mode(
        charls_jpegls_encoder* encoder,
        charls_interleave_mode interleave_mode) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_preset_coding_parameters(
        charls_jpegls_encoder* encoder,
        const charls_jpegls_pc_parameters* preset_coding_parameters) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_color_transformation(
        charls_jpegls_encoder* encoder,
        charls_color_transformation color_transformation) nogil

    charls_jpegls_errc charls_jpegls_encoder_get_estimated_destination_size(
        const charls_jpegls_encoder* encoder,
        size_t* size_in_bytes) nogil

    charls_jpegls_errc charls_jpegls_encoder_set_destination_buffer(
        charls_jpegls_encoder* encoder,
        void* destination_buffer,
        size_t destination_size) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_standard_spiff_header(
        charls_jpegls_encoder* encoder,
        charls_spiff_color_space color_space,
        charls_spiff_resolution_units resolution_units,
        uint32_t vertical_resolution,
        uint32_t horizontal_resolution) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_spiff_header(
        charls_jpegls_encoder* encoder,
        const charls_spiff_header* spiff_header) nogil

    charls_jpegls_errc charls_jpegls_encoder_write_spiff_entry(
        charls_jpegls_encoder* encoder,
        uint32_t entry_tag,
        const void* entry_data,
        size_t entry_data_size) nogil

    charls_jpegls_errc charls_jpegls_encoder_encode_from_buffer(
        charls_jpegls_encoder* encoder,
        const void* source_buffer,
        size_t source_size,
        uint32_t stride) nogil

    charls_jpegls_errc charls_jpegls_encoder_get_bytes_written(
        const charls_jpegls_encoder* encoder,
        size_t* bytes_written) nogil


_CHARLS_VERSION = ('%i.%i.%i' % (
    CHARLS_VERSION_MAJOR, CHARLS_VERSION_MINOR, CHARLS_VERSION_PATCH))


class JpegLsError(RuntimeError):
    """JPEG-LS Exceptions."""
    def __init__(self, func, err):
        cdef:
            char* error_message
            int32_t error_value
        try:
            error_value = int(err)
            error_message = charls_get_error_message(error_value)
            msg = error_message.decode('utf8').strip()
        except Exception:
            msg = 'NULL' if err is None else 'unknown error %s' % err
        msg = "%s returned '%s'" % (func, msg)
        RuntimeError.__init__(self, msg)


def jpegls_encode(data, level=None, out=None):
    """Return JPEG-LS image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        charls_jpegls_errc ret = CHARLS_JPEGLS_ERRC_SUCCESS
        charls_jpegls_encoder* encoder = NULL
        charls_frame_info frameinfo
        # charls_jpegls_pc_parameters preset_coding_parameters
        charls_interleave_mode interleave_mode
        int32_t near_lossless = _default_level(level, 0, 0, 9)
        uint32_t rowstride = src.strides[0]
        size_t bytes_written
        size_t size_in_bytes

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (data.dtype in (numpy.uint8, numpy.uint16)
            and data.ndim in (2, 3)
            and numpy.PyArray_ISCONTIGUOUS(data)):
        raise ValueError('invalid input shape, strides, or dtype')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.size * dst.itemsize
    elif dstsize > 0:
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)
        dst = out
        dstsize = dst.size * dst.itemsize

    # memset(&preset_coding_parameters, 0, sizeof(charls_jpegls_pc_parameters))
    # preset_coding_parameters.maximum_sample_value = 0
    # preset_coding_parameters.threshold1 = 0
    # preset_coding_parameters.threshold2 = 0
    # preset_coding_parameters.threshold3 = 0
    # preset_coding_parameters.reset_value = 0

    # memset(&frameinfo, 0, sizeof(charls_frame_info))
    frameinfo.width = <uint32_t>src.shape[1]
    frameinfo.height = <uint32_t>src.shape[0]
    frameinfo.bits_per_sample = <int32_t>(src.itemsize * 8)

    if src.ndim == 2 or src.shape[2] == 1:
        frameinfo.component_count = 1
        interleave_mode = CHARLS_INTERLEAVE_MODE_NONE
    elif src.shape[2] == 3:
        frameinfo.component_count = 3
        interleave_mode = CHARLS_INTERLEAVE_MODE_SAMPLE
    elif src.shape[2] == 4:
        frameinfo.component_count = 4
        interleave_mode = CHARLS_INTERLEAVE_MODE_LINE
    else:
        raise ValueError('invalid shape')

    try:
        with nogil:
            encoder = charls_jpegls_encoder_create()
            if encoder == NULL:
                raise JpegLsError('charls_jpegls_encoder_create', None)

            ret = charls_jpegls_encoder_set_frame_info(encoder, &frameinfo)
            if ret:
                raise JpegLsError('charls_jpegls_encoder_set_frame_info', ret)

            ret = charls_jpegls_encoder_set_near_lossless(
                encoder,
                near_lossless)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_set_near_lossless', ret)

            ret = charls_jpegls_encoder_set_interleave_mode(
                encoder,
                interleave_mode)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_set_interleave_mode', ret)

            # ret = charls_jpegls_encoder_set_color_transformation(
            #     encoder,
            #     color_transformation)
            # if ret:
            #     raise JpegLsError(
            #         'charls_jpegls_encoder_set_color_transformation', ret)

            # ret charls_jpegls_encoder_set_preset_coding_parameters(
            #     encoder,
            #     &preset_coding_parameters)
            # if ret:
            #    raise JpegLsError(
            #        'charls_jpegls_encoder_set_preset_coding_parameters', ret)

            if dstsize < 0:
                ret = charls_jpegls_encoder_get_estimated_destination_size(
                    encoder,
                    &size_in_bytes
                )
                if ret:
                    raise JpegLsError(
                        'charls_jpegls_encoder_get_estimated_destination_size',
                        ret)
                dstsize = size_in_bytes + sizeof(charls_spiff_header)
                with gil:
                    if out_type is bytes:
                        out = PyBytes_FromStringAndSize(NULL, dstsize)
                    else:
                        out = PyByteArray_FromStringAndSize(NULL, dstsize)
                    dst = out
                    dstsize = dst.size * dst.itemsize

            ret = charls_jpegls_encoder_set_destination_buffer(
                encoder,
                <void*>&dst[0],
                <size_t>dstsize)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_set_destination_buffer', ret)

            ret = charls_jpegls_encoder_write_standard_spiff_header(
                encoder,
                CHARLS_SPIFF_COLOR_SPACE_RGB,
                CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_INCH,
                300,
                300)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_write_standard_spiff_header', ret)

            ret = charls_jpegls_encoder_encode_from_buffer(
                encoder,
                <const void*>src.data,
                <size_t>srcsize,
                <uint32_t>rowstride)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_encode_from_buffer', ret)

            ret = charls_jpegls_encoder_get_bytes_written(
                encoder,
                &bytes_written)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_encoder_get_bytes_written', ret)
    finally:
        if encoder != NULL:
            charls_jpegls_encoder_destroy(encoder)

    if <ssize_t>bytes_written < dstsize:
        if out_given:
            out = memoryview(out)[:bytes_written]
        else:
            out = out[:bytes_written]

    return out


def jpegls_decode(data, out=None):
    """Return numpy array from JPEG LS image.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t itemsize = 0
        charls_jpegls_errc ret = CHARLS_JPEGLS_ERRC_SUCCESS
        charls_jpegls_decoder* decoder = NULL
        charls_interleave_mode interleave_mode
        charls_frame_info frameinfo
        # charls_spiff_header spiff_header
        # int32_t header_found

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            decoder = charls_jpegls_decoder_create()
            if decoder == NULL:
                raise JpegLsError('charls_jpegls_decoder_create', None)

            ret = charls_jpegls_decoder_set_source_buffer(
                decoder,
                <void*>&src[0],
                <size_t>srcsize)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_decoder_set_source_buffer', ret)

            # ret = charls_jpegls_decoder_read_spiff_header(
            #     decoder,
            #     &spiff_header,
            #     &header_found)
            # if ret:
            #     raise JpegLsError(
            #         'charls_jpegls_decoder_read_spiff_header', ret)

            ret = charls_jpegls_decoder_read_header(decoder)
            if ret:
                raise JpegLsError('charls_jpegls_decoder_read_header', ret)

            ret = charls_jpegls_decoder_get_frame_info(decoder, &frameinfo)
            if ret:
                raise JpegLsError('charls_jpegls_decoder_get_frame_info', ret)

            ret = charls_jpegls_decoder_get_interleave_mode(
                decoder,
                &interleave_mode)
            if ret:
                raise JpegLsError(
                    'charls_jpegls_decoder_get_interleave_mode', ret)

            with gil:
                if frameinfo.bits_per_sample <= 8:
                    dtype = numpy.uint8
                    itemsize = 1
                elif frameinfo.bits_per_sample <= 16:
                    dtype = numpy.uint16
                    itemsize = 2
                else:
                    raise ValueError(
                        'JpegLs bits_per_sample not supported: %i'
                        % frameinfo.bits_per_sample)

                if frameinfo.component_count == 1:
                    shape = (
                        frameinfo.height,
                        frameinfo.width
                    )
                    strides = (
                        frameinfo.width * itemsize,
                        itemsize
                    )
                elif interleave_mode == CHARLS_INTERLEAVE_MODE_NONE:
                    # planar
                    shape = (
                        frameinfo.component_count,
                        frameinfo.height,
                        frameinfo.width
                    )
                    strides = (
                        itemsize * frameinfo.width * frameinfo.height,
                        itemsize * frameinfo.width,
                        itemsize
                    )
                else:
                    # contig
                    # CHARLS_INTERLEAVE_MODE_LINE or
                    # CHARLS_INTERLEAVE_MODE_SAMPLE
                    shape = (
                        frameinfo.height,
                        frameinfo.width,
                        frameinfo.component_count
                    )
                    strides = (
                        itemsize * frameinfo.component_count * frameinfo.width,
                        itemsize * frameinfo.component_count,
                        itemsize
                    )
                out = _create_array(out, shape, dtype, strides=strides)
                dst = out
                dstsize = dst.size * dst.itemsize

            ret = charls_jpegls_decoder_decode_to_buffer(
                decoder,
                <void*>dst.data,
                <size_t>dstsize,
                0)

        if ret:
            raise JpegLsError('charls_jpegls_decoder_decode_to_buffer', ret)

    finally:
        if decoder != NULL:
            charls_jpegls_decoder_destroy(decoder)

    if (
        frameinfo.component_count > 1
        and interleave_mode == CHARLS_INTERLEAVE_MODE_NONE
    ):
        out = numpy.moveaxis(out, 0, -1)

    return out


###############################################################################

cdef _create_array(out, shape, dtype, strides=None):
    """Return numpy array of shape and dtype from output argument."""
    if out is None or isinstance(out, numbers.Integral):
        out = numpy.empty(shape, dtype)
    elif isinstance(out, numpy.ndarray):
        if out.shape != shape:
            raise ValueError('invalid output shape')
        if out.itemsize != numpy.dtype(dtype).itemsize:
            raise ValueError('invalid output dtype')
        if strides is not None:
            for i, j in zip(strides, out.strides):
                if i is not None and i != j:
                    raise ValueError('invalid output strides')
        elif not numpy.PyArray_ISCONTIGUOUS(out):
            raise ValueError('output is not contiguous')
    else:
        dstsize = 1
        for i in shape:
            dstsize *= i
        out = numpy.frombuffer(out, dtype, dstsize)
        out.shape = shape
    return out


cdef _parse_output(out, ssize_t out_size=-1, out_given=False, out_type=bytes):
    """Return out, out_size, out_given, out_type from output argument."""
    if out is None:
        pass
    elif out is bytes:
        out = None
        out_type = bytes
    elif out is bytearray:
        out = None
        out_type = bytearray
    elif isinstance(out, numbers.Integral):
        out_size = out
        out = None
    else:
        # out_size = len(out)
        # out_type = type(out)
        out_given = True
    return out, out_size, out_given, out_type


def _default_level(level, default, smallest, largest):
    """Return compression level in range."""
    if level is None:
        level = default
    if largest is not None:
        level = min(level, largest)
    if smallest is not None:
        level = max(level, smallest)
    return level
