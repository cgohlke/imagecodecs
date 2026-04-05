# imagecodecs/_exr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""EXR codec for the imagecodecs package."""

include '_shared.pxi'

from libc.string cimport strcmp
from openexr cimport *


class EXR:
    """EXR codec constants."""

    available = True

    class COMPRESSION(enum.IntEnum):
        """EXR codec compression types."""

        NONE = EXR_COMPRESSION_NONE
        RLE = EXR_COMPRESSION_RLE
        ZIPS = EXR_COMPRESSION_ZIPS
        ZIP = EXR_COMPRESSION_ZIP
        PIZ = EXR_COMPRESSION_PIZ
        PXR24 = EXR_COMPRESSION_PXR24
        B44 = EXR_COMPRESSION_B44
        B44A = EXR_COMPRESSION_B44A
        DWAA = EXR_COMPRESSION_DWAA
        DWAB = EXR_COMPRESSION_DWAB
        HTJ2K256 = EXR_COMPRESSION_HTJ2K256
        HTJ2K32 = EXR_COMPRESSION_HTJ2K32


class ExrError(RuntimeError):
    """EXR codec exceptions."""

    def __init__(self, func, err):
        if err is None:
            msg = 'NULL'
        elif isinstance(err, str):
            msg = err
        else:
            errstr = exr_get_error_code_as_string(err).decode()
            msg = f'{errstr} ({err})'
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def exr_version():
    """Return OpenEXR library version string."""
    cdef:
        int major, minor, patch
        const char* extra

    exr_get_library_version(&major, &minor, &patch, &extra)
    return f'openexr {major}.{minor}.{patch}'


def exr_check(const uint8_t[::1] data, /):
    """Return whether data is EXR encoded."""
    if data.shape[0] < 4:
        return False
    return (
        data[0] == 0x76
        and data[1] == 0x2f
        and data[2] == 0x31
        and data[3] == 0x01
    )


def exr_encode(
    data,
    /,
    level=None,
    *,
    compression=None,
    planar=None,
    frames=None,
    out=None,
):
    """Return EXR encoded data.

    Tiled, deep, and multi-resolution storage are not supported.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t* src_ptr
        const uint8_t* base_ptr
        ssize_t dstsize
        ssize_t pixel_stride, line_stride, frame_stride, ch_stride
        ssize_t i, c, y
        int bpe
        # c-level channel/part arrays
        int* ch_idx = NULL
        char** ch_names = NULL
        const char** part_names = NULL
        # EXR context and pipeline
        exr_context_t context = NULL
        exr_context_initializer_t cinit
        exr_result_t ret
        exr_pixel_type_t ptype
        exr_compression_t ctype
        exr_chunk_info_t cinfo
        exr_encode_pipeline_t encoder
        int encoder_initialized = 0
        int part_index = 0
        int pi
        int32_t scanlines_per_chunk
        # write stream and compression
        write_stream_t wstream
        int set_zip_level = 0
        int set_dwa_level = 0
        int clevel = -2
        float flevel = -1.0
        imagelayout_t layout

    if data is out:
        raise ValueError('cannot encode in-place')

    # validate dtype
    if src.dtype == numpy.float16:
        ptype = EXR_PIXEL_HALF
        bpe = 2
    elif src.dtype == numpy.float32:
        ptype = EXR_PIXEL_FLOAT
        bpe = 4
    elif src.dtype == numpy.uint32:
        ptype = EXR_PIXEL_UINT
        bpe = 4
    else:
        raise ValueError(
            f'invalid dtype {src.dtype!r}, '
            f'expected float16, float32, or uint32'
        )

    _image_layout(
        IC_FLOAT
        | IC_UINT
        | IC_SZ2
        | IC_SZ4
        | IC_GRAY
        | IC_RGB
        | IC_ALPHA
        | IC_EXTRA
        | IC_FRAMES
        | IC_PLANAR,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        None,  # bitspersample
        planar if planar is not None else False,
        frames if frames is not None else False,
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    # determine channel names (must be added alphabetically)
    if layout.samples == 1:
        channel_names = [b'Y']
    elif layout.samples == 2:
        channel_names = [b'A', b'Y']
    elif layout.samples == 3:
        channel_names = [b'B', b'G', b'R']
    elif layout.samples == 4:
        channel_names = [b'A', b'B', b'G', b'R']
    else:
        channel_names = sorted(
            [f'ch{i:02d}'.encode() for i in range(layout.samples)]
        )

    # compression type
    ctype = <exr_compression_t> <int> _enum_value(
        compression, EXR.COMPRESSION, EXR.COMPRESSION.ZIP
    )

    # pre-extract compression level to C types
    if level is not None:
        if ctype in (EXR_COMPRESSION_ZIP, EXR_COMPRESSION_ZIPS):
            clevel = <int> level
            set_zip_level = 1
        elif ctype in (EXR_COMPRESSION_DWAA, EXR_COMPRESSION_DWAB):
            flevel = <float> level
            set_dwa_level = 1

    # build c-level channel-to-data-index mapping
    ch_idx = <int*> malloc(layout.samples * sizeof(int))
    if ch_idx == NULL:
        raise MemoryError('failed to allocate channel index array')
    # reverse order for <= 4 RGBA samples: alphabetical EXR vs array order
    if layout.samples <= 4:
        for c in range(layout.samples):
            ch_idx[c] = <int> (layout.samples - 1 - c)
    else:
        for c in range(layout.samples):
            ch_idx[c] = <int> c

    # pre-extract channel name char* pointers from Python bytes
    ch_names = <char**> malloc(layout.samples * sizeof(char*))
    if ch_names == NULL:
        free(ch_idx)
        raise MemoryError('failed to allocate channel name array')
    for c in range(layout.samples):
        ch_names[c] = <char*> (<bytes> channel_names[c])

    # pre-build part names for multi-part files
    if layout.frames > 1:
        part_name_list = [f'frame{i}'.encode() for i in range(layout.frames)]
        part_names = <const char**> malloc(layout.frames * sizeof(const char*))
        if part_names == NULL:
            free(ch_idx)
            free(ch_names)
            raise MemoryError('failed to allocate part name array')
        for i in range(layout.frames):
            part_names[i] = <const char*> (<bytes> part_name_list[i])

    src_ptr = <const uint8_t*> src.data
    wstream.data = NULL

    try:
        with nogil:
            if layout.planar and layout.samples > 1:
                pixel_stride = bpe
                line_stride = layout.width * bpe
                frame_stride = (
                    layout.samples * layout.height * layout.width * bpe
                )
                ch_stride = layout.height * layout.width * bpe
            elif layout.samples == 1:
                pixel_stride = bpe
                line_stride = layout.width * bpe
                frame_stride = layout.height * layout.width * bpe
                ch_stride = bpe
            else:
                pixel_stride = layout.samples * bpe
                line_stride = layout.width * layout.samples * bpe
                frame_stride = (
                    layout.height * layout.width * layout.samples * bpe
                )
                ch_stride = bpe

            wstream.capacity = 1048576
            wstream.size = 0
            wstream.data = <uint8_t*> malloc(wstream.capacity)
            if wstream.data == NULL:
                raise MemoryError('failed to allocate write buffer')

            memset(&cinit, 0, sizeof(cinit))
            cinit.size = sizeof(cinit)
            cinit.user_data = &wstream
            cinit.write_fn = _exr_write_func
            cinit.size_fn = _exr_write_size_func
            cinit.destroy_fn = _exr_write_destroy_func
            cinit.zip_level = -2
            cinit.dwa_quality = -1.0

            ret = exr_start_write(
                &context, b'memory', EXR_WRITE_FILE_DIRECTLY, &cinit
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_start_write', ret)

            # header setup
            for i in range(layout.frames):
                ret = exr_add_part(
                    context, NULL, EXR_STORAGE_SCANLINE, &pi
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_add_part', ret)
                part_index = pi

                ret = exr_initialize_required_attr_simple(
                    context, part_index,
                    <int32_t> layout.width, <int32_t> layout.height, ctype
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError(
                        'exr_initialize_required_attr_simple', ret
                    )

                if set_zip_level:
                    ret = exr_set_zip_compression_level(
                        context, part_index, clevel
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError(
                            'exr_set_zip_compression_level', ret
                        )
                elif set_dwa_level:
                    ret = exr_set_dwa_compression_level(
                        context, part_index, flevel
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError(
                            'exr_set_dwa_compression_level', ret
                        )

                if part_names != NULL:
                    ret = exr_set_name(
                        context, part_index, part_names[i]
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError('exr_set_name', ret)

                for c in range(layout.samples):
                    ret = exr_add_channel(
                        context, part_index,
                        ch_names[c], ptype,
                        EXR_PERCEPTUALLY_LOGARITHMIC, 1, 1
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError('exr_add_channel', ret)

            ret = exr_write_header(context)
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_write_header', ret)

            # scanline encoding
            for i in range(layout.frames):
                part_index = <int> i

                ret = exr_get_scanlines_per_chunk(
                    <exr_const_context_t> context,
                    part_index, &scanlines_per_chunk
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_scanlines_per_chunk', ret)

                encoder_initialized = 0
                y = 0
                while y < layout.height:
                    ret = exr_write_scanline_chunk_info(
                        context, part_index, <int> y, &cinfo
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError(
                            'exr_write_scanline_chunk_info', ret
                        )

                    if not encoder_initialized:
                        ret = exr_encoding_initialize(
                            <exr_const_context_t> context,
                            part_index, &cinfo, &encoder
                        )
                        if ret != EXR_ERR_SUCCESS:
                            raise ExrError(
                                'exr_encoding_initialize', ret
                            )
                        encoder_initialized = 1

                        # set per-channel source pointers for this chunk
                        for c in range(encoder.channel_count):
                            base_ptr = (
                                src_ptr
                                + i * frame_stride
                                + ch_idx[c] * ch_stride
                                + y * line_stride
                            )
                            encoder.channels[c].encode_from_ptr = base_ptr
                            encoder.channels[c].user_pixel_stride = (
                                <int32_t> pixel_stride
                            )
                            encoder.channels[c].user_line_stride = (
                                <int32_t> line_stride
                            )
                            encoder.channels[c].user_bytes_per_element = (
                                <int16_t> bpe
                            )
                            encoder.channels[c].user_data_type = (
                                <uint16_t> encoder.channels[c].data_type
                            )

                        ret = exr_encoding_choose_default_routines(
                            <exr_const_context_t> context,
                            part_index, &encoder
                        )
                        if ret != EXR_ERR_SUCCESS:
                            raise ExrError(
                                'exr_encoding_choose_default_routines', ret
                            )
                    else:
                        ret = exr_encoding_update(
                            <exr_const_context_t> context,
                            part_index, &cinfo, &encoder
                        )
                        if ret != EXR_ERR_SUCCESS:
                            raise ExrError('exr_encoding_update', ret)

                        # set per-channel source pointers for this chunk
                        for c in range(encoder.channel_count):
                            base_ptr = (
                                src_ptr
                                + i * frame_stride
                                + ch_idx[c] * ch_stride
                                + y * line_stride
                            )
                            encoder.channels[c].encode_from_ptr = base_ptr
                            encoder.channels[c].user_pixel_stride = (
                                <int32_t> pixel_stride
                            )
                            encoder.channels[c].user_line_stride = (
                                <int32_t> line_stride
                            )
                            encoder.channels[c].user_bytes_per_element = (
                                <int16_t> bpe
                            )
                            encoder.channels[c].user_data_type = (
                                <uint16_t> encoder.channels[c].data_type
                            )

                    ret = exr_encoding_run(
                        <exr_const_context_t> context,
                        part_index, &encoder
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError('exr_encoding_run', ret)

                    y += scanlines_per_chunk

                if encoder_initialized:
                    exr_encoding_destroy(
                        <exr_const_context_t> context, &encoder
                    )
                    encoder_initialized = 0

            ret = exr_finish(&context)
            context = NULL
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_finish', ret)

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            if dstsize < 0:
                dstsize = <ssize_t> wstream.size
            out = _create_output(outtype, dstsize)

        dst = out
        if <uint64_t> dst.shape[0] < wstream.size:
            raise ValueError('output buffer too small')

        memcpy(<void*> &dst[0], wstream.data, wstream.size)
        return _return_output(out, dstsize, <ssize_t> wstream.size, outgiven)

    finally:
        if encoder_initialized:
            exr_encoding_destroy(<exr_const_context_t> context, &encoder)
        if context != NULL:
            exr_finish(&context)
        free(wstream.data)
        free(ch_idx)
        free(ch_names)
        if part_names != NULL:
            free(part_names)


def exr_decode(
    data,
    /,
    *,
    index=0,
    planar=None,
    out=None,
):
    """Return decoded EXR image.

    Deep storage and sub-sampled channels are not supported.

    """
    cdef:
        const uint8_t[::1] src = data
        numpy.ndarray dst
        uint8_t* dst_ptr
        int* ch_idx = NULL
        bint is_planar = planar
        ssize_t pixel_stride, line_stride, frame_stride, ch_stride
        ssize_t channels, frame, c
        ssize_t frame_offset
        int bpe
        # EXR context and pipeline
        exr_context_t context = NULL
        exr_context_initializer_t cinit
        exr_result_t ret
        mem_stream_t rstream
        # part and channel info
        int part_count = 0
        int part_start, part_end
        int part_index
        int num_channels
        int32_t width, height
        const exr_attr_chlist_t* chlist = NULL
        exr_attr_box2i_t datawin
        exr_pixel_type_t ptype
        # multi-part validation
        const exr_attr_chlist_t* other_chlist = NULL
        exr_attr_box2i_t other_datawin
        int32_t other_width, other_height
        int pi
        # storage type
        exr_storage_t storage
        exr_storage_t part_storage
        exr_storage_t other_storage
        exr_attr_box2i_t part_datawin
        int32_t part_datawin_min_y
        bint do_reverse

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.shape[0] == 0:
        raise ValueError('src is empty')

    rstream.data = &src[0]
    rstream.size = <uint64_t> src.shape[0]

    if index is not None:
        part_start = <int> index
        part_end = part_start + 1
    else:
        part_start = 0
        part_end = 0  # resolved after reading part count

    try:
        # open file and read header info
        with nogil:
            memset(&cinit, 0, sizeof(cinit))
            cinit.size = sizeof(cinit)
            cinit.user_data = &rstream
            cinit.read_fn = _exr_read_func
            cinit.size_fn = _exr_read_size_func
            cinit.zip_level = -2
            cinit.dwa_quality = -1.0

            ret = exr_start_read(&context, b'memory', &cinit)
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_start_read', ret)

            ret = exr_get_count(
                <exr_const_context_t> context, &part_count
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_get_count', ret)
            if part_count < 1:
                raise ExrError('exr_get_count', 'no parts found')

            if part_end == 0:
                part_end = part_count

            if part_start < 0 or part_start >= part_count:
                raise IndexError(
                    f'part index {part_start} out of range '
                    f'[0, {part_count})'
                )

            ret = exr_get_data_window(
                <exr_const_context_t> context, part_start, &datawin
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_get_data_window', ret)

            ret = exr_get_channels(
                <exr_const_context_t> context, part_start, &chlist
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_get_channels', ret)

            width = datawin.max.x - datawin.min.x + 1
            height = datawin.max.y - datawin.min.y + 1
            num_channels = chlist.num_channels
            if num_channels < 1:
                raise ValueError('no channels found')

            # check for unsupported sub-sampled channels
            for c in range(num_channels):
                if (
                    chlist.entries[c].x_sampling != 1
                    or chlist.entries[c].y_sampling != 1
                ):
                    raise ValueError(
                        'sub-sampled channels are not supported'
                    )

            # determine promoted pixel type across all channels
            ptype = chlist.entries[0].pixel_type
            for c in range(1, num_channels):
                if chlist.entries[c].pixel_type == EXR_PIXEL_FLOAT:
                    ptype = EXR_PIXEL_FLOAT
                elif (
                    chlist.entries[c].pixel_type == EXR_PIXEL_HALF
                    and ptype == EXR_PIXEL_UINT
                ):
                    ptype = EXR_PIXEL_HALF

            ret = exr_get_storage(
                <exr_const_context_t> context, part_start, &storage
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_get_storage', ret)
            if (
                storage != EXR_STORAGE_SCANLINE
                and storage != EXR_STORAGE_TILED
            ):
                raise ValueError(
                    f'unsupported storage type {<int> storage}'
                )

            # validate all parts have same shape and channel count
            for pi in range(part_start + 1, part_end):
                ret = exr_get_storage(
                    <exr_const_context_t> context, pi, &other_storage
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_storage', ret)
                if (
                    other_storage != EXR_STORAGE_SCANLINE
                    and other_storage != EXR_STORAGE_TILED
                ):
                    raise ValueError(
                        f'part {pi} has unsupported storage type '
                        f'{<int> other_storage}'
                    )

                ret = exr_get_data_window(
                    <exr_const_context_t> context, pi, &other_datawin
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_data_window', ret)
                other_width = (
                    other_datawin.max.x - other_datawin.min.x + 1
                )
                other_height = (
                    other_datawin.max.y - other_datawin.min.y + 1
                )
                if other_width != width or other_height != height:
                    raise ValueError(
                        f'part {pi} has shape '
                        f'{other_height}x{other_width}, '
                        f'expected {height}x{width}'
                    )
                ret = exr_get_channels(
                    <exr_const_context_t> context,
                    pi,
                    &other_chlist
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_channels', ret)
                if other_chlist.num_channels != num_channels:
                    raise ValueError(
                        f'part {pi} has {other_chlist.num_channels} channels, '
                        f'expected {num_channels}'
                    )
                # validate channel names and sampling match first part
                for c in range(num_channels):
                    if (
                        strcmp(
                            other_chlist.entries[c].name.str,
                            chlist.entries[c].name.str,
                        ) != 0
                    ):
                        raise ValueError(
                            f'part {pi} channel {c} name mismatch'
                        )
                    if (
                        other_chlist.entries[c].x_sampling != 1
                        or other_chlist.entries[c].y_sampling != 1
                    ):
                        raise ValueError(
                            'sub-sampled channels are not supported'
                        )
                # promote pixel type across parts
                for c in range(other_chlist.num_channels):
                    if other_chlist.entries[c].pixel_type == EXR_PIXEL_FLOAT:
                        ptype = EXR_PIXEL_FLOAT
                    elif (
                        other_chlist.entries[c].pixel_type == EXR_PIXEL_HALF
                        and ptype == EXR_PIXEL_UINT
                    ):
                        ptype = EXR_PIXEL_HALF

        # determine dtype and allocate output
        if ptype == EXR_PIXEL_HALF:
            dtype = numpy.float16
            bpe = 2
        elif ptype == EXR_PIXEL_FLOAT:
            dtype = numpy.float32
            bpe = 4
        elif ptype == EXR_PIXEL_UINT:
            dtype = numpy.uint32
            bpe = 4
        else:
            raise ValueError(f'unsupported pixel type {ptype}')

        channels = <ssize_t> num_channels

        ch_idx = <int*> malloc(channels * sizeof(int))
        if ch_idx == NULL:
            raise MemoryError('failed to allocate channel index array')
        # Reverse channel order only when names match the encoder's own
        # RGBA alphabetical scheme: Y, AY, BGR, ABGR.
        # External files with other names (RG, XA, layers, etc.) are
        # left in the on-disk alphabetical order instead.
        if channels == 1:
            do_reverse = (
                chlist.entries[0].name.str == b'Y'
            )
        elif channels == 2:
            do_reverse = (
                chlist.entries[0].name.str == b'A'
                and chlist.entries[1].name.str == b'Y'
            )
        elif channels == 3:
            do_reverse = (
                chlist.entries[0].name.str == b'B'
                and chlist.entries[1].name.str == b'G'
                and chlist.entries[2].name.str == b'R'
            )
        elif channels == 4:
            do_reverse = (
                chlist.entries[0].name.str == b'A'
                and chlist.entries[1].name.str == b'B'
                and chlist.entries[2].name.str == b'G'
                and chlist.entries[3].name.str == b'R'
            )
        else:
            do_reverse = False
        if do_reverse:
            for c in range(channels):
                ch_idx[c] = <int> (channels - 1 - c)
        else:
            for c in range(channels):
                ch_idx[c] = <int> c

        nframes = part_end - part_start
        if nframes == 1:
            if channels == 1:
                shape = (height, width)
            elif is_planar:
                shape = (channels, height, width)
            else:
                shape = (height, width, channels)
        else:
            if channels == 1:
                shape = (nframes, height, width)
            elif is_planar:
                shape = (nframes, channels, height, width)
            else:
                shape = (nframes, height, width, channels)

        out = _create_array(out, shape, dtype)
        dst = out
        dst_ptr = <uint8_t*> dst.data

        if is_planar and channels > 1:
            pixel_stride = bpe
            line_stride = width * bpe
            frame_stride = channels * height * width * bpe
            ch_stride = height * width * bpe
        elif channels == 1:
            pixel_stride = bpe
            line_stride = width * bpe
            frame_stride = height * width * bpe
            ch_stride = bpe
        else:
            pixel_stride = channels * bpe
            line_stride = width * channels * bpe
            frame_stride = height * width * channels * bpe
            ch_stride = bpe

        # decode all parts
        with nogil:
            for frame in range(nframes):
                part_index = part_start + <int> frame
                frame_offset = frame * frame_stride if nframes > 1 else 0

                ret = exr_get_storage(
                    <exr_const_context_t> context,
                    part_index, &part_storage
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_storage', ret)

                ret = exr_get_data_window(
                    <exr_const_context_t> context,
                    part_index, &part_datawin
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_get_data_window', ret)
                part_datawin_min_y = part_datawin.min.y

                if part_storage == EXR_STORAGE_TILED:
                    _exr_decode_tiled(
                        <exr_const_context_t> context,
                        part_index,
                        dst_ptr,
                        ch_idx,
                        frame_offset,
                        ch_stride,
                        line_stride,
                        pixel_stride,
                        <int16_t> bpe,
                        <uint16_t> ptype,
                        width, height,
                    )
                else:
                    _exr_decode_scanline(
                        <exr_const_context_t> context,
                        part_index,
                        dst_ptr,
                        ch_idx,
                        frame_offset,
                        ch_stride,
                        line_stride,
                        pixel_stride,
                        <int16_t> bpe,
                        <uint16_t> ptype,
                        width, height,
                        part_datawin_min_y,
                    )

    finally:
        if context != NULL:
            exr_finish(&context)
        free(ch_idx)

    return out


cdef int _exr_decode_tiled(
    exr_const_context_t context,
    int part_index,
    uint8_t* dst_ptr,
    const int* ch_idx,
    ssize_t frame_offset,
    ssize_t ch_stride,
    ssize_t line_stride,
    ssize_t pixel_stride,
    int16_t bpe,
    uint16_t ptype,
    int32_t width,
    int32_t height,
) except -1 nogil:
    """Decode one tiled part into output buffer."""
    cdef:
        exr_result_t ret
        exr_chunk_info_t cinfo
        exr_decode_pipeline_t decoder
        int decoder_initialized = 0
        uint32_t tile_xsize, tile_ysize
        int32_t tile_x, tile_y
        int32_t tiles_across, tiles_down
        uint8_t* base_ptr
        ssize_t c
        int32_t y

    ret = exr_get_tile_descriptor(
        context, part_index, &tile_xsize, &tile_ysize, NULL, NULL
    )
    if ret != EXR_ERR_SUCCESS:
        raise ExrError('exr_get_tile_descriptor', ret)

    tiles_across = (
        (width + <int32_t> tile_xsize - 1) / <int32_t> tile_xsize
    )
    tiles_down = (
        (height + <int32_t> tile_ysize - 1) / <int32_t> tile_ysize
    )

    try:
        for tile_y in range(tiles_down):
            for tile_x in range(tiles_across):
                ret = exr_read_tile_chunk_info(
                    context, part_index,
                    tile_x, tile_y, 0, 0, &cinfo
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_read_tile_chunk_info', ret)

                if not decoder_initialized:
                    ret = exr_decoding_initialize(
                        context, part_index, &cinfo, &decoder
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError(
                            'exr_decoding_initialize', ret
                        )
                    decoder_initialized = 1

                    y = tile_y * <int32_t> tile_ysize
                    for c in range(decoder.channel_count):
                        base_ptr = (
                            dst_ptr
                            + frame_offset
                            + ch_idx[c] * ch_stride
                            + y * line_stride
                            + tile_x
                            * <ssize_t> tile_xsize
                            * pixel_stride
                        )
                        decoder.channels[c].decode_to_ptr = base_ptr
                        decoder.channels[c].user_pixel_stride = (
                            <int32_t> pixel_stride
                        )
                        decoder.channels[c].user_line_stride = (
                            <int32_t> line_stride
                        )
                        decoder.channels[c].user_bytes_per_element = bpe
                        decoder.channels[c].user_data_type = ptype

                    ret = exr_decoding_choose_default_routines(
                        context, part_index, &decoder
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError(
                            'exr_decoding_choose_default_routines', ret
                        )
                else:
                    ret = exr_decoding_update(
                        context, part_index, &cinfo, &decoder
                    )
                    if ret != EXR_ERR_SUCCESS:
                        raise ExrError('exr_decoding_update', ret)

                    y = tile_y * <int32_t> tile_ysize
                    for c in range(decoder.channel_count):
                        base_ptr = (
                            dst_ptr
                            + frame_offset
                            + ch_idx[c] * ch_stride
                            + y * line_stride
                            + tile_x
                            * <ssize_t> tile_xsize
                            * pixel_stride
                        )
                        decoder.channels[c].decode_to_ptr = base_ptr

                ret = exr_decoding_run(
                    context, part_index, &decoder
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_decoding_run', ret)
    finally:
        if decoder_initialized:
            exr_decoding_destroy(context, &decoder)
    return 0


cdef int _exr_decode_scanline(
    exr_const_context_t context,
    int part_index,
    uint8_t* dst_ptr,
    const int* ch_idx,
    ssize_t frame_offset,
    ssize_t ch_stride,
    ssize_t line_stride,
    ssize_t pixel_stride,
    int16_t bpe,
    uint16_t ptype,
    int32_t width,
    int32_t height,
    int32_t data_window_min_y,
) except -1 nogil:
    """Decode one scanline part into output buffer."""
    cdef:
        exr_result_t ret
        exr_chunk_info_t cinfo
        exr_decode_pipeline_t decoder
        int decoder_initialized = 0
        int32_t scanlines_per_chunk
        int32_t scanline_y
        uint8_t* base_ptr
        ssize_t c
        int32_t y

    ret = exr_get_scanlines_per_chunk(
        context, part_index, &scanlines_per_chunk
    )
    if ret != EXR_ERR_SUCCESS:
        raise ExrError('exr_get_scanlines_per_chunk', ret)

    try:
        y = 0
        while y < height:
            scanline_y = data_window_min_y + y

            ret = exr_read_scanline_chunk_info(
                context, part_index, scanline_y, &cinfo
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_read_scanline_chunk_info', ret)

            if not decoder_initialized:
                ret = exr_decoding_initialize(
                    context, part_index, &cinfo, &decoder
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_decoding_initialize', ret)
                decoder_initialized = 1

                for c in range(decoder.channel_count):
                    base_ptr = (
                        dst_ptr
                        + frame_offset
                        + ch_idx[c] * ch_stride
                        + y * line_stride
                    )
                    decoder.channels[c].decode_to_ptr = base_ptr
                    decoder.channels[c].user_pixel_stride = (
                        <int32_t> pixel_stride
                    )
                    decoder.channels[c].user_line_stride = (
                        <int32_t> line_stride
                    )
                    decoder.channels[c].user_bytes_per_element = bpe
                    decoder.channels[c].user_data_type = ptype

                ret = exr_decoding_choose_default_routines(
                    context, part_index, &decoder
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError(
                        'exr_decoding_choose_default_routines', ret
                    )
            else:
                ret = exr_decoding_update(
                    context, part_index, &cinfo, &decoder
                )
                if ret != EXR_ERR_SUCCESS:
                    raise ExrError('exr_decoding_update', ret)

                for c in range(decoder.channel_count):
                    base_ptr = (
                        dst_ptr
                        + frame_offset
                        + ch_idx[c] * ch_stride
                        + y * line_stride
                    )
                    decoder.channels[c].decode_to_ptr = base_ptr

            ret = exr_decoding_run(
                context, part_index, &decoder
            )
            if ret != EXR_ERR_SUCCESS:
                raise ExrError('exr_decoding_run', ret)

            y += scanlines_per_chunk
    finally:
        if decoder_initialized:
            exr_decoding_destroy(context, &decoder)
    return 0


# Memory stream structures and callbacks for in-memory I/O

ctypedef struct mem_stream_t:
    const uint8_t* data
    uint64_t size

ctypedef struct write_stream_t:
    uint8_t* data
    uint64_t size
    uint64_t capacity


cdef int64_t _exr_read_func(
    exr_const_context_t ctxt,
    void* userdata,
    void* buffer,
    uint64_t sz,
    uint64_t offset,
    exr_stream_error_func_ptr_t error_cb
) noexcept nogil:
    """Read callback for memory-based I/O."""
    cdef:
        mem_stream_t* stream = <mem_stream_t*> userdata

    if offset + sz > stream.size:
        if offset >= stream.size:
            return 0
        sz = stream.size - offset
    memcpy(buffer, stream.data + offset, sz)
    return <int64_t> sz


cdef int64_t _exr_read_size_func(
    exr_const_context_t ctxt,
    void* userdata
) noexcept nogil:
    """Size query callback for memory-based I/O."""
    cdef:
        mem_stream_t* stream = <mem_stream_t*> userdata

    return <int64_t> stream.size


cdef int64_t _exr_write_func(
    exr_const_context_t ctxt,
    void* userdata,
    const void* buffer,
    uint64_t sz,
    uint64_t offset,
    exr_stream_error_func_ptr_t error_cb
) noexcept nogil:
    """Write callback for memory-based I/O."""
    cdef:
        write_stream_t* stream = <write_stream_t*> userdata
        uint64_t needed = offset + sz
        uint64_t new_capacity
        uint8_t* new_data

    if needed > stream.capacity:
        new_capacity = stream.capacity * 2
        if new_capacity < needed:
            new_capacity = needed
        new_data = <uint8_t*> realloc(stream.data, new_capacity)
        if new_data == NULL:
            return -1
        stream.data = new_data
        stream.capacity = new_capacity

    memcpy(stream.data + offset, buffer, sz)
    if needed > stream.size:
        stream.size = needed
    return <int64_t> sz


cdef int64_t _exr_write_size_func(
    exr_const_context_t ctxt,
    void* userdata
) noexcept nogil:
    """Size query callback for write stream."""
    cdef:
        write_stream_t* stream = <write_stream_t*> userdata

    return <int64_t> stream.size


cdef void _exr_write_destroy_func(
    exr_const_context_t ctxt,
    void* userdata,
    int failed
) noexcept nogil:
    """Destroy callback for write stream. Does not free data."""
    pass
