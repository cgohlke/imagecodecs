# imagecodecs/_pcx.pyx
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

"""PCX/DCX codec for the imagecodecs package."""

include '_shared.pxi'

cdef:
    const uint32_t PCX_MAGIC = 0x0A
    const uint32_t PCX_VERSION5 = 5
    const uint32_t PCX_ENCODING_RLE = 1
    const uint32_t PCX_PALETTE_MARKER = 0x0C
    const uint32_t DCX_MAGIC = 0x3ADE68B1

    packed struct pcx_header_t:
        uint8_t manufacturer  # must be 0x0A
        uint8_t version  # 0, 2, 3, 4, 5
        uint8_t encoding  # 0=uncompressed, 1=RLE
        uint8_t bits_per_pixel  # bits per pixel per plane
        uint16_t xmin
        uint16_t ymin
        uint16_t xmax
        uint16_t ymax
        uint16_t hdpi
        uint16_t vdpi
        uint8_t[48] ega_palette  # EGA palette
        uint8_t reserved
        uint8_t nplanes  # number of color planes
        uint16_t bytes_per_line  # bytes per scanline per plane (must be even)
        uint16_t palette_type  # 1=color/BW, 2=grayscale
        uint16_t hscreen_size
        uint16_t vscreen_size
        uint8_t[54] padding  # pad to 128 bytes

    struct pcx_page_t:
        pcx_header_t header
        ssize_t offset
        ssize_t end
        ssize_t width
        ssize_t height
        ssize_t samples
        ssize_t bytes_per_line
        uint8_t[768] palette


class PCX:
    """PCX codec constants."""

    available = True


class PcxError(RuntimeError):
    """PCX codec exceptions."""


def pcx_version():
    """Return PCX codec version string."""
    return 'pcx 2026.5.10'


def pcx_check(const uint8_t[::1] data, /):
    """Return whether data is PCX or DCX encoded or None if unknown."""
    cdef:
        ssize_t srcsize = data.nbytes

    if srcsize < 128:
        return False

    # check DCX magic (little-endian 0x3ADE68B1)
    if srcsize >= 4:
        if (
            <uint32_t> data[0]
            | (<uint32_t> data[1] << 8)
            | (<uint32_t> data[2] << 16)
            | (<uint32_t> data[3] << 24)
        ) == DCX_MAGIC:
            return True

    # check PCX header
    return (
        data[0] == PCX_MAGIC
        and data[1] <= 5  # version
        and data[2] in {0, 1}  # encoding
        and data[3] in {1, 2, 4, 8}  # bits_per_pixel
        and data[65] in {1, 2, 3, 4}  # nplanes
    )


def pcx_encode(
    data,
    /,
    *,
    out=None,
):
    """Return PCX or DCX encoded image.

    Supported:

    - 8-bit grayscale (uint8, 2D or 3D with frames): 1 plane, 8 bpp
    - 24-bit RGB (uint8, 3D or 4D with frames): 3 planes, 8 bpp
    - 32-bit RGBA (uint8, 3D or 4D with frames): 4 planes, 8 bpp

    Multi-frame input (4D, or 3D grayscale frames) is encoded as DCX.

    PCX always uses RLE encoding. Scanline data is stored per-plane.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst
        uint8_t* srcptr = <uint8_t*> src.data
        uint8_t* dstptr = NULL
        ssize_t dstsize, actual_size, page_size
        ssize_t frame_size, dcx_header_size
        ssize_t i, bytes_per_line, page_bounds
        imagelayout_t layout

    if data is out:
        raise ValueError('cannot encode in-place')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_RGB | IC_ALPHA | IC_FRAMES,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (
        layout.height > 0
        and layout.height <= UINT16_MAX
        and layout.width > 0
        and layout.width <= UINT16_MAX
    ):
        raise ValueError('invalid data shape or dtype')

    if layout.samples == 2:
        raise ValueError('grayscale+alpha not supported by PCX format')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            bytes_per_line = layout.width + layout.width % 2
            page_bounds = (
                128
                + layout.height * layout.samples * bytes_per_line * 2
                + (769 if layout.samples == 1 else 0)
            )
            if layout.frames > 1:
                dcx_header_size = 4 + (layout.frames + 1) * 4
                dstsize = dcx_header_size + layout.frames * page_bounds
            else:
                dstsize = page_bounds
        out = _create_output(outtype, dstsize)

    dst = out
    dstptr = <uint8_t*> &dst[0]
    dstsize = dst.nbytes

    with nogil:
        if layout.frames > 1:
            dcx_header_size = 4 + (layout.frames + 1) * 4
            if dstsize < dcx_header_size:
                raise ValueError(f'output too small {dstsize}')

            page_size = 0
            # write DCX magic (little-endian)
            dstptr[0] = <uint8_t> (DCX_MAGIC & 0xFF)
            dstptr[1] = <uint8_t> ((DCX_MAGIC >> 8) & 0xFF)
            dstptr[2] = <uint8_t> ((DCX_MAGIC >> 16) & 0xFF)
            dstptr[3] = <uint8_t> ((DCX_MAGIC >> 24) & 0xFF)

            actual_size = dcx_header_size
            frame_size = layout.height * layout.width * layout.samples

            for i in range(layout.frames):
                # write page offset to table
                dstptr[4 + i * 4] = <uint8_t> (actual_size & 0xFF)
                dstptr[4 + i * 4 + 1] = <uint8_t> ((actual_size >> 8) & 0xFF)
                dstptr[4 + i * 4 + 2] = <uint8_t> ((actual_size >> 16) & 0xFF)
                dstptr[4 + i * 4 + 3] = <uint8_t> ((actual_size >> 24) & 0xFF)

                page_size = _pcx_encode_page(
                    srcptr + i * frame_size,
                    dstptr + actual_size,
                    dstsize - actual_size,
                    layout.width,
                    layout.height,
                    layout.samples,
                )
                if page_size == 0:
                    break
                actual_size += page_size

            # terminating zero offset
            memset(dstptr + 4 + layout.frames * 4, 0, 4)

            if page_size == 0:
                raise ValueError('output too small')

        else:
            actual_size = _pcx_encode_page(
                srcptr,
                dstptr,
                dstsize,
                layout.width,
                layout.height,
                layout.samples,
            )

            if actual_size == 0:
                raise ValueError('output too small')

    del dst
    return _return_output(out, dstsize, actual_size, outgiven)


def pcx_decode(
    data,
    /,
    index=None,
    *,
    out=None,
):
    """Return decoded PCX or DCX image.

    Supported:

    - 8 bpp, 1 plane: 8-bit paletted or grayscale
    - 8 bpp, 3 planes: 24-bit RGB
    - 8 bpp, 4 planes: 32-bit RGBA
    - 4 bpp, 1 plane: 16-color paletted (decoded to RGB uint8)
    - 2 bpp, 1 plane: 4-color CGA paletted (decoded to RGB uint8)
    - 1 bpp, 1 plane: 1-bit monochrome (decoded to bool)
    - 1 bpp, 2-4 planes: multi-plane paletted (decoded to RGB uint8)

    DCX files (multi-page PCX container) are supported.
    If ``index`` is None for DCX, return all pages stacked along axis 0.
    If ``index`` is an integer, return a single page.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t* srcptr = <const uint8_t*> &src[0]
        numpy.ndarray dst
        uint8_t* dstptr
        ssize_t srcsize = src.nbytes
        ssize_t sidx, pi, nframes, page_size
        ssize_t pcx_end, ref_width, ref_height, ref_samples
        ssize_t[1024] page_offsets
        uint32_t page_offset
        pcx_page_t page

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize < 128:
        raise PcxError(f'invalid data size {srcsize} < 128')

    # check for DCX container
    if (
        <uint32_t> src[0]
        | (<uint32_t> src[1] << 8)
        | (<uint32_t> src[2] << 16)
        | (<uint32_t> src[3] << 24)
    ) == DCX_MAGIC:
        # collect all page offsets
        nframes = 0
        for pi in range(1024):
            if 4 + (pi + 1) * 4 > srcsize:
                break
            sidx = 4 + pi * 4
            page_offset = (
                <uint32_t> src[sidx]
                | (<uint32_t> src[sidx + 1] << 8)
                | (<uint32_t> src[sidx + 2] << 16)
                | (<uint32_t> src[sidx + 3] << 24)
            )
            if page_offset == 0:
                break
            page_offsets[nframes] = <ssize_t> page_offset
            nframes += 1

        if nframes == 0:
            raise PcxError('DCX contains no pages')

        if index is not None:
            page_index = int(index)
            if page_index < 0 or page_index >= nframes:
                raise PcxError(
                    f'page index {page_index} out of range '
                    f'(0..{nframes - 1})'
                )
            pcx_end = (
                page_offsets[page_index + 1]
                if page_index + 1 < nframes
                else srcsize
            )
            _pcx_read_page_info(
                srcptr, srcsize, page_offsets[page_index], pcx_end, &page
            )

            if page.samples > 1:
                shape = int(page.height), int(page.width), int(page.samples)
            else:
                shape = int(page.height), int(page.width)
            dtype = (
                numpy.bool_
                if page.header.bits_per_pixel == 1
                and page.header.nplanes == 1
                else numpy.uint8
            )
            out = _create_array(out, shape, dtype)
            dst = out
            dstptr = <uint8_t*> dst.data
            with nogil:
                _pcx_decode_page(srcptr, dstptr, &page)
            return out

        # index is None: parse first page for reference shape
        pcx_end = page_offsets[1] if nframes > 1 else srcsize
        _pcx_read_page_info(srcptr, srcsize, page_offsets[0], pcx_end, &page)
        ref_width = page.width
        ref_height = page.height
        ref_samples = page.samples

        # allocate output for all frames
        page_size = ref_height * ref_width * ref_samples
        if nframes > 1:
            if ref_samples > 1:
                shape = (
                    int(nframes),
                    int(ref_height),
                    int(ref_width),
                    int(ref_samples),
                )
            else:
                shape = int(nframes), int(ref_height), int(ref_width)
        else:
            if ref_samples > 1:
                shape = int(ref_height), int(ref_width), int(ref_samples)
            else:
                shape = int(ref_height), int(ref_width)
        dtype = (
            numpy.bool_
            if page.header.bits_per_pixel == 1
            and page.header.nplanes == 1
            else numpy.uint8
        )
        out = _create_array(out, shape, dtype)
        dst = out
        dstptr = <uint8_t*> dst.data

        with nogil:
            # decode first page (already parsed)
            _pcx_decode_page(srcptr, dstptr, &page)

            # decode remaining pages
            for pi in range(1, nframes):
                pcx_end = page_offsets[pi + 1] if pi + 1 < nframes else srcsize
                _pcx_read_page_info(
                    srcptr, srcsize, page_offsets[pi], pcx_end, &page
                )
                if (
                    page.width != ref_width
                    or page.height != ref_height
                    or page.samples != ref_samples
                ):
                    raise PcxError(
                        f'DCX page {pi} shape '
                        f'{page.height}x{page.width}x{page.samples} '
                        f'does not match first page '
                        f'{ref_height}x{ref_width}x{ref_samples}'
                    )
                _pcx_decode_page(srcptr, dstptr + pi * page_size, &page)

        return out

    # plain PCX (not DCX)
    _pcx_read_page_info(srcptr, srcsize, 0, srcsize, &page)

    if page.samples > 1:
        shape = int(page.height), int(page.width), int(page.samples)
    else:
        shape = int(page.height), int(page.width)
    dtype = (
        numpy.bool_
        if page.header.bits_per_pixel == 1
        and page.header.nplanes == 1
        else numpy.uint8
    )
    out = _create_array(out, shape, dtype)
    dst = out
    dstptr = <uint8_t*> dst.data
    with nogil:
        _pcx_decode_page(srcptr, dstptr, &page)
    return out


cdef ssize_t _pcx_encode_page(
    const uint8_t* src,
    uint8_t* dst,
    ssize_t dstsize,
    ssize_t width,
    ssize_t height,
    ssize_t samples,
) noexcept nogil:
    """Encode single PCX page. Return bytes written or 0 on error."""
    cdef:
        ssize_t bytes_per_line = width + width % 2
        ssize_t dstindex, i, j, plane, run
        uint8_t this_byte, next_byte
        pcx_header_t header

    if dstsize < 128:
        return 0

    memset(<void*> &header, 0, sizeof(pcx_header_t))
    header.manufacturer = PCX_MAGIC
    header.version = PCX_VERSION5
    header.encoding = PCX_ENCODING_RLE
    header.bits_per_pixel = 8
    header.xmax = <uint16_t> (width - 1)
    header.ymax = <uint16_t> (height - 1)
    header.hdpi = 72
    header.vdpi = 72
    header.nplanes = <uint8_t> samples
    header.bytes_per_line = <uint16_t> bytes_per_line
    header.palette_type = 2 if samples == 1 else 1

    memcpy(<void*> dst, <const void*> &header, sizeof(pcx_header_t))
    dstindex = 128

    for i in range(height):
        for plane in range(samples):
            j = 0
            while j < bytes_per_line:
                if j < width:
                    if samples == 1:
                        this_byte = src[i * width + j]
                    else:
                        this_byte = src[
                            i * width * samples
                            + j * samples
                            + plane
                        ]
                else:
                    this_byte = 0

                run = 1
                while run < 63 and j + run < bytes_per_line:
                    if j + run < width:
                        if samples == 1:
                            next_byte = src[i * width + j + run]
                        else:
                            next_byte = src[
                                i * width * samples
                                + (j + run) * samples
                                + plane
                            ]
                    else:
                        next_byte = 0
                    if next_byte != this_byte:
                        break
                    run += 1

                if dstindex + 2 > dstsize:
                    return 0

                if run > 1 or (this_byte & 0xC0) == 0xC0:
                    dst[dstindex] = <uint8_t> (0xC0 | run)
                    dstindex += 1
                    dst[dstindex] = this_byte
                    dstindex += 1
                else:
                    dst[dstindex] = this_byte
                    dstindex += 1

                j += run

    if samples == 1:
        if dstindex + 769 > dstsize:
            return 0
        dst[dstindex] = PCX_PALETTE_MARKER
        dstindex += 1
        for j in range(256):
            dst[dstindex] = <uint8_t> j  # R
            dstindex += 1
            dst[dstindex] = <uint8_t> j  # G
            dstindex += 1
            dst[dstindex] = <uint8_t> j  # B
            dstindex += 1

    return dstindex


cdef ssize_t _pcx_read_page_info(
    const uint8_t* src,
    ssize_t srcsize,
    ssize_t pcx_offset,
    ssize_t pcx_end,
    pcx_page_t* page,
) except -1 nogil:
    """Parse PCX header at offset and fill page info."""
    cdef:
        ssize_t i
        bint has_vga_palette, is_grayscale

    if pcx_offset + 128 > srcsize:
        raise PcxError(f'page offset {pcx_offset} past end of file')

    page.offset = pcx_offset
    page.end = pcx_end

    memcpy(
        <void*> &page.header,
        <const void*> &src[pcx_offset],
        sizeof(pcx_header_t),
    )

    if page.header.manufacturer != PCX_MAGIC:
        raise PcxError(
            f'invalid PCX manufacturer byte {page.header.manufacturer}'
        )
    if page.header.encoding not in {0, 1}:
        raise PcxError(f'unsupported encoding {page.header.encoding}')

    page.width = <ssize_t> page.header.xmax - <ssize_t> page.header.xmin + 1
    page.height = <ssize_t> page.header.ymax - <ssize_t> page.header.ymin + 1
    page.bytes_per_line = <ssize_t> page.header.bytes_per_line

    if page.width <= 0 or page.height <= 0:
        raise PcxError(f'invalid image dimensions {page.width}x{page.height}')
    if page.bytes_per_line <= 0:
        raise PcxError(f'invalid bytes_per_line={page.bytes_per_line}')

    # clamp width to what bytes_per_line can provide
    # some fax-format files have bytes_per_line < ceil(width * bpp / 8)
    if page.header.bits_per_pixel == 1:
        if page.width > page.bytes_per_line * 8:
            page.width = page.bytes_per_line * 8
    elif page.header.bits_per_pixel == 2:
        if page.width > page.bytes_per_line * 4:
            page.width = page.bytes_per_line * 4
    elif page.header.bits_per_pixel == 4:
        if page.width > page.bytes_per_line * 2:
            page.width = page.bytes_per_line * 2
    elif page.width > page.bytes_per_line:
        page.width = page.bytes_per_line

    # determine output format and read palette
    memset(<void*> page.palette, 0, 768)

    if page.header.bits_per_pixel == 8 and page.header.nplanes == 1:
        # 8-bit paletted or grayscale
        has_vga_palette = False
        is_grayscale = page.header.palette_type == 2

        if pcx_end - 769 >= pcx_offset + 128:
            if src[pcx_end - 769] == PCX_PALETTE_MARKER:
                has_vga_palette = True
                for i in range(768):
                    page.palette[i] = src[pcx_end - 768 + i]

                if not is_grayscale:
                    # check if palette is a grayscale ramp
                    is_grayscale = True
                    for i in range(256):
                        if (
                            page.palette[i * 3] != <uint8_t> i
                            or page.palette[i * 3 + 1] != <uint8_t> i
                            or page.palette[i * 3 + 2] != <uint8_t> i
                        ):
                            is_grayscale = False
                            break

        if is_grayscale or not has_vga_palette:
            page.samples = 1
        else:
            page.samples = 3  # expand palette to RGB

    elif page.header.bits_per_pixel == 8 and page.header.nplanes == 3:
        page.samples = 3
    elif page.header.bits_per_pixel == 8 and page.header.nplanes == 4:
        page.samples = 4
    elif (
        page.header.bits_per_pixel == 1
        and page.header.nplanes == 1
    ):
        page.samples = 1
    elif (
        page.header.bits_per_pixel == 1
        and page.header.nplanes in {2, 3, 4}
    ):
        # multi-plane bitplane: N planes x 1 bpp -> N-bit palette -> RGB
        page.samples = 3
        for i in range(48):
            page.palette[i] = page.header.ega_palette[i]
    elif (
        page.header.bits_per_pixel in {2, 4}
        and page.header.nplanes == 1
    ):
        # sub-byte paletted: 2 or 4 bpp -> palette -> RGB
        page.samples = 3
        for i in range(48):
            page.palette[i] = page.header.ega_palette[i]
    else:
        raise PcxError(
            f'unsupported format: {page.header.bits_per_pixel} bpp, '
            f'{page.header.nplanes} planes'
        )
    return 0


cdef void _pcx_decode_page(
    const uint8_t* src,
    uint8_t* dst,
    const pcx_page_t* page,
) noexcept nogil:
    """Decode single PCX page into dst buffer."""
    cdef:
        ssize_t sidx, i, j, k, plane, row_offset, rle_count
        ssize_t scanline_len, byte_idx, bit_shift, pal_idx
        ssize_t pixels_per_byte, pixel_in_byte, bit
        uint8_t rle_byte, val
        uint8_t* rowbuf = NULL

    # RLE runs can span plane boundaries within a scanline, so
    # multi-plane formats must be decoded as a flat buffer first
    if page.header.nplanes > 1:
        scanline_len = <ssize_t> page.header.nplanes * page.bytes_per_line
        rowbuf = <uint8_t*> malloc(scanline_len)
        if rowbuf == NULL:
            return
    else:
        scanline_len = 0

    sidx = page.offset + 128

    for i in range(page.height):
        if page.header.nplanes > 1:
            # decode entire scanline (all planes) into flat buffer
            memset(<void*> rowbuf, 0, scanline_len)
            j = 0
            while j < scanline_len and sidx < page.end:
                if page.header.encoding == PCX_ENCODING_RLE:
                    rle_byte = src[sidx]
                    sidx += 1
                    if (rle_byte & 0xC0) == 0xC0:
                        rle_count = rle_byte & 0x3F
                        if sidx >= page.end:
                            break
                        val = src[sidx]
                        sidx += 1
                    else:
                        rle_count = 1
                        val = rle_byte
                else:
                    rle_count = 1
                    val = src[sidx]
                    sidx += 1
                for k in range(rle_count):
                    if j >= scanline_len:
                        break
                    rowbuf[j] = val
                    j += 1

            if page.header.bits_per_pixel == 1:
                # multi-plane bitplane: combine N bit planes
                row_offset = i * page.width * 3
                for j in range(page.width):
                    byte_idx = j >> 3
                    bit_shift = 7 - (j & 7)
                    pal_idx = 0
                    for plane in range(<ssize_t> page.header.nplanes):
                        pal_idx |= (((
                            rowbuf[plane * page.bytes_per_line + byte_idx]
                            >> bit_shift
                        ) & 1) << plane)
                    dst[row_offset + j * 3] = page.palette[pal_idx * 3]
                    dst[row_offset + j * 3 + 1] = page.palette[pal_idx * 3 + 1]
                    dst[row_offset + j * 3 + 2] = page.palette[pal_idx * 3 + 2]
            else:
                # 8bpp multi-plane: scatter to interleaved output
                row_offset = i * page.width * page.samples
                for plane in range(<ssize_t> page.header.nplanes):
                    for j in range(page.width):
                        dst[row_offset + j * page.samples + plane] = rowbuf[
                            plane * page.bytes_per_line + j
                        ]

        elif page.header.bits_per_pixel in {2, 4}:
            # sub-byte paletted (single plane): 2 or 4 bpp
            pixels_per_byte = 8 // page.header.bits_per_pixel
            row_offset = i * page.width * 3
            j = 0
            while j < page.bytes_per_line and sidx < page.end:
                if page.header.encoding == PCX_ENCODING_RLE:
                    rle_byte = src[sidx]
                    sidx += 1
                    if (rle_byte & 0xC0) == 0xC0:
                        rle_count = rle_byte & 0x3F
                        if sidx >= page.end:
                            break
                        val = src[sidx]
                        sidx += 1
                    else:
                        rle_count = 1
                        val = rle_byte
                else:
                    rle_count = 1
                    val = src[sidx]
                    sidx += 1

                for k in range(rle_count):
                    if j >= page.bytes_per_line:
                        break
                    for pixel_in_byte in range(pixels_per_byte):
                        bit_shift = (
                            8
                            - page.header.bits_per_pixel
                            * (pixel_in_byte + 1)
                        )
                        pal_idx = (
                            val >> bit_shift
                        ) & ((1 << page.header.bits_per_pixel) - 1)
                        if (
                            j * pixels_per_byte + pixel_in_byte
                            < page.width
                        ):
                            dst[
                                row_offset
                                + (j * pixels_per_byte + pixel_in_byte)
                                * 3
                            ] = page.palette[pal_idx * 3]
                            dst[
                                row_offset
                                + (j * pixels_per_byte + pixel_in_byte)
                                * 3 + 1
                            ] = page.palette[pal_idx * 3 + 1]
                            dst[
                                row_offset
                                + (j * pixels_per_byte + pixel_in_byte)
                                * 3 + 2
                            ] = page.palette[pal_idx * 3 + 2]
                    j += 1

        elif page.header.bits_per_pixel == 1:
            # 1-bit monochrome (single plane)
            row_offset = i * page.width
            j = 0
            while j < page.bytes_per_line and sidx < page.end:
                if page.header.encoding == PCX_ENCODING_RLE:
                    rle_byte = src[sidx]
                    sidx += 1
                    if (rle_byte & 0xC0) == 0xC0:
                        rle_count = rle_byte & 0x3F
                        if sidx >= page.end:
                            break
                        val = src[sidx]
                        sidx += 1
                    else:
                        rle_count = 1
                        val = rle_byte
                else:
                    rle_count = 1
                    val = src[sidx]
                    sidx += 1

                for k in range(rle_count):
                    if j >= page.bytes_per_line:
                        break
                    for bit in range(8):
                        if j * 8 + bit < page.width:
                            dst[row_offset + j * 8 + bit] = (
                                1 if (val & (0x80 >> bit)) else 0
                            )
                    j += 1

        else:
            # 8-bit single plane (paletted or grayscale)
            row_offset = i * page.width * page.samples
            j = 0
            while j < page.bytes_per_line and sidx < page.end:
                if page.header.encoding == PCX_ENCODING_RLE:
                    rle_byte = src[sidx]
                    sidx += 1
                    if (rle_byte & 0xC0) == 0xC0:
                        rle_count = rle_byte & 0x3F
                        if sidx >= page.end:
                            break
                        val = src[sidx]
                        sidx += 1
                    else:
                        rle_count = 1
                        val = rle_byte
                else:
                    rle_count = 1
                    val = src[sidx]
                    sidx += 1

                for k in range(rle_count):
                    if j >= page.bytes_per_line:
                        break
                    if j < page.width:
                        if page.samples == 1:
                            dst[row_offset + j] = val
                        else:
                            # paletted: look up RGB
                            dst[row_offset + j * 3] = page.palette[val * 3]
                            dst[row_offset + j * 3 + 1] = (
                                page.palette[val * 3 + 1]
                            )
                            dst[row_offset + j * 3 + 2] = (
                                page.palette[val * 3 + 2]
                            )
                    j += 1

    if rowbuf != NULL:
        free(rowbuf)
