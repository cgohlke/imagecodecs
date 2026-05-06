# imagecodecs/_tga.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2023-2026, Christoph Gohlke
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

"""TGA (TARGA) codec for the imagecodecs package."""

include '_shared.pxi'


class TGA:
    """TGA codec constants."""

    available = True


class TgaError(RuntimeError):
    """TGA codec exceptions."""


def tga_version():
    """Return tga codec version string."""
    return 'tga 2026.5.10'


def tga_check(const uint8_t[::1] data, /):
    """Return whether data is TGA encoded or None if unknown."""
    if data.nbytes < 18:
        return False
    # TGA 2.0: last 18 bytes are signature "TRUEVISION-XFILE.\0"
    if data.nbytes >= 44:
        if bytes(
            data[data.nbytes - 18:data.nbytes]
        ) == b'TRUEVISION-XFILE.\x00':
            return True
    # heuristic: validate header fields for TGA 1.0 files
    if data[1] not in {0, 1}:  # colormap_type
        return False
    if data[2] not in {0, 1, 2, 3, 9, 10, 11}:  # image_type
        return False
    if data[16] not in {0, 8, 15, 16, 24, 32}:  # pixel_depth
        return False
    return None


def tga_encode(
    data,
    /,
    *,
    rle=False,
    out=None,
):
    """Return TGA encoded image.

    Supported:

    - 8-bit grayscale (uint8, 2D): type 3 or RLE type 11
    - 16-bit grayscale+alpha (uint8, 3D, 2 samples): type 3 or RLE type 11
    - 24-bit RGB (uint8, 3D, 3 samples): type 2 or RLE type 10, stored BGR
    - 32-bit RGBA (uint8, 3D, 4 samples): type 2 or RLE type 10, stored BGRA

    Not supported:

    - Paletted (colormap) images
    - 16-bit direct-color (RGB555)

    A TGA 2.0 footer is appended so that tga_check returns True.
    Images are encoded bottom-up (origin at lower-left), which is standard.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst
        uint8_t* srcptr = <uint8_t*> src.data
        uint8_t* dstptr = NULL
        uint8_t r0, g0, b0, a0
        ssize_t dstsize, dstindex, actual_size, minsize, err
        ssize_t i, j, k, srow, sroff, nxt, nxtoff, px
        ssize_t run, raw
        bint rle_ = bool(rle)
        tga_header_t header
        imagelayout_t layout

    if data is out:
        raise ValueError('cannot encode in-place')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_RGB | IC_ALPHA,
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

    # 18-byte header + pixel data (worst case) + 26-byte TGA 2.0 footer
    if rle_:
        minsize = 18 + 26 + layout.height * layout.width * (layout.samples + 1)
    else:
        minsize = 18 + 26 + layout.height * layout.width * layout.samples

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = minsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstptr = <uint8_t*> &dst[0]
    dstsize = dst.nbytes

    if rle_:
        # exact RLE size is not known upfront; require room for header+footer
        if dstsize < 44:
            raise ValueError(f'output too small {dstsize} < 44')
    elif dstsize < minsize:
        raise ValueError(f'output too small {dstsize} < {minsize}')

    err = 0
    with nogil:
        memset(<void*> &header, 0, sizeof(tga_header_t))
        header.colormap_type = 0
        if layout.samples == 1 or layout.samples == 2:
            header.image_type = (
                TGA_TYPE_RLE_GRAYSCALE if rle_ else TGA_TYPE_GRAYSCALE
            )
        else:
            header.image_type = (
                TGA_TYPE_RLE_TRUECOLOR if rle_ else TGA_TYPE_TRUECOLOR
            )
        header.width = <uint16_t> layout.width
        header.height = <uint16_t> layout.height
        header.pixel_depth = <uint8_t> (layout.samples * 8)
        # bit5=0: bottom-up; bits 0-3: alpha channel depth
        header.image_descriptor = (
            8 if (layout.samples == 2 or layout.samples == 4) else 0
        )
        memcpy(<void*> dstptr, <const void*> &header, sizeof(tga_header_t))
        dstindex = 18  # no image ID, no colormap

        if layout.samples == 1:
            # 8-bit grayscale
            if not rle_:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width
                    memcpy(
                        <void*> (dstptr + dstindex),
                        <const void*> (srcptr + sroff),
                        layout.width,
                    )
                    dstindex += layout.width
            else:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width
                    if dstindex + layout.width * 2 + 26 > dstsize:
                        err = 1
                        break
                    j = 0
                    while j < layout.width:
                        r0 = srcptr[sroff + j]
                        run = 1
                        while (
                            run < 128
                            and j + run < layout.width
                            and srcptr[sroff + j + run] == r0
                        ):
                            run += 1
                        if run >= 2:
                            dstptr[dstindex] = 0x80 | <uint8_t> (run - 1)
                            dstindex += 1
                            dstptr[dstindex] = r0
                            dstindex += 1
                            j += run
                        else:
                            raw = 1
                            while raw < 128 and j + raw < layout.width:
                                nxt = j + raw
                                if (
                                    nxt + 1 < layout.width
                                    and srcptr[sroff + nxt]
                                    == srcptr[sroff + nxt + 1]
                                ):
                                    break
                                raw += 1
                            dstptr[dstindex] = <uint8_t> (raw - 1)
                            dstindex += 1
                            for k in range(raw):
                                dstptr[dstindex] = srcptr[sroff + j + k]
                                dstindex += 1
                            j += raw
        elif layout.samples == 2:
            # 16-bit grayscale+alpha: store as L, A (no reordering needed)
            if not rle_:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 2
                    memcpy(
                        <void*> (dstptr + dstindex),
                        <const void*> (srcptr + sroff),
                        layout.width * 2,
                    )
                    dstindex += layout.width * 2
            else:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 2
                    if dstindex + layout.width * 3 + 26 > dstsize:
                        err = 1
                        break
                    j = 0
                    while j < layout.width:
                        r0 = srcptr[sroff + j * 2]
                        g0 = srcptr[sroff + j * 2 + 1]
                        run = 1
                        while (
                            run < 128
                            and j + run < layout.width
                            and srcptr[sroff + (j + run) * 2] == r0
                            and srcptr[sroff + (j + run) * 2 + 1] == g0
                        ):
                            run += 1
                        if run >= 2:
                            dstptr[dstindex] = 0x80 | <uint8_t> (run - 1)
                            dstindex += 1
                            dstptr[dstindex] = r0  # L
                            dstindex += 1
                            dstptr[dstindex] = g0  # A
                            dstindex += 1
                            j += run
                        else:
                            raw = 1
                            while raw < 128 and j + raw < layout.width:
                                nxt = j + raw
                                nxtoff = sroff + nxt * 2
                                if (
                                    nxt + 1 < layout.width
                                    and srcptr[nxtoff]
                                    == srcptr[sroff + (nxt + 1) * 2]
                                    and srcptr[nxtoff + 1]
                                    == srcptr[sroff + (nxt + 1) * 2 + 1]
                                ):
                                    break
                                raw += 1
                            dstptr[dstindex] = <uint8_t> (raw - 1)
                            dstindex += 1
                            for k in range(raw):
                                px = j + k
                                dstptr[dstindex] = srcptr[sroff + px * 2]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 2 + 1]
                                dstindex += 1
                            j += raw
        elif layout.samples == 3:
            # 24-bit RGB -> store as BGR
            if not rle_:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 3
                    for j in range(layout.width):
                        dstptr[dstindex] = srcptr[sroff + j * 3 + 2]  # B
                        dstindex += 1
                        dstptr[dstindex] = srcptr[sroff + j * 3 + 1]  # G
                        dstindex += 1
                        dstptr[dstindex] = srcptr[sroff + j * 3]  # R
                        dstindex += 1
            else:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 3
                    if dstindex + layout.width * 4 + 26 > dstsize:
                        err = 1
                        break
                    j = 0
                    while j < layout.width:
                        r0 = srcptr[sroff + j * 3]
                        g0 = srcptr[sroff + j * 3 + 1]
                        b0 = srcptr[sroff + j * 3 + 2]
                        run = 1
                        while run < 128 and j + run < layout.width:
                            nxtoff = sroff + (j + run) * 3
                            if (
                                srcptr[nxtoff] != r0
                                or srcptr[nxtoff + 1] != g0
                                or srcptr[nxtoff + 2] != b0
                            ):
                                break
                            run += 1
                        if run >= 2:
                            dstptr[dstindex] = 0x80 | <uint8_t> (run - 1)
                            dstindex += 1
                            dstptr[dstindex] = b0  # B
                            dstindex += 1
                            dstptr[dstindex] = g0  # G
                            dstindex += 1
                            dstptr[dstindex] = r0  # R
                            dstindex += 1
                            j += run
                        else:
                            raw = 1
                            while raw < 128 and j + raw < layout.width:
                                nxt = j + raw
                                nxtoff = sroff + nxt * 3
                                if (
                                    nxt + 1 < layout.width
                                    and srcptr[nxtoff]
                                    == srcptr[sroff + (nxt + 1) * 3]
                                    and srcptr[nxtoff + 1]
                                    == srcptr[sroff + (nxt + 1) * 3 + 1]
                                    and srcptr[nxtoff + 2]
                                    == srcptr[sroff + (nxt + 1) * 3 + 2]
                                ):
                                    break
                                raw += 1
                            dstptr[dstindex] = <uint8_t> (raw - 1)
                            dstindex += 1
                            for k in range(raw):
                                px = j + k
                                dstptr[dstindex] = srcptr[sroff + px * 3 + 2]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 3 + 1]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 3]
                                dstindex += 1
                            j += raw

        else:
            # 32-bit RGBA -> store as BGRA (layout.samples == 4)
            if not rle_:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 4
                    for j in range(layout.width):
                        dstptr[dstindex] = srcptr[sroff + j * 4 + 2]  # B
                        dstindex += 1
                        dstptr[dstindex] = srcptr[sroff + j * 4 + 1]  # G
                        dstindex += 1
                        dstptr[dstindex] = srcptr[sroff + j * 4]  # R
                        dstindex += 1
                        dstptr[dstindex] = srcptr[sroff + j * 4 + 3]  # A
                        dstindex += 1
            else:
                for i in range(layout.height):
                    srow = layout.height - 1 - i
                    sroff = srow * layout.width * 4
                    if dstindex + layout.width * 5 + 26 > dstsize:
                        err = 1
                        break
                    j = 0
                    while j < layout.width:
                        r0 = srcptr[sroff + j * 4]
                        g0 = srcptr[sroff + j * 4 + 1]
                        b0 = srcptr[sroff + j * 4 + 2]
                        a0 = srcptr[sroff + j * 4 + 3]
                        run = 1
                        while run < 128 and j + run < layout.width:
                            nxtoff = sroff + (j + run) * 4
                            if (
                                srcptr[nxtoff] != r0
                                or srcptr[nxtoff + 1] != g0
                                or srcptr[nxtoff + 2] != b0
                                or srcptr[nxtoff + 3] != a0
                            ):
                                break
                            run += 1
                        if run >= 2:
                            dstptr[dstindex] = 0x80 | <uint8_t> (run - 1)
                            dstindex += 1
                            dstptr[dstindex] = b0
                            dstindex += 1
                            dstptr[dstindex] = g0
                            dstindex += 1
                            dstptr[dstindex] = r0
                            dstindex += 1
                            dstptr[dstindex] = a0
                            dstindex += 1
                            j += run
                        else:
                            raw = 1
                            while raw < 128 and j + raw < layout.width:
                                nxt = j + raw
                                nxtoff = sroff + nxt * 4
                                if (
                                    nxt + 1 < layout.width
                                    and (
                                        srcptr[nxtoff]
                                        == srcptr[sroff + (nxt + 1) * 4]
                                    )
                                    and (
                                        srcptr[nxtoff + 1]
                                        == srcptr[sroff + (nxt + 1) * 4 + 1]
                                    )
                                    and (
                                        srcptr[nxtoff + 2]
                                        == srcptr[sroff + (nxt + 1) * 4 + 2]
                                    )
                                    and (
                                        srcptr[nxtoff + 3]
                                        == srcptr[sroff + (nxt + 1) * 4 + 3]
                                    )
                                ):
                                    break
                                raw += 1
                            dstptr[dstindex] = <uint8_t> (raw - 1)
                            dstindex += 1
                            for k in range(raw):
                                px = j + k
                                dstptr[dstindex] = srcptr[sroff + px * 4 + 2]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 4 + 1]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 4]
                                dstindex += 1
                                dstptr[dstindex] = srcptr[sroff + px * 4 + 3]
                                dstindex += 1
                            j += raw

        # TGA 2.0 footer (26 bytes)
        # extension area offset = 0 (absent)
        dstptr[dstindex] = 0
        dstptr[dstindex + 1] = 0
        dstptr[dstindex + 2] = 0
        dstptr[dstindex + 3] = 0
        # developer directory offset = 0 (absent)
        dstptr[dstindex + 4] = 0
        dstptr[dstindex + 5] = 0
        dstptr[dstindex + 6] = 0
        dstptr[dstindex + 7] = 0
        # "TRUEVISION-XFILE.\0"
        dstptr[dstindex + 8] = 84   # T
        dstptr[dstindex + 9] = 82   # R
        dstptr[dstindex + 10] = 85  # U
        dstptr[dstindex + 11] = 69  # E
        dstptr[dstindex + 12] = 86  # V
        dstptr[dstindex + 13] = 73  # I
        dstptr[dstindex + 14] = 83  # S
        dstptr[dstindex + 15] = 73  # I
        dstptr[dstindex + 16] = 79  # O
        dstptr[dstindex + 17] = 78  # N
        dstptr[dstindex + 18] = 45  # -
        dstptr[dstindex + 19] = 88  # X
        dstptr[dstindex + 20] = 70  # F
        dstptr[dstindex + 21] = 73  # I
        dstptr[dstindex + 22] = 76  # L
        dstptr[dstindex + 23] = 69  # E
        dstptr[dstindex + 24] = 46  # .
        dstptr[dstindex + 25] = 0   # \0
        dstindex += 26
        actual_size = dstindex

    if err:
        raise ValueError(
            f'output too small: buffer exhausted after {dstindex} bytes'
        )

    del dst
    return _return_output(out, dstsize, actual_size, outgiven)


def tga_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded TGA image.

    Supported:

    - Type 2 true-color: 24-bit (BGR->RGB), 32-bit (BGRA->RGBA),
      16-bit (RGB555->RGB)
    - Type 3 grayscale: 8-bit
    - Type 10 RLE true-color: same as type 2 but run-length encoded
    - Type 11 RLE grayscale: same as type 3 but run-length encoded
    - Type 1 color-mapped: 8-bit indices with 24 or 32-bit BGR(A) colormap
    - Type 9 RLE color-mapped: same as type 1 but RLE
    - Bottom-up (standard) and top-down (image_descriptor bit 5) layouts

    Not supported:

    - 16-bit colormap entries
    - Types 0 (no image), 32, 33 (compressed)

    """
    cdef:
        numpy.ndarray dst
        numpy.ndarray pixbuf
        uint8_t* dstptr = NULL
        uint8_t* pixptr = NULL
        const uint8_t* src_base
        const uint8_t* pline
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t height, width, samples
        ssize_t i, j, k, dstindex, row, file_row
        ssize_t pixel_offset, pixel_total, pixel_size
        ssize_t colormap_offset, colormap_origin_s, cmap_entry_size
        ssize_t srclimit, sroff, cmap_ptr, rle_total
        tga_header_t header
        bint topdown, has_rle
        uint8_t rle_hdr, rle_count
        uint8_t r, g, b, palval
        uint16_t pixel16

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize < 18:
        raise TgaError(f'invalid TGA size {srcsize} < 18')

    memcpy(<void*> &header, <const void*> &src[0], sizeof(tga_header_t))

    # validate header
    if header.colormap_type > 1:
        raise TgaError(f'unsupported {header.colormap_type=}')
    if header.image_type not in {
        TGA_TYPE_COLORMAP,
        TGA_TYPE_TRUECOLOR,
        TGA_TYPE_GRAYSCALE,
        TGA_TYPE_RLE_COLORMAP,
        TGA_TYPE_RLE_TRUECOLOR,
        TGA_TYPE_RLE_GRAYSCALE,
    }:
        raise TgaError(f'unsupported {header.image_type=}')
    if header.width == 0 or header.height == 0:
        raise TgaError('zero-dimension image')

    height = <ssize_t> header.height
    width = <ssize_t> header.width
    topdown = (header.image_descriptor & 0x20) != 0
    has_rle = header.image_type in (
        TGA_TYPE_RLE_COLORMAP,
        TGA_TYPE_RLE_TRUECOLOR,
        TGA_TYPE_RLE_GRAYSCALE,
    )

    # compute colormap and pixel offsets
    cmap_entry_size = (
        (<ssize_t> header.colormap_depth + 7) // 8
        if header.colormap_type == 1
        else 0
    )
    colormap_offset = 18 + <ssize_t> header.id_length
    pixel_offset = (
        colormap_offset + <ssize_t> header.colormap_length * cmap_entry_size
    )

    if pixel_offset > srcsize:
        raise TgaError('invalid header: pixel data offset past end of file')

    colormap_origin_s = <ssize_t> header.colormap_origin

    # determine output samples and encoded pixel_size
    if header.image_type in (
        TGA_TYPE_GRAYSCALE, TGA_TYPE_RLE_GRAYSCALE
    ):
        if header.pixel_depth == 8:
            samples = 1
            pixel_size = 1
        elif (
            header.pixel_depth == 16
            and (header.image_descriptor & 0xF) == 8
        ):
            # 16-bit grayscale+alpha: 8-bit luminance + 8-bit alpha (LA)
            samples = 2
            pixel_size = 2
        else:
            raise TgaError(f'unsupported grayscale {header.pixel_depth=}')
    elif header.image_type in (
        TGA_TYPE_COLORMAP, TGA_TYPE_RLE_COLORMAP
    ):
        if header.pixel_depth != 8:
            raise TgaError(f'unsupported colormap {header.pixel_depth=}')
        if cmap_entry_size not in (3, 4):
            raise TgaError(f'unsupported {header.colormap_depth=}')
        samples = cmap_entry_size  # 3 (RGB) or 4 (RGBA)
        pixel_size = 1
    else:
        # true-color
        if header.pixel_depth == 8:
            raise TgaError(
                'true-color with pixel_depth=8: use colormap type instead'
            )
        elif header.pixel_depth in (15, 16):
            samples = 3
            pixel_size = 2
        elif header.pixel_depth == 24:
            samples = 3
            pixel_size = 3
        elif header.pixel_depth == 32:
            samples = 4
            pixel_size = 4
        else:
            raise TgaError(f'unsupported {header.pixel_depth=}')

    pixel_total = height * width * pixel_size

    if not has_rle:
        if pixel_offset + pixel_total > srcsize:
            raise TgaError('pixel data extends past end of file')

    if header.image_type in (
        TGA_TYPE_COLORMAP, TGA_TYPE_RLE_COLORMAP
    ):
        if (
            colormap_offset
            + (colormap_origin_s + 256) * cmap_entry_size
            > srcsize
        ):
            raise TgaError('colormap extends past end of file')

    # allocate RLE decode buffer if needed
    if has_rle:
        pixbuf = numpy.empty(pixel_total, numpy.uint8)
        pixptr = <uint8_t*> pixbuf.data
    else:
        pixbuf = None
        pixptr = NULL

    # create output array
    if samples > 1:
        shape = (int(height), int(width), samples)
    else:
        shape = (int(height), int(width))
    out = _create_array(out, shape, numpy.uint8)
    dst = out
    dstptr = <uint8_t*> dst.data

    src_base = &src[0]

    with nogil:
        if has_rle:
            # decode RLE stream from src[pixel_offset:] into pixptr
            srclimit = srcsize
            i = pixel_offset
            rle_total = 0
            while rle_total < height * width and i < srclimit:
                rle_hdr = src[i]
                i += 1
                rle_count = (rle_hdr & 0x7F) + 1
                if rle_hdr & 0x80:
                    # RLE packet: repeat next pixel rle_count times
                    if i + pixel_size > srclimit:
                        break
                    for j in range(rle_count):
                        if rle_total >= height * width:
                            break
                        for k in range(pixel_size):
                            pixptr[rle_total * pixel_size + k] = src[i + k]
                        rle_total += 1
                    i += pixel_size
                else:
                    # raw packet: rle_count literal pixels follow
                    for j in range(rle_count):
                        if (
                            rle_total >= height * width
                            or i + pixel_size > srclimit
                        ):
                            break
                        for k in range(pixel_size):
                            pixptr[rle_total * pixel_size + k] = src[i + k]
                        rle_total += 1
                        i += pixel_size

        # convert decoded (or raw) pixels to output array
        for row in range(height):
            if topdown:
                file_row = row
            else:
                file_row = height - 1 - row

            sroff = file_row * width * pixel_size
            if has_rle:
                pline = pixptr + sroff
            else:
                pline = src_base + pixel_offset + sroff

            dstindex = row * width * samples

            if pixel_size == 1 and samples == 1:
                # 8-bit grayscale: copy row directly
                memcpy(
                    <void*> (dstptr + dstindex),
                    <const void*> pline,
                    width,
                )
                dstindex += width

            elif pixel_size == 1:
                # 8-bit colormap: look up BGR(A) entry in src
                for j in range(width):
                    palval = pline[j]
                    cmap_ptr = (
                        colormap_offset
                        + (colormap_origin_s + <ssize_t> palval)
                        * cmap_entry_size
                    )
                    dstptr[dstindex] = src[cmap_ptr + 2]  # R
                    dstptr[dstindex + 1] = src[cmap_ptr + 1]  # G
                    dstptr[dstindex + 2] = src[cmap_ptr]  # B
                    if samples == 4:
                        dstptr[dstindex + 3] = src[cmap_ptr + 3]  # A
                    dstindex += samples

            elif pixel_size == 2 and samples == 2:
                # 16-bit grayscale+alpha: copy row directly (L, A)
                memcpy(
                    <void*> (dstptr + dstindex),
                    <const void*> pline,
                    width * 2,
                )
                dstindex += width * 2

            elif pixel_size == 2:
                # 16-bit RGB555: expand to 8-bit per channel
                for j in range(width):
                    pixel16 = (
                        (<uint16_t> pline[j * 2 + 1] << 8)
                        | pline[j * 2]
                    )
                    r = <uint8_t> ((pixel16 >> 10) & 0x1F)
                    g = <uint8_t> ((pixel16 >> 5) & 0x1F)
                    b = <uint8_t> (pixel16 & 0x1F)
                    # scale 5-bit -> 8-bit: (v<<3)|(v>>2)
                    dstptr[dstindex] = (r << 3) | (r >> 2)
                    dstptr[dstindex + 1] = (g << 3) | (g >> 2)
                    dstptr[dstindex + 2] = (b << 3) | (b >> 2)
                    dstindex += 3

            elif pixel_size == 3:
                # 24-bit BGR -> RGB
                for j in range(width):
                    dstptr[dstindex] = pline[j * 3 + 2]  # R
                    dstptr[dstindex + 1] = pline[j * 3 + 1]  # G
                    dstptr[dstindex + 2] = pline[j * 3]  # B
                    dstindex += 3

            else:
                # 32-bit BGRA -> RGBA (pixel_size == 4)
                for j in range(width):
                    dstptr[dstindex] = pline[j * 4 + 2]  # R
                    dstptr[dstindex + 1] = pline[j * 4 + 1]  # G
                    dstptr[dstindex + 2] = pline[j * 4]  # B
                    dstptr[dstindex + 3] = pline[j * 4 + 3]  # A
                    dstindex += 4

    return out


cdef enum tga_image_type_t:
    TGA_TYPE_NONE = 0
    TGA_TYPE_COLORMAP = 1
    TGA_TYPE_TRUECOLOR = 2
    TGA_TYPE_GRAYSCALE = 3
    TGA_TYPE_RLE_COLORMAP = 9
    TGA_TYPE_RLE_TRUECOLOR = 10
    TGA_TYPE_RLE_GRAYSCALE = 11


cdef packed struct tga_header_t:
    uint8_t id_length  # length of image ID string
    uint8_t colormap_type  # 0=none, 1=present
    uint8_t image_type  # see tga_image_type_t
    uint16_t colormap_origin  # first colormap entry index
    uint16_t colormap_length  # number of colormap entries
    uint8_t colormap_depth  # bits per colormap entry
    uint16_t x_origin  # lower-left x
    uint16_t y_origin  # lower-left y
    uint16_t width  # image width in pixels
    uint16_t height  # image height in pixels
    uint8_t pixel_depth  # bits per pixel
    uint8_t image_descriptor  # bits 0-3: alpha depth; bit 5: top-down
