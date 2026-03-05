# imagecodecs/_bmp.pyx
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

"""BMP (Bitmap) codec for the imagecodecs package."""

include '_shared.pxi'

from bmptypes cimport *

from . import jpeg8_decode, png_decode


class BMP:
    """BMP codec constants."""

    available = True


class BmpError(RuntimeError):
    """BMP codec exceptions."""


def bmp_version():
    """Return bmp codec version string."""
    return 'bmp 2026.3.6'


def bmp_check(const uint8_t[::1] data, /):
    """Return whether data is BMP encoded or None if unknown."""
    return data.nbytes > 54 and data[0] == 66 and data[1] == 77  # b'BM'


def bmp_encode(
    data,
    /,
    *,
    ppm=None,
    out=None,
):
    """Return BMP encoded image.

    Supported:
    - 8-bit grayscale (uint8, 2D): written as 8-bit paletted with identity
      palette
    - 24-bit RGB (uint8, 3D, 3 samples): written as 24-bit uncompressed BI_RGB
    - 32-bit RGBA (uint8, 3D, 4 samples): written as 32-bit BI_BITFIELDS BGRA
      using BITMAPV4HEADER with explicit BGRA channel masks

    Not supported (encode):
    - 1 and 4-bit paletted
    - 16-bit direct-color
    - Any compression (BI_RLE8, BI_RLE4)
    - OS/2 file types (BA, CI, CP, IC, PT)

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* srcptr = <uint8_t*> src.data
        uint8_t* dstptr = NULL
        ssize_t dstsize, rowpad, srcindex, dstindex, i, j
        ssize_t samples = src.shape[2] if src.ndim == 3 else 1
        int32_t ppm_ = 3780 if ppm is None else max(1, ppm)  # 96 DPI
        bmp_fileheader_t fileheader
        bmp_infoheader_t infoheader

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        src.dtype == numpy.uint8
        and (src.ndim == 2 or src.ndim == 3)
        and (samples == 1 or samples == 3 or samples == 4)
        and src.shape[0] <= INT32_MAX
        and src.shape[1] <= INT32_MAX
    ):
        raise ValueError('invalid data shape or dtype')

    # infoheader
    memset(<void*> &infoheader, 0, sizeof(bmp_infoheader_t))
    infoheader.size = 108 if samples == 4 else 40  # BITMAPV4HEADER for RGBA
    infoheader.width = <int32_t> src.shape[1]
    infoheader.height = <int32_t> src.shape[0]
    infoheader.planes = 1
    infoheader.bitcount = 8 if samples == 1 else (24 if samples == 3 else 32)
    infoheader.compression_type = BI_BITFIELDS if samples == 4 else BI_RGB
    infoheader.x_ppm = ppm_
    infoheader.y_ppm = ppm_
    infoheader.clr_used = 0
    infoheader.clr_important = 0
    if samples == 4:
        infoheader.red_mask = 0x00FF0000
        infoheader.green_mask = 0x0000FF00
        infoheader.blue_mask = 0x000000FF
        infoheader.alpha_mask = 0xFF000000

    rowpad = (
        ((<ssize_t> infoheader.width * <ssize_t> infoheader.bitcount) // 8) % 4
    )
    rowpad = 0 if rowpad == 0 else 4 - rowpad

    infoheader.size_image = <uint32_t> (
        infoheader.height * (infoheader.width * samples + rowpad)
    )

    # fileheader
    memset(<void*> &fileheader, 0, sizeof(bmp_fileheader_t))
    fileheader.type = 0x4d42  # b'BM'
    fileheader.offbits = (
        1078 if samples == 1 else (122 if samples == 4 else 54)
    )  # 14 + infoheader.size, no palette
    fileheader.size = fileheader.offbits + infoheader.size_image

    # create output
    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        out = _create_output(outtype, fileheader.size)
    dst = out
    dstptr = <uint8_t*> &dst[0]
    dstsize = dst.nbytes
    if dstsize < <ssize_t> fileheader.size:
        raise ValueError(f'output too small {dstsize} < {fileheader.size}')

    # copy to output
    with nogil:
        memcpy(<void*> dstptr, <const void*> &fileheader, 14)
        memcpy(
            <void*> (dstptr + 14), <const void*> &infoheader, infoheader.size
        )

        if samples == 1:
            # write palette
            dstindex = 54
            for i in range(256):
                dstptr[dstindex] = <uint8_t> i
                dstindex += 1
                dstptr[dstindex] = <uint8_t> i
                dstindex += 1
                dstptr[dstindex] = <uint8_t> i
                dstindex += 1
                dstptr[dstindex] = 255
                dstindex += 1
            # write index
            srcindex = 0
            for i in range(<ssize_t> infoheader.height):
                dstindex = <ssize_t> (
                    fileheader.offbits
                    + (infoheader.height - 1 - i) * (infoheader.width + rowpad)
                )
                for j in range(<ssize_t> infoheader.width):
                    dstptr[dstindex] = srcptr[srcindex]
                    dstindex += 1
                    srcindex += 1
                for j in range(rowpad):
                    dstptr[dstindex] = 0
                    dstindex += 1

        elif samples == 3:
            # write BGR
            srcindex = 0
            for i in range(<ssize_t> infoheader.height):
                dstindex = <ssize_t> (
                    fileheader.offbits
                    + (infoheader.height - 1 - i)
                    * (infoheader.width * 3 + rowpad)
                )
                for j in range(<ssize_t> infoheader.width):
                    dstptr[dstindex] = srcptr[srcindex + 2]
                    dstindex += 1
                    dstptr[dstindex] = srcptr[srcindex + 1]
                    dstindex += 1
                    dstptr[dstindex] = srcptr[srcindex]
                    dstindex += 1
                    srcindex += 3
                for j in range(rowpad):
                    dstptr[dstindex] = 0
                    dstindex += 1

        elif samples == 4:
            # write BGRA (32-bit rows are always DWORD-aligned, no rowpad)
            srcindex = 0
            for i in range(<ssize_t> infoheader.height):
                dstindex = <ssize_t> (
                    fileheader.offbits
                    + (infoheader.height - 1 - i) * infoheader.width * 4
                )
                for j in range(<ssize_t> infoheader.width):
                    dstptr[dstindex] = srcptr[srcindex + 2]  # B
                    dstindex += 1
                    dstptr[dstindex] = srcptr[srcindex + 1]  # G
                    dstindex += 1
                    dstptr[dstindex] = srcptr[srcindex]  # R
                    dstindex += 1
                    dstptr[dstindex] = srcptr[srcindex + 3]  # A
                    dstindex += 1
                    srcindex += 4

    del dst
    return _return_output(out, dstsize, fileheader.size, outgiven)


def bmp_decode(
    data,
    /,
    *,
    asrgb=None,
    out=None,
):
    """Return decoded BMP image.

    Supported:
    - 1 and 4-bit paletted (BI_RGB): expanded via palette to grayscale or RGB
    - 8-bit paletted (BI_RGB): grayscale or RGB depending on palette content
    - 16-bit direct-color (BI_RGB RGB555, BI_BITFIELDS RGB555/RGB565/RGBA)
    - 24-bit direct-color (BI_RGB BGR)
    - 32-bit direct-color (BI_RGB BGRX treated as RGB, BI_BITFIELDS with
      optional alpha if alpha mask is present)
    - Embedded JPEG (BI_JPEG) and PNG (BI_PNG): delegated to jpeg8/png decoders
    - Bottom-up (positive height) and top-down (negative height) layouts

    Not supported (decode):
    - BI_RLE8 and BI_RLE4 run-length encoding
    - OS/2 file types (BA, CI, CP, IC, PT)

    """
    cdef:
        numpy.ndarray dst
        uint8_t* dstptr = NULL
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t height, width, samples, rowpad, rowstride, i, j
        ssize_t offset, palindex, dstindex, srcindex
        bmp_fileheader_t fileheader
        bmp_infoheader_t infoheader
        uint32_t infoheader_size
        uint32_t rmask, gmask, bmask, amask
        int rshift, gshift, bshift, ashift
        int rbits, gbits, bbits, abits
        uint32_t pixel32, m, rv, gv, bv, av
        uint16_t pixel16
        bint has_alpha

    if data is out:
        raise ValueError('cannot decode in-place')
    if srcsize > UINT32_MAX:
        raise ValueError('input too large')
    if srcsize < 54:
        raise BmpError(f'invalid BMP size {srcsize} < 54')

    # read file header
    memcpy(
        <void*> &fileheader, <const void*> &src[0], sizeof(bmp_fileheader_t)
    )
    if fileheader.type != 0x4d42:  # b'BM'
        raise BmpError(f'invalid file header {bytes(src[:2])}')
    if fileheader.size > srcsize:
        raise BmpError(f'invalid file size {fileheader.size} > {srcsize}')
    if fileheader.offbits >= srcsize:
        raise BmpError(f'invalid file size {fileheader.offbits} > {srcsize}')
    # if fileheader.reserved1 != 0 or fileheader.reserved2 != 0:
    #     raise BmpError('invalid file header')

    # read info header
    offset = sizeof(bmp_fileheader_t)
    memcpy(
        <void*> &infoheader_size, <const void*> &src[offset], sizeof(uint32_t)
    )
    if (
        infoheader_size < 16
        or infoheader_size > sizeof(bmp_infoheader_t)
        or infoheader_size + offset > srcsize
    ):
        raise BmpError(f'invalid {infoheader_size=}')
    memset(<void*> &infoheader, 0, sizeof(bmp_infoheader_t))
    memcpy(<void*> &infoheader, <const void*> &src[offset], infoheader_size)
    offset += infoheader_size  # offset to palette
    if infoheader.compression_type == BI_JPEG:
        return jpeg8_decode(src[fileheader.offbits:], out=out)
    if infoheader.compression_type == BI_PNG:
        return png_decode(src[fileheader.offbits:], out=out)
    if infoheader.compression_type == BI_BITFIELDS:
        if infoheader.bitcount not in (16, 32):
            raise BmpError(
                f'BI_BITFIELDS requires 16 or 32-bit, '
                f'got {infoheader.bitcount=}'
            )
    elif infoheader.compression_type != BI_RGB:
        raise BmpError(f'{infoheader.compression_type=} not implemented')
    if infoheader.bitcount not in (1, 4, 8, 16, 24, 32):
        raise BmpError(f'{infoheader.bitcount=} not implemented')
    if infoheader.clr_used == 0 and infoheader.bitcount < 16:
        infoheader.clr_used = 2 ** infoheader.bitcount
    if infoheader.clr_used > 256:
        raise BmpError(f'{infoheader.clr_used=} not implemented')
    if infoheader.clr_used * 4 + offset > fileheader.offbits:
        raise BmpError('invalid palette')
    if infoheader.size_image == 0:
        if infoheader.compression_type not in (BI_RGB, BI_BITFIELDS):
            raise BmpError(f'invalid {infoheader.size_image=}')
        rowpad = (<ssize_t> infoheader.width * infoheader.bitcount // 8) % 4
        rowpad = 0 if rowpad == 0 else 4 - rowpad
        infoheader.size_image = <uint32_t> (
            abs(infoheader.height)
            * (infoheader.width * infoheader.bitcount // 8 + rowpad)
        )
    if infoheader.size_image + fileheader.offbits > srcsize:
        raise BmpError(
            f'invalid {infoheader.size_image=} + {fileheader.offbits=}'
            f' > {srcsize=}'
        )

    samples = 3
    has_alpha = False
    rmask = 0
    gmask = 0
    bmask = 0
    amask = 0
    rshift = 0
    gshift = 0
    bshift = 0
    ashift = 0
    rbits = 8
    gbits = 8
    bbits = 8
    abits = 0
    if infoheader.bitcount <= 8:
        if asrgb is None:
            # detect grayscale palette
            samples = 1
            palindex = offset
            for i in range(infoheader.clr_used):
                if src[palindex] != i:
                    samples = 3
                    break
                palindex += 1
                if src[palindex] != i:
                    samples = 3
                    break
                palindex += 1
                if src[palindex] != i:
                    samples = 3
                    break
                palindex += 2
        elif not asrgb:
            samples = 1
    elif infoheader.bitcount == 16:
        if infoheader.compression_type == BI_BITFIELDS:
            if infoheader_size >= 52:
                rmask = infoheader.red_mask
                gmask = infoheader.green_mask
                bmask = infoheader.blue_mask
                amask = infoheader.alpha_mask
            elif offset + 12 <= srcsize:
                memcpy(<void*> &rmask, <const void*> &src[offset], 4)
                memcpy(<void*> &gmask, <const void*> &src[offset + 4], 4)
                memcpy(<void*> &bmask, <const void*> &src[offset + 8], 4)
        else:
            # RGB555
            rmask = 0x7C00
            gmask = 0x03E0
            bmask = 0x001F
        # compute shifts and bit depths from masks
        m = rmask
        rshift = 0
        while m != 0 and (m & 1) == 0:
            rshift += 1
            m >>= 1
        m = gmask
        gshift = 0
        while m != 0 and (m & 1) == 0:
            gshift += 1
            m >>= 1
        m = bmask
        bshift = 0
        while m != 0 and (m & 1) == 0:
            bshift += 1
            m >>= 1
        m = amask
        ashift = 0
        while m != 0 and (m & 1) == 0:
            ashift += 1
            m >>= 1
        m = rmask >> rshift
        rbits = 0
        while m:
            rbits += m & 1
            m >>= 1
        m = gmask >> gshift
        gbits = 0
        while m:
            gbits += m & 1
            m >>= 1
        m = bmask >> bshift
        bbits = 0
        while m:
            bbits += m & 1
            m >>= 1
        m = amask >> ashift
        abits = 0
        while m:
            abits += m & 1
            m >>= 1
        has_alpha = amask != 0
        samples = 4 if has_alpha else 3
    elif infoheader.bitcount == 32:
        if infoheader.compression_type == BI_BITFIELDS:
            if infoheader_size >= 52:
                rmask = infoheader.red_mask
                gmask = infoheader.green_mask
                bmask = infoheader.blue_mask
                amask = infoheader.alpha_mask
            elif offset + 12 <= srcsize:
                memcpy(<void*> &rmask, <const void*> &src[offset], 4)
                memcpy(<void*> &gmask, <const void*> &src[offset + 4], 4)
                memcpy(<void*> &bmask, <const void*> &src[offset + 8], 4)
                if offset + 16 <= srcsize:
                    memcpy(<void*> &amask, <const void*> &src[offset + 12], 4)
        else:
            # BI_RGB 32-bit: BGRX layout, high byte is reserved padding
            rmask = 0x00FF0000
            gmask = 0x0000FF00
            bmask = 0x000000FF
            amask = 0  # X byte is not alpha
        # compute shifts and bit depths from masks
        m = rmask
        rshift = 0
        while m != 0 and (m & 1) == 0:
            rshift += 1
            m >>= 1
        m = gmask
        gshift = 0
        while m != 0 and (m & 1) == 0:
            gshift += 1
            m >>= 1
        m = bmask
        bshift = 0
        while m != 0 and (m & 1) == 0:
            bshift += 1
            m >>= 1
        m = amask
        ashift = 0
        while m != 0 and (m & 1) == 0:
            ashift += 1
            m >>= 1
        m = rmask >> rshift
        rbits = 0
        while m:
            rbits += m & 1
            m >>= 1
        m = gmask >> gshift
        gbits = 0
        while m:
            gbits += m & 1
            m >>= 1
        m = bmask >> bshift
        bbits = 0
        while m:
            bbits += m & 1
            m >>= 1
        m = amask >> ashift
        abits = 0
        while m:
            abits += m & 1
            m >>= 1
        has_alpha = amask != 0
        samples = 4 if has_alpha else 3

    # create output array
    height = <ssize_t> abs(infoheader.height)
    width = <ssize_t> infoheader.width
    rowpad = <ssize_t> ((width * infoheader.bitcount) // 8) % 4
    rowpad = 0 if rowpad == 0 else 4 - rowpad
    rowstride = <ssize_t> (
        ((width * <ssize_t> infoheader.bitcount + 31) // 32) * 4
    )

    if samples > 1:
        shape = (int(height), int(width), samples)
    else:
        shape = (int(height), int(width))
        samples = 1
    out = _create_array(out, shape, numpy.uint8)
    dst = out
    dstptr = <uint8_t*> dst.data

    # copy image
    with nogil:
        srcindex = fileheader.offbits
        dstindex = 0

        if infoheader.bitcount == 8 and infoheader.compression_type == 0:
            if samples == 3:
                # apply palette
                for i in range(height):
                    if infoheader.height > 0:
                        dstindex = (height - 1 - i) * width * 3
                    for j in range(width):
                        palindex = src[srcindex]
                        if palindex >= infoheader.clr_used:
                            raise IndexError(
                                f'{palindex=} >= {infoheader.clr_used=}'
                            )
                        palindex = offset + palindex * 4
                        dstptr[dstindex] = src[palindex + 2]  # R
                        dstindex += 1
                        dstptr[dstindex] = src[palindex + 1]  # G
                        dstindex += 1
                        dstptr[dstindex] = src[palindex]  # B
                        dstindex += 1
                        srcindex += 1
                    srcindex += rowpad

            elif samples == 1:
                # copy palette index
                for i in range(height):
                    if infoheader.height > 0:
                        dstindex = (height - 1 - i) * width
                    for j in range(width):
                        dstptr[dstindex] = src[srcindex]
                        dstindex += 1
                        srcindex += 1
                    srcindex += rowpad

        elif infoheader.bitcount == 24 and infoheader.compression_type == 0:
            for i in range(height):
                if infoheader.height > 0:
                    dstindex = (height - 1 - i) * width * 3
                for j in range(width):
                    dstptr[dstindex] = src[srcindex + 2]  # R
                    dstindex += 1
                    dstptr[dstindex] = src[srcindex + 1]  # G
                    dstindex += 1
                    dstptr[dstindex] = src[srcindex]  # B
                    dstindex += 1
                    srcindex += 3
                srcindex += rowpad

        elif (
            infoheader.bitcount == 1 and infoheader.compression_type == BI_RGB
        ):
            for i in range(height):
                if infoheader.height > 0:
                    dstindex = (height - 1 - i) * width * samples
                else:
                    dstindex = i * width * samples
                srcindex = <ssize_t> fileheader.offbits + i * rowstride
                for j in range(width):
                    palindex = (src[srcindex + j // 8] >> (7 - (j % 8))) & 1
                    if palindex >= infoheader.clr_used:
                        raise IndexError(
                            f'{palindex=} >= {infoheader.clr_used=}'
                        )
                    palindex = offset + palindex * 4
                    if samples == 3:
                        dstptr[dstindex] = src[palindex + 2]  # R
                        dstptr[dstindex + 1] = src[palindex + 1]  # G
                        dstptr[dstindex + 2] = src[palindex]  # B
                        dstindex += 3
                    else:
                        dstptr[dstindex] = src[palindex + 2]
                        dstindex += 1

        elif (
            infoheader.bitcount == 4 and infoheader.compression_type == BI_RGB
        ):
            for i in range(height):
                if infoheader.height > 0:
                    dstindex = (height - 1 - i) * width * samples
                else:
                    dstindex = i * width * samples
                srcindex = <ssize_t> fileheader.offbits + i * rowstride
                for j in range(width):
                    if j % 2 == 0:
                        palindex = (src[srcindex + j // 2] >> 4) & 0xF
                    else:
                        palindex = src[srcindex + j // 2] & 0xF
                    if palindex >= infoheader.clr_used:
                        raise IndexError(
                            f'{palindex=} >= {infoheader.clr_used=}'
                        )
                    palindex = offset + palindex * 4
                    if samples == 3:
                        dstptr[dstindex] = src[palindex + 2]  # R
                        dstptr[dstindex + 1] = src[palindex + 1]  # G
                        dstptr[dstindex + 2] = src[palindex]  # B
                        dstindex += 3
                    else:
                        dstptr[dstindex] = src[palindex + 2]
                        dstindex += 1

        elif infoheader.bitcount == 16:
            for i in range(height):
                if infoheader.height > 0:
                    dstindex = (height - 1 - i) * width * samples
                else:
                    dstindex = i * width * samples
                srcindex = <ssize_t> fileheader.offbits + i * rowstride
                for j in range(width):
                    memcpy(
                        <void*> &pixel16,
                        <const void*> &src[srcindex + j * 2],
                        2
                    )
                    pixel32 = <uint32_t> pixel16
                    rv = (pixel32 & rmask) >> rshift
                    gv = (pixel32 & gmask) >> gshift
                    bv = (pixel32 & bmask) >> bshift
                    dstptr[dstindex] = (
                        <uint8_t> (rv * 255 / ((1 << rbits) - 1))
                        if rbits < 8 else <uint8_t> rv
                    )
                    dstptr[dstindex + 1] = (
                        <uint8_t> (gv * 255 / ((1 << gbits) - 1))
                        if gbits < 8 else <uint8_t> gv
                    )
                    dstptr[dstindex + 2] = (
                        <uint8_t> (bv * 255 / ((1 << bbits) - 1))
                        if bbits < 8 else <uint8_t> bv
                    )
                    if samples == 4:
                        av = (pixel32 & amask) >> ashift
                        dstptr[dstindex + 3] = (
                            <uint8_t> (av * 255 / ((1 << abits) - 1))
                            if abits > 0 and abits < 8 else <uint8_t> av
                        )
                    dstindex += samples

        elif infoheader.bitcount == 32:
            for i in range(height):
                if infoheader.height > 0:
                    dstindex = (height - 1 - i) * width * samples
                else:
                    dstindex = i * width * samples
                srcindex = <ssize_t> fileheader.offbits + i * rowstride
                for j in range(width):
                    memcpy(
                        <void*> &pixel32,
                        <const void*> &src[srcindex + j * 4],
                        4
                    )
                    rv = (pixel32 & rmask) >> rshift
                    gv = (pixel32 & gmask) >> gshift
                    bv = (pixel32 & bmask) >> bshift
                    dstptr[dstindex] = <uint8_t> rv
                    dstptr[dstindex + 1] = <uint8_t> gv
                    dstptr[dstindex + 2] = <uint8_t> bv
                    if samples == 4:
                        av = (pixel32 & amask) >> ashift
                        dstptr[dstindex + 3] = <uint8_t> av
                    dstindex += samples

        else:
            raise NotImplementedError(
                f'{infoheader.bitcount=} and {infoheader.compression_type=}'
            )

    return out
