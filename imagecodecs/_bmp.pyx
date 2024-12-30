# imagecodecs/_bmp.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2023-2025, Christoph Gohlke
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

"""BMP codec for the imagecodecs package."""

include '_shared.pxi'

from bmptypes cimport *

try:
    from ._jpeg8 import jpeg8_decode
except ImportError as exc:
    jpeg8_decode = None

try:
    from ._png import png_decode
except ImportError as exc:
    png_decode = None


class BMP:
    """BMP codec constants."""

    available = True


class BmpError(RuntimeError):
    """BMP codec exceptions."""


def bmp_version():
    """Return bmp codec version string."""
    return 'bmp 2024.1.1'


def bmp_check(const uint8_t[::1] data):
    """Return whether data is BMP encoded."""
    # TODO: b'BA', b'CI', b'CP', b'IC', b'PT'
    return data.size > 54 and data[0] == 66 and data[1] == 77  # 'BM'


def bmp_encode(data, ppm=None, out=None):
    """Return BMP encoded image."""
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

    if not (
        src.dtype == numpy.uint8
        and (src.ndim == 2 or src.ndim == 3)
        and (samples == 1 or samples == 3)
        and src.shape[0] <= 2147483647
        and src.shape[1] <= 2147483647
    ):
        raise ValueError('invalid data shape or dtype')

    # infoheader
    memset(<void*> &infoheader, 0, sizeof(bmp_infoheader_t))
    infoheader.size = 40
    infoheader.width = <int32_t> src.shape[1]
    infoheader.height = <int32_t> src.shape[0]
    infoheader.planes = 1
    infoheader.bitcount = 8 if samples == 1 else 24
    infoheader.compression_type = 0
    infoheader.x_ppm = ppm_
    infoheader.y_ppm = ppm_
    infoheader.clr_used = 0
    infoheader.clr_important = 0

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
    fileheader.offbits = 1078 if samples == 1 else 54
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
        memcpy(<void*> dstptr, <void*> &fileheader, 14)
        memcpy(<void*> (dstptr + 14), <void*> &infoheader, 40)

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

    del dst
    return _return_output(out, dstsize, fileheader.size, outgiven)


def bmp_decode(data, asrgb=None, out=None):
    """Return decoded BMP image.

    Only 8 and 24-bit BMP files are supported.

    """
    cdef:
        numpy.ndarray dst
        uint8_t* dstptr = NULL
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t height, width, samples, rowpad, i, j
        ssize_t offset, palindex, dstindex, srcindex
        bmp_fileheader_t fileheader
        bmp_infoheader_t infoheader
        uint32_t infoheader_size

    if data is out:
        raise ValueError('cannot decode in-place')
    if srcsize > 4294967295U:
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
        infoheader_size > sizeof(bmp_infoheader_t)
        or infoheader_size + offset > srcsize
    ):
        raise BmpError(f'invalid {infoheader_size=}')
    memset(<void*> &infoheader, 0, sizeof(bmp_infoheader_t))
    memcpy(<void*> &infoheader, <const void*> &src[offset], infoheader_size)
    offset += infoheader_size  # offset to palette
    if infoheader.compression_type == 4 and jpeg8_decode is not None:
        return jpeg8_decode(src[fileheader.offbits:], out=out)
    if infoheader.compression_type == 5 and png_decode is not None:
        return png_decode(src[fileheader.offbits:], out=out)
    if infoheader.compression_type != 0:
        raise BmpError(f'{infoheader.compression_type=} not implemented')
    if infoheader.bitcount != 8 and infoheader.bitcount != 24:
        raise BmpError(f'{infoheader.bitcount=} not implemented')
    if infoheader.clr_used == 0 and infoheader.bitcount < 16:
        infoheader.clr_used = 2 ** infoheader.bitcount
    if infoheader.clr_used > 256:
        raise BmpError(f'{infoheader.clr_used=} not implemented')
    if infoheader.clr_used * 4 + offset > fileheader.offbits:
        raise BmpError('invalid palette')
    if infoheader.size_image == 0:
        if infoheader.compression_type != 0:
            raise BmpError(f'invalid {infoheader.size_image=}')
        # infoheader.size_image = TODO
    if infoheader.size_image + fileheader.offbits > srcsize:
        raise BmpError(
            f'invalid {infoheader.size_image=} + {fileheader.offbits=}'
            f' > {srcsize=}'
        )

    samples = 3
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

    # create output array
    height = <ssize_t> abs(infoheader.height)
    width = <ssize_t> infoheader.width
    rowpad = <ssize_t> ((width * infoheader.bitcount) // 8) % 4
    rowpad = 0 if rowpad == 0 else 4 - rowpad

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
                                f'{palindex=} out of range '
                                f'{infoheader.clr_used=}'
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

        else:
            raise NotImplementedError(
                f'{infoheader.bitcount=} and {infoheader.compression_type=}'
            )

    return out
