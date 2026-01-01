# type: ignore
"""..."""

import os
import random

import numpy
import tifffile
from noise import snoise3, snoise4
from skimage.util.dtype import convert

import imagecodecs

numpy.random.seed(42)
random.seed(42)

OUTDIR = 'testimages'


def imcwrite(filename, *args, **kwargs):
    """..."""
    # print(filename, args)
    filename = os.path.join(OUTDIR, filename)
    return imagecodecs.imwrite(filename, *args, **kwargs)


def tifwrite(filename, *args, **kwargs):
    """..."""
    filename = os.path.join(OUTDIR, filename)
    return tifffile.imwrite(filename, *args, **kwargs)


def gennoise(shape=(11, 32, 31, 9), octaves=1):
    """..."""
    noise = numpy.empty(shape, 'f8')
    freq = shape[1] * octaves / 2.0
    channels = shape[3] - 1

    for i in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                for c in range(channels):
                    noise[i, y, x, c] = snoise4(
                        1.5 * i / freq, x / freq, y / freq, c / 6, octaves
                    )

    # freq = 24 * octaves
    for i in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                noise[i, y, x, -1] = snoise3(
                    i / freq, x / freq, y / freq, octaves
                )

    noise += 1.0
    noise /= 2.0

    noise[..., -1] *= 0.9
    noise[..., -1] += 0.1

    return noise


def getdata(data, itype, planar=False, frames=0):
    """..."""
    # if itype == 'frames':
    #    return data[..., 0]
    if frames is not None:
        data = data[frames]
    if itype == 'rgba':
        data = data[..., [0, 2, 4, -1]]
    elif itype == 'rgb':
        data = data[..., [0, 2, 4]]
    elif itype == 'cmyk':
        data = data[..., [0, 2, 4, 6]]
    elif itype == 'cmyka':
        data = data[..., [0, 2, 4, 6, -1]]
    elif itype == 'gray':
        data = data[..., 0]
        planar = False
    elif itype == 'graya':
        data = data[..., [0, -1]]
    # elif itype == 'rrggbbaa':
    #     data = numpy.moveaxis(data[..., [0, 2, 4, -1]], -1, 0).copy()
    # elif itype == 'rrggbb':
    #     data = numpy.moveaxis(data[..., [0, 2, 4]], -1, 0).copy()
    elif itype == 'channels':
        data = data[..., :-1]
    elif itype == 'channelsa':
        data = data[..., :]
    else:
        raise ValueError('unknown image type')
    if planar:
        data = numpy.moveaxis(data, -1, -3)
    return data.copy()


def raw(data):
    """..."""
    return
    encode = imagecodecs.none_encode
    ext = 'bin'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb_planar' + ext, getdata(data, 'rgb', planar=True), encode)
    imcwrite(
        'rgb_alpha_planar' + ext, getdata(data, 'rgba', planar=True), encode
    )
    imcwrite('cmyk' + ext, getdata(data, 'cmyk'), encode)
    imcwrite('cmyk_alpha' + ext, getdata(data, 'cmyka'), encode)
    imcwrite('gray_extrasamples' + ext, getdata(data, 'channels'), encode)
    imcwrite(
        'gray_alpha_extrasamples' + ext, getdata(data, 'channelsa'), encode
    )
    imcwrite('gray_frames' + ext, getdata(data, 'gray', frames=None), encode)
    imcwrite('rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode)


def npy(data):
    """..."""
    encode = imagecodecs.numpy_encode
    ext = 'npy'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    # imcwrite('data' + ext, data, encode)
    imcwrite('npy/gray' + ext, getdata(data, 'gray'), encode)
    if data.dtype.kind != 'b':
        imcwrite('npy/gray_alpha' + ext, getdata(data, 'graya'), encode)
        imcwrite('npy/rgb' + ext, getdata(data, 'rgb'), encode)
        imcwrite('npy/rgb_alpha' + ext, getdata(data, 'rgba'), encode)
        imcwrite(
            'npy/rgb_planar' + ext, getdata(data, 'rgb', planar=True), encode
        )
        imcwrite(
            'npy/rgb_alpha_planar' + ext,
            getdata(data, 'rgba', planar=True),
            encode,
        )
        imcwrite('npy/cmyk' + ext, getdata(data, 'cmyk'), encode)
        imcwrite('npy/cmyk_alpha' + ext, getdata(data, 'cmyka'), encode)
        imcwrite(
            'npy/gray_extrasamples' + ext, getdata(data, 'channels'), encode
        )
        imcwrite(
            'npy/gray_alpha_extrasamples' + ext,
            getdata(data, 'channelsa'),
            encode,
        )
        imcwrite(
            'npy/gray_frames' + ext, getdata(data, 'gray', frames=None), encode
        )
        imcwrite(
            'npy/rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode
        )


def jxl(data):
    """..."""
    encode = imagecodecs.jpegxl_encode
    kwargs = {}  # dict(level=100)
    ext = 'jxl'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)

    # kwargs['photometric'] = 'GRAY'
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)

    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
    imcwrite(
        'gray_frames' + ext,
        getdata(data, 'gray', frames=None)[..., numpy.newaxis],
        encode,
        photometric='GRAY',
        usecontainer=True,
    )

    # kwargs['photometric'] = 'RGB'
    # if data.dtype.kind != '':
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    imcwrite(
        'rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode, **kwargs
    )

    imcwrite(
        'gray_extrasamples_planar' + ext,
        getdata(data, 'channels', planar=True),
        encode,
        planar=True,
        **kwargs,
    )

    # if data.dtype.kind != 'f':
    #    imcwrite(
    #        'gray_extrasamples' +  ext, getdata(data, 'channels'), encode, **kwargs
    #    )
    #    imcwrite(
    #        'gray_alpha_extrasamples' +  ext, getdata(data, 'channelsa'), encode, **kwargs
    #    )


def jxs(data):
    """..."""
    encode = imagecodecs.jpegxs_encode
    kwargs = {}  # dict(level=100)
    ext = 'jxs'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)

    kwargs = {'bitspersample': 12} if data.dtype.char == 'H' else {}

    # kwargs['photometric'] = 'GRAY'
    # imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)

    # imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
    # imcwrite(
    #     'gray_frames' + ext,
    #     getdata(data, 'gray', frames=None)[..., numpy.newaxis],
    #     encode,
    #     photometric='GRAY',
    #     usecontainer=True,
    # )

    # kwargs['photometric'] = 'RGB'
    # if data.dtype.kind != '':
    # imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    # imcwrite(
    #     'rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode, **kwargs
    # )

    # imcwrite(
    #     'gray_extrasamples_planar' + ext,
    #     getdata(data, 'channels', planar=True),
    #     encode,
    #     planar=True,
    #     **kwargs,
    # )


def heif(data):
    """..."""
    encode = imagecodecs.heif_encode
    kwargs = {}  # , compression='AV1')
    ext = 'heif'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)

    if data.dtype.char in 'BH':
        if data.dtype.char == 'H':
            # uint16 requires special build
            # data >>= 2
            kwargs['bitspersample'] = 12
            kwargs['compression'] = 'AV1'
        imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)
        imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
        imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
        imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
        imcwrite(
            'gray_frames' + ext,
            getdata(data, 'gray', frames=None),
            encode,
            photometric='GRAY',
            **kwargs,
        )
        imcwrite(
            'rgb_frames' + ext,
            getdata(data, 'rgb', frames=None),
            encode,
            photometric='RGB',
            **kwargs,
        )
    # if data.dtype.kind != 'f':
    #    imcwrite(
    #       'gray_extrasamples' +  ext, getdata(data, 'channels'), encode, **kwargs
    #    )
    #    imcwrite(
    #       'gray_alpha_extrasamples' +  ext, getdata(data, 'channelsa'), encode, **kwargs
    #    )


def udr(data):
    """..."""
    encode = imagecodecs.ultrahdr_encode
    kwargs = dict(level=100)
    ext = 'uhdr'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)


def jxr(data):
    """..."""
    encode = imagecodecs.jpegxr_encode
    kwargs = {}  # dict(level=100)
    ext = 'jxr'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)
    if data.dtype == numpy.bool_:
        return
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    if data.dtype.kind == 'f':
        return

    imcwrite(
        'gray_extrasamples' + ext, getdata(data, 'channels'), encode, **kwargs
    )
    imcwrite(
        'gray_alpha_extrasamples' + ext,
        getdata(data, 'channelsa'),
        encode,
        **kwargs,
    )

    kwargs['photometric'] = 'cmyk'
    imcwrite('cmyk' + ext, getdata(data, 'cmyk'), encode, **kwargs)
    imcwrite('cmyk_alpha' + ext, getdata(data, 'cmyka'), encode, **kwargs)


def zfp(data):
    """..."""
    kwargs = {}
    kwargs = dict(execution='omp', mode='R')  # , level=24)
    encode = imagecodecs.zfp_encode
    ext = 'zfp'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
    imcwrite(
        'rgb_planar' + ext, getdata(data, 'rgb', planar=True), encode, **kwargs
    )
    imcwrite(
        'rgb_alpha_planar' + ext,
        getdata(data, 'rgba', planar=True),
        encode,
        **kwargs,
    )
    imcwrite(
        'gray_extrasamples' + ext, getdata(data, 'channels'), encode, **kwargs
    )
    imcwrite(
        'gray_alpha_extrasamples' + ext,
        getdata(data, 'channelsa'),
        encode,
        **kwargs,
    )
    imcwrite(
        'gray_frames' + ext,
        getdata(data, 'gray', frames=None),
        encode,
        **kwargs,
    )
    imcwrite(
        'rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode, **kwargs
    )

    # with open('gray_extrasamples' +  ext, 'wb') as fh:
    #     fh.write(zfp_encode(getdata(data, 'channels')))


def lerc(data):
    """..."""
    kwargs = {}
    encode = imagecodecs.lerc_encode
    ext = 'lerc2'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
    imcwrite(
        'gray_extrasamples' + ext, getdata(data, 'channels'), encode, **kwargs
    )
    imcwrite(
        'gray_alpha_extrasamples' + ext,
        getdata(data, 'channelsa'),
        encode,
        **kwargs,
    )
    kwargs = dict(planar=True)
    imcwrite(
        'rgb_planar' + ext, getdata(data, 'rgb', planar=True), encode, **kwargs
    )
    imcwrite(
        'rgb_alpha_planar' + ext,
        getdata(data, 'rgba', planar=True),
        encode,
        **kwargs,
    )


def avif(data):
    """..."""
    if data.itemsize > 1:

        def encode(data):
            return imagecodecs.avif_encode(data, bitspersample=12)

    else:
        encode = imagecodecs.avif_encode

    # print('>', getdata(data, 'rgba').shape)

    ext = 'avif'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)

    imcwrite('rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode)
    # imcwrite(
    #    'rgb_alpha_frames' + ext, getdata(data, 'rgba', frames=None), encode
    # )

    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)

    imcwrite('gray_frames' + ext, getdata(data, 'gray', frames=None), encode)
    # imcwrite(
    #    'gray_alpha_frames' + ext, getdata(data, 'graya', frames=None), encode
    # )


def gif(data):
    """..."""
    encode = imagecodecs.gif_encode
    ext = 'gif'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    imcwrite('gray_frames' + ext, getdata(data, 'gray', frames=None), encode)


def bmp(data):
    """..."""
    encode = imagecodecs.bmp_encode
    ext = 'bmp'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)


def png(data):
    """..."""
    encode = imagecodecs.png_encode
    ext = 'png'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)


def qoi(data):
    """..."""
    encode = imagecodecs.qoi_encode
    ext = 'qoi'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    # imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    # imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)


def apng(data):
    """..."""
    encode = imagecodecs.apng_encode
    ext = 'apng'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('gray_frames' + ext, getdata(data, 'gray', frames=None), encode)
    imcwrite('rgb_frames' + ext, getdata(data, 'rgb', frames=None), encode)


def jpg(data):
    """..."""
    encode = imagecodecs.jpeg8_encode
    ext = 'jpg'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    kwargs = dict(bitspersample=12 if data.dtype.itemsize == 2 else None)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)
    # imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode, **kwargs)
    # lossless
    ext = 'ljpg'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    kwargs['lossless'] = True
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode, **kwargs)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, **kwargs)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, **kwargs)


def jls(data):
    """..."""
    encode = imagecodecs.jpegls_encode
    ext = 'jls'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    # imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)


def j2k(data):
    """..."""
    encode = imagecodecs.j2k_encode
    ext = 'j2k'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    imcwrite('gray_alpha' + ext, getdata(data, 'graya'), encode)

    ext = 'jp2'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite(
        'rgb_alpha' + ext, getdata(data, 'rgba'), encode, codecformat='JP2'
    )
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode, codecformat='JP2')
    imcwrite('gray' + ext, getdata(data, 'gray'), encode, codecformat='JP2')
    imcwrite(
        'gray_alpha' + ext, getdata(data, 'graya'), encode, codecformat='JP2'
    )


def jph(data):
    """..."""
    encode = imagecodecs.htj2k_encode
    ext = 'jph'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)
    imcwrite('gray' + ext, getdata(data, 'gray'), encode)
    # imcwrite('gray_extrasamples' + ext, getdata(data, 'channels'), encode)

    # if data.dtype != 'uint8':
    #     return

    imcwrite(
        'rgb_planar' + ext,
        getdata(data, 'rgb', planar=True),
        encode,
        planar=True,
    )
    # imcwrite(
    #     'rgb_alpha_planar' + ext,
    #     getdata(data, 'rgba', planar=True),
    #     encode,
    #     planar=True,
    # )
    # imcwrite(
    #     'gray_extrasamples' + ext, getdata(data, 'channels'), encode,
    # )
    # imcwrite(
    #     'gray_extrasamples_planar' + ext,
    #     getdata(data, 'channels', planar=True),
    #     encode,
    #     planar=True,
    # )


def webp(data):
    """..."""
    encode = imagecodecs.webp_encode
    ext = 'webp'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb_alpha' + ext, getdata(data, 'rgba'), encode)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)  # TODO: check no copy


def rgbe(data):
    """..."""
    encode = imagecodecs.rgbe_encode
    ext = 'rgbe'
    ext = '_%s%i.%s' % (data.dtype.kind, data.itemsize, ext)
    imcwrite('rgb' + ext, getdata(data, 'rgb'), encode)


def tif(data):
    """..."""
    ext = '_%s%i.tif' % (data.dtype.kind, data.itemsize)
    kwargs = dict(metadata=False, datetime=False, software='')
    tifwrite('gray' + ext, getdata(data, 'gray'), rowsperstrip=17, **kwargs)
    tifwrite(
        'gray_tiled' + ext, getdata(data, 'gray'), tile=(16, 16), **kwargs
    )
    if data.dtype == numpy.bool_:
        return
    tifwrite(
        'rgb_alpha' + ext,
        getdata(data, 'rgba'),
        rowsperstrip=17,
        extrasamples=['UNASSALPHA'],
        **kwargs,
    )
    tifwrite('rgb' + ext, getdata(data, 'rgb'), rowsperstrip=17, **kwargs)
    tifwrite('rgb_tiled' + ext, getdata(data, 'rgb'), tile=(16, 16), **kwargs)

    tifwrite(
        'gray_alpha' + ext,
        getdata(data, 'graya'),
        photometric='minisblack',
        extrasamples=['UNASSALPHA'],
        rowsperstrip=17,
        **kwargs,
    )
    tifwrite(
        'gray_extrasamples' + ext,
        getdata(data, 'channels'),
        photometric='minisblack',
        planarconfig='contig',
        rowsperstrip=17,
        **kwargs,
    )
    tifwrite(
        'gray_frames' + ext,
        getdata(data, 'gray', frames=None),
        photometric='minisblack',
        rowsperstrip=17,
        **kwargs,
    )
    tifwrite(
        'rgb_frames' + ext,
        getdata(data, 'rgb', frames=None),
        photometric='rgb',
        rowsperstrip=17,
        **kwargs,
    )
    tifwrite(
        'gray_volumetric' + ext,
        getdata(data, 'gray', frames=None),
        photometric='minisblack',
        volumetric=True,
        tile=(16, 16),
        **kwargs,
    )
    tifwrite(
        'rgb_planar' + ext,
        getdata(data, 'rgb', planar=True),
        photometric='rgb',
        planarconfig='separate',
        rowsperstrip=17,
        **kwargs,
    )
    tifwrite(
        'rgb_planar_tiled' + ext,
        getdata(data, 'rgb', planar=True),
        photometric='rgb',
        planarconfig='separate',
        tile=(16, 16),
        **kwargs,
    )
    tifwrite(
        'rgb_alpha_planar' + ext,
        getdata(data, 'rgba', planar=True),
        photometric='rgb',
        planarconfig='separate',
        extrasamples=['UNASSALPHA'],
        rowsperstrip=17,
        **kwargs,
    )

    if data.dtype == 'uint8':
        # CMYK
        tifwrite(
            'cmyk' + ext,
            getdata(data, 'cmyk'),
            photometric='separated',
            rowsperstrip=17,
            **kwargs,
        )
        tifwrite(
            'cmyk_planar' + ext,
            getdata(data, 'cmyk', planar=True),
            photometric='separated',
            planarconfig='separate',
            rowsperstrip=17,
            **kwargs,
        )
        tifwrite(
            'cmyk_alpha' + ext,
            getdata(data, 'cmyka'),
            photometric='separated',
            extrasamples=['UNASSALPHA'],
            rowsperstrip=17,
            **kwargs,
        )
        tifwrite(
            'cmyk_alpha_planar' + ext,
            getdata(data, 'cmyka', planar=True),
            photometric='separated',
            planarconfig='separate',
            extrasamples=['UNASSALPHA'],
            rowsperstrip=17,
            **kwargs,
        )

    data = getdata(data, 'rgb')
    if data.dtype == 'uint8':
        for compression in (
            'webp',
            'png',
            'jpeg',
            'jpegxr',
            'jpeg2000',
            'jpegxl',
            'packbits',
        ):
            for tile in ('', '_tiled'):
                ext = '_%s%i%s_%s.tif' % (
                    data.dtype.kind,
                    data.itemsize,
                    tile,
                    compression,
                )
                tifwrite(
                    'rgb' + ext,
                    data,
                    compression=compression,
                    rowsperstrip=17 if tile is None else None,
                    tile=(16, 16) if tile else None,
                    **kwargs,
                )

    if data.dtype == 'uint16':
        for compression in (
            'png',
            'jpeg',
            'jpegxr',
            'jpeg2000',
            'jpegxl',
            'packbits',
        ):
            ext = '_%s%i_%s.tif' % (
                data.dtype.kind,
                data.itemsize,
                compression,
            )
            tifwrite(
                'rgb' + ext,
                data,
                compression=compression,
                rowsperstrip=17,
                **kwargs,
            )

    if data.dtype == 'uint8' or data.dtype == 'uint16':
        for compression in (None, 'deflate', 'zstd'):
            ext = '_%s%i_lerc%s.tif' % (
                data.dtype.kind,
                data.itemsize,
                '' if compression is None else ('_' + compression),
            )
            tifwrite(
                'rgb' + ext,
                data,
                compression='lerc',
                compressionargs={'compression': compression},
                rowsperstrip=17,
                **kwargs,
            )

    for compression in ('none', 'deflate', 'zstd', 'lzma', 'lzw'):
        kwargs['predictor'] = (
            compression != 'none' and data.dtype.char not in 'Qq'
        )
        ext = '_%s%i%s.tif' % (
            data.dtype.kind,
            data.itemsize,
            '' if compression == 'none' else ('_' + compression),
        )
        tifwrite(
            'rgb' + ext,
            data,
            compression=compression,
            rowsperstrip=17,
            **kwargs,
        )


def tif_f3(data):
    """Write float24 TIFF"""
    from tifffile import TiffFile

    from imagecodecs import float24_encode

    kwargs = dict(metadata=False, datetime=False, software='')
    tifwrite('gray_f3.tif', getdata(data, 'gray'), rowsperstrip=17, **kwargs)
    with TiffFile(OUTDIR + '/gray_f3.tif', mode='r+') as tif:
        fh = tif.filehandle
        page = tif.pages.first
        # print(page.tags['BitsPerSample'])
        # print(page.dataoffsets)
        bytecounts = []
        for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
            fh.seek(offset)
            strip = fh.read(bytecount)
            strip = numpy.frombuffer(strip, dtype='float32')
            strip = float24_encode(strip)
            fh.seek(offset)
            fh.write(strip)
            bytecounts.append(len(strip))
        page.tags['BitsPerSample'].overwrite(24)
        page.tags['StripByteCounts'].overwrite(bytecounts)


if 1:
    NOISE = gennoise()

    DATA = convert(NOISE, 'b1', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    jxr(DATA)

    DATA = convert(NOISE, 'f8', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    zfp(DATA)
    lerc(DATA)
    # jxl(DATA)

    DATA = convert(NOISE, 'f4', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    zfp(DATA)
    jxr(DATA)
    lerc(DATA)
    jxl(DATA)
    rgbe(DATA)
    tif_f3(DATA)

    DATA = convert(NOISE, 'f2', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    jxr(DATA)  # regression saving extra channels
    # lerc(DATA)
    jxl(DATA)
    udr(DATA)
    # avif(DATA)

    DATA = convert(NOISE, 'i1', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    j2k(DATA)
    jph(DATA)
    lerc(DATA)

    DATA = convert(NOISE, 'i2', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)
    j2k(DATA)
    jph(DATA)
    lerc(DATA)

    DATA = convert(NOISE, 'i4', force_copy=True)
    DATA //= 64  # 26 bit
    raw(DATA)
    npy(DATA)
    tif(DATA)
    j2k(DATA)
    jph(DATA)
    zfp(DATA)
    lerc(DATA)

    DATA = convert(NOISE, 'i8', force_copy=True)
    # DATA >>= 2
    raw(DATA)
    npy(DATA)
    tif(DATA)
    zfp(DATA)

    DATA = convert(NOISE, 'u1', force_copy=True)
    raw(DATA)
    npy(DATA)
    bmp(DATA)
    gif(DATA)
    tif(DATA)
    png(DATA)
    apng(DATA)
    jpg(DATA)
    j2k(DATA)
    jph(DATA)
    jls(DATA)
    webp(DATA)
    jxr(DATA)
    jxs(DATA)
    lerc(DATA)
    avif(DATA)
    jxl(DATA)
    heif(DATA)
    qoi(DATA)

    DATA = convert(NOISE, 'u2', force_copy=True)
    DATA //= 16  # 12 bit
    raw(DATA)
    npy(DATA)
    tif(DATA)
    png(DATA)
    apng(DATA)
    jpg(DATA)
    j2k(DATA)
    jph(DATA)
    jls(DATA)
    jxr(DATA)
    jxs(DATA)
    lerc(DATA)
    avif(DATA)
    jxl(DATA)
    heif(DATA)

    DATA = convert(NOISE, 'u4', force_copy=True)
    DATA //= 64  # 26 bit
    raw(DATA)
    npy(DATA)
    tif(DATA)
    lerc(DATA)
    j2k(DATA)
    jph(DATA)
    # jxl(DATA)

    DATA = convert(NOISE, 'u8', force_copy=True)
    raw(DATA)
    npy(DATA)
    tif(DATA)


if 0:
    tifwrite('rgba16.tif', DATA)
    tifwrite('rgb16.tif', DATA[..., :3])
    tifwrite('gray16.tif', DATA[..., -1])
    tifwrite(
        'graya16.tif',
        DATA[..., -2:],
        photometric='minisblack',
        extrasamples=['UNASSALPHA'],
    )

    DATA //= 256
    DATA = DATA.astype('uint8')

    tifwrite('rgba8.tif', DATA)
    tifwrite('rgb8.tif', DATA[..., :3])
    tifwrite('channels.tif', DATA[..., 0])
    tifwrite(
        'graya8.tif',
        DATA[..., -2:],
        photometric='minisblack',
        extrasamples=['UNASSALPHA'],
    )


def im16(data):
    """..."""
    array_buffer = data.tobytes()
    samples = data.shape[-1]
    if samples == 1:
        mode = 'I;16'
    elif samples == 3:
        mode = 'RGB;16'
    elif samples == 4:
        mode = 'RGBA;16'

    img = Image.new('I', data.T.shape[-2:])
    img.frombytes(array_buffer, 'raw', mode)
    return img


if 0:
    from PIL import Image

    im = im16(DATA)
    im.save('rgba16.png', 'PNG')

    im = im16(DATA[..., :3])
    im.save('rgb16.png', 'PNG')

    im = im16(DATA[..., :1])
    im.save('gray16.png', 'PNG')

    DATA //= 256
    DATA = DATA.astype('uint8')

    im = Image.fromarray(DATA)
    im.save('rgba8.png', 'PNG')

    im = Image.fromarray(DATA[..., :3])
    im.save('rgb8.png', 'PNG')

    im = Image.fromarray(DATA[..., 0])
    im.save('channels.png', 'PNG')
