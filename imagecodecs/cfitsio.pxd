# imagecodecs/cfitsio.pxd
# cython: language_level = 3

# Cython declarations for the `cfitsio 3.49` library.
# https://heasarc.gsfc.nasa.gov/fitsio/


cdef extern from 'fitsio.h':

    int CFITSIO_VERSION
    int CFITSIO_MINOR
    int CFITSIO_MAJOR
    int CFITSIO_SONAME

    int ffgmsg(
        char *err_message
    ) nogil


cdef extern from 'fitsio2.h':

    int fits_rcomp(
        int a[],
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    ) nogil

    int fits_rcomp_short(
        short a[],
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    ) nogil

    int fits_rcomp_byte(
        signed char a[],
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    ) nogil

    int fits_rdecomp(
        unsigned char* c,
        int clen,
        unsigned int array[],
        int nx,
        int nblock
    ) nogil

    int fits_rdecomp_short(
        unsigned char* c,
        int clen,
        unsigned short array[],
        int nx,
        int nblock
    ) nogil

    int fits_rdecomp_byte(
        unsigned char* c,
        int clen,
        unsigned char array[],
        int nx,
        int nblock
    ) nogil
