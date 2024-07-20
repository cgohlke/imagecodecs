# imagecodecs/ricecomp.pxd
# cython: language_level = 3

# Cython declarations for the `ricecomp 2023.7.4` library.
# Forked from `cfitsio 3.49`
# https://heasarc.gsfc.nasa.gov/fitsio/


cdef extern from 'ricecomp.h' nogil:

    const char* RCOMP_VERSION

    int RCOMP_OK
    int RCOMP_ERROR_MEMORY
    int RCOMP_ERROR_EOB
    int RCOMP_ERROR_EOS
    int RCOMP_WARN_UNUSED

    int rcomp_int(
        int[] a,
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    )

    int rcomp_short(
        short[] a,
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    )

    int rcomp_byte(
        signed char[] a,
        int nx,
        unsigned char* c,
        int clen,
        int nblock
    )

    int rdecomp_int(
        unsigned char* c,
        int clen,
        unsigned int[] array,
        int nx,
        int nblock
    )

    int rdecomp_short(
        unsigned char* c,
        int clen,
        unsigned short[] array,
        int nx,
        int nblock
    )

    int rdecomp_byte(
        unsigned char* c,
        int clen,
        unsigned char[] array,
        int nx,
        int nblock
    )
