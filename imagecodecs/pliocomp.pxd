# imagecodecs/pliocomp.pxd

# Cython declarations for the `pliocomp 2026.5.10` library.
# Forked from `cfitsio 4.6.3`
# https://heasarc.gsfc.nasa.gov/fitsio/

cdef extern from 'pliocomp.h' nogil:

    const char* PLIO_VERSION

    int PLIO_OK
    int PLIO_ERROR_MEMORY
    int PLIO_ERROR_OVERFLOW
    int PLIO_ERROR_FORMAT
    int PLIO_ERROR

    int PLIO_HEADER_SIZE
    int PLIO_MAX_VALUE

    int plio_encode_ "plio_encode"(
        const int* src,
        int npix,
        short* dst,
        int maxout,
        int* nout
    )

    int plio_decode_ "plio_decode"(
        const short* src,
        int nsrc,
        int* dst,
        int npix
    )
