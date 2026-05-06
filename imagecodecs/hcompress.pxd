# imagecodecs/hcompress.pxd

# Cython declarations for the `hcompress 2026.5.10` library.
# Forked from `cfitsio 4.6.3`
# https://heasarc.gsfc.nasa.gov/fitsio/

cdef extern from 'hcompress.h' nogil:

    const char* HCOMP_VERSION

    int HCOMP_OK
    int HCOMP_ERROR_MEMORY
    int HCOMP_ERROR_OVERFLOW
    int HCOMP_ERROR_FORMAT
    int HCOMP_ERROR

    int hcomp_compress(
        int* a,
        int ny,
        int nx,
        int scale,
        char* output,
        long* nbytes,
        int* status
    )

    int hcomp_compress64(
        long long* a,
        int ny,
        int nx,
        int scale,
        char* output,
        long* nbytes,
        int* status
    )

    int hcomp_decompress(
        unsigned char* input,
        int nbin,
        int smooth,
        int* a,
        int na,
        int* ny,
        int* nx,
        int* scale,
        int* status
    )

    int hcomp_decompress64(
        unsigned char* input,
        int nbin,
        int smooth,
        long long* a,
        int na,
        int* ny,
        int* nx,
        int* scale,
        int* status
    )
