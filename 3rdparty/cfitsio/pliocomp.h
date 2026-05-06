/* pliocomp.h */

/*
  IRAF PLIO (pixel list I/O) line-list compression.
  Forked from cfitsio.
  https://heasarc.gsfc.nasa.gov/fitsio/
*/

#ifndef PLIOCOMP_H
#define PLIOCOMP_H

#include <stddef.h>

#define PLIO_VERSION "2026.5.10"

#define PLIO_OK (0)
#define PLIO_ERROR_MEMORY (-1)
#define PLIO_ERROR_OVERFLOW (-2)
#define PLIO_ERROR_FORMAT (-3)
#define PLIO_ERROR (-4)

/* PLIO line-list header size in shorts. */
#define PLIO_HEADER_SIZE 7

/* Maximum pixel value for PLIO encoding (24 bits) */
#define PLIO_MAX_VALUE 16777215

int plio_encode(
    const int *src,
    int npix,
    short *dst,
    int maxout,
    int *nout
);

int plio_decode(
    const short *src,
    int nsrc,
    int *dst,
    int npix
);

#endif
