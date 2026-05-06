/* hcompress.h */

#ifndef HCOMPRESS_H
#define HCOMPRESS_H

#define HCOMP_VERSION "2026.5.10"

#define HCOMP_OK (0)
#define HCOMP_ERROR_MEMORY (-1)
#define HCOMP_ERROR_OVERFLOW (-2)
#define HCOMP_ERROR_FORMAT (-3)
#define HCOMP_ERROR (-4)

int hcomp_compress(
    int a[],
    int ny,
    int nx,
    int scale,
    char *output,
    long *nbytes,
    int *status
);

int hcomp_compress64(
    long long a[],
    int ny,
    int nx,
    int scale,
    char *output,
    long *nbytes,
    int *status
);

int hcomp_decompress(
    unsigned char *input,
    int nbin,
    int smooth,
    int *a,
    int na,
    int *ny,
    int *nx,
    int *scale,
    int *status
);

int hcomp_decompress64(
    unsigned char *input,
    int nbin,
    int smooth,
    long long *a,
    int na,
    int *ny,
    int *nx,
    int *scale,
    int *status
);

#endif
