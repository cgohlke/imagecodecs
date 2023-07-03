/* ricecomp.h */

#ifndef RCOMP_H
#define RCOMP_H

#define RCOMP_VERSION "2023.7.4"

#define RCOMP_OK (0)
#define RCOMP_ERROR_MEMORY (-1)
#define RCOMP_ERROR_EOB (-2)
#define RCOMP_ERROR_EOS (-3)
#define RCOMP_WARN_UNUSED (-4)

int rcomp_int(
    int a[],
    int nx,
    unsigned char *c,
    int clen,
    int nblock
);

int rcomp_short(
    short a[],
    int nx,
    unsigned char *c,
    int clen,
    int nblock
);

int rcomp_byte(
    signed char a[],
    int nx,
    unsigned char *c,
    int clen,
    int nblock
);

int rdecomp_int(
    unsigned char *c,
    int clen,
    unsigned int array[],
    int nx,
    int nblock
);

int rdecomp_short(
    unsigned char *c,
    int clen,
    unsigned short array[],
    int nx,
    int nblock
);

int rdecomp_byte(
    unsigned char *c,
    int clen,
    unsigned char array[],
    int nx,
    int nblock
);

#endif