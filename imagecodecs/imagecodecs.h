/* imagecodecs.h */

#ifndef ICD_H
#define ICD_H

#define ICD_VERSION "2019.1.1"

#include <stdint.h>

#ifndef HAVE_SSIZE_T
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define HAVE_SSIZE_T 1
#else
#include <sys/types.h>
#endif
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX INTPTR_MAX
#define SSIZE_MIN INTPTR_MIN
#endif


/* Endianness */

#ifndef ICD_MSB
#define ICD_MSB 1
#endif

#if ICD_MSB
#define ICD_LSB 0
#define ICD_BOC '<'
#else
#define ICD_LSB 1
#define ICD_BOC '>'
#endif

#if ICD_LSB
#error Big-endian platforms not supported
#endif


/* Error codes */

#define ICD_OK 0
#define ICD_ERROR -1
#define ICD_MEMORY_ERROR -2
#define ICD_RUNTIME_ERROR -3
#define ICD_NOTIMPLEMENTED_ERROR -4
#define ICD_VALUE_ERROR -5

#define ICD_LZW_INVALID -10
#define ICD_LZW_NOTIMPLEMENTED -11
#define ICD_LZW_BUFFER_TOO_SMALL -12
#define ICD_LZW_TABLE_TOO_SMALL -13

/* Function declarations */

void icd_swapbytes(
    void *src,
    const ssize_t srcsize,
    const ssize_t itemsize);


unsigned char icd_bitmask(
    const int numbits);


ssize_t icd_packints_decode(
    const uint8_t *src,
    const ssize_t srcsize,
    uint8_t *dst,
    const ssize_t dstsize,
    const int numbits);


ssize_t icd_packbits_size(
    const uint8_t *src,
    const ssize_t srcsize);


ssize_t icd_packbits_decode(
    const uint8_t *src,
    const ssize_t srcsize,
    uint8_t *dst,
    const ssize_t dstsize);


ssize_t icd_packbits_encode(
    const uint8_t *src,
    const ssize_t srcsize,
    uint8_t *dst,
    const ssize_t dstsize);


ssize_t icd_delta(
    void *src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void *dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode);


ssize_t icd_diff(
    void *src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void *dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const char itemtype,
    const int decode);


ssize_t icd_xor(
    void *src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void *dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const int decode);


ssize_t icd_floatpred(
    void *src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    void *dst,
    const ssize_t dstsize,
    const ssize_t dststride,
    const ssize_t itemsize,
    const ssize_t samples,
    const char byteorder,
    const int decode);


ssize_t icd_bitorder(
    uint8_t* src,
    const ssize_t srcsize,
    const ssize_t srcstride,
    const ssize_t itemsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t dststride);


typedef struct {
    ssize_t len;
    uint8_t *buf;
} icd_lzw_table_t;


typedef struct icd_lzw_handle {
    icd_lzw_table_t *table;
    uint8_t *buffer;
    ssize_t buffersize;
    ssize_t dummy;  /* icd_lzw_handle_t multiple of icd_lzw_table_t */
} icd_lzw_handle_t;


icd_lzw_handle_t *icd_lzw_new(
    ssize_t buffersize);

void icd_lzw_del(
    icd_lzw_handle_t *handle);

ssize_t icd_lzw_size(
    icd_lzw_handle_t *handle,
    const uint8_t *src,
    const ssize_t srcsize);


ssize_t icd_lzw_decode(
    icd_lzw_handle_t *handle,
    const uint8_t *src,
    const ssize_t srcsize,
    uint8_t *dst,
    const ssize_t dstsize);


#endif /* ICD_H */
