/* h5checksum.h */

#ifndef H5CHECKSUM_H
#define H5CHECKSUM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define FUNC_ENTER_NOAPI_NOINIT_NOERR
#define FUNC_ENTER_PACKAGE_NOERR
#define H5_ATTR_FALLTHROUGH
#define FUNC_LEAVE_NOAPI_VOID
#define FUNC_LEAVE_NOAPI(ret_value) return (ret_value);

#define H5_VERS_MAJOR 1
#define H5_VERS_MINOR 14
#define H5_VERS_RELEASE 3

uint32_t H5_checksum_fletcher32(const void *data, size_t len);
uint32_t H5_checksum_crc(const void *data, size_t len);
uint32_t H5_checksum_lookup3(const void *data, size_t len, uint32_t initval);
uint32_t H5_checksum_metadata(const void *data, size_t len, uint32_t initval);
uint32_t H5_hash_string(const char *str);

#endif
