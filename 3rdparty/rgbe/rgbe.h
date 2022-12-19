#ifndef _H_RGBE
#define _H_RGBE
/* THIS CODE CARRIES NO GUARANTEE OF USABILITY OR FITNESS FOR ANY PURPOSE.
 * WHILE THE AUTHORS HAVE TRIED TO ENSURE THE PROGRAM WORKS CORRECTLY,
 * IT IS STRICTLY USE AT YOUR OWN RISK.  */

/* utility for reading and writing Ward's rgbe image format.
   See rgbe.txt file for more details.
*/

#include <stddef.h>

#define RGBE_VERSION "2022.12.22"

#ifndef HAVE_SSIZE_T
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define HAVE_SSIZE_T 1
#else
#include <sys/types.h>
#endif
#endif

/* flags indicating which fields in an rgbe_header_info are valid */
#define RGBE_VALID_PROGRAMTYPE 0x01
#define RGBE_VALID_GAMMA 0x02
#define RGBE_VALID_EXPOSURE 0x04

/* return codes for rgbe routines */
#define RGBE_RETURN_SUCCESS 0
#define RGBE_RETURN_FAILURE -1
#define RGBE_READ_ERROR -2
#define RGBE_WRITE_ERROR -3
#define RGBE_FORMAT_ERROR -4
#define RGBE_MEMORY_ERROR -5

typedef struct {
    int valid;            /* indicate which fields are valid */
    char programtype[16]; /* listed at beginning of file to identify it
                           * after "#?".  defaults to "RGBE" */
    float gamma;          /* image has already been gamma corrected with
                           * given gamma.  defaults to 1.0 (no correction) */
    float exposure;       /* a value of 1.0 in an image corresponds to
                           * <exposure> watts/steradian/m^2.
                           * defaults to 1.0 */
} rgbe_header_info;

typedef struct {
    char *data;
    size_t size;
    size_t pos;
    char owner;
} rgbe_stream_t;

/* read or write headers */
/* you may set rgbe_header_info to null if you want to */
int
RGBE_WriteHeader(
    rgbe_stream_t *fp, int width, int height, rgbe_header_info *info);

int
RGBE_ReadHeader(
    rgbe_stream_t *fp, int *width, int *height, rgbe_header_info *info);

/* read or write pixels */
/* can read or write pixels in chunks of any size including single pixels*/
int
RGBE_WritePixels(rgbe_stream_t *fp, float *data, int numpixels);

int
RGBE_ReadPixels(rgbe_stream_t *fp, float *data, int numpixels);

/* read or write run length encoded files */
/* must be called to read or write whole scanlines */
int
RGBE_WritePixels_RLE(
    rgbe_stream_t *fp, float *data, int scanline_width, int num_scanlines);

int
RGBE_ReadPixels_RLE(
    rgbe_stream_t *fp, float *data, int scanline_width, int num_scanlines);

/* FILE-like memory-stream */

rgbe_stream_t *
rgbe_stream_new(size_t size, char *data);

void
rgbe_stream_del(rgbe_stream_t *stream);

size_t
rgbe_stream_read(void *ptr, size_t size, size_t nmemb, rgbe_stream_t *stream);

size_t
rgbe_stream_write(
    const void *ptr, size_t size, size_t nmemb, rgbe_stream_t *stream);

int
rgbe_stream_printf(rgbe_stream_t *stream, const char *format, ...);

char *
rgbe_stream_gets(char *str, size_t n, rgbe_stream_t *stream);

#endif /* _H_RGBE */
