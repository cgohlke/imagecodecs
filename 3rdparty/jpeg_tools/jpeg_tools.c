/* jpeg_tools.c */

/* Merge, crop, and mosaic JPEG streams without lossy recompression. */

/* Initially generated with assistance from Claude (Anthropic),
   then reviewed and adapted by Christoph Gohlke. */

/*
SPDX-License-Identifier: 0BSD

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "jpeg_tools.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jpeglib.h"
#include "jerror.h"

#define JT_MAX_DIMENSION 65535


/*****************************************************************************/
/* Marker segment parser */

typedef struct jt_parser_t {
    const uint8_t* buf;
    ssize_t size;
    ssize_t pos;
    int tables;
    int done;
} jt_parser_t;


static int
jt_parser_init(
    jt_parser_t* p,
    const uint8_t* buf,
    const ssize_t size,
    const int tables)
{
    p->buf = buf;
    p->size = size;
    p->pos = 2;
    p->tables = tables;
    p->done = 0;
    if (buf == NULL || size < 2 || buf[0] != 0xFF || buf[1] != 0xD8) {
        return JPEG_TOOLS_INVALID_HEADER;
    }
    return JPEG_TOOLS_OK;
}


/* Return 1 and next marker segment, 0 at end, or negative error code.
   Header streams end after the SOS segment, tables-only streams at EOI. */
static int
jt_parser_next(
    jt_parser_t* p,
    int* marker,
    const uint8_t** segment,
    ssize_t* segsize)
{
    const uint8_t* buf = p->buf;
    const ssize_t size = p->size;
    ssize_t pos = p->pos;
    ssize_t length;
    int m;

    if (p->done) {
        return 0;
    }
    if (p->tables && pos >= size) {
        p->done = 1;
        return 0;
    }
    if (pos >= size || buf[pos] != 0xFF) {
        return JPEG_TOOLS_INVALID_HEADER;
    }
    while (pos + 1 < size && buf[pos + 1] == 0xFF) {
        pos++;  /* fill bytes */
    }
    if (pos + 1 >= size) {
        return JPEG_TOOLS_DATA_TRUNCATED;
    }
    m = buf[pos + 1];
    if (m == 0xD9 && p->tables) {
        p->pos = pos + 2;
        p->done = 1;
        return 0;
    }
    if (m == 0x01 || (m >= 0xD0 && m <= 0xD9)) {
        return JPEG_TOOLS_INVALID_HEADER;
    }
    if (pos + 4 > size) {
        return JPEG_TOOLS_DATA_TRUNCATED;
    }
    length = ((ssize_t)buf[pos + 2] << 8) | (ssize_t)buf[pos + 3];
    if (length < 2 || length > size - pos - 2) {
        return JPEG_TOOLS_DATA_TRUNCATED;
    }
    *marker = m;
    *segment = buf + pos;
    *segsize = length + 2;
    p->pos = pos + 2 + length;
    if (m == 0xDA) {
        if (p->tables) {
            return JPEG_TOOLS_INVALID_HEADER;
        }
        p->done = 1;
    }
    return 1;
}


static inline int
jt_is_app(
    const int marker)
{
    return marker >= 0xE0 && marker <= 0xEF;
}


static inline int
jt_is_sof(
    const int marker)
{
    return (
        marker >= 0xC0 && marker <= 0xCF
        && marker != 0xC4 && marker != 0xC8 && marker != 0xCC
    );
}


static inline int
jt_is_adobe(
    const int marker,
    const uint8_t* segment,
    const ssize_t segsize)
{
    return (
        marker == 0xEE && segsize >= 9 && memcmp(segment + 4, "Adobe", 5) == 0
    );
}


static inline int
jt_is_jfif(
    const int marker,
    const uint8_t* segment,
    const ssize_t segsize)
{
    return (
        marker == 0xE0 && segsize >= 9 && memcmp(segment + 4, "JFIF", 5) == 0
    );
}


/* Return whether segment is required for decoding entropy-coded data. */
static inline int
jt_is_critical(
    const int marker)
{
    return (
        jt_is_sof(marker)
        || marker == 0xC4  /* DHT */
        || marker == 0xCC  /* DAC */
        || marker == 0xDB  /* DQT */
        || marker == 0xDD  /* DRI */
        || marker == 0xDA  /* SOS */
    );
}


/*****************************************************************************/
/* Header information */

typedef struct jt_header_t {
    ssize_t scanpos;  /* position after SOS segment */
    ssize_t height;
    ssize_t width;
    ssize_t restart;  /* restart interval in MCUs or 0 */
    int sof;          /* SOF marker or 0 */
    int ncomponents;
    int nscancomponents;
    int hmax;
    int vmax;
} jt_header_t;


static int
jt_header_parse(
    jt_header_t* h,
    const uint8_t* buf,
    const ssize_t size)
{
    jt_parser_t p;
    const uint8_t* seg = NULL;
    ssize_t segsize = 0;
    int marker = 0;
    int ret, i, hs, vs;

    memset(h, 0, sizeof(jt_header_t));
    ret = jt_parser_init(&p, buf, size, 0);
    if (ret < 0) {
        return ret;
    }
    while ((ret = jt_parser_next(&p, &marker, &seg, &segsize)) > 0) {
        if (jt_is_sof(marker) && h->sof == 0) {
            if (segsize < 10) {
                return JPEG_TOOLS_INVALID_HEADER;
            }
            h->sof = marker;
            h->height = ((ssize_t)seg[5] << 8) | seg[6];
            h->width = ((ssize_t)seg[7] << 8) | seg[8];
            h->ncomponents = seg[9];
            if (segsize < 10 + 3 * (ssize_t)h->ncomponents) {
                return JPEG_TOOLS_INVALID_HEADER;
            }
            for (i = 0; i < h->ncomponents; i++) {
                hs = seg[11 + 3 * i] >> 4;
                vs = seg[11 + 3 * i] & 0x0F;
                if (hs > h->hmax) {
                    h->hmax = hs;
                }
                if (vs > h->vmax) {
                    h->vmax = vs;
                }
            }
        }
        else if (marker == 0xDD) {
            if (segsize < 6) {
                return JPEG_TOOLS_INVALID_HEADER;
            }
            h->restart = ((ssize_t)seg[4] << 8) | seg[5];
        }
        else if (marker == 0xDA) {
            if (segsize < 5) {
                return JPEG_TOOLS_INVALID_HEADER;
            }
            h->nscancomponents = seg[4];
        }
    }
    if (ret < 0) {
        return ret;
    }
    h->scanpos = p.pos;
    return JPEG_TOOLS_OK;
}


/* Compare critical segments of two headers, ignoring SOF dimensions. */
static int
jt_header_compare(
    const uint8_t* abuf,
    const ssize_t asize,
    const uint8_t* bbuf,
    const ssize_t bsize)
{
    jt_parser_t a, b;
    const uint8_t* aseg = NULL;
    const uint8_t* bseg = NULL;
    ssize_t aseglen = 0;
    ssize_t bseglen = 0;
    int amarker = 0;
    int bmarker = 0;
    int aret, bret;

    if ((aret = jt_parser_init(&a, abuf, asize, 0)) < 0) {
        return aret;
    }
    if ((bret = jt_parser_init(&b, bbuf, bsize, 0)) < 0) {
        return bret;
    }
    for (;;) {
        do {
            aret = jt_parser_next(&a, &amarker, &aseg, &aseglen);
        } while (aret > 0 && !jt_is_critical(amarker));
        do {
            bret = jt_parser_next(&b, &bmarker, &bseg, &bseglen);
        } while (bret > 0 && !jt_is_critical(bmarker));
        if (aret < 0) {
            return aret;
        }
        if (bret < 0) {
            return bret;
        }
        if (aret == 0 || bret == 0) {
            return (aret == bret) ? JPEG_TOOLS_OK : JPEG_TOOLS_SHAPE_MISMATCH;
        }
        if (amarker != bmarker || aseglen != bseglen) {
            return JPEG_TOOLS_SHAPE_MISMATCH;
        }
        if (jt_is_sof(amarker)) {
            if (aseglen < 10) {
                return JPEG_TOOLS_INVALID_HEADER;
            }
            if (
                memcmp(aseg, bseg, 5) != 0
                || memcmp(aseg + 9, bseg + 9, (size_t)(aseglen - 9)) != 0)
            {
                return JPEG_TOOLS_SHAPE_MISMATCH;
            }
        }
        else if (memcmp(aseg, bseg, (size_t)aseglen) != 0) {
            return JPEG_TOOLS_SHAPE_MISMATCH;
        }
    }
}


/*****************************************************************************/
/* Output writer; counts bytes if dst is NULL */

typedef struct jt_writer_t {
    uint8_t* dst;
    ssize_t dstsize;
    ssize_t pos;
} jt_writer_t;


static int
jt_write(
    jt_writer_t* w,
    const uint8_t* src,
    const ssize_t size)
{
    if (size <= 0) {
        return JPEG_TOOLS_OK;
    }
    if (size > PTRDIFF_MAX - w->pos) {
        return JPEG_TOOLS_IMAGE_TOO_LARGE;
    }
    if (w->dst != NULL) {
        if (size > w->dstsize - w->pos) {
            return JPEG_TOOLS_OUTPUT_TOO_SMALL;
        }
        memcpy(w->dst + w->pos, src, (size_t)size);
    }
    w->pos += size;
    return JPEG_TOOLS_OK;
}


/* Return next restart interval [pos, *end) in entropy-coded data.
   Return 1 if interval ends at RST marker, 0 at EOI or end of data. */
static int
jt_interval(
    const uint8_t* buf,
    const ssize_t size,
    const ssize_t pos,
    ssize_t* end,
    ssize_t* next)
{
    ssize_t i = pos;
    int m;

    while (i < size) {
        if (buf[i] != 0xFF) {
            i++;
            continue;
        }
        if (i + 1 >= size) {
            break;
        }
        m = buf[i + 1];
        if (m == 0x00) {
            i += 2;  /* stuffed byte */
        }
        else if (m == 0xFF) {
            i++;  /* fill byte */
        }
        else if (m >= 0xD0 && m <= 0xD7) {
            *end = i;
            *next = i + 2;
            return 1;
        }
        else if (m == 0xD9) {
            *end = i;
            *next = size;
            return 0;
        }
        else {
            return JPEG_TOOLS_UNSUPPORTED_FORMAT;
        }
    }
    *end = size;
    *next = size;
    return 0;
}


/* Write count restart intervals of tile starting at *pos. */
static int
jt_write_intervals(
    jt_writer_t* w,
    const uint8_t* buf,
    const ssize_t size,
    ssize_t* pos,
    const ssize_t count,
    ssize_t* rst)
{
    uint8_t marker[2] = { 0xFF, 0xD0 };
    ssize_t k, end, next;
    int ret;

    for (k = 0; k < count; k++) {
        if (*pos >= size) {
            return JPEG_TOOLS_DATA_TRUNCATED;
        }
        ret = jt_interval(buf, size, *pos, &end, &next);
        if (ret < 0) {
            return ret;
        }
        if (ret == 0 && k + 1 < count) {
            return JPEG_TOOLS_DATA_TRUNCATED;
        }
        if (*rst >= 0) {
            marker[1] = (uint8_t)(0xD0 + (*rst & 7));
            if ((ret = jt_write(w, marker, 2)) < 0) {
                return ret;
            }
        }
        *rst += 1;
        if ((ret = jt_write(w, buf + *pos, end - *pos)) < 0) {
            return ret;
        }
        *pos = next;
    }
    return JPEG_TOOLS_OK;
}


typedef struct jt_tile_t {
    ssize_t scanpos;
    ssize_t height;
    ssize_t width;
} jt_tile_t;


static int
jt_merge_write(
    const jpeg_tools_input_t* in,
    jt_writer_t* w)
{
    static const uint8_t soi[2] = { 0xFF, 0xD8 };
    static const uint8_t eoi[2] = { 0xFF, 0xD9 };
    uint8_t adobe[16] = {
        0xFF, 0xEE, 0x00, 0x0E, 'A', 'd', 'o', 'b', 'e',
        0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    uint8_t buf8[8];
    jt_parser_t p;
    jt_header_t hdr;
    jt_header_t th;
    jt_tile_t* tiles = NULL;
    ssize_t* cursor = NULL;
    const uint8_t* hbuf;
    const uint8_t* seg = NULL;
    const uint8_t* scan;
    ssize_t hsize, scansize, segsize = 0;
    ssize_t ntiles, i, r, c, t, m;
    ssize_t mcuw = 8;
    ssize_t mcuh = 8;
    ssize_t height = 0;
    ssize_t width = 0;
    ssize_t restart = 0;
    ssize_t rst = -1;
    ssize_t tw, tc, count;
    int transform = -1;
    int ncomponents = 0;
    int marker = 0;
    int leading, ret;
    int multi, nojfif;

    if (
        in->rows < 1 || in->columns < 1 || in->data == NULL
        || in->datasize == NULL
        || in->rows > PTRDIFF_MAX / in->columns)
    {
        return JPEG_TOOLS_VALUE_ERROR;
    }
    ntiles = in->rows * in->columns;
    multi = ntiles > 1;

    switch (in->colorspace)
    {
        case 0:
            break;
        case JCS_GRAYSCALE:
            ncomponents = 1;
            break;
        case JCS_RGB:
            ncomponents = 3;
            transform = 0;
            break;
        case JCS_CMYK:
            ncomponents = 4;
            transform = 0;
            break;
        case JCS_YCbCr:
            ncomponents = 3;
            transform = 1;
            break;
        case JCS_YCCK:
            ncomponents = 4;
            transform = 2;
            break;
        default:
            return JPEG_TOOLS_VALUE_ERROR;
    }
    adobe[15] = (uint8_t)(transform < 0 ? 0 : transform);
    /* JFIF implies YCbCr or grayscale and takes precedence over Adobe */
    nojfif = transform == 0 || transform == 2;

    for (i = 0; i < ntiles; i++) {
        if (in->data[i] == NULL || in->datasize[i] <= 0) {
            return JPEG_TOOLS_TILE_MISSING;
        }
    }

    if (in->header != NULL) {
        hbuf = in->header;
        hsize = in->headersize;
    }
    else {
        hbuf = in->data[0];
        hsize = in->datasize[0];
    }
    ret = jt_header_parse(&hdr, hbuf, hsize);
    if (ret < 0) {
        return ret;
    }
    if (in->header != NULL && hdr.scanpos != hsize) {
        return JPEG_TOOLS_VALUE_ERROR;  /* header does not end with SOS */
    }
    if (ncomponents > 0 && hdr.sof != 0 && hdr.ncomponents != ncomponents) {
        return JPEG_TOOLS_VALUE_ERROR;
    }

    if (multi) {
        if (hdr.sof == 0) {
            return JPEG_TOOLS_INVALID_HEADER;
        }
        if (
            (hdr.sof != 0xC0 && hdr.sof != 0xC1 && hdr.sof != 0xC9)
            || hdr.nscancomponents != hdr.ncomponents
            || hdr.ncomponents < 1
            || hdr.hmax < 1
            || hdr.vmax < 1)
        {
            return JPEG_TOOLS_UNSUPPORTED_FORMAT;
        }
        if (hdr.ncomponents > 1) {
            mcuw = 8 * (ssize_t)hdr.hmax;
            mcuh = 8 * (ssize_t)hdr.vmax;
        }

        tiles = (jt_tile_t*)calloc((size_t)ntiles, sizeof(jt_tile_t));
        cursor = (ssize_t*)calloc((size_t)in->columns, sizeof(ssize_t));
        if (tiles == NULL || cursor == NULL) {
            ret = JPEG_TOOLS_MEMORY_ERROR;
            goto cleanup;
        }
        for (t = 0; t < ntiles; t++) {
            if (in->header != NULL) {
                tiles[t].scanpos = 0;
                tiles[t].height = hdr.height;
                tiles[t].width = hdr.width;
                continue;
            }
            ret = jt_header_parse(&th, in->data[t], in->datasize[t]);
            if (ret < 0) {
                goto cleanup;
            }
            if (t > 0) {
                ret = jt_header_compare(
                    in->data[0],
                    hdr.scanpos,
                    in->data[t],
                    th.scanpos
                );
                if (ret < 0) {
                    goto cleanup;
                }
            }
            tiles[t].scanpos = th.scanpos;
            tiles[t].height = th.height;
            tiles[t].width = th.width;
        }

        ret = JPEG_TOOLS_SHAPE_MISMATCH;
        for (r = 0; r < in->rows; r++) {
            for (c = 0; c < in->columns; c++) {
                t = r * in->columns + c;
                if (tiles[t].height < 1 || tiles[t].width < 1) {
                    ret = JPEG_TOOLS_UNSUPPORTED_FORMAT;
                    goto cleanup;
                }
                if (
                    tiles[t].height != tiles[r * in->columns].height
                    || tiles[t].width != tiles[c].width
                    || (r + 1 < in->rows && tiles[t].height % mcuh != 0)
                    || (c + 1 < in->columns && tiles[t].width % mcuw != 0))
                {
                    goto cleanup;
                }
            }
        }
        for (r = 0; r < in->rows; r++) {
            height += tiles[r * in->columns].height;
        }
        for (c = 0; c < in->columns; c++) {
            width += tiles[c].width;
        }
        if (height > JT_MAX_DIMENSION || width > JT_MAX_DIMENSION) {
            ret = JPEG_TOOLS_IMAGE_TOO_LARGE;
            goto cleanup;
        }

        restart = hdr.restart;
        if (restart == 0) {
            restart = (
                ((tiles[0].width + mcuw - 1) / mcuw)
                * ((tiles[0].height + mcuh - 1) / mcuh)
            );
        }
        if (restart > 65535) {
            ret = JPEG_TOOLS_IMAGE_TOO_LARGE;
            goto cleanup;
        }
        for (t = 0; t < ntiles; t++) {
            r = t / in->columns;
            c = t % in->columns;
            tw = (tiles[t].width + mcuw - 1) / mcuw;
            count = tw * ((tiles[t].height + mcuh - 1) / mcuh);
            if (hdr.restart == 0 && count != restart) {
                if (
                    !(in->columns == 1 && t + 1 == ntiles &&
                    count < restart))
                {
                    ret = JPEG_TOOLS_SHAPE_MISMATCH;
                    goto cleanup;
                }
            }
            if (in->columns > 1 && tw % restart != 0) {
                ret = JPEG_TOOLS_SHAPE_MISMATCH;
                goto cleanup;
            }
            if (in->columns == 1 && t + 1 < ntiles && count % restart != 0) {
                ret = JPEG_TOOLS_SHAPE_MISMATCH;
                goto cleanup;
            }
        }
    }

    /* SOI and leading APPn segments; JFIF requires APP0 after SOI */
    if ((ret = jt_write(w, soi, 2)) < 0) {
        goto cleanup;
    }
    jt_parser_init(&p, hbuf, hsize, 0);
    while ((ret = jt_parser_next(&p, &marker, &seg, &segsize)) > 0) {
        if (transform >= 0 && jt_is_adobe(marker, seg, segsize)) {
            continue;  /* decoders use last Adobe marker; avoid conflict */
        }
        if (nojfif && jt_is_jfif(marker, seg, segsize)) {
            continue;
        }
        if (!jt_is_app(marker)) {
            break;
        }
        if ((ret = jt_write(w, seg, segsize)) < 0) {
            goto cleanup;
        }
    }
    if (ret < 0) {
        goto cleanup;
    }
    if (transform >= 0) {
        if ((ret = jt_write(w, adobe, 16)) < 0) {
            goto cleanup;
        }
    }

    /* segments of tables-only stream without SOI and EOI */
    if (in->tables != NULL && in->tablessize > 0) {
        ret = jt_parser_init(&p, in->tables, in->tablessize, 1);
        if (ret < 0) {
            goto cleanup;
        }
        while ((ret = jt_parser_next(&p, &marker, &seg, &segsize)) > 0) {
            if ((ret = jt_write(w, seg, segsize)) < 0) {
                goto cleanup;
            }
        }
        if (ret < 0) {
            goto cleanup;
        }
    }

    /* remaining header segments up to SOS */
    jt_parser_init(&p, hbuf, hsize, 0);
    leading = 1;
    while ((ret = jt_parser_next(&p, &marker, &seg, &segsize)) > 0) {
        if (transform >= 0 && jt_is_adobe(marker, seg, segsize)) {
            continue;
        }
        if (nojfif && jt_is_jfif(marker, seg, segsize)) {
            continue;
        }
        if (leading && jt_is_app(marker)) {
            continue;
        }
        leading = 0;
        if (multi && marker == hdr.sof && segsize >= 10) {
            if ((ret = jt_write(w, seg, 5)) < 0) {
                goto cleanup;
            }
            buf8[0] = (uint8_t)(height >> 8);
            buf8[1] = (uint8_t)(height & 0xFF);
            buf8[2] = (uint8_t)(width >> 8);
            buf8[3] = (uint8_t)(width & 0xFF);
            if ((ret = jt_write(w, buf8, 4)) < 0) {
                goto cleanup;
            }
            if ((ret = jt_write(w, seg + 9, segsize - 9)) < 0) {
                goto cleanup;
            }
            continue;
        }
        if (multi && marker == 0xDA && hdr.restart == 0) {
            buf8[0] = 0xFF;
            buf8[1] = 0xDD;
            buf8[2] = 0x00;
            buf8[3] = 0x04;
            buf8[4] = (uint8_t)(restart >> 8);
            buf8[5] = (uint8_t)(restart & 0xFF);
            if ((ret = jt_write(w, buf8, 6)) < 0) {
                goto cleanup;
            }
        }
        if ((ret = jt_write(w, seg, segsize)) < 0) {
            goto cleanup;
        }
    }
    if (ret < 0) {
        goto cleanup;
    }

    if (!multi) {
        /* entropy-coded data is not modified */
        if (in->header != NULL) {
            scan = in->data[0];
            scansize = in->datasize[0];
        }
        else {
            scan = in->data[0] + hdr.scanpos;
            scansize = in->datasize[0] - hdr.scanpos;
        }
        if ((ret = jt_write(w, scan, scansize)) < 0) {
            goto cleanup;
        }
        if (
            scansize < 2
            || scan[scansize - 2] != 0xFF
            || scan[scansize - 1] != 0xD9)
        {
            ret = jt_write(w, eoi, 2);
        }
        goto cleanup;
    }

    if (in->columns == 1) {
        for (t = 0; t < ntiles; t++) {
            tw = (tiles[t].width + mcuw - 1) / mcuw;
            count = tw * ((tiles[t].height + mcuh - 1) / mcuh);
            i = tiles[t].scanpos;
            ret = jt_write_intervals(
                w,
                in->data[t],
                in->datasize[t],
                &i,
                (count + restart - 1) / restart,
                &rst
            );
            if (ret < 0) {
                goto cleanup;
            }
        }
    }
    else {
        /* interleave restart intervals of tiles in each MCU row */
        for (r = 0; r < in->rows; r++) {
            for (c = 0; c < in->columns; c++) {
                cursor[c] = tiles[r * in->columns + c].scanpos;
            }
            tc = (tiles[r * in->columns].height + mcuh - 1) / mcuh;
            for (m = 0; m < tc; m++) {
                for (c = 0; c < in->columns; c++) {
                    t = r * in->columns + c;
                    tw = (tiles[t].width + mcuw - 1) / mcuw;
                    ret = jt_write_intervals(
                        w,
                        in->data[t],
                        in->datasize[t],
                        &cursor[c],
                        tw / restart,
                        &rst
                    );
                    if (ret < 0) {
                        goto cleanup;
                    }
                }
            }
        }
    }
    ret = jt_write(w, eoi, 2);

cleanup:
    free(tiles);
    free(cursor);
    return ret;
}


ssize_t
jpeg_tools_merge(
    const jpeg_tools_input_t* input,
    uint8_t* dst,
    const ssize_t dstsize)
{
    jt_writer_t w;
    int ret;

    if (input == NULL || (dst != NULL && dstsize < 0)) {
        return JPEG_TOOLS_VALUE_ERROR;
    }
    w.dst = dst;
    w.dstsize = dstsize;
    w.pos = 0;
    ret = jt_merge_write(input, &w);
    if (ret < 0) {
        return ret;
    }
    return w.pos;
}


/*****************************************************************************/
/* libjpeg error and source managers */

typedef struct jt_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
} jt_error_mgr;


static void
jt_error_exit(
    j_common_ptr cinfo)
{
    jt_error_mgr* err = (jt_error_mgr*)cinfo->err;
    longjmp(err->setjmp_buffer, 1);
}


static void
jt_output_message(
    j_common_ptr cinfo)
{
    (void)cinfo;
}


/* Source manager reading consecutive memory buffers; fails at end of data */
typedef struct jt_source_mgr {
    struct jpeg_source_mgr pub;
    const uint8_t* buffer[3];
    size_t size[3];
    int index;
} jt_source_mgr;


static const JOCTET jt_eoi[2] = { 0xFF, 0xD9 };


static void
jt_init_source(
    j_decompress_ptr cinfo)
{
    (void)cinfo;
}


static boolean
jt_fill_input_buffer(
    j_decompress_ptr cinfo)
{
    jt_source_mgr* src = (jt_source_mgr*)cinfo->src;
    const uint8_t* buf;
    size_t size;

    while (src->index < 3) {
        buf = src->buffer[src->index];
        size = src->size[src->index];
        src->index++;
        if (buf != NULL && size > 0) {
            src->pub.next_input_byte = (const JOCTET*)buf;
            src->pub.bytes_in_buffer = size;
            return TRUE;
        }
    }
    ERREXIT(cinfo, JERR_INPUT_EOF);
    return FALSE;
}


static void
jt_skip_input_data(
    j_decompress_ptr cinfo,
    long num_bytes)
{
    jt_source_mgr* src = (jt_source_mgr*)cinfo->src;
    size_t remaining = num_bytes > 0 ? (size_t)num_bytes : 0;
    size_t n;

    while (remaining > 0) {
        if (src->pub.bytes_in_buffer == 0) {
            (void)jt_fill_input_buffer(cinfo);
        }
        n = src->pub.bytes_in_buffer;
        if (n > remaining) {
            n = remaining;
        }
        src->pub.next_input_byte += n;
        src->pub.bytes_in_buffer -= n;
        remaining -= n;
    }
}


static void
jt_term_source(
    j_decompress_ptr cinfo)
{
    (void)cinfo;
}


static void
jt_source(
    j_decompress_ptr cinfo,
    jt_source_mgr* src,
    const uint8_t* buf0,
    const ssize_t size0,
    const uint8_t* buf1,
    const ssize_t size1,
    const uint8_t* buf2,
    const ssize_t size2)
{
    src->pub.init_source = jt_init_source;
    src->pub.fill_input_buffer = jt_fill_input_buffer;
    src->pub.skip_input_data = jt_skip_input_data;
    src->pub.resync_to_restart = jpeg_resync_to_restart;
    src->pub.term_source = jt_term_source;
    src->pub.next_input_byte = NULL;
    src->pub.bytes_in_buffer = 0;
    src->buffer[0] = buf0;
    src->buffer[1] = buf1;
    src->buffer[2] = buf2;
    src->size[0] = size0 > 0 ? (size_t)size0 : 0;
    src->size[1] = size1 > 0 ? (size_t)size1 : 0;
    src->size[2] = size2 > 0 ? (size_t)size2 : 0;
    src->index = 0;
    cinfo->src = &src->pub;
}


static void
jt_tile_source(
    j_decompress_ptr cinfo,
    jt_source_mgr* src,
    const jpeg_tools_input_t* in,
    const ssize_t index)
{
    if (in->header != NULL && in->headersize > 0) {
        jt_source(
            cinfo,
            src,
            in->header,
            in->headersize,
            in->data[index],
            in->datasize[index],
            jt_eoi,
            2
        );
    }
    else {
        /* EOI is only read if data lacks one */
        jt_source(
            cinfo,
            src,
            in->data[index],
            in->datasize[index],
            jt_eoi,
            2,
            NULL,
            0
        );
    }
}


/*****************************************************************************/
/* Transcode */

static inline ssize_t
jt_round_up(
    const ssize_t a,
    const ssize_t b)
{
    return ((a + b - 1) / b) * b;
}


/* Copy DCT blocks of tile at pixel position (ty, tx) to crop window. */
static void
jt_copy_blocks(
    j_compress_ptr out,
    jvirt_barray_ptr* dstcoef,
    const ssize_t* dstwidth,
    const ssize_t* dstheight,
    j_decompress_ptr src,
    jvirt_barray_ptr* srccoef,
    const ssize_t ty,
    const ssize_t tx,
    const ssize_t top,
    const ssize_t left)
{
    const ssize_t imcuw = (ssize_t)src->max_h_samp_factor * DCTSIZE;
    const ssize_t imcuh = (ssize_t)src->max_v_samp_factor * DCTSIZE;
    jpeg_component_info* comp;
    JBLOCKARRAY srow, drow;
    ssize_t h, v, sbx, sby, cbx, cby, sw, sh, x0, x1, y0, y1, y;
    int ci;

    for (ci = 0; ci < out->num_components; ci++) {
        comp = &src->comp_info[ci];
        h = comp->h_samp_factor;
        v = comp->v_samp_factor;
        sbx = tx / imcuw * h;
        sby = ty / imcuh * v;
        cbx = left / imcuw * h;
        cby = top / imcuh * v;
        sw = jt_round_up((ssize_t)comp->width_in_blocks, h);
        sh = jt_round_up((ssize_t)comp->height_in_blocks, v);
        x0 = sbx > cbx ? sbx : cbx;
        x1 = sbx + sw;
        if (x1 > cbx + dstwidth[ci]) {
            x1 = cbx + dstwidth[ci];
        }
        y0 = sby > cby ? sby : cby;
        y1 = sby + sh;
        if (y1 > cby + dstheight[ci]) {
            y1 = cby + dstheight[ci];
        }
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }
        for (y = y0; y < y1; y++) {
            srow = (*src->mem->access_virt_barray)(
                (j_common_ptr)src,
                srccoef[ci],
                (JDIMENSION)(y - sby),
                1,
                FALSE
            );
            drow = (*out->mem->access_virt_barray)(
                (j_common_ptr)out,
                dstcoef[ci],
                (JDIMENSION)(y - cby),
                1,
                TRUE
            );
            memcpy(
                drow[0] + (x0 - cbx),
                srow[0] + (x0 - sbx),
                (size_t)(x1 - x0) * sizeof(JBLOCK)
            );
        }
    }
}


/* Return whether tile is compatible with reference tile. */
static int
jt_compatible(
    j_decompress_ptr ref,
    j_decompress_ptr tile,
    const int coefficients,
    const int samecolor)
{
    jpeg_component_info* rc;
    jpeg_component_info* tc;
    int ci;

    if (
        tile->num_components != ref->num_components
        || tile->data_precision != ref->data_precision
        || (samecolor && tile->jpeg_color_space != ref->jpeg_color_space))
    {
        return 0;
    }
#if JPEG_LIB_VERSION >= 80
    if (tile->block_size != DCTSIZE) {
        return 0;
    }
#endif
    for (ci = 0; ci < ref->num_components; ci++) {
        rc = &ref->comp_info[ci];
        tc = &tile->comp_info[ci];
        if (
            tc->h_samp_factor != rc->h_samp_factor
            || tc->v_samp_factor != rc->v_samp_factor)
        {
            return 0;
        }
        if (
            coefficients
            && rc->quant_table != NULL
            && tc->quant_table != NULL
            && memcmp(
            rc->quant_table->quantval,
            tc->quant_table->quantval,
            sizeof(rc->quant_table->quantval)
            ) != 0)
        {
            return 0;
        }
    }
    return 1;
}


/* Return whether tile size fits grid cell at tile index. */
static int
jt_tile_fits(
    j_decompress_ptr tile,
    const jpeg_tools_input_t* in,
    const ssize_t index,
    const ssize_t cellh,
    const ssize_t cellw)
{
    const ssize_t h = (ssize_t)tile->image_height;
    const ssize_t w = (ssize_t)tile->image_width;

    return (
        h <= cellh && w <= cellw
        && (index / in->columns + 1 == in->rows || h == cellh)
        && (index % in->columns + 1 == in->columns || w == cellw)
    );
}


/* Return whether saved marker is copied to output. */
static int
jt_keep_marker(
    j_compress_ptr out,
    const jpeg_saved_marker_ptr marker,
    const int copy)
{
    const unsigned int length = marker->data_length;
    const int code = marker->marker;

    switch (copy) {
        case 1:
            return code == JPEG_COM;
        case 2:
            return (
                code == JPEG_APP0 + 2
                && length >= 12
                && memcmp(marker->data, "ICC_PROFILE\0", 12) == 0
            );
        case 3:
            if (
                code == JPEG_APP0
                && length >= 5
                && memcmp(marker->data, "JFIF\0", 5) == 0)
            {
                /* JFIF is written by libjpeg or invalid for color space */
                return (
                    !out->write_JFIF_header
                    && (
                        out->jpeg_color_space == JCS_GRAYSCALE
                        || out->jpeg_color_space == JCS_YCbCr
                    )
                );
            }
            if (
                code == JPEG_APP0 + 14
                && length >= 5
                && memcmp(marker->data, "Adobe", 5) == 0)
            {
                return !out->write_Adobe_marker;
            }
            return 1;
        default:
            return 0;
    }
}


/* libjpeg objects are not local to the function calling setjmp */
typedef struct jt_transcode_t {
    struct jpeg_decompress_struct ref;
    struct jpeg_decompress_struct tmp;
    struct jpeg_compress_struct out;
    jt_error_mgr err;
    jt_source_mgr refsrc;
    jt_source_mgr tmpsrc;
} jt_transcode_t;


static int
jt_transcode(
    jt_transcode_t* ctx,
    const jpeg_tools_input_t* input,
    const jpeg_tools_options_t* options,
    const ssize_t refindex,
    uint8_t** dst,
    unsigned long* dstsize,
    char* msg)
{
    j_decompress_ptr ref = &ctx->ref;
    j_decompress_ptr tmp = &ctx->tmp;
    j_compress_ptr out = &ctx->out;
    jt_header_t hdr;
    jvirt_barray_ptr dstcoef[MAX_COMPONENTS];
    ssize_t dstwidth[MAX_COMPONENTS];
    ssize_t dstheight[MAX_COMPONENTS];
    jvirt_barray_ptr* refcoef = NULL;
    jvirt_barray_ptr* tilecoef = NULL;
    jpeg_component_info* comp;
    jpeg_saved_marker_ptr marker;
    const ssize_t ntiles = input->rows * input->columns;
    const int samecolor = input->colorspace == 0;
    ssize_t t, r, c, y;
    ssize_t cellh, cellw, imcuh, imcuw, top, left, height, width, ty, tx;
    int ci, hmax, vmax, hfound, wfound;
    int tmptables = 0;

    out->err = jpeg_std_error(&ctx->err.pub);
    ctx->err.pub.error_exit = jt_error_exit;
    ctx->err.pub.output_message = jt_output_message;
    ref->err = &ctx->err.pub;
    tmp->err = &ctx->err.pub;

    if (setjmp(ctx->err.setjmp_buffer)) {
        if (msg != NULL) {
            (*ctx->err.pub.format_message)((j_common_ptr)out, msg);
        }
        return JPEG_TOOLS_LIBJPEG_ERROR;
    }

    jpeg_create_compress(out);
    jpeg_create_decompress(ref);
    jpeg_create_decompress(tmp);

    /* markers must be saved before reading the header */
    if (options->copy == 1) {
        jpeg_save_markers(ref, JPEG_COM, 0xFFFF);
    }
    else if (options->copy == 2) {
        jpeg_save_markers(ref, JPEG_APP0 + 2, 0xFFFF);
    }
    else if (options->copy == 3) {
        jpeg_save_markers(ref, JPEG_COM, 0xFFFF);
        for (ci = 0; ci < 16; ci++) {
            jpeg_save_markers(ref, JPEG_APP0 + ci, 0xFFFF);
        }
    }

    if (input->tables != NULL && input->tablessize > 0) {
        jt_source(
            ref,
            &ctx->refsrc,
            input->tables,
            input->tablessize,
            jt_eoi,
            2,
            NULL,
            0
        );
        jpeg_read_header(ref, FALSE);
    }
    jt_tile_source(ref, &ctx->refsrc, input, refindex);
    jpeg_read_header(ref, TRUE);
#if JPEG_LIB_VERSION >= 80
    if (ref->block_size != DCTSIZE) {
        return JPEG_TOOLS_UNSUPPORTED_FORMAT;
    }
#endif
    if (input->colorspace > 0) {
        ref->jpeg_color_space = (J_COLOR_SPACE)input->colorspace;
    }

    /* cell size from tiles outside the last row and column if possible */
    cellh = (ssize_t)ref->image_height;
    cellw = (ssize_t)ref->image_width;
    hfound = (
        input->header != NULL || input->rows == 1
        || refindex / input->columns + 1 < input->rows
    );
    wfound = (
        input->header != NULL || input->columns == 1
        || refindex % input->columns + 1 < input->columns
    );
    for (t = refindex + 1; t < ntiles && !(hfound && wfound); t++) {
        r = t / input->columns;
        c = t % input->columns;
        if (
            input->data[t] == NULL || input->datasize[t] <= 0
            || !((!hfound && r + 1 < input->rows)
            || (!wfound && c + 1 < input->columns))
            || jt_header_parse(&hdr, input->data[t], input->datasize[t]) < 0
            || hdr.sof == 0)
        {
            continue;
        }
        if (!hfound && r + 1 < input->rows) {
            cellh = hdr.height;
            hfound = 1;
        }
        if (!wfound && c + 1 < input->columns) {
            cellw = hdr.width;
            wfound = 1;
        }
    }

    /* tile grid and crop window */
    imcuh = (ssize_t)ref->max_v_samp_factor * DCTSIZE;
    imcuw = (ssize_t)ref->max_h_samp_factor * DCTSIZE;
    if (cellh < 1 || cellw < 1) {
        return JPEG_TOOLS_UNSUPPORTED_FORMAT;
    }
    if (
        (input->rows > 1 && cellh % imcuh != 0)
        || (input->columns > 1 && cellw % imcuw != 0))
    {
        return JPEG_TOOLS_SHAPE_MISMATCH;
    }
    top = options->crop_top;
    left = options->crop_left;
    if (top % imcuh != 0 || left % imcuw != 0) {
        return JPEG_TOOLS_VALUE_ERROR;
    }
    if (
        input->rows > JT_MAX_DIMENSION
        || input->columns > JT_MAX_DIMENSION
        || top > PTRDIFF_MAX / 2
        || left > PTRDIFF_MAX / 2)
    {
        return JPEG_TOOLS_IMAGE_TOO_LARGE;
    }
    height = options->crop_height;
    if (height == 0) {
        height = input->rows * cellh - top;
    }
    width = options->crop_width;
    if (width == 0) {
        width = input->columns * cellw - left;
    }
    if (height < 1 || width < 1) {
        return JPEG_TOOLS_VALUE_ERROR;
    }
    if (height > JPEG_MAX_DIMENSION || width > JPEG_MAX_DIMENSION) {
        return JPEG_TOOLS_IMAGE_TOO_LARGE;
    }

    refcoef = jpeg_read_coefficients(ref);

    jpeg_copy_critical_parameters(ref, out);
    out->image_width = (JDIMENSION)width;
    out->image_height = (JDIMENSION)height;
#if JPEG_LIB_VERSION >= 70
    out->jpeg_width = (JDIMENSION)width;
    out->jpeg_height = (JDIMENSION)height;
#endif

    if (options->grayscale) {
        /* keep luma component with its quantization and entropy tables */
        if (
            out->num_components != 1
            && !(
                out->jpeg_color_space == JCS_YCbCr
                && out->num_components == 3
            ))
        {
            return JPEG_TOOLS_UNSUPPORTED_FORMAT;
        }
        out->num_components = 1;
        out->jpeg_color_space = JCS_GRAYSCALE;
        out->comp_info[0].h_samp_factor = 1;
        out->comp_info[0].v_samp_factor = 1;
        out->write_JFIF_header = TRUE;
        out->write_Adobe_marker = FALSE;
    }
    if (options->arithmetic) {
        out->arith_code = TRUE;
    }
    if (options->optimize >= 0) {
        out->optimize_coding = options->optimize > 0 ? TRUE : FALSE;
    }
    if (options->restart > 0) {
        out->restart_in_rows = options->restart;
    }
    if (options->progressive > 0) {
        jpeg_simple_progression(out);
    }

    /* output coefficient arrays; same sample scaling as source required */
    hmax = 1;
    vmax = 1;
    for (ci = 0; ci < out->num_components; ci++) {
        comp = &out->comp_info[ci];
        if (comp->h_samp_factor > hmax) {
            hmax = comp->h_samp_factor;
        }
        if (comp->v_samp_factor > vmax) {
            vmax = comp->v_samp_factor;
        }
    }
    for (ci = 0; ci < out->num_components; ci++) {
        comp = &out->comp_info[ci];
        if (
            comp->h_samp_factor * ref->max_h_samp_factor
            != ref->comp_info[ci].h_samp_factor * hmax
            || comp->v_samp_factor * ref->max_v_samp_factor
            != ref->comp_info[ci].v_samp_factor * vmax)
        {
            return JPEG_TOOLS_UNSUPPORTED_FORMAT;
        }
        dstwidth[ci] = jt_round_up(
            (width * comp->h_samp_factor + hmax * DCTSIZE - 1)
            / (hmax * DCTSIZE),
            comp->h_samp_factor
        );
        dstheight[ci] = jt_round_up(
            (height * comp->v_samp_factor + vmax * DCTSIZE - 1)
            / (vmax * DCTSIZE),
            comp->v_samp_factor
        );
        dstcoef[ci] = (*out->mem->request_virt_barray)(
            (j_common_ptr)out,
            JPOOL_IMAGE,
            TRUE,
            (JDIMENSION)dstwidth[ci],
            (JDIMENSION)dstheight[ci],
            (JDIMENSION)comp->v_samp_factor
        );
    }
    (*out->mem->realize_virt_arrays)((j_common_ptr)out);
    for (ci = 0; ci < out->num_components; ci++) {
        /* zero all rows so tiles can be written in any order */
        for (y = 0; y < dstheight[ci]; y++) {
            (void)(*out->mem->access_virt_barray)(
                (j_common_ptr)out,
                dstcoef[ci],
                (JDIMENSION)y,
                1,
                TRUE
            );
        }
    }

    for (t = 0; t < ntiles; t++) {
        ty = (t / input->columns) * cellh;
        tx = (t % input->columns) * cellw;
        if (
            ty >= top + height || ty + cellh <= top
            || tx >= left + width || tx + cellw <= left
            || input->data[t] == NULL || input->datasize[t] <= 0)
        {
            continue;  /* missing tiles remain zero */
        }
        if (t == refindex) {
            if (!jt_tile_fits(ref, input, t, cellh, cellw)) {
                return JPEG_TOOLS_SHAPE_MISMATCH;
            }
            jt_copy_blocks(
                out,
                dstcoef,
                dstwidth,
                dstheight,
                ref,
                refcoef,
                ty,
                tx,
                top,
                left
            );
            continue;
        }
        if (!tmptables) {
            if (input->tables != NULL && input->tablessize > 0) {
                jt_source(
                    tmp,
                    &ctx->tmpsrc,
                    input->tables,
                    input->tablessize,
                    jt_eoi,
                    2,
                    NULL,
                    0
                );
                jpeg_read_header(tmp, FALSE);
            }
            tmptables = 1;
        }
        jt_tile_source(tmp, &ctx->tmpsrc, input, t);
        jpeg_read_header(tmp, TRUE);
        if (
            !jt_compatible(ref, tmp, 0, samecolor)
            || !jt_tile_fits(tmp, input, t, cellh, cellw))
        {
            return JPEG_TOOLS_SHAPE_MISMATCH;
        }
        tilecoef = jpeg_read_coefficients(tmp);
        if (!jt_compatible(ref, tmp, 1, samecolor)) {
            return JPEG_TOOLS_SHAPE_MISMATCH;
        }
        jt_copy_blocks(
            out,
            dstcoef,
            dstwidth,
            dstheight,
            tmp,
            tilecoef,
            ty,
            tx,
            top,
            left
        );
        jpeg_abort_decompress(tmp);
    }

    jpeg_mem_dest(out, (unsigned char**)dst, dstsize);
    jpeg_write_coefficients(out, dstcoef);
    for (marker = ref->marker_list; marker != NULL; marker = marker->next) {
        if (jt_keep_marker(out, marker, options->copy)) {
            jpeg_write_marker(
                out,
                marker->marker,
                marker->data,
                marker->data_length
            );
        }
    }
    jpeg_finish_compress(out);
    return JPEG_TOOLS_OK;
}


int
jpeg_tools_transcode(
    const jpeg_tools_input_t* input,
    const jpeg_tools_options_t* options,
    uint8_t** dst,
    unsigned long* dstsize,
    char* msg)
{
    jt_transcode_t ctx;
    uint8_t* origdst;
    unsigned long origsize;
    ssize_t ntiles, t, refindex;
    int ret;

    if (msg != NULL) {
        msg[0] = '\0';
    }
    if (
        input == NULL || options == NULL || dst == NULL || dstsize == NULL
        || input->data == NULL || input->datasize == NULL
        || input->rows < 1 || input->columns < 1
        || input->rows > PTRDIFF_MAX / input->columns
        || options->crop_top < 0 || options->crop_left < 0
        || options->crop_height < 0 || options->crop_width < 0
        || options->copy < 0 || options->copy > 3
        || input->colorspace < 0 || input->colorspace > JCS_YCCK)
    {
        return JPEG_TOOLS_VALUE_ERROR;
    }
    ntiles = input->rows * input->columns;
    refindex = -1;
    for (t = 0; t < ntiles; t++) {
        if (input->data[t] != NULL && input->datasize[t] > 0) {
            refindex = t;
            break;
        }
    }
    if (refindex < 0) {
        return JPEG_TOOLS_TILE_MISSING;
    }

    origdst = *dst;
    origsize = *dstsize;
    memset(&ctx, 0, sizeof(ctx));

    ret = jt_transcode(&ctx, input, options, refindex, dst, dstsize, msg);

    if (ret < 0 && ctx.out.dest != NULL) {
        /* set *dst to buffer currently used by jpeg_mem_dest */
        (*ctx.out.dest->term_destination)(&ctx.out);
    }
    jpeg_destroy_compress(&ctx.out);
    jpeg_destroy_decompress(&ctx.ref);
    jpeg_destroy_decompress(&ctx.tmp);
    if (ret < 0) {
        if (*dst != origdst) {
            free(*dst);
        }
        *dst = origdst;
        *dstsize = origsize;
    }
    return ret;
}
