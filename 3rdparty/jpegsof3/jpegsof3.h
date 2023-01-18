/* jpegsof3.h */

/* This file is a modified version of

https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.h

which is distributed under the BSD 3-Clause License:

The Software has been developed for research purposes only and
is not a clinical tool.

Copyright(c) 2014 - 2016 Chris Rorden. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met :
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright owner nor the name of this project
(dcm2niix) may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT OWNER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO
EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
Decode DICOM Transfer Syntax 1.2.840.10008.1.2.4.70 and
1.2.840.10008.1.2.4.57
JPEG Lossless, Nonhierarchical
see ISO/IEC 10918-1 / ITU T.81
specifically, format with 'Start of Frame' (SOF) code 0xC3
http://www.w3.org/Graphics/JPEG/itu-t81.pdf
This code decodes data with 1..16 bits per pixel
It appears unique to medical imaging, and is not supported by most
JPEG libraries
http://www.dicomlibrary.com/dicom/transfer-syntax/
https://en.wikipedia.org/wiki/Lossless_JPEG#Lossless_mode_of_operation
*/

#ifndef _JPEGSOF3_
#define _JPEGSOF3_

#ifndef HAVE_SSIZE_T
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define HAVE_SSIZE_T 1
#else
#include <sys/types.h>
#endif
#endif

#define JPEGSOF3_VERSION "2023.1.23"

#define JPEGSOF3_OK 0
#define JPEGSOF3_INVALID_OUTPUT -1
#define JPEGSOF3_INVALID_SIGNATURE -2
#define JPEGSOF3_INVALID_HEADER_TAG -3
#define JPEGSOF3_SEGMENT_GT_IMAGE -4
#define JPEGSOF3_INVALID_ITU_T81 -5
#define JPEGSOF3_INVALID_BIT_DEPTH -6
#define JPEGSOF3_TABLE_CORRUPTED -7
#define JPEGSOF3_TABLE_SIZE_CORRUPTED -8
#define JPEGSOF3_INVALID_RESTART_SEGMENTS -9
#define JPEGSOF3_NO_TABLE -10

#ifdef __cplusplus
extern "C" {
#endif

int
decode_jpegsof3(
    unsigned char *lRawRA,
    ssize_t lRawSz,
    unsigned char *lImgRA8,
    ssize_t lImgSz,
    int *dimX,
    int *dimY,
    int *bits,
    int *frames);

#ifdef __cplusplus
}
#endif

#endif
