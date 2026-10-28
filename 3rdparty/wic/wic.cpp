/* wic.cpp */

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

/*
 * Windows Imaging Component (WIC) decode and encode implementations.
 *
 * Wraps COM-based WIC calls behind a plain C API.
 * Windows only.
 */

#include "wic.h"

#include <initguid.h>
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cwctype>
#include <mutex>
#include <new>

using Microsoft::WRL::ComPtr;

/*
 * SEH-safe wrappers around WIC COM calls.
 *
 * WIC can raise structured exceptions (e.g. STATUS_NOT_IMPLEMENTED)
 * when parsing truncated or malformed images. These helpers catch
 * any SEH exception and return a non-zero HRESULT so callers can
 * report a clean error instead of crashing.
 */

static __declspec(noinline) HRESULT
seh_create_decoder(
    IWICImagingFactory* factory,
    IStream* stream,
    IWICBitmapDecoder** decoder)
{
    __try {
        return factory->CreateDecoderFromStream(
            stream,
            nullptr,
            WICDecodeMetadataCacheOnDemand,
            decoder
        );
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_get_frame_count(
    IWICBitmapDecoder* decoder,
    UINT* count)
{
    __try {
        return decoder->GetFrameCount(count);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_get_frame(
    IWICBitmapDecoder* decoder,
    UINT index,
    IWICBitmapFrameDecode** frame)
{
    __try {
        return decoder->GetFrame(index, frame);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_get_pixel_format(
    IWICBitmapFrameDecode* frame,
    WICPixelFormatGUID* fmt)
{
    __try {
        return frame->GetPixelFormat(fmt);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_get_size(
    IWICBitmapFrameDecode* frame,
    UINT* w,
    UINT* h)
{
    __try {
        return frame->GetSize(w, h);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_copy_pixels_frame(
    IWICBitmapFrameDecode* frame,
    UINT stride,
    UINT bufsize,
    BYTE* buf)
{
    __try {
        return frame->CopyPixels(nullptr, stride, bufsize, buf);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_create_format_converter(
    IWICImagingFactory* factory,
    IWICFormatConverter** conv)
{
    __try {
        return factory->CreateFormatConverter(conv);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_converter_init(
    IWICFormatConverter* conv,
    IWICBitmapFrameDecode* frame,
    const WICPixelFormatGUID& fmt)
{
    __try {
        return conv->Initialize(
            frame,
            fmt,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom
        );
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

static __declspec(noinline) HRESULT
seh_copy_pixels_conv(
    IWICFormatConverter* conv,
    UINT stride,
    UINT bufsize,
    BYTE* buf)
{
    __try {
        return conv->CopyPixels(nullptr, stride, bufsize, buf);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return (HRESULT)GetExceptionCode();
    }
}

struct pixel_info {
    UINT channels;
    UINT bitsPerPixel;
    UINT bitsPerSample;  /* max set bits in any channel mask */
    WICPixelFormatNumericRepresentation numeric;
    WCHAR name[64];
};

/* SEH-safe query of source pixel format properties.
 * Uses raw pointers only in the __try scope. */
static __declspec(noinline) void
seh_query_pixel_format_info(
    IWICImagingFactory* factory,
    const WICPixelFormatGUID& srcFormat,
    pixel_info* info)
{
    memset(info, 0, sizeof(pixel_info));
    __try {
        IWICComponentInfo* compInfo = nullptr;
        if (FAILED(factory->CreateComponentInfo(srcFormat, &compInfo)))
            return;
        UINT actual = 0;
        if (FAILED(compInfo->GetFriendlyName(64, info->name, &actual)))
            info->name[0] = 0;
        info->name[63] = 0;
        IWICPixelFormatInfo2* pixInfo = nullptr;
        if (
            SUCCEEDED(
            compInfo->QueryInterface(
            __uuidof(IWICPixelFormatInfo2),
            reinterpret_cast<void**>(&pixInfo)
            )
        ))
        {
            pixInfo->GetChannelCount(&info->channels);
            pixInfo->GetBitsPerPixel(&info->bitsPerPixel);
            pixInfo->GetNumericRepresentation(&info->numeric);
            for (UINT i = 0; i < info->channels; i++) {
                BYTE mask[64];
                actual = 0;
                if (FAILED(pixInfo->GetChannelMask(
                    i, sizeof(mask), mask, &actual)))
                    continue;
                UINT bits = 0;
                for (UINT j = 0; j < actual && j < sizeof(mask); j++) {
                    for (BYTE b = mask[j]; b; b &= (BYTE)(b - 1)) bits++;
                }
                if (bits > info->bitsPerSample) info->bitsPerSample = bits;
            }
            pixInfo->Release();
        }
        compInfo->Release();
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        memset(info, 0, sizeof(pixel_info));
    }
}

namespace {

/* Module-level factory singleton.
 *
 * COM is initialized once at module load (wic_factory_init) and the
 * factory is shared across all decode calls. IWICImagingFactory is
 * MTA-safe: CreateDecoderFromStream/CreateFormatConverter return
 * independent per-call COM objects with no shared mutable state.
 */
ComPtr<IWICImagingFactory> g_factory;
bool g_com_initialized = false;
std::once_flag g_init_once;
HRESULT g_init_hr = S_FALSE;

/*
 * Dedicated MTA thread that owns the COM apartment and factory lifetime.
 *
 * WIC's IWICImagingFactory has ThreadingModel=Both, so the factory pointer
 * is usable from any calling thread without marshaling. Keeping COM init on
 * a separate thread avoids polluting the main thread's apartment, allowing
 * Qt (COINIT_APARTMENTTHREADED / OleInitialize) to co-exist without
 * triggering RPC_E_CHANGED_MODE.
 */
static HANDLE g_host_thread = NULL;
static HANDLE g_exit_event  = NULL;

static DWORD WINAPI
wic_host_thread_proc(LPVOID lpReady)
{
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) {
        g_com_initialized = true;
        hr = CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&g_factory)
        );
        if (FAILED(hr)) {
            CoUninitialize();
            g_com_initialized = false;
        }
    }
    g_init_hr = hr;
    /* Unblock wic_factory_init(). */
    SetEvent(*static_cast<HANDLE*>(lpReady));
    /* Stay alive until wic_factory_destroy() signals exit. */
    WaitForSingleObject(g_exit_event, INFINITE);
    g_factory.Reset();
    if (g_com_initialized) {
        CoUninitialize();
        g_com_initialized = false;
    }
    return 0;
}

/*
 * MemIStream: read-only in-memory IStream with no data copy.
 *
 * Wraps a const uint8_t* directly. The buffer must remain valid for
 * the lifetime of the stream (until IStream::Release returns 0).
 */
class MemIStream final : public IStream {
public:
    MemIStream(const uint8_t* data, size_t size) noexcept
        : m_refcount(1), m_data(data), m_size(size), m_pos(0) {}

    /* IUnknown */

    HRESULT STDMETHODCALLTYPE QueryInterface(
        REFIID riid, void** ppv) noexcept override
    {
        if (!ppv) return E_POINTER;
        if (
            IsEqualGUID(riid, __uuidof(IUnknown))
            || IsEqualGUID(riid, __uuidof(ISequentialStream))
            || IsEqualGUID(riid, __uuidof(IStream)))
        {
            *ppv = static_cast<IStream*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() noexcept override
    {
        return (ULONG)InterlockedIncrement(&m_refcount);
    }

    ULONG STDMETHODCALLTYPE Release() noexcept override
    {
        LONG rc = InterlockedDecrement(&m_refcount);
        if (rc == 0) delete this;
        return (ULONG)rc;
    }

    /* ISequentialStream */

    HRESULT STDMETHODCALLTYPE Read(
        void* pv, ULONG cb, ULONG* pcbRead) noexcept override
    {
        ULONG avail = (ULONG)(m_size - m_pos);
        ULONG n = cb < avail ? cb : avail;
        memcpy(pv, m_data + m_pos, n);
        m_pos += n;
        if (pcbRead) *pcbRead = n;
        return n == cb ? S_OK : S_FALSE;
    }

    HRESULT STDMETHODCALLTYPE Write(
        const void*, ULONG, ULONG*) noexcept override
    {
        return STG_E_ACCESSDENIED;
    }

    /* IStream */

    HRESULT STDMETHODCALLTYPE Seek(
        LARGE_INTEGER dlibMove,
        DWORD dwOrigin,
        ULARGE_INTEGER* plibNewPosition) noexcept override
    {
        LONGLONG np;
        switch (dwOrigin)
        {
            case STREAM_SEEK_SET:
                np = dlibMove.QuadPart; break;
            case STREAM_SEEK_CUR:
                np = (LONGLONG)m_pos + dlibMove.QuadPart; break;
            case STREAM_SEEK_END:
                np = (LONGLONG)m_size + dlibMove.QuadPart; break;
            default: return STG_E_INVALIDFUNCTION;
        }
        if (np < 0 || (size_t)np > m_size)
            return STG_E_INVALIDFUNCTION;
        m_pos = (size_t)np;
        if (plibNewPosition) plibNewPosition->QuadPart = (ULONGLONG)m_pos;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetSize(ULARGE_INTEGER) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE CopyTo(
        IStream*,
        ULARGE_INTEGER,
        ULARGE_INTEGER*,
        ULARGE_INTEGER*) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE Commit(DWORD) noexcept override
    {
        return S_OK;      /* no-op for read-only stream */
    }

    HRESULT STDMETHODCALLTYPE Revert() noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE LockRegion(
        ULARGE_INTEGER, ULARGE_INTEGER, DWORD) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE UnlockRegion(
        ULARGE_INTEGER, ULARGE_INTEGER, DWORD) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE Stat(
        STATSTG* pstatstg, DWORD) noexcept override
    {
        if (!pstatstg) return STG_E_INVALIDPOINTER;
        memset(pstatstg, 0, sizeof(STATSTG));
        pstatstg->type = STGTY_STREAM;
        pstatstg->cbSize.QuadPart = (ULONGLONG)m_size;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Clone(IStream**) noexcept override
    {
        return E_NOTIMPL;
    }

private:
LONG m_refcount;
const uint8_t* m_data;
size_t m_size;
size_t m_pos;
};

/* Create a read-only IStream wrapping [data, data+size).
 * Returns an empty ComPtr on allocation failure. */
ComPtr<IStream>
make_stream(const uint8_t* data, size_t size) noexcept
{
    ComPtr<IStream> s;
    MemIStream* raw = new(std::nothrow) MemIStream(data, size);
    if (raw) s.Attach(raw);  /* Attach: takes ownership, no AddRef */
    return s;
}

/*
 * GrowIStream: writable, growable in-memory IStream for encoding.
 *
 * Owns a heap-allocated buffer that doubles in capacity as needed.
 * After encoding, the caller calls detach() to take ownership of the buffer.
 */
class GrowIStream final : public IStream {
public:
    GrowIStream() noexcept
        : m_refcount(1), m_data(nullptr), m_size(0),
        m_capacity(0), m_pos(0) {}

    ~GrowIStream()
    {
        free(m_data);
    }

    /* detach the buffer; caller takes ownership and must free(). */
    uint8_t* detach(size_t* out_size) noexcept
    {
        uint8_t* p = m_data;
        if (out_size) *out_size = m_size;
        m_data = nullptr;
        m_size = 0;
        m_capacity = 0;
        m_pos = 0;
        return p;
    }

    /* IUnknown */

    HRESULT STDMETHODCALLTYPE QueryInterface(
        REFIID riid,
        void** ppv) noexcept override
    {
        if (!ppv) return E_POINTER;
        if (
            IsEqualGUID(riid, __uuidof(IUnknown))
            || IsEqualGUID(riid, __uuidof(ISequentialStream))
            || IsEqualGUID(riid, __uuidof(IStream)))
        {
            *ppv = static_cast<IStream*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() noexcept override
    {
        return (ULONG)InterlockedIncrement(&m_refcount);
    }

    ULONG STDMETHODCALLTYPE Release() noexcept override
    {
        LONG rc = InterlockedDecrement(&m_refcount);
        if (rc == 0) delete this;
        return (ULONG)rc;
    }

    /* ISequentialStream */

    HRESULT STDMETHODCALLTYPE Read(
        void* pv, ULONG cb, ULONG* pcbRead) noexcept override
    {
        ULONG avail = (m_pos < m_size)
                ? (ULONG)(m_size - m_pos) : 0;
        ULONG n = cb < avail ? cb : avail;
        if (n > 0) memcpy(pv, m_data + m_pos, n);
        m_pos += n;
        if (pcbRead) *pcbRead = n;
        return n == cb ? S_OK : S_FALSE;
    }

    HRESULT STDMETHODCALLTYPE Write(
        const void* pv, ULONG cb, ULONG* pcbWritten) noexcept override
    {
        if (cb == 0) {
            if (pcbWritten) *pcbWritten = 0;
            return S_OK;
        }
        size_t end = m_pos + cb;
        if (end < m_pos) return STG_E_MEDIUMFULL;      /* overflow */
        if (end > m_capacity) {
            size_t newcap = m_capacity ? m_capacity : 4096;
            while (newcap < end) {
                if (newcap > SIZE_MAX / 2) return STG_E_MEDIUMFULL;
                newcap *= 2;
            }
            uint8_t* p = (uint8_t*)realloc(m_data, newcap);
            if (!p) return STG_E_MEDIUMFULL;
            m_data = p;
            m_capacity = newcap;
        }
        memcpy(m_data + m_pos, pv, cb);
        m_pos += cb;
        if (m_pos > m_size) m_size = m_pos;
        if (pcbWritten) *pcbWritten = cb;
        return S_OK;
    }

    /* IStream */

    HRESULT STDMETHODCALLTYPE Seek(
        LARGE_INTEGER dlibMove,
        DWORD dwOrigin,
        ULARGE_INTEGER* plibNewPosition) noexcept override
    {
        LONGLONG np;
        switch (dwOrigin)
        {
            case STREAM_SEEK_SET:
                np = dlibMove.QuadPart; break;
            case STREAM_SEEK_CUR:
                np = (LONGLONG)m_pos + dlibMove.QuadPart; break;
            case STREAM_SEEK_END:
                np = (LONGLONG)m_size + dlibMove.QuadPart; break;
            default: return STG_E_INVALIDFUNCTION;
        }
        if (np < 0) return STG_E_INVALIDFUNCTION;
        m_pos = (size_t)np;
        /* allow seeking past end; write will grow */
        if (plibNewPosition) plibNewPosition->QuadPart = (ULONGLONG)m_pos;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetSize(
        ULARGE_INTEGER libNewSize) noexcept override
    {
        size_t ns = (size_t)libNewSize.QuadPart;
        if (ns > m_capacity) {
            uint8_t* p = (uint8_t*)realloc(m_data, ns);
            if (!p) return STG_E_MEDIUMFULL;
            m_data = p;
            m_capacity = ns;
        }
        if (ns > m_size)
            memset(m_data + m_size, 0, ns - m_size);
        m_size = ns;
        if (m_pos > m_size) m_pos = m_size;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CopyTo(
        IStream*,
        ULARGE_INTEGER,
        ULARGE_INTEGER*,
        ULARGE_INTEGER*) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE Commit(DWORD) noexcept override
    {
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Revert() noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE LockRegion(
        ULARGE_INTEGER, ULARGE_INTEGER, DWORD) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE UnlockRegion(
        ULARGE_INTEGER, ULARGE_INTEGER, DWORD) noexcept override
    {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE Stat(
        STATSTG* pstatstg, DWORD) noexcept override
    {
        if (!pstatstg) return STG_E_INVALIDPOINTER;
        memset(pstatstg, 0, sizeof(STATSTG));
        pstatstg->type = STGTY_STREAM;
        pstatstg->cbSize.QuadPart = (ULONGLONG)m_size;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Clone(
        IStream**) noexcept override
    {
        return E_NOTIMPL;
    }

    private:
    LONG m_refcount;
    uint8_t* m_data;
    size_t m_size;
    size_t m_capacity;
    size_t m_pos;
};

/*
 * Determine the best canonical target pixel format for decoding.
 *
 * Fast-path: six common formats are resolved by GUID comparison alone.
 * Fallback: uses channel count and significant bits per channel from
 * `pinfo`, or queries them if `pinfo` is NULL, then maps to
 * Gray/RGB/RGBA at 8 or 16 bpc.
 */
void
determine_target_format(
    IWICImagingFactory* pFactory,
    const WICPixelFormatGUID& srcFormat,
    const pixel_info* pinfo,
    WICPixelFormatGUID& dstFormat,
    uint32_t& components,
    uint32_t& bpc) noexcept
{
    /* fast-path: common formats needing no conversion */
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat16bppGray)) {
        dstFormat = GUID_WICPixelFormat16bppGray;
        components = 1; bpc = 16; return;
    }
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat8bppGray)) {
        dstFormat = GUID_WICPixelFormat8bppGray;
        components = 1; bpc = 8; return;
    }
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat24bppRGB)) {
        dstFormat = GUID_WICPixelFormat24bppRGB;
        components = 3; bpc = 8; return;
    }
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat48bppRGB)) {
        dstFormat = GUID_WICPixelFormat48bppRGB;
        components = 3; bpc = 16; return;
    }
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat32bppRGBA)) {
        dstFormat = GUID_WICPixelFormat32bppRGBA;
        components = 4; bpc = 8; return;
    }
    if (IsEqualGUID(srcFormat, GUID_WICPixelFormat64bppRGBA)) {
        dstFormat = GUID_WICPixelFormat64bppRGBA;
        components = 4; bpc = 16; return;
    }

    /* safe default for unknown formats */
    dstFormat = GUID_WICPixelFormat32bppRGBA;
    components = 4;
    bpc = 8;

    pixel_info local;
    if (!pinfo) {
        seh_query_pixel_format_info(pFactory, srcFormat, &local);
        pinfo = &local;
    }
    const UINT channelCount = pinfo->channels;
    if (channelCount == 0) return;

    /* padding bits (e.g. 32bppBGR) must not count as sample bits */
    const UINT bpcSource = pinfo->bitsPerSample
        ? pinfo->bitsPerSample
        : pinfo->bitsPerPixel / channelCount;
    const bool highBPC = bpcSource > 8;

    /* identify grayscale by GUID or single non-indexed channel */
    const bool isGray = (
        IsEqualGUID(srcFormat, GUID_WICPixelFormatBlackWhite)
        || IsEqualGUID(srcFormat, GUID_WICPixelFormat2bppGray)
        || IsEqualGUID(srcFormat, GUID_WICPixelFormat4bppGray)
        || IsEqualGUID(srcFormat, GUID_WICPixelFormat8bppGray)
        || IsEqualGUID(srcFormat, GUID_WICPixelFormat16bppGray)
        || IsEqualGUID(srcFormat, GUID_WICPixelFormat32bppGrayFloat)
        || (
            channelCount == 1
            && pinfo->numeric != WICPixelFormatNumericRepresentationIndexed
        )
    );

    /* alpha heuristic: 2 channels (gray+alpha) or 4+ (RGBA, CMYKA, ...) */
    const bool hasAlpha = (channelCount == 2 || channelCount >= 4);

    if (isGray) {
        dstFormat = highBPC
            ? GUID_WICPixelFormat16bppGray
            : GUID_WICPixelFormat8bppGray;
        components = 1;
        bpc = highBPC ? 16 : 8;
    }
    else if (hasAlpha) {
        dstFormat = highBPC
            ? GUID_WICPixelFormat64bppRGBA
            : GUID_WICPixelFormat32bppRGBA;
        components = 4;
        bpc = highBPC ? 16 : 8;
    }
    else {
        dstFormat = highBPC
            ? GUID_WICPixelFormat48bppRGB
            : GUID_WICPixelFormat24bppRGB;
        components = 3;
        bpc = highBPC ? 16 : 8;
    }
}

/* Map WIC_FORMAT_* constant to container format GUID. */
bool
format_to_container_guid(int32_t format, GUID& guid) noexcept
{
    switch (format)
    {
        case WIC_FORMAT_BMP:  guid = GUID_ContainerFormatBmp;  return true;
        case WIC_FORMAT_PNG:  guid = GUID_ContainerFormatPng;  return true;
        case WIC_FORMAT_JPEG: guid = GUID_ContainerFormatJpeg; return true;
        case WIC_FORMAT_TIFF: guid = GUID_ContainerFormatTiff; return true;
        case WIC_FORMAT_GIF:  guid = GUID_ContainerFormatGif;  return true;
        case WIC_FORMAT_WMP:  guid = GUID_ContainerFormatWmp;  return true;
        case WIC_FORMAT_HEIF: guid = GUID_ContainerFormatHeif; return true;
        case WIC_FORMAT_WEBP: guid = GUID_ContainerFormatWebp; return true;
        default: return false;
    }
}

/* Map (components, bpc) to WIC pixel format GUID. */
bool
pixel_format_guid(
    uint32_t components, uint32_t bpc, WICPixelFormatGUID& fmt) noexcept
{
    if (bpc == 8) {
        switch (components)
        {
            case 1: fmt = GUID_WICPixelFormat8bppGray;  return true;
            case 3: fmt = GUID_WICPixelFormat24bppRGB;  return true;
            case 4: fmt = GUID_WICPixelFormat32bppRGBA; return true;
        }
    }
    else if (bpc == 16) {
        switch (components)
        {
            case 1: fmt = GUID_WICPixelFormat16bppGray; return true;
            case 3: fmt = GUID_WICPixelFormat48bppRGB;  return true;
            case 4: fmt = GUID_WICPixelFormat64bppRGBA; return true;
        }
    }
    return false;
}

/* Write short lowercase container format name, e.g. "png", to out. */
void
container_name(
    IWICBitmapDecoder* decoder, char* out, size_t size) noexcept
{
    static const struct { const GUID* guid; const char* name; } names[] = {
        {&GUID_ContainerFormatBmp, "bmp"},
        {&GUID_ContainerFormatPng, "png"},
        {&GUID_ContainerFormatIco, "ico"},
        {&GUID_ContainerFormatJpeg, "jpeg"},
        {&GUID_ContainerFormatTiff, "tiff"},
        {&GUID_ContainerFormatGif, "gif"},
        {&GUID_ContainerFormatWmp, "wmp"},
        {&GUID_ContainerFormatDds, "dds"},
        {&GUID_ContainerFormatAdng, "dng"},
        {&GUID_ContainerFormatHeif, "heif"},
        {&GUID_ContainerFormatWebp, "webp"},
        {&GUID_ContainerFormatRaw, "raw"},
    };
    out[0] = 0;
    GUID guid;
    if (SUCCEEDED(decoder->GetContainerFormat(&guid))) {
        for (const auto& n : names) {
            if (IsEqualGUID(guid, *n.guid)) {
                snprintf(out, size, "%s", n.name);
                return;
            }
        }
    }
    /* unknown container: use first file extension of decoder, e.g. ".jxl" */
    ComPtr<IWICBitmapDecoderInfo> pInfo;
    if (FAILED(decoder->GetDecoderInfo(&pInfo))) return;
    WCHAR ext[64] = {};
    UINT actual = 0;
    if (FAILED(pInfo->GetFileExtensions(64, ext, &actual))) return;
    size_t i = 0;
    for (const WCHAR* p = ext[0] == L'.' ? ext + 1 : ext;
         *p && *p != L',' && i + 1 < size; p++) {
        if (*p > 127) break;
        out[i++] = (char)towlower(*p);
    }
    out[i] = 0;
}

}  /* namespace */

extern "C" int32_t
wic_factory_init(void)
{
    std::call_once(
        g_init_once,
        []() {
            HANDLE hReady = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!hReady) {
                g_init_hr = HRESULT_FROM_WIN32(GetLastError());
                return;
            }
            g_exit_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!g_exit_event) {
                CloseHandle(hReady);
                g_init_hr = HRESULT_FROM_WIN32(GetLastError());
                return;
            }
            g_host_thread = CreateThread(
                nullptr,
                0,
                wic_host_thread_proc,
                &hReady,
                0,
                nullptr
            );
            if (!g_host_thread) {
                g_init_hr = HRESULT_FROM_WIN32(GetLastError());
                CloseHandle(g_exit_event); g_exit_event = nullptr;
                CloseHandle(hReady);
                return;
            }
            /* wait for the background thread to finish COM/factory init. */
            WaitForSingleObject(hReady, INFINITE);
            CloseHandle(hReady);
            if (FAILED(g_init_hr)) {
                /* init failed; clean up the background thread. */
                SetEvent(g_exit_event);
                WaitForSingleObject(g_host_thread, INFINITE);
                CloseHandle(g_host_thread); g_host_thread = nullptr;
                CloseHandle(g_exit_event); g_exit_event = nullptr;
            }
        }
    );
    return (int32_t)g_init_hr;
}

extern "C" void
wic_factory_destroy(void)
{
    if (g_exit_event) {
        SetEvent(g_exit_event);
    }
    if (g_host_thread) {
        WaitForSingleObject(g_host_thread, INFINITE);
        CloseHandle(g_host_thread);
        g_host_thread = nullptr;
    }
    if (g_exit_event) {
        CloseHandle(g_exit_event);
        g_exit_event = nullptr;
    }
}

extern "C" int32_t
wic_get_info(
    const uint8_t* src,
    size_t srcsize,
    uint32_t frame_index,
    wic_info_t* info)
{
    if (!info)
        return (int32_t)E_INVALIDARG;
    memset(info, 0, sizeof(wic_info_t));
    if (!src || !srcsize)
        return (int32_t)E_INVALIDARG;
    if (!g_factory)
        return (int32_t)CO_E_NOTINITIALIZED;

    ComPtr<IStream> pStream = make_stream(src, srcsize);
    if (!pStream) return (int32_t)E_OUTOFMEMORY;

    ComPtr<IWICBitmapDecoder> pDecoder;
    HRESULT hr = seh_create_decoder(g_factory.Get(), pStream.Get(), &pDecoder);
    if (FAILED(hr)) return (int32_t)hr;

    UINT frameCount = 0;
    hr = seh_get_frame_count(pDecoder.Get(), &frameCount);
    if (FAILED(hr)) return (int32_t)hr;
    info->frame_count = frameCount;
    if (frame_index >= frameCount) return (int32_t)E_INVALIDARG;

    container_name(pDecoder.Get(), info->format, sizeof(info->format));

    ComPtr<IWICBitmapFrameDecode> pFrame;
    hr = seh_get_frame(pDecoder.Get(), frame_index, &pFrame);
    if (FAILED(hr)) return (int32_t)hr;

    WICPixelFormatGUID srcFmt, dstFmt;
    hr = seh_get_pixel_format(pFrame.Get(), &srcFmt);
    if (FAILED(hr)) return (int32_t)hr;

    pixel_info pinfo;
    seh_query_pixel_format_info(g_factory.Get(), srcFmt, &pinfo);

    uint32_t comp = 0, bpcVal = 0;
    determine_target_format(
        g_factory.Get(), srcFmt, &pinfo, dstFmt, comp, bpcVal
    );

    UINT w = 0, h = 0;
    hr = seh_get_size(pFrame.Get(), &w, &h);
    if (FAILED(hr)) return (int32_t)hr;

    info->width = w;
    info->height = h;
    info->components = comp;
    info->bpc = bpcVal;
    info->bitspersample = pinfo.bitsPerSample;
    if (!WideCharToMultiByte(
        CP_UTF8, 0, pinfo.name, -1,
        info->pixelformat, (int)sizeof(info->pixelformat), nullptr, nullptr))
        info->pixelformat[0] = 0;
    return S_OK;
}

extern "C" int32_t
wic_copy_pixels(
    const uint8_t* src,
    size_t srcsize,
    uint32_t frame_index,
    uint8_t* dst,
    uint32_t dst_stride,
    size_t dst_size)
{
    if (!src || !srcsize || !dst || !dst_size)
        return (int32_t)E_INVALIDARG;
    if (dst_size > (size_t)UINT_MAX)
        return (int32_t)E_INVALIDARG;
    if (!g_factory)
        return (int32_t)CO_E_NOTINITIALIZED;

    ComPtr<IStream> pStream = make_stream(src, srcsize);
    if (!pStream) return (int32_t)E_OUTOFMEMORY;

    ComPtr<IWICBitmapDecoder> pDecoder;
    HRESULT hr = seh_create_decoder(g_factory.Get(), pStream.Get(), &pDecoder);
    if (FAILED(hr)) return (int32_t)hr;

    UINT frameCount = 0;
    hr = seh_get_frame_count(pDecoder.Get(), &frameCount);
    if (FAILED(hr)) return (int32_t)hr;
    if (frame_index >= frameCount) return (int32_t)E_INVALIDARG;

    ComPtr<IWICBitmapFrameDecode> pFrame;
    hr = seh_get_frame(pDecoder.Get(), frame_index, &pFrame);
    if (FAILED(hr)) return (int32_t)hr;

    WICPixelFormatGUID srcFmt, dstFmt;
    hr = seh_get_pixel_format(pFrame.Get(), &srcFmt);
    if (FAILED(hr)) return (int32_t)hr;

    uint32_t components = 0, bpcVal = 0;
    determine_target_format(
        g_factory.Get(),
        srcFmt,
        nullptr,
        dstFmt,
        components,
        bpcVal
    );

    /* If already in the target format, copy directly from the frame.
     * Avoids WINCODEC_ERR_WRONGSTATE from codecs (e.g. JPEG XR) that
     * reject same-format IWICFormatConverter::Initialize calls.
     */
    if (IsEqualGUID(srcFmt, dstFmt)) {
        hr = seh_copy_pixels_frame(
            pFrame.Get(),
            dst_stride,
            (UINT)dst_size,
            dst
        );
        return (int32_t)hr;
    }

    /* No fallback format: the caller sized and interprets dst for dstFmt. */
    ComPtr<IWICFormatConverter> pConv;
    hr = seh_create_format_converter(g_factory.Get(), &pConv);
    if (FAILED(hr)) return (int32_t)hr;
    hr = seh_converter_init(pConv.Get(), pFrame.Get(), dstFmt);
    if (FAILED(hr)) return (int32_t)hr;
    hr = seh_copy_pixels_conv(
        pConv.Get(),
        dst_stride,
        (UINT)dst_size,
        dst
    );
    return (int32_t)hr;
}

/* Legacy single-call API: calls wic_get_info + malloc + wic_copy_pixels. */
extern "C" int32_t
wic_decode_(
    const uint8_t* src,
    size_t srcsize,
    uint32_t frame_index,
    wic_decode_result_t* result)
{
    if (!src || !srcsize || !result)
        return (int32_t)E_INVALIDARG;

    memset(result, 0, sizeof(wic_decode_result_t));

    wic_info_t info;
    int32_t hr = wic_get_info(src, srcsize, frame_index, &info);
    result->frame_count = info.frame_count;
    if (hr != 0) return hr;

    const uint32_t width = info.width;
    const uint32_t height = info.height;
    const uint32_t components = info.components;
    const uint32_t bpcVal = info.bpc;

    const uint32_t bytes_per_pixel = components * (bpcVal / 8);
    if (bytes_per_pixel == 0 || width > UINT32_MAX / bytes_per_pixel)
        return (int32_t)E_OUTOFMEMORY;
    const uint32_t stride = width * bytes_per_pixel;
    const size_t bufSize = (size_t)stride * height;
    if (bufSize == 0 || bufSize > (size_t)0x7FFFFFFF)
        return (int32_t)E_OUTOFMEMORY;

    uint8_t* buf = (uint8_t*)malloc(bufSize);
    if (!buf) return (int32_t)E_OUTOFMEMORY;

    hr = wic_copy_pixels(src, srcsize, frame_index, buf, stride, bufSize);
    if (hr != 0) {
        free(buf); return hr;
    }

    result->data = buf;
    result->width = width;
    result->height = height;
    result->stride = stride;
    result->components = components;
    result->bpc = bpcVal;
    return 0;
}

extern "C" int32_t
wic_check_(const uint8_t* src, size_t srcsize)
{
    if (!src || !srcsize) return 0;
    if (!g_factory) return -1;

    ComPtr<IStream> pStream = make_stream(src, srcsize);
    if (!pStream) return -1;

    ComPtr<IWICBitmapDecoder> pDecoder;
    HRESULT hr = seh_create_decoder(
        g_factory.Get(),
        pStream.Get(),
        &pDecoder
    );
    return SUCCEEDED(hr) ? 1 : 0;
}

extern "C" void
wic_decode_free(uint8_t* data)
{
    free(data);
}

extern "C" int32_t
wic_encode_(
    const uint8_t* src,
    uint32_t width,
    uint32_t height,
    uint32_t components,
    uint32_t bpc,
    int32_t format,
    int32_t quality,
    uint8_t** dst,
    size_t* dstsize)
{
    if (!src || !width || !height || !dst || !dstsize)
        return (int32_t)E_INVALIDARG;
    if (!g_factory)
        return (int32_t)CO_E_NOTINITIALIZED;

    *dst = nullptr;
    *dstsize = 0;

    GUID containerGuid;
    if (!format_to_container_guid(format, containerGuid))
        return (int32_t)E_INVALIDARG;

    WICPixelFormatGUID pixelFormat;
    if (!pixel_format_guid(components, bpc, pixelFormat))
        return (int32_t)E_INVALIDARG;

    const uint32_t bytes_per_pixel = components * (bpc / 8);
    if (bytes_per_pixel == 0 || width > UINT32_MAX / bytes_per_pixel)
        return (int32_t)E_INVALIDARG;
    const uint32_t stride = width * bytes_per_pixel;
    const size_t bufSize = (size_t)stride * height;
    if (bufSize > (size_t)UINT_MAX)
        return (int32_t)E_INVALIDARG;

    /* create growable output stream */
    GrowIStream* rawStream = new(std::nothrow) GrowIStream();
    if (!rawStream) return (int32_t)E_OUTOFMEMORY;
    ComPtr<IStream> pStream;
    pStream.Attach(rawStream);

    /* create encoder */
    ComPtr<IWICBitmapEncoder> pEncoder;
    HRESULT hr = g_factory->CreateEncoder(containerGuid, nullptr, &pEncoder);
    if (FAILED(hr)) return (int32_t)hr;

    hr = pEncoder->Initialize(pStream.Get(), WICBitmapEncoderNoCache);
    if (FAILED(hr)) return (int32_t)hr;

    /* create frame */
    ComPtr<IWICBitmapFrameEncode> pFrame;
    ComPtr<IPropertyBag2> pPropBag;
    hr = pEncoder->CreateNewFrame(&pFrame, &pPropBag);
    if (FAILED(hr)) return (int32_t)hr;

    /* set quality for lossy formats */
    if (quality >= 0 && pPropBag) {
        float q = (float)quality / 100.0f;
        PROPBAG2 option = {};
        option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT varValue;
        VariantInit(&varValue);
        varValue.vt = VT_R4;
        varValue.fltVal = q;
        pPropBag->Write(1, &option, &varValue);
        /* ignore errors - not all formats support ImageQuality */
    }

    hr = pFrame->Initialize(pPropBag.Get());
    if (FAILED(hr)) return (int32_t)hr;

    hr = pFrame->SetSize(width, height);
    if (FAILED(hr)) return (int32_t)hr;

    WICPixelFormatGUID frameFormat = pixelFormat;
    hr = pFrame->SetPixelFormat(&frameFormat);
    if (FAILED(hr)) return (int32_t)hr;

    /* if the encoder changed the pixel format, we need conversion */
    if (!IsEqualGUID(frameFormat, pixelFormat)) {
        /* create a WIC bitmap from the source pixels */
        ComPtr<IWICBitmap> pBitmap;
        hr = g_factory->CreateBitmapFromMemory(
            width,
            height,
            pixelFormat,
            stride,
            (UINT)bufSize,
            const_cast<BYTE*>(src),
            &pBitmap
        );
        if (FAILED(hr)) return (int32_t)hr;

        ComPtr<IWICFormatConverter> pConv;
        hr = g_factory->CreateFormatConverter(&pConv);
        if (FAILED(hr)) return (int32_t)hr;

        /* indexed target formats (e.g. BMP gray, GIF) require a palette */
        ComPtr<IWICPalette> pPalette;
        WICBitmapPaletteType paletteType = WICBitmapPaletteTypeCustom;
        if (
            IsEqualGUID(frameFormat, GUID_WICPixelFormat8bppIndexed)
            || IsEqualGUID(frameFormat, GUID_WICPixelFormat4bppIndexed)
            || IsEqualGUID(frameFormat, GUID_WICPixelFormat2bppIndexed)
            || IsEqualGUID(frameFormat, GUID_WICPixelFormat1bppIndexed))
        {
            hr = g_factory->CreatePalette(&pPalette);
            if (FAILED(hr)) return (int32_t)hr;
            if (components == 1) {
                paletteType = WICBitmapPaletteTypeFixedGray256;
                hr = pPalette->InitializePredefined(paletteType, FALSE);
            }
            else {
                hr = pPalette->InitializeFromBitmap(
                    pBitmap.Get(), 256, components == 4 ? TRUE : FALSE
                );
            }
            if (FAILED(hr)) return (int32_t)hr;
            hr = pFrame->SetPalette(pPalette.Get());
            if (FAILED(hr)) return (int32_t)hr;
        }

        hr = pConv->Initialize(
            pBitmap.Get(),
            frameFormat,
            WICBitmapDitherTypeNone,
            pPalette.Get(),
            0.0,
            paletteType
        );
        if (FAILED(hr)) return (int32_t)hr;

        hr = pFrame->WriteSource(pConv.Get(), nullptr);
        if (FAILED(hr)) return (int32_t)hr;
    }
    else {
        hr = pFrame->WritePixels(
            height,
            stride,
            (UINT)bufSize,
            const_cast<BYTE*>(src)
        );
        if (FAILED(hr)) return (int32_t)hr;
    }

    hr = pFrame->Commit();
    if (FAILED(hr)) return (int32_t)hr;

    hr = pEncoder->Commit();
    if (FAILED(hr)) return (int32_t)hr;

    *dst = rawStream->detach(dstsize);
    if (!*dst) return (int32_t)E_OUTOFMEMORY;

    return 0;
}

extern "C" void
wic_encode_free(uint8_t* data)
{
    free(data);
}

extern "C" const char*
wic_version_string(void)
{
    /* C++11 guarantees this static is initialized exactly once,
       thread-safely.
    */
    static const char* const s = []() -> const char* {
        static char buf[32];
        using RtlGetVersionFunc = NTSTATUS(WINAPI*)(POSVERSIONINFOEXW);
        HMODULE hNtDll = GetModuleHandleW(L"ntdll.dll");
        auto pRtlGetVersion = reinterpret_cast<RtlGetVersionFunc>(
            hNtDll ? GetProcAddress(hNtDll, "RtlGetVersion") : nullptr);
        if (pRtlGetVersion) {
            OSVERSIONINFOEXW osvi = {};
            osvi.dwOSVersionInfoSize = sizeof(osvi);
            if (pRtlGetVersion(&osvi) == 0) {
                snprintf(
                    buf,
                    sizeof(buf),
                    "%lu.%lu.%lu",
                    (unsigned long)osvi.dwMajorVersion,
                    (unsigned long)osvi.dwMinorVersion,
                    (unsigned long)osvi.dwBuildNumber
                );
                return buf;
            }
        }
        snprintf(buf, sizeof(buf), "n/a");
        return buf;
    }();
    return s;
}
