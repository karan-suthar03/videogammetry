#include "camera_reader.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace {

constexpr DWORD kVideoStream = static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM);

template <class T>
void release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

std::string toUtf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string hrError(const char* what, HRESULT hr) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s failed (0x%08lX)", what, static_cast<unsigned long>(hr));
    return buf;
}

// COM + Media Foundation for the calling thread, undone in reverse order.
struct MfScope {
    bool com = false, mf = false;
    MfScope() {
        com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
        mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
    }
    ~MfScope() {
        if (mf) MFShutdown();
        if (com) CoUninitialize();
    }
};

}  // namespace

std::vector<CameraReader::Device> CameraReader::list() {
    std::vector<Device> out;
    MfScope scope;
    IMFAttributes* attr = nullptr;
    if (FAILED(MFCreateAttributes(&attr, 1))) return out;
    attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    if (SUCCEEDED(MFEnumDeviceSources(attr, &devices, &count))) {
        for (UINT32 i = 0; i < count; ++i) {
            wchar_t* name = nullptr;
            wchar_t* link = nullptr;
            UINT32 len = 0;
            devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &len);
            devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &link, &len);
            if (link) out.push_back({name ? toUtf8(name) : "camera", link});
            CoTaskMemFree(name);
            CoTaskMemFree(link);
            devices[i]->Release();
        }
        CoTaskMemFree(devices);
    }
    attr->Release();
    return out;
}

struct CameraReader::Impl {
    MfScope scope;  // first: outlives the MF objects below
    IMFMediaSource* source = nullptr;
    IMFSourceReader* reader = nullptr;
    int width = 0, height = 0, outWidth = 0, outHeight = 0;
    LONG defaultStride = 0;
    SwsContext* sws = nullptr;
    AVFrame* in = nullptr;
    AVFrame* out = nullptr;

    std::thread thread;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::uint8_t> back, ready, current;  // RGB8 frames
    double backTime = 0, readyTime = 0;
    long sequence = 0, taken = 0;
    bool closed = false;
    std::string error;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }
        changed.notify_all();
        if (thread.joinable()) thread.join();
        sws_free_context(&sws);
        av_frame_free(&in);
        av_frame_free(&out);
        release(reader);
        if (source) source->Shutdown();
        release(source);
    }

    // Converts one BGRX sample into `back` (RGB8, downscaled).
    bool convert(IMFSample* sample) {
        IMFMediaBuffer* buffer = nullptr;
        if (FAILED(sample->ConvertToContiguousBuffer(&buffer))) return false;
        IMF2DBuffer* buffer2d = nullptr;
        BYTE* scan0 = nullptr;
        LONG pitch = 0;
        bool locked2d = false;
        if (SUCCEEDED(buffer->QueryInterface(IID_PPV_ARGS(&buffer2d))) && SUCCEEDED(buffer2d->Lock2D(&scan0, &pitch))) {
            locked2d = true;
        } else {
            BYTE* data = nullptr;
            DWORD maxLen = 0, len = 0;
            if (FAILED(buffer->Lock(&data, &maxLen, &len))) {
                release(buffer2d);
                buffer->Release();
                return false;
            }
            pitch = defaultStride ? defaultStride : width * 4;
            scan0 = pitch < 0 ? data + static_cast<std::ptrdiff_t>(-pitch) * (height - 1) : data;
        }

        in->data[0] = scan0;
        in->linesize[0] = pitch;
        back.resize(static_cast<std::size_t>(outWidth) * outHeight * 3);
        out->data[0] = back.data();
        out->linesize[0] = outWidth * 3;
        const int ret = sws_scale_frame(sws, out, in);
        in->data[0] = nullptr;
        out->data[0] = nullptr;

        if (locked2d) {
            buffer2d->Unlock2D();
        } else {
            buffer->Unlock();
        }
        release(buffer2d);
        buffer->Release();
        return ret >= 0;
    }

    void capture() {
        MfScope threadScope;
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (closed) return;
            }
            DWORD stream = 0, flags = 0;
            LONGLONG time = 0;
            IMFSample* sample = nullptr;
            const HRESULT hr = reader->ReadSample(kVideoStream, 0, &stream, &flags, &time, &sample);
            std::string failure;
            if (FAILED(hr)) failure = hrError("camera read", hr);
            else if (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM)) failure = "camera stopped";
            if (!failure.empty()) {
                release(sample);
                std::lock_guard<std::mutex> lock(mutex);
                error = failure;
                closed = true;
                changed.notify_all();
                return;
            }
            if (!sample) continue;  // stream tick without a frame
            const bool ok = convert(sample);
            sample->Release();
            if (!ok) continue;
            backTime = time * 1e-7;
            {
                std::lock_guard<std::mutex> lock(mutex);
                std::swap(back, ready);
                readyTime = backTime;
                ++sequence;
            }
            changed.notify_all();
        }
    }
};

CameraReader::CameraReader() = default;
CameraReader::~CameraReader() = default;

bool CameraReader::open(const std::wstring& id, int longSide, std::string& error) {
    impl_ = std::make_unique<Impl>();
    Impl& d = *impl_;

    IMFAttributes* attr = nullptr;
    HRESULT hr = MFCreateAttributes(&attr, 2);
    if (SUCCEEDED(hr)) {
        attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        attr->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, id.c_str());
        hr = MFCreateDeviceSource(attr, &d.source);
        attr->Release();
    }
    if (FAILED(hr)) {
        error = hrError("open camera", hr);
        return false;
    }
    attr = nullptr;
    MFCreateAttributes(&attr, 1);
    if (attr) attr->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, TRUE);  // decode + convert to RGB
    hr = MFCreateSourceReaderFromMediaSource(d.source, attr, &d.reader);
    release(attr);
    if (FAILED(hr)) {
        error = hrError("camera reader", hr);
        return false;
    }

    // Mode: at least 25 fps first, then the smallest size covering `longSide`
    // (or the largest if none does), then the higher frame rate.
    IMFMediaType* best = nullptr;
    UINT32 bestW = 0, bestH = 0;
    double bestFps = 0;
    const auto better = [&](UINT32 w, UINT32 h, double fps) {
        if (!best) return true;
        const UINT32 l = std::max(w, h), bl = std::max(bestW, bestH);
        if ((fps >= 25) != (bestFps >= 25)) return fps >= 25;
        const bool covers = longSide <= 0 ? false : l >= static_cast<UINT32>(longSide);
        const bool bestCovers = longSide <= 0 ? false : bl >= static_cast<UINT32>(longSide);
        if (covers != bestCovers) return covers;
        if (l != bl) return covers ? l < bl : l > bl;
        return fps > bestFps;
    };
    for (DWORD i = 0;; ++i) {
        IMFMediaType* type = nullptr;
        if (FAILED(d.reader->GetNativeMediaType(kVideoStream, i, &type))) break;
        UINT32 w = 0, h = 0, num = 0, den = 0;
        MFGetAttributeSize(type, MF_MT_FRAME_SIZE, &w, &h);
        MFGetAttributeRatio(type, MF_MT_FRAME_RATE, &num, &den);
        const double fps = den ? double(num) / den : 0;
        if (w > 0 && h > 0 && better(w, h, fps)) {
            release(best);
            best = type;
            bestW = w;
            bestH = h;
            bestFps = fps;
        } else {
            type->Release();
        }
    }
    if (!best) {
        error = "the camera offers no video format";
        return false;
    }
    hr = d.reader->SetCurrentMediaType(kVideoStream, nullptr, best);
    best->Release();
    if (FAILED(hr)) {
        error = hrError("set camera format", hr);
        return false;
    }
    IMFMediaType* rgb = nullptr;
    MFCreateMediaType(&rgb);
    rgb->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    rgb->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    MFSetAttributeSize(rgb, MF_MT_FRAME_SIZE, bestW, bestH);
    hr = d.reader->SetCurrentMediaType(kVideoStream, nullptr, rgb);
    rgb->Release();
    if (FAILED(hr)) {
        error = hrError("convert camera frames to RGB", hr);
        return false;
    }
    IMFMediaType* current = nullptr;
    if (SUCCEEDED(d.reader->GetCurrentMediaType(kVideoStream, &current))) {
        UINT32 w = 0, h = 0;
        MFGetAttributeSize(current, MF_MT_FRAME_SIZE, &w, &h);
        if (w && h) {
            bestW = w;
            bestH = h;
        }
        d.defaultStride = static_cast<LONG>(MFGetAttributeUINT32(current, MF_MT_DEFAULT_STRIDE, 0));
        current->Release();
    }
    d.width = static_cast<int>(bestW);
    d.height = static_cast<int>(bestH);
    d.outWidth = d.width;
    d.outHeight = d.height;
    const int l = std::max(d.width, d.height);
    if (longSide > 0 && l > longSide) {
        const double s = double(longSide) / l;
        d.outWidth = static_cast<int>(d.width * s + 0.5);
        d.outHeight = static_cast<int>(d.height * s + 0.5);
    }

    d.sws = sws_alloc_context();
    d.in = av_frame_alloc();
    d.out = av_frame_alloc();
    if (!d.sws || !d.in || !d.out) {
        error = "out of memory";
        return false;
    }
    d.sws->flags = SWS_AREA;
    d.in->width = d.width;
    d.in->height = d.height;
    d.in->format = AV_PIX_FMT_BGR0;
    d.out->width = d.outWidth;
    d.out->height = d.outHeight;
    d.out->format = AV_PIX_FMT_RGB24;

    d.thread = std::thread([&d] { d.capture(); });
    return true;
}

bool CameraReader::read(vg::FrameView& frame) {
    Impl& d = *impl_;
    std::unique_lock<std::mutex> lock(d.mutex);
    d.changed.wait(lock, [&] { return d.closed || d.sequence > d.taken; });
    if (d.closed) return false;
    std::swap(d.current, d.ready);
    d.taken = d.sequence;
    frame.data = d.current.data();
    frame.width = d.outWidth;
    frame.height = d.outHeight;
    frame.stride = d.outWidth * 3;
    frame.format = vg::FrameView::Format::RGB8;
    frame.timestamp = d.readyTime;
    return true;
}

void CameraReader::close() {
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->closed = true;
    }
    impl_->changed.notify_all();
}

const std::string& CameraReader::lastError() const {
    static const std::string none;
    return impl_ ? impl_->error : none;
}
