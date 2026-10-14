#include "video_reader.h"

#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace {

std::string avErrorString(int code) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buf, sizeof(buf));
    return buf;
}

}  // namespace

struct VideoReader::Impl {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* sws = nullptr;
    int streamIndex = -1;
    AVRational timeBase{0, 1};

    bool flushing = false;   // demuxer exhausted, draining the decoder
    bool finished = false;

    std::vector<std::uint8_t> rgb;
    int rgbStride = 0;

    Info info;
    std::string error;

    ~Impl() {
        sws_freeContext(sws);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
    }

    bool fail(const std::string& what, int code) {
        error = what + ": " + avErrorString(code);
        finished = true;
        return false;
    }

    // Pulls one decoded frame into `frame`, feeding packets as needed.
    bool decodeNext() {
        while (true) {
            int ret = avcodec_receive_frame(codec, frame);
            if (ret == 0) return true;
            if (ret == AVERROR_EOF) {
                finished = true;
                return false;
            }
            if (ret != AVERROR(EAGAIN)) return fail("decode", ret);

            if (flushing) {
                // Decoder wants input but there is none left.
                finished = true;
                return false;
            }

            ret = av_read_frame(format, packet);
            if (ret == AVERROR_EOF) {
                flushing = true;
                avcodec_send_packet(codec, nullptr);
                continue;
            }
            if (ret < 0) return fail("read packet", ret);

            if (packet->stream_index == streamIndex) {
                ret = avcodec_send_packet(codec, packet);
                av_packet_unref(packet);
                if (ret < 0 && ret != AVERROR(EAGAIN)) return fail("send packet", ret);
            } else {
                av_packet_unref(packet);
            }
        }
    }

    bool convertToRgb() {
        sws = sws_getCachedContext(sws,
                                   frame->width, frame->height,
                                   static_cast<AVPixelFormat>(frame->format),
                                   frame->width, frame->height, AV_PIX_FMT_RGB24,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws) {
            error = "could not create pixel format converter";
            finished = true;
            return false;
        }

        rgbStride = frame->width * 3;
        rgb.resize(static_cast<std::size_t>(rgbStride) * frame->height);

        std::uint8_t* dst[4] = {rgb.data(), nullptr, nullptr, nullptr};
        int dstStride[4] = {rgbStride, 0, 0, 0};
        sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst, dstStride);
        return true;
    }
};

VideoReader::VideoReader() : impl_(std::make_unique<Impl>()) {}

VideoReader::~VideoReader() = default;

bool VideoReader::open(const std::string& path, std::string& error) {
    impl_ = std::make_unique<Impl>();
    Impl& d = *impl_;

    int ret = avformat_open_input(&d.format, path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        error = "cannot open '" + path + "': " + avErrorString(ret);
        return false;
    }
    ret = avformat_find_stream_info(d.format, nullptr);
    if (ret < 0) {
        error = "cannot read stream info: " + avErrorString(ret);
        return false;
    }

    const AVCodec* decoder = nullptr;
    d.streamIndex = av_find_best_stream(d.format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (d.streamIndex < 0 || !decoder) {
        error = "no decodable video stream in '" + path + "'";
        return false;
    }
    AVStream* stream = d.format->streams[d.streamIndex];
    d.timeBase = stream->time_base;

    d.codec = avcodec_alloc_context3(decoder);
    if (!d.codec) {
        error = "out of memory";
        return false;
    }
    avcodec_parameters_to_context(d.codec, stream->codecpar);
    d.codec->thread_count = 0;  // let FFmpeg pick based on CPU cores

    ret = avcodec_open2(d.codec, decoder, nullptr);
    if (ret < 0) {
        error = "cannot open decoder: " + avErrorString(ret);
        return false;
    }

    d.packet = av_packet_alloc();
    d.frame = av_frame_alloc();
    if (!d.packet || !d.frame) {
        error = "out of memory";
        return false;
    }

    d.info.width = d.codec->width;
    d.info.height = d.codec->height;
    d.info.codec = decoder->name;
    AVRational rate = av_guess_frame_rate(d.format, stream, nullptr);
    d.info.fps = rate.den ? av_q2d(rate) : 0.0;
    if (stream->duration != AV_NOPTS_VALUE) {
        d.info.durationSec = stream->duration * av_q2d(stream->time_base);
    } else if (d.format->duration != AV_NOPTS_VALUE) {
        d.info.durationSec = static_cast<double>(d.format->duration) / AV_TIME_BASE;
    }
    d.info.frameCountEstimate = stream->nb_frames;
    return true;
}

const VideoReader::Info& VideoReader::info() const { return impl_->info; }

const std::string& VideoReader::lastError() const { return impl_->error; }

bool VideoReader::read(vg::FrameView& out) {
    Impl& d = *impl_;
    if (d.finished || !d.codec) return false;
    if (!d.decodeNext()) return false;

    const std::int64_t pts = d.frame->best_effort_timestamp;
    if (!d.convertToRgb()) return false;

    out.data = d.rgb.data();
    out.width = d.frame->width;
    out.height = d.frame->height;
    out.stride = d.rgbStride;
    out.format = vg::FrameView::Format::RGB8;
    out.timestamp = pts == AV_NOPTS_VALUE ? 0.0 : pts * av_q2d(d.timeBase);

    av_frame_unref(d.frame);
    return true;
}
