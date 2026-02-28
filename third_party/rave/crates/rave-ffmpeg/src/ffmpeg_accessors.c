#include <libavformat/avformat.h>

AVStream *rave_avformat_stream(AVFormatContext *ctx, int index) {
    if (!ctx || index < 0 || !ctx->streams || (unsigned int)index >= ctx->nb_streams) {
        return NULL;
    }
    return ctx->streams[index];
}

int64_t rave_avformat_duration(const AVFormatContext *ctx) {
    if (!ctx) {
        return 0;
    }
    return ctx->duration;
}

const AVOutputFormat *rave_avformat_oformat(const AVFormatContext *ctx) {
    if (!ctx) {
        return NULL;
    }
    return ctx->oformat;
}

AVIOContext **rave_avformat_pb(AVFormatContext *ctx) {
    if (!ctx) {
        return NULL;
    }
    return &ctx->pb;
}
