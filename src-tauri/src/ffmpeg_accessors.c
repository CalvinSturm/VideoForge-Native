#include <libavformat/avformat.h>

AVStream *videoforge_avformat_stream(AVFormatContext *ctx, int index) {
    if (!ctx || index < 0 || !ctx->streams || (unsigned int)index >= ctx->nb_streams) {
        return NULL;
    }
    return ctx->streams[index];
}

const AVOutputFormat *videoforge_avformat_oformat(const AVFormatContext *ctx) {
    if (!ctx) {
        return NULL;
    }
    return ctx->oformat;
}

AVIOContext **videoforge_avformat_pb(AVFormatContext *ctx) {
    if (!ctx) {
        return NULL;
    }
    return &ctx->pb;
}
