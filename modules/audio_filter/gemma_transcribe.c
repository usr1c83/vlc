/*****************************************************************************
 * gemma_transcribe.c : live speech transcription/translation with Gemma 4
 *****************************************************************************
 * Copyright (C) 2026 VLC authors and VideoLAN
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 *****************************************************************************/

/*
 * This audio filter taps the decoded audio, slices it into short segments
 * and submits each segment to a Gemma 4 multimodal inference server for
 * speech transcription, optionally translating it into a target language.
 * The resulting text is published on the libvlc instance through the
 * "gemma-transcript-text" string variable, which the "transcript" sub
 * source picks up to render live subtitles on top of the video:
 *
 *   vlc --audio-filter=gemma_transcribe --sub-source=transcript \
 *       --gemma-transcribe-language=Russian input.mkv
 *
 * By default the filter speaks the OpenAI-compatible chat completions
 * protocol with base64 WAV "input_audio" content parts, as implemented by
 * llama.cpp llama-server running a Gemma 4 audio model. The Google
 * Generative Language API ("gemini" format) is also supported for hosted
 * Gemma 4 models.
 *
 * The filter can additionally voice the transcription over the original
 * audio track (voice-over translation). Gemma 4's speaker diarization is
 * then used to split the transcription into per-speaker utterances, each
 * synthesized with its own voice through an OpenAI-compatible text-to-speech
 * endpoint (/v1/audio/speech) and mixed into the audio output while the
 * original sound is ducked:
 *
 *   vlc --audio-filter=gemma_transcribe --sub-source=transcript \
 *       --gemma-transcribe-language=Russian --gemma-transcribe-voiceover \
 *       input.mkv
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <ctype.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_plugin.h>
#include <vlc_aout.h>
#include <vlc_filter.h>
#include <vlc_block.h>
#include <vlc_memstream.h>
#include <vlc_queue.h>
#include <vlc_strings.h>
#include <vlc_url.h>
#include <vlc_variables.h>

#include "../access/http/message.h"
#include "../access/http/connmgr.h"
#include "../misc/webservices/json_helper.h"

#define GEMMA_TRANSCRIPT_VAR "gemma-transcript-text"

/* Gemma 4 ingests audio as 16 kHz mono PCM and accepts clips of at most
 * 30 seconds. */
#define TRANSCRIBE_RATE 16000
#define SEGMENT_MS_MIN  1000
#define SEGMENT_MS_MAX  30000

/* Number of segments allowed to sit in the inference pipeline before the
 * filter starts skipping segments to remain (near) real-time. */
#define MAX_PENDING 3

/* Maximum number of utterances voiced per segment */
#define MAX_UTTERANCES 32

/* Maximum backlog of synthesized voice-over audio, in seconds */
#define MIX_MAX_DEPTH 60

#define CFG_PREFIX "gemma-transcribe-"

struct gemma_endpoint
{
    char *host;
    char *authority;
    char *path;
    unsigned port;
    bool secure;
};

typedef struct
{
    /* Endpoints (constant after Open) */
    struct gemma_endpoint ep;      /* Gemma 4 inference server */
    bool gemini;
    char *key;
    char *model;
    char *prompt;

    /* Voice-over synthesis (constant after Open) */
    bool voiceover;
    struct gemma_endpoint tts_ep;  /* OpenAI-compatible TTS server */
    char *tts_key;
    char *tts_model;
    char **voices;                 /* voice per diarized speaker */
    size_t voice_count;

    /* Mono 16 kHz conversion state (audio thread only) */
    unsigned channels;
    unsigned rate;
    double step;   /* input samples per output sample */
    double pos;    /* fractional read position between prev and current */
    float prev;
    int16_t *buf;
    size_t fill;
    size_t target;

    /* Inference pipeline */
    struct vlc_http_mgr *http;
    vlc_queue_t queue;
    bool dead;
    atomic_uint pending;
    vlc_thread_t thread;

    /* Voice-over mix FIFO (audio thread <-> worker thread) */
    vlc_mutex_t mix_lock;
    block_t *mix_first;            /* chain of mono FL32 blocks at 'rate' */
    block_t **mix_lastp;
    size_t mix_offset;             /* samples consumed in the head block */
    size_t mix_depth;              /* total queued samples */
    float duck;                    /* original audio gain under voice-over */
    float gain;                    /* smoothed current original audio gain */
} filter_sys_t;

/*****************************************************************************
 * Request marshalling
 *****************************************************************************/

static void JsonWriteString(struct vlc_memstream *ms, const char *str)
{
    vlc_memstream_putc(ms, '"');
    for (const unsigned char *p = (const unsigned char *)str; *p != '\0'; p++)
    {
        switch (*p)
        {
            case '"':  vlc_memstream_puts(ms, "\\\""); break;
            case '\\': vlc_memstream_puts(ms, "\\\\"); break;
            case '\n': vlc_memstream_puts(ms, "\\n");  break;
            case '\r': vlc_memstream_puts(ms, "\\r");  break;
            case '\t': vlc_memstream_puts(ms, "\\t");  break;
            default:
                if (*p < 0x20)
                    vlc_memstream_printf(ms, "\\u%04x", *p);
                else
                    vlc_memstream_putc(ms, *p);
        }
    }
    vlc_memstream_putc(ms, '"');
}

static void *BuildWav(const int16_t *samples, size_t count, size_t *sizep)
{
    size_t data_size = count * 2;
    uint8_t *buf = malloc(44 + data_size);
    if (unlikely(buf == NULL))
        return NULL;

    memcpy(buf, "RIFF", 4);
    SetDWLE(buf + 4, 36 + data_size);
    memcpy(buf + 8, "WAVEfmt ", 8);
    SetDWLE(buf + 16, 16);                     /* fmt chunk size */
    SetWLE(buf + 20, 1);                       /* PCM */
    SetWLE(buf + 22, 1);                       /* mono */
    SetDWLE(buf + 24, TRANSCRIBE_RATE);
    SetDWLE(buf + 28, TRANSCRIBE_RATE * 2);    /* byte rate */
    SetWLE(buf + 32, 2);                       /* block align */
    SetWLE(buf + 34, 16);                      /* bits per sample */
    memcpy(buf + 36, "data", 4);
    SetDWLE(buf + 40, data_size);

    for (size_t i = 0; i < count; i++)
        SetWLE(buf + 44 + 2 * i, (uint16_t)samples[i]);

    *sizep = 44 + data_size;
    return buf;
}

/**
 * Serializes the inference request for one audio segment.
 * The audio part is placed before the instruction text as recommended for
 * Gemma 4 multimodal prompts.
 */
static int BuildRequestBody(filter_sys_t *sys, const char *b64,
                            struct vlc_memstream *ms)
{
    if (vlc_memstream_open(ms))
        return -1;

    if (sys->gemini)
    {
        vlc_memstream_puts(ms, "{\"contents\":[{\"parts\":["
                               "{\"inline_data\":{\"mime_type\":\"audio/wav\","
                               "\"data\":\"");
        vlc_memstream_puts(ms, b64);
        vlc_memstream_puts(ms, "\"}},{\"text\":");
        JsonWriteString(ms, sys->prompt);
        vlc_memstream_puts(ms, "}]}],\"generationConfig\":"
                               "{\"temperature\":0,\"maxOutputTokens\":512}}");
    }
    else
    {
        vlc_memstream_puts(ms, "{\"model\":");
        JsonWriteString(ms, sys->model);
        vlc_memstream_puts(ms, ",\"temperature\":0,\"max_tokens\":512,"
                               "\"messages\":[{\"role\":\"user\",\"content\":["
                               "{\"type\":\"input_audio\",\"input_audio\":"
                               "{\"format\":\"wav\",\"data\":\"");
        vlc_memstream_puts(ms, b64);
        vlc_memstream_puts(ms, "\"}},{\"type\":\"text\",\"text\":");
        JsonWriteString(ms, sys->prompt);
        vlc_memstream_puts(ms, "}]}]}");
    }

    return vlc_memstream_close(ms);
}

/*****************************************************************************
 * HTTP transport
 *****************************************************************************/

/**
 * POSTs a JSON document (taking ownership of the heap buffer) and returns
 * the response body as a NUL-terminated heap buffer, or NULL on error.
 */
static char *HttpPost(filter_t *filter, const struct gemma_endpoint *ep,
                      bool goog_auth, const char *key,
                      char *body, size_t body_len, size_t *restrict resp_len)
{
    filter_sys_t *sys = filter->p_sys;

    block_t *payload = block_heap_Alloc(body, body_len);
    if (unlikely(payload == NULL))
        return NULL;

    struct vlc_http_msg *req =
        vlc_http_req_create("POST", ep->secure ? "https" : "http",
                            ep->authority, ep->path);
    if (unlikely(req == NULL))
    {
        block_Release(payload);
        return NULL;
    }

    vlc_http_msg_add_header(req, "Content-Type", "application/json");
    vlc_http_msg_add_header(req, "Expect", "100-continue");
    if (key != NULL)
    {
        if (goog_auth)
            vlc_http_msg_add_header(req, "x-goog-api-key", "%s", key);
        else
            vlc_http_msg_add_header(req, "Authorization", "Bearer %s", key);
    }
    vlc_http_msg_add_agent(req, PACKAGE_NAME "/" PACKAGE_VERSION);

    struct vlc_http_msg *resp =
        vlc_http_mgr_request(sys->http, ep->secure, ep->host, ep->port,
                             req, false, true);
    vlc_http_msg_destroy(req);
    if (resp == NULL)
    {
        block_Release(payload);
        msg_Warn(filter, "cannot reach server %s", ep->authority);
        return NULL;
    }

    int status = vlc_http_msg_get_status(resp);
    if (status >= 100 && status < 200)
    {
        if (vlc_http_msg_write(resp, payload, true) < 0)
        {
            msg_Warn(filter, "cannot send request payload");
            vlc_http_msg_destroy(resp);
            return NULL;
        }

        resp = vlc_http_msg_get_final(resp);
        if (resp == NULL)
        {
            msg_Warn(filter, "no response from server %s", ep->authority);
            return NULL;
        }
        status = vlc_http_msg_get_status(resp);
    }
    else /* the server answered without waiting for the payload */
        block_Release(payload);

    struct vlc_memstream ms;
    if (vlc_memstream_open(&ms))
    {
        vlc_http_msg_destroy(resp);
        return NULL;
    }

    block_t *b;
    while ((b = vlc_http_msg_read(resp)) != NULL)
    {
        if (ms.length < (16u << 20)) /* sanity cap on the answer size */
            vlc_memstream_write(&ms, b->p_buffer, b->i_buffer);
        block_Release(b);
    }
    vlc_http_msg_destroy(resp);

    if (vlc_memstream_close(&ms))
        return NULL;

    if (status < 200 || status >= 300)
    {
        msg_Warn(filter, "request to %s failed (HTTP %d): %.200s",
                 ep->authority, status, ms.ptr);
        free(ms.ptr);
        return NULL;
    }

    *resp_len = ms.length;
    return ms.ptr;
}

/*****************************************************************************
 * Response parsing
 *****************************************************************************/

static char *TrimDup(const char *str)
{
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
        str++;

    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == ' ' || str[len - 1] == '\t'
                    || str[len - 1] == '\n' || str[len - 1] == '\r'))
        len--;

    return strndup(str, len);
}

static char *ExtractText(filter_t *filter, const char *doc, size_t len)
{
    filter_sys_t *sys = filter->p_sys;
    struct json_helper_sys jsys = {
        .logger = vlc_object_logger(filter),
        .buffer = doc,
        .size = len,
    };
    struct json_object json;

    if (json_parse(&jsys, &json))
    {
        msg_Warn(filter, "invalid JSON response from inference server");
        return NULL;
    }

    const char *text = NULL;

    if (sys->gemini)
    {
        const struct json_array *cands = json_get_array(&json, "candidates");
        if (cands != NULL && cands->size > 0
         && cands->entries[0].type == JSON_OBJECT)
        {
            const struct json_object *content =
                json_get_object(&cands->entries[0].object, "content");
            const struct json_array *parts = (content != NULL)
                ? json_get_array(content, "parts") : NULL;
            if (parts != NULL && parts->size > 0
             && parts->entries[0].type == JSON_OBJECT)
                text = json_get_str(&parts->entries[0].object, "text");
        }
    }
    else
    {
        const struct json_array *choices = json_get_array(&json, "choices");
        if (choices != NULL && choices->size > 0
         && choices->entries[0].type == JSON_OBJECT)
        {
            const struct json_object *message =
                json_get_object(&choices->entries[0].object, "message");
            if (message != NULL)
                text = json_get_str(message, "content");
        }
    }

    char *ret;
    if (text != NULL)
        ret = TrimDup(text);
    else
    {
        const struct json_object *error = json_get_object(&json, "error");
        const char *errmsg = (error != NULL)
            ? json_get_str(error, "message") : NULL;
        msg_Warn(filter, "no transcription in response%s%s",
                 (errmsg != NULL) ? ": " : "",
                 (errmsg != NULL) ? errmsg : "");
        ret = NULL;
    }

    json_free(&json);
    return ret;
}

/*****************************************************************************
 * Voice-over synthesis
 *****************************************************************************/

/**
 * Locates the PCM payload of a 16-bit little-endian WAV file.
 */
static int WavParse(const uint8_t *buf, size_t size,
                    unsigned *restrict ratep, unsigned *restrict channelsp,
                    const uint8_t **restrict datap, size_t *restrict framesp)
{
    if (size < 12 || memcmp(buf, "RIFF", 4) != 0
     || memcmp(buf + 8, "WAVE", 4) != 0)
        return -1;

    unsigned rate = 0, channels = 0;
    bool fmt_ok = false;

    for (size_t off = 12; off + 8 <= size;)
    {
        size_t csize = GetDWLE(buf + off + 4);
        const uint8_t *chunk = buf + off + 8;

        if (csize > size - off - 8) /* tolerate a truncated last chunk */
            csize = size - off - 8;

        if (memcmp(buf + off, "fmt ", 4) == 0 && csize >= 16)
        {
            channels = GetWLE(chunk + 2);
            rate = GetDWLE(chunk + 4);
            fmt_ok = GetWLE(chunk) == 1 /* PCM */
                  && GetWLE(chunk + 14) == 16 /* bits */
                  && channels > 0 && rate > 0;
        }
        else if (memcmp(buf + off, "data", 4) == 0)
        {
            if (!fmt_ok)
                return -1;
            *ratep = rate;
            *channelsp = channels;
            *datap = chunk;
            *framesp = csize / (2 * channels);
            return 0;
        }

        off += 8 + csize + (csize & 1);
    }
    return -1;
}

static float TtsMonoSample(const uint8_t *data, size_t frame,
                           unsigned channels)
{
    float acc = 0.f;
    for (unsigned c = 0; c < channels; c++)
        acc += (int16_t)GetWLE(data + (frame * channels + c) * 2);
    return acc / (channels * 32768.f);
}

/**
 * Converts synthesized PCM to a mono FL32 block at the stream sample rate.
 */
static block_t *TtsResample(const uint8_t *data, size_t frames,
                            unsigned in_rate, unsigned channels,
                            unsigned out_rate)
{
    if (frames < 2)
        return NULL;

    size_t out_frames = (uint64_t)frames * out_rate / in_rate;
    if (out_frames == 0)
        return NULL;

    block_t *b = block_Alloc(out_frames * sizeof (float));
    if (unlikely(b == NULL))
        return NULL;
    b->i_nb_samples = out_frames;

    float *out = (float *)b->p_buffer;
    const double step = (double)in_rate / out_rate;
    double pos = 0.;

    for (size_t i = 0; i < out_frames; i++, pos += step)
    {
        size_t idx = (size_t)pos;
        if (idx > frames - 2)
            idx = frames - 2;

        float frac = pos - idx;
        float s0 = TtsMonoSample(data, idx, channels);
        float s1 = TtsMonoSample(data, idx + 1, channels);
        out[i] = s0 + (s1 - s0) * frac;
    }
    return b;
}

static void MixEnqueue(filter_t *filter, block_t *b)
{
    filter_sys_t *sys = filter->p_sys;

    vlc_mutex_lock(&sys->mix_lock);
    if (sys->mix_depth + b->i_nb_samples
            > (size_t)sys->rate * MIX_MAX_DEPTH)
    {
        vlc_mutex_unlock(&sys->mix_lock);
        msg_Warn(filter, "voice-over backlog full, dropping utterance");
        block_Release(b);
        return;
    }
    *sys->mix_lastp = b;
    sys->mix_lastp = &b->p_next;
    sys->mix_depth += b->i_nb_samples;
    vlc_mutex_unlock(&sys->mix_lock);
}

/**
 * Synthesizes one utterance through the OpenAI-compatible speech endpoint
 * and queues the audio for mixing. Speakers are mapped round-robin onto the
 * configured voice list.
 */
static void TtsSpeak(filter_t *filter, unsigned speaker, const char *text)
{
    filter_sys_t *sys = filter->p_sys;
    const char *voice = sys->voices[(speaker - 1) % sys->voice_count];

    struct vlc_memstream ms;
    if (vlc_memstream_open(&ms))
        return;
    vlc_memstream_puts(&ms, "{\"model\":");
    JsonWriteString(&ms, sys->tts_model);
    vlc_memstream_puts(&ms, ",\"voice\":");
    JsonWriteString(&ms, voice);
    vlc_memstream_puts(&ms, ",\"response_format\":\"wav\",\"input\":");
    JsonWriteString(&ms, text);
    vlc_memstream_putc(&ms, '}');
    if (vlc_memstream_close(&ms))
        return;

    size_t resp_len;
    char *resp = HttpPost(filter, &sys->tts_ep, false, sys->tts_key,
                          ms.ptr, ms.length, &resp_len);
    if (resp == NULL)
        return;

    unsigned rate, channels;
    const uint8_t *data;
    size_t frames;

    if (WavParse((const uint8_t *)resp, resp_len, &rate, &channels,
                 &data, &frames) == 0)
    {
        block_t *b = TtsResample(data, frames, rate, channels, sys->rate);
        if (b != NULL)
            MixEnqueue(filter, b);
    }
    else
        msg_Warn(filter, "unsupported TTS response (16-bit PCM WAV needed)");

    free(resp);
}

/**
 * Extracts the speaker number from a diarized transcription line of the
 * form "S1: text" / "Speaker 2: text". Returns the utterance text.
 */
static const char *ParseSpeakerLine(const char *line, unsigned *speaker)
{
    const char *p = line;
    while (*p == ' ' || *p == '\t' || *p == '-')
        p++;

    if (tolower((unsigned char)p[0]) == 's')
    {
        const char *r = p + 1;
        static const char word[] = "peaker";
        size_t w = 0;
        while (word[w] != '\0' && tolower((unsigned char)r[w]) == word[w])
            w++;
        if (w == sizeof (word) - 1)
            r += w;
        while (*r == ' ')
            r++;
        if (*r >= '0' && *r <= '9')
        {
            unsigned n = 0;
            while (*r >= '0' && *r <= '9')
                n = n * 10 + (*r++ - '0');
            while (*r == ' ')
                r++;
            if (*r == ':' || *r == '.' || *r == ')')
            {
                r++;
                while (*r == ' ' || *r == '\t')
                    r++;
                *speaker = (n > 0) ? n : 1;
                return r;
            }
        }
    }

    *speaker = 1;
    return p;
}

/**
 * Splits a diarized transcription into utterances, publishes the subtitle
 * text and synthesizes the voice-over.
 */
static void HandleUtterances(filter_t *filter, char *text)
{
    struct
    {
        unsigned speaker;
        const char *text;
    } utt[MAX_UTTERANCES];
    size_t count = 0;

    for (char *line = text, *next; line != NULL && count < MAX_UTTERANCES;
         line = next)
    {
        next = strchr(line, '\n');
        if (next != NULL)
            *(next++) = '\0';

        unsigned speaker;
        const char *utt_text = ParseSpeakerLine(line, &speaker);
        if (utt_text[0] == '\0')
            continue;

        utt[count].speaker = speaker;
        utt[count].text = utt_text;
        count++;
    }

    /* Publish the subtitle text, one dash-prefixed line per speaker turn */
    struct vlc_memstream ms;
    if (vlc_memstream_open(&ms) == 0)
    {
        for (size_t i = 0; i < count; i++)
        {
            if (count > 1)
                vlc_memstream_puts(&ms, (i > 0) ? "\n- " : "- ");
            vlc_memstream_puts(&ms, utt[i].text);
        }
        if (vlc_memstream_close(&ms) == 0)
        {
            var_SetString(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR,
                          ms.ptr);
            free(ms.ptr);
        }
    }

    for (size_t i = 0; i < count; i++)
        TtsSpeak(filter, utt[i].speaker, utt[i].text);
}

/*****************************************************************************
 * Worker thread
 *****************************************************************************/

static char *Transcribe(filter_t *filter, const block_t *segment)
{
    filter_sys_t *sys = filter->p_sys;
    size_t wav_size;
    void *wav = BuildWav((const int16_t *)segment->p_buffer,
                         segment->i_nb_samples, &wav_size);
    if (unlikely(wav == NULL))
        return NULL;

    char *b64 = vlc_b64_encode_binary(wav, wav_size);
    free(wav);
    if (unlikely(b64 == NULL))
        return NULL;

    struct vlc_memstream ms;
    int err = BuildRequestBody(sys, b64, &ms);
    free(b64);
    if (unlikely(err))
        return NULL;

    size_t resp_len;
    char *resp = HttpPost(filter, &sys->ep, sys->gemini, sys->key,
                          ms.ptr, ms.length, &resp_len);
    if (resp == NULL)
        return NULL;

    char *text = ExtractText(filter, resp, resp_len);
    free(resp);
    return text;
}

static void *Worker(void *data)
{
    filter_t *filter = data;
    filter_sys_t *sys = filter->p_sys;
    block_t *segment;

    vlc_thread_set_name("vlc-gemma");

    while ((segment = vlc_queue_DequeueKillable(&sys->queue,
                                                &sys->dead)) != NULL)
    {
        char *text = Transcribe(filter, segment);
        block_Release(segment);

        if (text != NULL)
        {
            if (sys->voiceover)
                HandleUtterances(filter, text);
            else
                /* An empty answer means silence: clear the subtitle. */
                var_SetString(vlc_object_instance(filter),
                              GEMMA_TRANSCRIPT_VAR, text);
            free(text);
        }
        atomic_fetch_sub(&sys->pending, 1);
    }
    return NULL;
}

/*****************************************************************************
 * Audio path
 *****************************************************************************/

static void QueueSegment(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    if (sys->fill == 0)
        return;

    if (atomic_load(&sys->pending) >= MAX_PENDING)
    {   /* Inference is slower than real-time: skip this segment. */
        msg_Warn(filter, "inference server too slow, skipping segment");
        sys->fill = 0;
        return;
    }

    block_t *b = block_Alloc(sys->fill * sizeof (int16_t));
    if (unlikely(b == NULL))
    {
        sys->fill = 0;
        return;
    }
    memcpy(b->p_buffer, sys->buf, sys->fill * sizeof (int16_t));
    b->i_nb_samples = sys->fill;
    sys->fill = 0;

    atomic_fetch_add(&sys->pending, 1);
    vlc_queue_Enqueue(&sys->queue, b);
}

/**
 * Overlays pending synthesized speech onto the outgoing audio, smoothly
 * ducking the original program under the voice-over.
 */
static void MixVoiceover(filter_t *filter, block_t *b)
{
    filter_sys_t *sys = filter->p_sys;
    float *pcm = (float *)b->p_buffer;
    const unsigned channels = sys->channels;

    vlc_mutex_lock(&sys->mix_lock);
    if (sys->mix_first == NULL && sys->gain > 0.999f)
    {
        sys->gain = 1.f;
        vlc_mutex_unlock(&sys->mix_lock);
        return;
    }

    for (size_t i = 0; i < b->i_nb_samples; i++)
    {
        float tts = 0.f;
        float gain_target = 1.f;
        block_t *m = sys->mix_first;

        if (m != NULL)
        {
            tts = ((const float *)m->p_buffer)[sys->mix_offset];
            gain_target = sys->duck;
            sys->mix_depth--;
            if (++sys->mix_offset >= m->i_nb_samples)
            {
                sys->mix_first = m->p_next;
                if (sys->mix_first == NULL)
                    sys->mix_lastp = &sys->mix_first;
                block_Release(m);
                sys->mix_offset = 0;
            }
        }

        /* ~40 ms gain slew at 48 kHz to avoid ducking clicks */
        sys->gain += (gain_target - sys->gain) * 0.0005f;

        for (unsigned c = 0; c < channels; c++)
        {
            float s = pcm[i * channels + c] * sys->gain + tts;
            if (s > 1.f)
                s = 1.f;
            else if (s < -1.f)
                s = -1.f;
            pcm[i * channels + c] = s;
        }
    }
    vlc_mutex_unlock(&sys->mix_lock);
}

static block_t *Process(filter_t *filter, block_t *in)
{
    filter_sys_t *sys = filter->p_sys;
    const float *pcm = (const float *)in->p_buffer;
    const unsigned channels = sys->channels;

    for (size_t i = 0; i < in->i_nb_samples; i++)
    {
        float acc = 0.f;
        for (unsigned c = 0; c < channels; c++)
            acc += pcm[i * channels + c];

        const float cur = acc / channels;

        /* Linear resampling to 16 kHz mono */
        while (sys->pos < 1.0)
        {
            float v = sys->prev + (cur - sys->prev) * (float)sys->pos;
            long s = lrintf(v * 32767.f);
            if (s > INT16_MAX)
                s = INT16_MAX;
            else if (s < INT16_MIN)
                s = INT16_MIN;

            sys->buf[sys->fill++] = s;
            if (sys->fill == sys->target)
                QueueSegment(filter);

            sys->pos += sys->step;
        }
        sys->pos -= 1.0;
        sys->prev = cur;
    }

    if (sys->voiceover)
        MixVoiceover(filter, in);

    return in;
}

static void Flush(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    sys->fill = 0;
    sys->pos = 0.;
    sys->prev = 0.f;

    if (sys->voiceover)
    {   /* Drop the now stale voice-over audio */
        vlc_mutex_lock(&sys->mix_lock);
        block_ChainRelease(sys->mix_first);
        sys->mix_first = NULL;
        sys->mix_lastp = &sys->mix_first;
        sys->mix_offset = 0;
        sys->mix_depth = 0;
        sys->gain = 1.f;
        vlc_mutex_unlock(&sys->mix_lock);
    }
}

/*****************************************************************************
 * Open/Close
 *****************************************************************************/

static void EndpointClean(struct gemma_endpoint *ep)
{
    free(ep->path);
    free(ep->authority);
    free(ep->host);
}

static int EndpointParse(filter_t *filter, struct gemma_endpoint *ep,
                         const char *option)
{
    char *url = var_InheritString(filter, option);
    if (url == NULL)
    {
        msg_Err(filter, "missing %s", option);
        return VLC_EGENERIC;
    }

    vlc_url_t u;
    int err = vlc_UrlParse(&u, url);
    free(url);
    if (err || u.psz_protocol == NULL || u.psz_host == NULL)
    {
        msg_Err(filter, "invalid %s", option);
        goto error;
    }

    if (!vlc_ascii_strcasecmp(u.psz_protocol, "https"))
        ep->secure = true;
    else if (!vlc_ascii_strcasecmp(u.psz_protocol, "http"))
        ep->secure = false;
    else
    {
        msg_Err(filter, "unsupported protocol \"%s\"", u.psz_protocol);
        goto error;
    }

    ep->host = strdup(u.psz_host);
    ep->port = u.i_port;
    ep->authority = vlc_http_authority(u.psz_host, u.i_port);

    const char *path = (u.psz_path != NULL) ? u.psz_path : "/";
    if (u.psz_option != NULL)
    {
        if (asprintf(&ep->path, "%s?%s", path, u.psz_option) < 0)
            ep->path = NULL;
    }
    else
        ep->path = strdup(path);

    if (ep->host == NULL || ep->authority == NULL || ep->path == NULL)
        goto error;

    vlc_UrlClean(&u);
    return VLC_SUCCESS;

error:
    vlc_UrlClean(&u);
    EndpointClean(ep);
    ep->host = ep->authority = ep->path = NULL;
    return VLC_EGENERIC;
}

static char *BuildPrompt(filter_t *filter, bool diarize)
{
    char *custom = var_InheritString(filter, CFG_PREFIX "prompt");
    if (custom != NULL)
        return custom;

    char *language = var_InheritString(filter, CFG_PREFIX "language");
    char *task;
    int ret;

    if (language != NULL)
        ret = asprintf(&task, "translate the speech into %s", language);
    else
        ret = asprintf(&task, "transcribe the speech verbatim");
    free(language);
    if (ret < 0)
        return NULL;

    char *prompt;
    if (diarize)
        ret = asprintf(&prompt, "You are a live voice-over interpreter. "
                       "Listen to this audio clip and %s. Distinguish the "
                       "speakers. Reply with one line per utterance, each "
                       "formatted exactly as 'S<speaker number>: <text>' "
                       "(for example 'S1: Hello.'). Number the speakers "
                       "consistently within the clip. Do not add anything "
                       "else. If there is no speech, reply with an empty "
                       "message.", task);
    else
        ret = asprintf(&prompt, "You are a live subtitler. Listen to this "
                       "audio clip and %s. Reply with the resulting text "
                       "only, without timestamps, speaker labels or "
                       "commentary. If there is no speech, reply with an "
                       "empty message.", task);
    free(task);
    return (ret >= 0) ? prompt : NULL;
}

static int ParseVoices(filter_sys_t *sys, char *list)
{
    size_t count = 0;
    char **voices = NULL;

    for (char *item = list, *next; item != NULL; item = next)
    {
        next = strchr(item, ',');
        if (next != NULL)
            *(next++) = '\0';

        char *name = TrimDup(item);
        if (name == NULL)
            break;
        if (name[0] == '\0')
        {
            free(name);
            continue;
        }

        char **grown = realloc(voices, (count + 1) * sizeof (*voices));
        if (grown == NULL)
        {
            free(name);
            break;
        }
        voices = grown;
        voices[count++] = name;
    }

    if (count == 0)
    {
        free(voices);
        return VLC_EGENERIC;
    }
    sys->voices = voices;
    sys->voice_count = count;
    return VLC_SUCCESS;
}

static void Close(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    vlc_queue_Kill(&sys->queue, &sys->dead);
    vlc_join(sys->thread, NULL);
    block_ChainRelease(vlc_queue_DequeueAll(&sys->queue));
    block_ChainRelease(sys->mix_first);

    vlc_http_mgr_destroy(sys->http);
    var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);

    for (size_t i = 0; i < sys->voice_count; i++)
        free(sys->voices[i]);
    free(sys->voices);
    free(sys->tts_model);
    free(sys->tts_key);
    EndpointClean(&sys->tts_ep);

    free(sys->buf);
    free(sys->prompt);
    free(sys->model);
    free(sys->key);
    EndpointClean(&sys->ep);
    free(sys);
}

static int Open(vlc_object_t *obj)
{
    filter_t *filter = (filter_t *)obj;

    static const char *const options[] = {
        "url", "api", "key", "model", "language", "segment-ms", "prompt",
        "voiceover", "tts-url", "tts-key", "tts-model", "voices", "duck",
        NULL
    };
    config_ChainParse(filter, CFG_PREFIX, options, filter->p_cfg);

    filter_sys_t *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
        return VLC_ENOMEM;
    filter->p_sys = sys;

    int ret = EndpointParse(filter, &sys->ep, CFG_PREFIX "url");
    if (ret != VLC_SUCCESS)
    {
        free(sys);
        return ret;
    }

    char *api = var_InheritString(filter, CFG_PREFIX "api");
    sys->gemini = api != NULL && !strcmp(api, "gemini");
    free(api);

    sys->voiceover = var_InheritBool(filter, CFG_PREFIX "voiceover");

    ret = VLC_ENOMEM;
    sys->key = var_InheritString(filter, CFG_PREFIX "key");
    sys->model = var_InheritString(filter, CFG_PREFIX "model");
    sys->prompt = BuildPrompt(filter, sys->voiceover);
    if (sys->model == NULL || sys->prompt == NULL)
        goto error;

    if (sys->voiceover)
    {
        ret = EndpointParse(filter, &sys->tts_ep, CFG_PREFIX "tts-url");
        if (ret != VLC_SUCCESS)
            goto error;

        ret = VLC_ENOMEM;
        sys->tts_key = var_InheritString(filter, CFG_PREFIX "tts-key");
        sys->tts_model = var_InheritString(filter, CFG_PREFIX "tts-model");
        char *voices = var_InheritString(filter, CFG_PREFIX "voices");
        if (sys->tts_model == NULL || voices == NULL)
            goto error;

        int verr = ParseVoices(sys, voices);
        free(voices);
        if (verr != VLC_SUCCESS)
        {
            msg_Err(filter, "no usable voice-over voices");
            ret = VLC_EGENERIC;
            goto error;
        }

        float duck = var_InheritFloat(filter, CFG_PREFIX "duck");
        sys->duck = VLC_CLIP(duck, 0.f, 1.f);
        sys->gain = 1.f;
        vlc_mutex_init(&sys->mix_lock);
        sys->mix_lastp = &sys->mix_first;
    }

    int64_t segment_ms = var_InheritInteger(filter, CFG_PREFIX "segment-ms");
    segment_ms = VLC_CLIP(segment_ms, SEGMENT_MS_MIN, SEGMENT_MS_MAX);
    sys->target = TRANSCRIBE_RATE * segment_ms / 1000;
    sys->buf = vlc_alloc(sys->target, sizeof (int16_t));
    if (unlikely(sys->buf == NULL))
    {
        ret = VLC_ENOMEM;
        goto error;
    }

    filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    aout_FormatPrepare(&filter->fmt_in.audio);
    filter->fmt_out.audio = filter->fmt_in.audio;

    sys->channels = aout_FormatNbChannels(&filter->fmt_in.audio);
    sys->rate = filter->fmt_in.audio.i_rate;
    sys->step = (double)sys->rate / TRANSCRIBE_RATE;
    if (sys->channels == 0 || sys->rate == 0)
    {
        msg_Err(filter, "invalid audio format");
        ret = VLC_EGENERIC;
        goto error;
    }

    sys->http = vlc_http_mgr_create(VLC_OBJECT(filter), NULL);
    if (sys->http == NULL)
    {
        ret = VLC_EGENERIC;
        goto error;
    }

    vlc_queue_Init(&sys->queue, offsetof (block_t, p_next));
    atomic_init(&sys->pending, 0);
    sys->dead = false;

    var_Create(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR,
               VLC_VAR_STRING);

    if (vlc_clone(&sys->thread, Worker, filter))
    {
        var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);
        vlc_http_mgr_destroy(sys->http);
        ret = VLC_EGENERIC;
        goto error;
    }

    static const struct vlc_filter_operations filter_ops =
    {
        .filter_audio = Process, .flush = Flush, .close = Close,
    };
    filter->ops = &filter_ops;

    msg_Dbg(filter, "transcribing via %s (%s API, model %s, %"PRId64" ms "
            "segments%s)", sys->ep.authority,
            sys->gemini ? "Gemini" : "OpenAI", sys->model, segment_ms,
            sys->voiceover ? ", with voice-over" : "");
    return VLC_SUCCESS;

error:
    for (size_t i = 0; i < sys->voice_count; i++)
        free(sys->voices[i]);
    free(sys->voices);
    free(sys->tts_model);
    free(sys->tts_key);
    EndpointClean(&sys->tts_ep);
    free(sys->buf);
    free(sys->prompt);
    free(sys->model);
    free(sys->key);
    EndpointClean(&sys->ep);
    free(sys);
    return ret;
}

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/

#define URL_TEXT N_("Inference server URL")
#define URL_LONGTEXT N_("HTTP(S) URL of the Gemma 4 inference endpoint. " \
    "For a local llama.cpp server this is typically " \
    "\"http://127.0.0.1:8080/v1/chat/completions\". For the Google " \
    "Generative Language API, use something like " \
    "\"https://generativelanguage.googleapis.com/v1beta/models/" \
    "gemma-4-12b-it:generateContent\" together with the \"gemini\" " \
    "API format.")

#define API_TEXT N_("API format")
#define API_LONGTEXT N_("Protocol spoken by the inference server: " \
    "\"openai\" for OpenAI-compatible chat completions servers " \
    "(llama.cpp llama-server, vLLM, LM Studio…), \"gemini\" for the " \
    "Google Generative Language API.")

#define KEY_TEXT N_("API key")
#define KEY_LONGTEXT N_("Credential sent to the inference server, if it " \
    "requires one. Sent as a Bearer token (OpenAI format) or as the " \
    "x-goog-api-key header (Gemini format).")

#define MODEL_TEXT N_("Model name")
#define MODEL_LONGTEXT N_("Name of the Gemma 4 audio-capable model to " \
    "request from the server. Audio input requires the 12B, E2B or E4B " \
    "variants.")

#define LANGUAGE_TEXT N_("Target language")
#define LANGUAGE_LONGTEXT N_("Language the speech is translated into, " \
    "e.g. \"English\", \"Russian\" or \"fr\". When left empty, the speech " \
    "is transcribed in its original language.")

#define SEGMENT_TEXT N_("Segment length (ms)")
#define SEGMENT_LONGTEXT N_("Duration of the audio slices submitted for " \
    "transcription, in milliseconds. Shorter segments lower the subtitle " \
    "latency, longer segments give the model more context. Gemma 4 " \
    "accepts at most 30000 ms of audio per request.")

#define PROMPT_TEXT N_("Custom instruction")
#define PROMPT_LONGTEXT N_("Overrides the instruction sent to the model " \
    "along with each audio segment. When set, the target language option " \
    "is ignored, and voice-over diarization requires the model to answer " \
    "with 'S<n>: <text>' lines.")

#define VOICEOVER_TEXT N_("Voice-over synthesis")
#define VOICEOVER_LONGTEXT N_("Speaks the transcription over the original " \
    "audio (voice-over translation). The transcription is split by " \
    "speaker using Gemma 4 diarization, each speaker is synthesized with " \
    "its own voice through the text-to-speech server, and the original " \
    "sound is ducked while the voice-over plays.")

#define TTS_URL_TEXT N_("Text-to-speech server URL")
#define TTS_URL_LONGTEXT N_("HTTP(S) URL of an OpenAI-compatible speech " \
    "synthesis endpoint (/v1/audio/speech), such as a local " \
    "Kokoro-FastAPI, OpenedAI-Speech or LocalAI server. The server must " \
    "support WAV output.")

#define TTS_KEY_TEXT N_("Text-to-speech API key")
#define TTS_KEY_LONGTEXT N_("Credential sent to the text-to-speech " \
    "server as a Bearer token, if it requires one.")

#define TTS_MODEL_TEXT N_("Text-to-speech model")
#define TTS_MODEL_LONGTEXT N_("Model name requested from the " \
    "text-to-speech server.")

#define VOICES_TEXT N_("Voice-over voices")
#define VOICES_LONGTEXT N_("Comma-separated list of text-to-speech voice " \
    "names. Diarized speakers are assigned these voices in order (speaker " \
    "1 gets the first voice, speaker 2 the second one, and so on).")

#define DUCK_TEXT N_("Original audio level under voice-over")
#define DUCK_LONGTEXT N_("Gain applied to the original audio while the " \
    "voice-over is speaking, between 0.0 (mute the program) and 1.0 " \
    "(no ducking).")

#define TRANSCRIBE_HELP N_("Transcribe or translate speech into live " \
    "subtitles and voice-over with a Gemma 4 model (use with the " \
    "\"transcript\" sub source)")

static const char *const api_values[] = { "openai", "gemini" };
static const char *const api_texts[] = {
    N_("OpenAI-compatible (llama.cpp, vLLM…)"),
    N_("Google Generative Language API"),
};

vlc_module_begin ()
    set_shortname( N_("Gemma transcribe") )
    set_description( N_("Gemma 4 live speech transcription") )
    set_help( TRANSCRIBE_HELP )
    set_capability( "audio filter", 0 )
    set_subcategory( SUBCAT_AUDIO_AFILTER )

    add_string( CFG_PREFIX "url", "http://127.0.0.1:8080/v1/chat/completions",
                URL_TEXT, URL_LONGTEXT )
    add_string( CFG_PREFIX "api", "openai", API_TEXT, API_LONGTEXT )
        change_string_list( api_values, api_texts )
    add_password( CFG_PREFIX "key", NULL, KEY_TEXT, KEY_LONGTEXT )
    add_string( CFG_PREFIX "model", "gemma-4-12b-it", MODEL_TEXT,
                MODEL_LONGTEXT )
    add_string( CFG_PREFIX "language", NULL, LANGUAGE_TEXT,
                LANGUAGE_LONGTEXT )
    add_integer_with_range( CFG_PREFIX "segment-ms", 5000,
                            SEGMENT_MS_MIN, SEGMENT_MS_MAX,
                            SEGMENT_TEXT, SEGMENT_LONGTEXT )
    add_string( CFG_PREFIX "prompt", NULL, PROMPT_TEXT, PROMPT_LONGTEXT )

    set_section( N_("Voice-over"), NULL )
    add_bool( CFG_PREFIX "voiceover", false, VOICEOVER_TEXT,
              VOICEOVER_LONGTEXT )
    add_string( CFG_PREFIX "tts-url",
                "http://127.0.0.1:8880/v1/audio/speech",
                TTS_URL_TEXT, TTS_URL_LONGTEXT )
    add_password( CFG_PREFIX "tts-key", NULL, TTS_KEY_TEXT, TTS_KEY_LONGTEXT )
    add_string( CFG_PREFIX "tts-model", "tts-1", TTS_MODEL_TEXT,
                TTS_MODEL_LONGTEXT )
    add_string( CFG_PREFIX "voices", "alloy,onyx,nova,echo",
                VOICES_TEXT, VOICES_LONGTEXT )
    add_float_with_range( CFG_PREFIX "duck", 0.25, 0., 1.,
                          DUCK_TEXT, DUCK_LONGTEXT )

    add_shortcut( "gemma_transcribe", "gemma" )
    set_callback( Open )
vlc_module_end ()
