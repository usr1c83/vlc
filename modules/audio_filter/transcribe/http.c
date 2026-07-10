/*****************************************************************************
 * http.c : HTTP inference backends for the transcription filter
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
 * Speech-to-text over HTTP: OpenAI-compatible chat completions with base64
 * WAV "input_audio" content parts (llama.cpp llama-server, vLLM…) and the
 * Google Generative Language API. Text-to-speech over HTTP: OpenAI
 * compatible /v1/audio/speech endpoints returning PCM WAV.
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdint.h>

#include <vlc_common.h>
#include <vlc_block.h>
#include <vlc_configuration.h>
#include <vlc_filter.h>
#include <vlc_memstream.h>
#include <vlc_strings.h>
#include <vlc_url.h>
#include <vlc_variables.h>

#include "../../access/http/message.h"
#include "../../access/http/connmgr.h"
#include "../../misc/webservices/json_helper.h"

#include "transcribe.h"

struct http_endpoint
{
    char *host;
    char *authority;
    char *path;
    unsigned port;
    bool secure;
};

/*****************************************************************************
 * Common HTTP/JSON plumbing
 *****************************************************************************/

static void EndpointClean(struct http_endpoint *ep)
{
    free(ep->path);
    free(ep->authority);
    free(ep->host);
    ep->host = ep->authority = ep->path = NULL;
}

static int EndpointParse(filter_t *filter, struct http_endpoint *ep,
                         const char *option)
{
    char name[64];

    snprintf(name, sizeof (name), CFG_PREFIX "%s", option);

    char *url = var_InheritString(filter, name);
    if (url == NULL)
    {
        msg_Err(filter, "missing %s", name);
        return VLC_EGENERIC;
    }

    vlc_url_t u;
    int err = vlc_UrlParse(&u, url);
    free(url);
    if (err || u.psz_protocol == NULL || u.psz_host == NULL)
    {
        msg_Err(filter, "invalid %s", name);
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
    return VLC_EGENERIC;
}

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

/**
 * POSTs a JSON document (taking ownership of the heap buffer) and returns
 * the response body as a NUL-terminated heap buffer, or NULL on error.
 */
static char *HttpPost(filter_t *filter, struct vlc_http_mgr *mgr,
                      const struct http_endpoint *ep,
                      bool goog_auth, const char *key,
                      char *body, size_t body_len, size_t *restrict resp_len)
{
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
        vlc_http_mgr_request(mgr, ep->secure, ep->host, ep->port,
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

    size_t total = 0;
    block_t *b;
    while ((b = vlc_http_msg_read(resp)) != NULL)
    {
        if (total < (16u << 20)) /* sanity cap on the answer size */
        {
            vlc_memstream_write(&ms, b->p_buffer, b->i_buffer);
            total += b->i_buffer;
        }
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
 * Speech-to-text backend
 *****************************************************************************/

struct http_transcribe_sys
{
    struct http_endpoint ep;
    bool gemini;
    char *key;
    char *model;
    const char *prompt;
    struct vlc_http_mgr *mgr;
};

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
static int BuildRequestBody(struct http_transcribe_sys *sys, const char *b64,
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

static char *ExtractText(filter_t *filter, bool gemini,
                         const char *doc, size_t len)
{
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

    if (gemini)
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
        ret = transcribe_TrimDup(text);
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

static char *HttpTranscribe(filter_t *filter, void *opaque,
                            const int16_t *pcm, size_t frames)
{
    struct http_transcribe_sys *sys = opaque;
    size_t wav_size;
    void *wav = BuildWav(pcm, frames, &wav_size);
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
    char *resp = HttpPost(filter, sys->mgr, &sys->ep, sys->gemini, sys->key,
                          ms.ptr, ms.length, &resp_len);
    if (resp == NULL)
        return NULL;

    char *text = ExtractText(filter, sys->gemini, resp, resp_len);
    free(resp);
    return text;
}

static void HttpTranscribeClose(void *opaque)
{
    struct http_transcribe_sys *sys = opaque;

    vlc_http_mgr_destroy(sys->mgr);
    free(sys->model);
    free(sys->key);
    EndpointClean(&sys->ep);
    free(sys);
}

int transcribe_HttpOpen(filter_t *filter, struct transcribe_backend *backend,
                        const char *prompt, bool gemini)
{
    struct http_transcribe_sys *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
        return VLC_ENOMEM;

    int ret = EndpointParse(filter, &sys->ep, "url");
    if (ret != VLC_SUCCESS)
    {
        free(sys);
        return ret;
    }

    sys->gemini = gemini;
    sys->prompt = prompt;
    sys->key = var_InheritString(filter, CFG_PREFIX "api-key");
    sys->model = var_InheritString(filter, CFG_PREFIX "model");
    sys->mgr = vlc_http_mgr_create(VLC_OBJECT(filter), NULL);
    if (sys->model == NULL || sys->mgr == NULL)
    {
        HttpTranscribeClose(sys);
        return VLC_EGENERIC;
    }

    msg_Dbg(filter, "transcribing via %s (%s API, model %s)",
            sys->ep.authority, gemini ? "Gemini" : "OpenAI", sys->model);

    backend->sys = sys;
    backend->run = HttpTranscribe;
    backend->close = HttpTranscribeClose;
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Text-to-speech backend
 *****************************************************************************/

struct http_tts_sys
{
    struct http_endpoint ep;
    char *key;
    char *model;
    char **voices;
    size_t voice_count;
    struct vlc_http_mgr *mgr;
};

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

/**
 * Converts 16-bit little-endian PCM to a mono FL32 block.
 */
static block_t *WavToBlock(const uint8_t *data, size_t frames,
                           unsigned channels)
{
    block_t *b = (frames > 0) ? block_Alloc(frames * sizeof (float)) : NULL;
    if (b == NULL)
        return NULL;
    b->i_nb_samples = frames;

    float *out = (float *)b->p_buffer;
    for (size_t i = 0; i < frames; i++)
    {
        float acc = 0.f;
        for (unsigned c = 0; c < channels; c++)
            acc += (int16_t)GetWLE(data + (i * channels + c) * 2);
        out[i] = acc / (channels * 32768.f);
    }
    return b;
}

static block_t *HttpTtsSpeak(filter_t *filter, void *opaque,
                             unsigned speaker, const char *text,
                             unsigned *restrict ratep)
{
    struct http_tts_sys *sys = opaque;
    const char *voice = sys->voices[(speaker - 1) % sys->voice_count];

    struct vlc_memstream ms;
    if (vlc_memstream_open(&ms))
        return NULL;
    vlc_memstream_puts(&ms, "{\"model\":");
    JsonWriteString(&ms, sys->model);
    vlc_memstream_puts(&ms, ",\"voice\":");
    JsonWriteString(&ms, voice);
    vlc_memstream_puts(&ms, ",\"response_format\":\"wav\",\"input\":");
    JsonWriteString(&ms, text);
    vlc_memstream_putc(&ms, '}');
    if (vlc_memstream_close(&ms))
        return NULL;

    size_t resp_len;
    char *resp = HttpPost(filter, sys->mgr, &sys->ep, false, sys->key,
                          ms.ptr, ms.length, &resp_len);
    if (resp == NULL)
        return NULL;

    unsigned rate, channels;
    const uint8_t *data;
    size_t frames;
    block_t *b = NULL;

    if (WavParse((const uint8_t *)resp, resp_len, &rate, &channels,
                 &data, &frames) == 0)
    {
        b = WavToBlock(data, frames, channels);
        *ratep = rate;
    }
    else
        msg_Warn(filter, "unsupported TTS response (16-bit PCM WAV needed)");

    free(resp);
    return b;
}

static void HttpTtsClose(void *opaque)
{
    struct http_tts_sys *sys = opaque;

    vlc_http_mgr_destroy(sys->mgr);
    transcribe_FreeVoiceList(sys->voices, sys->voice_count);
    free(sys->model);
    free(sys->key);
    EndpointClean(&sys->ep);
    free(sys);
}

int transcribe_TtsServerOpen(filter_t *filter, struct tts_backend *backend,
                             const char *voices)
{
    struct http_tts_sys *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
        return VLC_ENOMEM;

    int ret = EndpointParse(filter, &sys->ep, "tts-url");
    if (ret != VLC_SUCCESS)
    {
        free(sys);
        return ret;
    }

    if (voices == NULL || voices[0] == '\0')
        voices = "alloy,onyx,nova,echo"; /* standard OpenAI voice names */
    if (transcribe_ParseVoiceList(voices, &sys->voices, &sys->voice_count))
    {
        msg_Err(filter, "no usable voice-over voices");
        EndpointClean(&sys->ep);
        free(sys);
        return VLC_EGENERIC;
    }

    sys->key = var_InheritString(filter, CFG_PREFIX "tts-api-key");
    sys->model = var_InheritString(filter, CFG_PREFIX "tts-model");
    sys->mgr = vlc_http_mgr_create(VLC_OBJECT(filter), NULL);
    if (sys->model == NULL || sys->mgr == NULL)
    {
        HttpTtsClose(sys);
        return VLC_EGENERIC;
    }

    msg_Dbg(filter, "voice-over via %s (%zu voices)", sys->ep.authority,
            sys->voice_count);

    backend->sys = sys;
    backend->speak = HttpTtsSpeak;
    backend->close = HttpTtsClose;
    return VLC_SUCCESS;
}
