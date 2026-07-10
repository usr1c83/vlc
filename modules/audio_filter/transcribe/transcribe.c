/*****************************************************************************
 * transcribe.c : live speech transcription/translation with Gemma 4
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
 * and transcribes them on the fly with a Gemma 4 multimodal model,
 * optionally translating the speech into a target language. The resulting
 * text is published on the libvlc instance through the
 * "gemma-transcript-text" string variable, which the "transcript" sub
 * source picks up to render live subtitles on top of the video:
 *
 *   vlc --audio-filter=gemma_transcribe --sub-source=transcript \
 *       --gemma-transcribe-language=Russian input.mkv
 *
 * Inference runs either in-process through llama.cpp/libmtmd on a local
 * GGUF model (no external service needed), or against an HTTP inference
 * server: OpenAI-compatible chat completions (llama.cpp llama-server,
 * vLLM…) with base64 WAV "input_audio" content parts, or the Google
 * Generative Language API for hosted Gemma 4 models.
 *
 * The filter can additionally voice the transcription over the original
 * audio track (voice-over translation). Gemma 4's speaker diarization is
 * then used to split the transcription into per-speaker utterances, each
 * synthesized with its own voice — either through the built-in espeak-ng
 * synthesizer or an OpenAI-compatible text-to-speech server — and mixed
 * into the audio output while the original sound is ducked:
 *
 *   vlc --audio-filter=gemma_transcribe --sub-source=transcript \
 *       --gemma-transcribe-language=Russian --gemma-transcribe-voiceover \
 *       input.mkv
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <math.h>
#include <stdatomic.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_plugin.h>
#include <vlc_aout.h>
#include <vlc_block.h>
#include <vlc_filter.h>
#include <vlc_memstream.h>
#include <vlc_queue.h>
#include <vlc_variables.h>

#include "transcribe.h"

#define SEGMENT_MS_MIN  1000
#define SEGMENT_MS_MAX  30000

/* Number of segments allowed to sit in the inference pipeline before the
 * filter starts skipping segments to remain (near) real-time. */
#define MAX_PENDING 3

/* Maximum number of utterances voiced per segment */
#define MAX_UTTERANCES 32

/* Maximum backlog of synthesized voice-over audio, in seconds */
#define MIX_MAX_DEPTH 60

typedef struct
{
    struct transcribe_backend backend;
    bool voiceover;
    struct tts_backend tts;
    char *prompt;

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
 * Shared helpers
 *****************************************************************************/

char *transcribe_TrimDup(const char *str)
{
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
        str++;

    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == ' ' || str[len - 1] == '\t'
                    || str[len - 1] == '\n' || str[len - 1] == '\r'))
        len--;

    return strndup(str, len);
}

void transcribe_FreeVoiceList(char **voices, size_t count)
{
    for (size_t i = 0; i < count; i++)
        free(voices[i]);
    free(voices);
}

int transcribe_ParseVoiceList(const char *list, char ***voicesp,
                              size_t *countp)
{
    char **voices = NULL;
    size_t count = 0;

    for (const char *item = list; item != NULL && *item != '\0';)
    {
        const char *next = strchr(item, ',');
        size_t len = (next != NULL) ? (size_t)(next - item) : strlen(item);

        while (len > 0 && (*item == ' ' || *item == '\t'))
            item++, len--;
        while (len > 0 && (item[len - 1] == ' ' || item[len - 1] == '\t'))
            len--;

        if (len > 0)
        {
            char *name = strndup(item, len);
            char **grown = (name != NULL)
                ? realloc(voices, (count + 1) * sizeof (*voices)) : NULL;
            if (grown == NULL)
            {
                free(name);
                transcribe_FreeVoiceList(voices, count);
                return VLC_ENOMEM;
            }
            voices = grown;
            voices[count++] = name;
        }

        item = (next != NULL) ? next + 1 : NULL;
    }

    if (count == 0)
        return VLC_EGENERIC;

    *voicesp = voices;
    *countp = count;
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Voice-over mixing
 *****************************************************************************/

/**
 * Converts a mono FL32 block to the output stream sample rate.
 */
static block_t *ResampleMono(block_t *in, unsigned in_rate,
                             unsigned out_rate)
{
    if (in_rate == out_rate)
        return in;

    const float *src = (const float *)in->p_buffer;
    size_t frames = in->i_nb_samples;

    if (frames < 2)
    {
        block_Release(in);
        return NULL;
    }

    size_t out_frames = (uint64_t)frames * out_rate / in_rate;
    block_t *out = (out_frames > 0)
        ? block_Alloc(out_frames * sizeof (float)) : NULL;
    if (out == NULL)
    {
        block_Release(in);
        return NULL;
    }
    out->i_nb_samples = out_frames;

    float *dst = (float *)out->p_buffer;
    const double step = (double)in_rate / out_rate;
    double pos = 0.;

    for (size_t i = 0; i < out_frames; i++, pos += step)
    {
        size_t idx = (size_t)pos;
        if (idx > frames - 2)
            idx = frames - 2;

        float frac = pos - idx;
        dst[i] = src[idx] + (src[idx + 1] - src[idx]) * frac;
    }

    block_Release(in);
    return out;
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

/*****************************************************************************
 * Worker thread
 *****************************************************************************/

/**
 * Extracts the speaker number from a diarized transcription line of the
 * form "S1: text" / "Speaker 2: text". Returns the utterance text.
 */
static const char *ParseSpeakerLine(const char *line, unsigned *speaker)
{
    const char *p = line;
    while (*p == ' ' || *p == '\t' || *p == '-')
        p++;

    if (p[0] == 'S' || p[0] == 's')
    {
        const char *r = p + 1;
        static const char word[] = "peaker";
        size_t w = 0;
        while (word[w] != '\0'
            && (r[w] == word[w] || r[w] == word[w] - 'a' + 'A'))
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
    filter_sys_t *sys = filter->p_sys;
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
    {
        unsigned rate;
        block_t *b = sys->tts.speak(filter, sys->tts.sys, utt[i].speaker,
                                    utt[i].text, &rate);
        if (b == NULL || rate == 0)
        {
            if (b != NULL)
                block_Release(b);
            continue;
        }

        b = ResampleMono(b, rate, sys->rate);
        if (b != NULL)
            MixEnqueue(filter, b);
    }
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
        char *text = sys->backend.run(filter, sys->backend.sys,
                                      (const int16_t *)segment->p_buffer,
                                      segment->i_nb_samples);
        block_Release(segment);

        if (text != NULL)
        {
            msg_Dbg(filter, "transcript: %s", text);
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
        msg_Warn(filter, "transcription too slow, skipping segment");
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

static int OpenTranscriber(filter_t *filter, filter_sys_t *sys)
{
    char *backend = var_InheritString(filter, CFG_PREFIX "backend");
    int ret;

    if (backend != NULL && !strcmp(backend, "local"))
    {
#ifdef HAVE_LLAMACPP_MTMD
        ret = transcribe_LocalOpen(filter, &sys->backend, sys->prompt);
#else
        msg_Err(filter, "VLC was built without llama.cpp support: "
                "use an inference server backend instead");
        ret = VLC_EGENERIC;
#endif
    }
    else
    {
        bool gemini = backend != NULL && !strcmp(backend, "gemini");
        ret = transcribe_HttpOpen(filter, &sys->backend, sys->prompt,
                                  gemini);
    }
    free(backend);
    return ret;
}

static int OpenSynthesizer(filter_t *filter, filter_sys_t *sys)
{
    char *tts = var_InheritString(filter, CFG_PREFIX "tts");
    char *voices = var_InheritString(filter, CFG_PREFIX "voices");
    int ret;

    if (tts != NULL && !strcmp(tts, "espeak"))
    {
#ifdef HAVE_ESPEAK_NG
        char *language = var_InheritString(filter, CFG_PREFIX "language");
        ret = transcribe_EspeakOpen(filter, &sys->tts, voices, language);
        free(language);
#else
        msg_Err(filter, "VLC was built without espeak-ng support: "
                "use a text-to-speech server instead");
        ret = VLC_EGENERIC;
#endif
    }
    else
        ret = transcribe_TtsServerOpen(filter, &sys->tts, voices);

    free(voices);
    free(tts);
    return ret;
}

static void Close(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    vlc_queue_Kill(&sys->queue, &sys->dead);
    vlc_join(sys->thread, NULL);
    block_ChainRelease(vlc_queue_DequeueAll(&sys->queue));
    block_ChainRelease(sys->mix_first);

    var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);

    if (sys->voiceover)
        sys->tts.close(sys->tts.sys);
    sys->backend.close(sys->backend.sys);

    free(sys->buf);
    free(sys->prompt);
    free(sys);
}

static int Open(vlc_object_t *obj)
{
    filter_t *filter = (filter_t *)obj;

    static const char *const options[] = {
        "backend", "url", "api-key", "model", "language", "segment-ms",
        "prompt", "model-path", "mmproj-path", "threads",
        "voiceover", "tts", "tts-url", "tts-api-key", "tts-model",
        "voices", "duck", NULL
    };
    config_ChainParse(filter, CFG_PREFIX, options, filter->p_cfg);

    filter_sys_t *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
        return VLC_ENOMEM;
    filter->p_sys = sys;

    filter->fmt_in.audio.i_format = VLC_CODEC_FL32;
    aout_FormatPrepare(&filter->fmt_in.audio);
    filter->fmt_out.audio = filter->fmt_in.audio;

    sys->channels = aout_FormatNbChannels(&filter->fmt_in.audio);
    sys->rate = filter->fmt_in.audio.i_rate;
    sys->step = (double)sys->rate / TRANSCRIBE_RATE;

    int ret = VLC_EGENERIC;
    if (sys->channels == 0 || sys->rate == 0)
    {
        msg_Err(filter, "invalid audio format");
        goto error;
    }

    sys->voiceover = var_InheritBool(filter, CFG_PREFIX "voiceover");
    sys->prompt = BuildPrompt(filter, sys->voiceover);
    if (sys->prompt == NULL)
    {
        ret = VLC_ENOMEM;
        goto error;
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

    ret = OpenTranscriber(filter, sys);
    if (ret != VLC_SUCCESS)
        goto error;

    if (sys->voiceover)
    {
        ret = OpenSynthesizer(filter, sys);
        if (ret != VLC_SUCCESS)
        {
            sys->backend.close(sys->backend.sys);
            goto error;
        }

        float duck = var_InheritFloat(filter, CFG_PREFIX "duck");
        sys->duck = VLC_CLIP(duck, 0.f, 1.f);
        sys->gain = 1.f;
        vlc_mutex_init(&sys->mix_lock);
        sys->mix_lastp = &sys->mix_first;
    }

    vlc_queue_Init(&sys->queue, offsetof (block_t, p_next));
    atomic_init(&sys->pending, 0);
    sys->dead = false;

    var_Create(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR,
               VLC_VAR_STRING);

    if (vlc_clone(&sys->thread, Worker, filter))
    {
        var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);
        if (sys->voiceover)
            sys->tts.close(sys->tts.sys);
        sys->backend.close(sys->backend.sys);
        ret = VLC_EGENERIC;
        goto error;
    }

    static const struct vlc_filter_operations filter_ops =
    {
        .filter_audio = Process, .flush = Flush, .close = Close,
    };
    filter->ops = &filter_ops;

    msg_Dbg(filter, "transcribing %"PRId64" ms segments%s", segment_ms,
            sys->voiceover ? " with voice-over" : "");
    return VLC_SUCCESS;

error:
    free(sys->buf);
    free(sys->prompt);
    free(sys);
    return ret;
}

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/

#ifdef HAVE_LLAMACPP_MTMD
# define BACKEND_DEFAULT "local"
#else
# define BACKEND_DEFAULT "openai"
#endif

#ifdef HAVE_ESPEAK_NG
# define TTS_DEFAULT "espeak"
#else
# define TTS_DEFAULT "server"
#endif

#define BACKEND_TEXT N_("Inference backend")
#define BACKEND_LONGTEXT N_("Where to run the Gemma 4 model: \"local\" " \
    "runs the model inside VLC through llama.cpp (no external service " \
    "needed), \"openai\" talks to an OpenAI-compatible chat completions " \
    "server (llama.cpp llama-server, vLLM, LM Studio…), \"gemini\" talks " \
    "to the Google Generative Language API.")

#define URL_TEXT N_("Inference server URL")
#define URL_LONGTEXT N_("HTTP(S) URL of the Gemma 4 inference endpoint. " \
    "For a local llama.cpp server this is typically " \
    "\"http://127.0.0.1:8080/v1/chat/completions\". For the Google " \
    "Generative Language API, use something like " \
    "\"https://generativelanguage.googleapis.com/v1beta/models/" \
    "gemma-4-12b-it:generateContent\" together with the \"gemini\" " \
    "backend.")

#define KEY_TEXT N_("API key")
#define KEY_LONGTEXT N_("Credential sent to the inference server, if it " \
    "requires one. Sent as a Bearer token (OpenAI format) or as the " \
    "x-goog-api-key header (Gemini format).")

#define MODEL_TEXT N_("Model name")
#define MODEL_LONGTEXT N_("Name of the Gemma 4 audio-capable model to " \
    "request from the inference server. Audio input requires the 12B, " \
    "E2B or E4B variants.")

#define MODEL_PATH_TEXT N_("Model file")
#define MODEL_PATH_LONGTEXT N_("GGUF file of the Gemma 4 audio-capable " \
    "model used by the local backend, e.g. gemma-4-E2B-it-Q4_K_M.gguf.")

#define MMPROJ_PATH_TEXT N_("Multimodal projector file")
#define MMPROJ_PATH_LONGTEXT N_("GGUF file of the multimodal projector " \
    "(mmproj) matching the model file, required for audio input with the " \
    "local backend.")

#define THREADS_TEXT N_("Inference threads")
#define THREADS_LONGTEXT N_("Number of CPU threads used by the local " \
    "backend (0 = automatic).")

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
    "its own voice, and the original sound is ducked while the voice-over " \
    "plays.")

#define TTS_TEXT N_("Speech synthesizer")
#define TTS_LONGTEXT N_("How the voice-over speech is synthesized: " \
    "\"espeak\" uses the built-in espeak-ng synthesizer (no external " \
    "service needed), \"server\" uses an OpenAI-compatible speech server.")

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
#define VOICES_LONGTEXT N_("Comma-separated list of voice names. Diarized " \
    "speakers are assigned these voices in order (speaker 1 gets the " \
    "first voice, speaker 2 the second one, and so on). When left empty, " \
    "a default voice set is derived from the synthesizer and the target " \
    "language.")

#define DUCK_TEXT N_("Original audio level under voice-over")
#define DUCK_LONGTEXT N_("Gain applied to the original audio while the " \
    "voice-over is speaking, between 0.0 (mute the program) and 1.0 " \
    "(no ducking).")

#define TRANSCRIBE_HELP N_("Transcribe or translate speech into live " \
    "subtitles and voice-over with a Gemma 4 model (use with the " \
    "\"transcript\" sub source)")

static const char *const backend_values[] = { "local", "openai", "gemini" };
static const char *const backend_texts[] = {
    N_("Built-in (llama.cpp)"),
    N_("OpenAI-compatible server (llama.cpp, vLLM…)"),
    N_("Google Generative Language API"),
};

static const char *const tts_values[] = { "espeak", "server" };
static const char *const tts_texts[] = {
    N_("Built-in (espeak-ng)"),
    N_("OpenAI-compatible speech server"),
};

vlc_module_begin ()
    set_shortname( N_("Gemma transcribe") )
    set_description( N_("Gemma 4 live speech transcription") )
    set_help( TRANSCRIBE_HELP )
    set_capability( "audio filter", 0 )
    set_subcategory( SUBCAT_AUDIO_AFILTER )

    add_string( CFG_PREFIX "backend", BACKEND_DEFAULT,
                BACKEND_TEXT, BACKEND_LONGTEXT )
        change_string_list( backend_values, backend_texts )
    add_string( CFG_PREFIX "language", NULL, LANGUAGE_TEXT,
                LANGUAGE_LONGTEXT )
    add_integer_with_range( CFG_PREFIX "segment-ms", 5000,
                            SEGMENT_MS_MIN, SEGMENT_MS_MAX,
                            SEGMENT_TEXT, SEGMENT_LONGTEXT )
    add_string( CFG_PREFIX "prompt", NULL, PROMPT_TEXT, PROMPT_LONGTEXT )

    set_section( N_("Local inference"), NULL )
    add_loadfile( CFG_PREFIX "model-path", NULL,
                  MODEL_PATH_TEXT, MODEL_PATH_LONGTEXT )
    add_loadfile( CFG_PREFIX "mmproj-path", NULL,
                  MMPROJ_PATH_TEXT, MMPROJ_PATH_LONGTEXT )
    add_integer_with_range( CFG_PREFIX "threads", 0, 0, 256,
                            THREADS_TEXT, THREADS_LONGTEXT )

    set_section( N_("Inference server"), NULL )
    add_string( CFG_PREFIX "url", "http://127.0.0.1:8080/v1/chat/completions",
                URL_TEXT, URL_LONGTEXT )
    add_password( CFG_PREFIX "api-key", NULL, KEY_TEXT, KEY_LONGTEXT )
    add_string( CFG_PREFIX "model", "gemma-4-12b-it", MODEL_TEXT,
                MODEL_LONGTEXT )

    set_section( N_("Voice-over"), NULL )
    add_bool( CFG_PREFIX "voiceover", false, VOICEOVER_TEXT,
              VOICEOVER_LONGTEXT )
    add_string( CFG_PREFIX "tts", TTS_DEFAULT, TTS_TEXT, TTS_LONGTEXT )
        change_string_list( tts_values, tts_texts )
    add_string( CFG_PREFIX "voices", NULL, VOICES_TEXT, VOICES_LONGTEXT )
    add_float_with_range( CFG_PREFIX "duck", 0.25, 0., 1.,
                          DUCK_TEXT, DUCK_LONGTEXT )
    add_string( CFG_PREFIX "tts-url",
                "http://127.0.0.1:8880/v1/audio/speech",
                TTS_URL_TEXT, TTS_URL_LONGTEXT )
    add_password( CFG_PREFIX "tts-api-key", NULL, TTS_KEY_TEXT,
                  TTS_KEY_LONGTEXT )
    add_string( CFG_PREFIX "tts-model", "tts-1", TTS_MODEL_TEXT,
                TTS_MODEL_LONGTEXT )

    add_shortcut( "gemma_transcribe", "gemma" )
    set_callback( Open )
vlc_module_end ()
