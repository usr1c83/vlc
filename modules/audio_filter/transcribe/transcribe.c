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
 *
 * The transcription can be exported as a SubRip subtitle file and the
 * synthesized voice-over as a timeline-aligned WAV track. In listen mode
 * the filter translates whatever another application is playing: feed it
 * an audio capture of the system output and VLC outputs only the
 * translation:
 *
 *   vlc pulse://$(pactl get-default-sink).monitor \
 *       --audio-filter=gemma_transcribe --gemma-transcribe-listen \
 *       --gemma-transcribe-voiceover --gemma-transcribe-language=Russian
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <errno.h>
#include <math.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_plugin.h>
#include <vlc_aout.h>
#include <vlc_block.h>
#include <vlc_filter.h>
#include <vlc_fs.h>
#include <vlc_memstream.h>
#include <vlc_queue.h>
#include <vlc_variables.h>

#include "transcribe.h"

#define SEGMENT_MS_MIN  1000
#define SEGMENT_MS_MAX  30000

/* Maximum number of segments kept waiting in the inference pipeline. When
 * inference cannot keep up with playback, the filter drops the oldest
 * waiting segments and keeps the newest ones, so the subtitles and
 * voice-over track the current playback position instead of falling
 * further and further behind. A small backlog keeps latency low. */
#define MAX_PENDING 2

/* Maximum number of utterances voiced per segment */
#define MAX_UTTERANCES 32

/* Maximum backlog of synthesized voice-over audio, in seconds */
#define MIX_MAX_DEPTH 60

typedef struct
{
    struct transcribe_backend backend;
    bool voiceover;
    bool listen;   /* translate-only mode: never output the original */
    float idle_gain;
    struct tts_backend tts;
    char *prompt;

    /* Subtitle/audio export (worker thread only after Open) */
    FILE *srt;
    unsigned srt_seq;
    FILE *wav;
    uint64_t wav_samples;

    /* Mono 16 kHz conversion state (audio thread only) */
    unsigned channels;
    unsigned rate;
    double step;   /* input samples per output sample */
    double pos;    /* fractional read position between prev and current */
    float prev;
    int16_t *buf;
    size_t fill;
    size_t target;
    vlc_tick_t seg_pts;   /* PTS of the first sample of the segment */

    /* Inference pipeline */
    vlc_queue_t queue;
    bool dead;
    unsigned queued;      /* segments waiting in the queue (queue lock) */
    unsigned dropped;     /* segments dropped while behind (queue lock) */
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
 * ducking the original program under the voice-over. In listen mode the
 * original is muted entirely and only the voice-over is audible.
 */
static void MixVoiceover(filter_t *filter, block_t *b)
{
    filter_sys_t *sys = filter->p_sys;
    float *pcm = (float *)b->p_buffer;
    const unsigned channels = sys->channels;
    const float idle = sys->idle_gain;

    vlc_mutex_lock(&sys->mix_lock);
    if (sys->mix_first == NULL && idle == 1.f && sys->gain > 0.999f)
    {
        sys->gain = 1.f;
        vlc_mutex_unlock(&sys->mix_lock);
        return;
    }

    for (size_t i = 0; i < b->i_nb_samples; i++)
    {
        float tts = 0.f;
        float gain_target = idle;
        block_t *m = sys->mix_first;

        if (m != NULL)
        {
            tts = ((const float *)m->p_buffer)[sys->mix_offset];
            gain_target = (sys->duck < idle) ? sys->duck : idle;
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
 * Subtitle and voice-over export (worker thread only)
 *****************************************************************************/

/**
 * Appends one numbered SRT entry covering the segment interval.
 */
static void SrtWrite(filter_t *filter, vlc_tick_t pts, vlc_tick_t length,
                     const char *text)
{
    filter_sys_t *sys = filter->p_sys;

    if (sys->srt == NULL || text[0] == '\0' || pts == VLC_TICK_INVALID)
        return;

    uint64_t t0 = (pts > VLC_TICK_0) ? MS_FROM_VLC_TICK(pts - VLC_TICK_0) : 0;
    uint64_t t1 = t0 + MS_FROM_VLC_TICK(length);

    fprintf(sys->srt, "%u\n"
            "%02"PRIu64":%02"PRIu64":%02"PRIu64",%03"PRIu64" --> "
            "%02"PRIu64":%02"PRIu64":%02"PRIu64",%03"PRIu64"\n%s\n\n",
            ++sys->srt_seq,
            t0 / 3600000, (t0 / 60000) % 60, (t0 / 1000) % 60, t0 % 1000,
            t1 / 3600000, (t1 / 60000) % 60, (t1 / 1000) % 60, t1 % 1000,
            text);
    fflush(sys->srt);
}

static void WavWriteHeader(FILE *f, unsigned rate, uint64_t samples)
{
    uint8_t hdr[44];
    uint32_t data_size = (samples * 2 > UINT32_MAX - 36)
        ? UINT32_MAX - 36 : samples * 2;

    memcpy(hdr, "RIFF", 4);
    SetDWLE(hdr + 4, 36 + data_size);
    memcpy(hdr + 8, "WAVEfmt ", 8);
    SetDWLE(hdr + 16, 16);
    SetWLE(hdr + 20, 1);              /* PCM */
    SetWLE(hdr + 22, 1);              /* mono */
    SetDWLE(hdr + 24, rate);
    SetDWLE(hdr + 28, rate * 2);      /* byte rate */
    SetWLE(hdr + 32, 2);              /* block align */
    SetWLE(hdr + 34, 16);             /* bits per sample */
    memcpy(hdr + 36, "data", 4);
    SetDWLE(hdr + 40, data_size);
    fwrite(hdr, 1, sizeof (hdr), f);
}

/**
 * Appends one synthesized utterance to the exported translation track,
 * padding with silence up to its timeline position.
 */
static void WavExport(filter_t *filter, vlc_tick_t pts, const block_t *b)
{
    filter_sys_t *sys = filter->p_sys;

    if (sys->wav == NULL)
        return;

    if (pts != VLC_TICK_INVALID && pts > VLC_TICK_0)
    {
        uint64_t start = samples_from_vlc_tick(pts - VLC_TICK_0, sys->rate);
        static const uint8_t zeros[8192] = { 0 };

        while (sys->wav_samples < start)
        {
            size_t n = start - sys->wav_samples;
            if (n > sizeof (zeros) / 2)
                n = sizeof (zeros) / 2;
            if (fwrite(zeros, 2, n, sys->wav) != n)
                return;
            sys->wav_samples += n;
        }
    }

    const float *in = (const float *)b->p_buffer;
    for (size_t i = 0; i < b->i_nb_samples; i++)
    {
        long v = lrintf(in[i] * 32767.f);
        if (v > INT16_MAX)
            v = INT16_MAX;
        else if (v < INT16_MIN)
            v = INT16_MIN;

        uint8_t sample[2];
        SetWLE(sample, (uint16_t)(int16_t)v);
        if (fwrite(sample, 2, 1, sys->wav) != 1)
            return;
    }
    sys->wav_samples += b->i_nb_samples;
}

/**
 * Publishes the transcription: subtitle variable and SRT export.
 */
static void PublishText(filter_t *filter, const char *text, vlc_tick_t pts,
                        vlc_tick_t length)
{
    var_SetString(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR, text);
    SrtWrite(filter, pts, length, text);
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
static void HandleUtterances(filter_t *filter, char *text, vlc_tick_t pts,
                             vlc_tick_t length)
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
            PublishText(filter, ms.ptr, pts, length);
            free(ms.ptr);
        }
    }

    vlc_tick_t voice_pts = pts;
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
        {
            WavExport(filter, voice_pts, b);
            if (voice_pts != VLC_TICK_INVALID)
                voice_pts += vlc_tick_from_samples(b->i_nb_samples,
                                                   sys->rate);
            MixEnqueue(filter, b);
        }
    }
}

/**
 * Dequeues the next segment to transcribe, keeping the queue-depth counter
 * consistent with the dequeue under a single lock. Returns NULL when the
 * filter is being closed.
 */
static block_t *NextSegment(filter_sys_t *sys)
{
    vlc_queue_Lock(&sys->queue);
    while (vlc_queue_IsEmpty(&sys->queue) && !sys->dead)
        vlc_queue_Wait(&sys->queue);

    block_t *segment = vlc_queue_DequeueUnlocked(&sys->queue);
    if (segment != NULL)
        sys->queued--;
    vlc_queue_Unlock(&sys->queue);
    return segment;
}

static void *Worker(void *data)
{
    filter_t *filter = data;
    filter_sys_t *sys = filter->p_sys;
    block_t *segment;

    vlc_thread_set_name("vlc-gemma");

    while ((segment = NextSegment(sys)) != NULL)
    {
        const vlc_tick_t pts = segment->i_pts;
        const vlc_tick_t length = segment->i_length;
        char *text = sys->backend.run(filter, sys->backend.sys,
                                      (const int16_t *)segment->p_buffer,
                                      segment->i_nb_samples);
        block_Release(segment);

        if (text != NULL)
        {
            msg_Dbg(filter, "transcript: %s", text);
            if (sys->voiceover)
                HandleUtterances(filter, text, pts, length);
            else
                /* An empty answer means silence: clear the subtitle. */
                PublishText(filter, text, pts, length);
            free(text);
        }
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

    block_t *b = block_Alloc(sys->fill * sizeof (int16_t));
    if (unlikely(b == NULL))
    {
        sys->fill = 0;
        return;
    }
    memcpy(b->p_buffer, sys->buf, sys->fill * sizeof (int16_t));
    b->i_nb_samples = sys->fill;
    b->i_pts = sys->seg_pts;
    b->i_length = vlc_tick_from_samples(sys->fill, TRANSCRIBE_RATE);
    sys->fill = 0;

    unsigned dropped = 0;

    vlc_queue_Lock(&sys->queue);
    /* When inference falls behind playback, drop the oldest waiting
     * segments so the pipeline always works on the most recent audio and
     * the subtitles stay close to real time (at the cost of skipping the
     * speech in between — enabling GPU offload avoids this). */
    while (sys->queued >= MAX_PENDING)
    {
        block_t *old = vlc_queue_DequeueUnlocked(&sys->queue);
        if (old == NULL)
            break;
        block_Release(old);
        sys->queued--;
        dropped = ++sys->dropped;
    }
    vlc_queue_EnqueueUnlocked(&sys->queue, b);
    sys->queued++;
    vlc_queue_Signal(&sys->queue);
    vlc_queue_Unlock(&sys->queue);

    /* Warn once, then every ~100 dropped segments, to hint at GPU offload
     * without flooding the log. */
    if (dropped == 1 || (dropped > 0 && dropped % 100 == 0))
        msg_Warn(filter, "inference cannot keep up with playback, skipping "
                 "audio; enable GPU offload (%sgpu-layers) or a shorter "
                 "%ssegment-ms for lower latency", CFG_PREFIX, CFG_PREFIX);
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

            if (sys->fill == 0)
                sys->seg_pts = (in->i_pts != VLC_TICK_INVALID)
                    ? in->i_pts + vlc_tick_from_samples(i, sys->rate)
                    : VLC_TICK_INVALID;

            sys->buf[sys->fill++] = s;
            if (sys->fill == sys->target)
                QueueSegment(filter);

            sys->pos += sys->step;
        }
        sys->pos -= 1.0;
        sys->prev = cur;
    }

    if (sys->voiceover || sys->listen)
        MixVoiceover(filter, in);

    return in;
}

static void Flush(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    sys->fill = 0;
    sys->pos = 0.;
    sys->prev = 0.f;

    if (sys->voiceover || sys->listen)
    {   /* Drop the now stale voice-over audio */
        vlc_mutex_lock(&sys->mix_lock);
        block_ChainRelease(sys->mix_first);
        sys->mix_first = NULL;
        sys->mix_lastp = &sys->mix_first;
        sys->mix_offset = 0;
        sys->mix_depth = 0;
        sys->gain = sys->idle_gain;
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

static void CloseExports(filter_sys_t *sys)
{
    if (sys->srt != NULL)
    {
        fclose(sys->srt);
        sys->srt = NULL;
    }
    if (sys->wav != NULL)
    {   /* patch the header now that the stream size is known */
        if (fseek(sys->wav, 0, SEEK_SET) == 0)
            WavWriteHeader(sys->wav, sys->rate, sys->wav_samples);
        fclose(sys->wav);
        sys->wav = NULL;
    }
}

static void Close(filter_t *filter)
{
    filter_sys_t *sys = filter->p_sys;

    vlc_queue_Kill(&sys->queue, &sys->dead);
    vlc_join(sys->thread, NULL);
    block_ChainRelease(vlc_queue_DequeueAll(&sys->queue));
    block_ChainRelease(sys->mix_first);

    var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);

    CloseExports(sys);
    if (sys->voiceover)
        sys->tts.close(sys->tts.sys);
    sys->backend.close(sys->backend.sys);

    free(sys->buf);
    free(sys->prompt);
    free(sys);
}

static void OpenExports(filter_t *filter, filter_sys_t *sys)
{
    char *path = var_InheritString(filter, CFG_PREFIX "export-file");
    if (path != NULL)
    {
        sys->srt = vlc_fopen(path, "wt");
        if (sys->srt == NULL)
            msg_Err(filter, "cannot export subtitles to %s: %s", path,
                    vlc_strerror_c(errno));
        else
            msg_Dbg(filter, "exporting subtitles to %s", path);
        free(path);
    }

    path = var_InheritString(filter, CFG_PREFIX "export-audio");
    if (path != NULL)
    {
        if (!sys->voiceover)
            msg_Warn(filter, "audio export needs %svoiceover", CFG_PREFIX);
        else
        {
            sys->wav = vlc_fopen(path, "wb");
            if (sys->wav == NULL)
                msg_Err(filter, "cannot export voice-over to %s: %s", path,
                        vlc_strerror_c(errno));
            else
            {
                WavWriteHeader(sys->wav, sys->rate, 0);
                msg_Dbg(filter, "exporting voice-over to %s", path);
            }
        }
        free(path);
    }
}

static int Open(vlc_object_t *obj)
{
    filter_t *filter = (filter_t *)obj;

    static const char *const options[] = {
        "backend", "url", "api-key", "model", "language", "segment-ms",
        "prompt", "model-path", "mmproj-path", "threads", "gpu-layers",
        "download", "model-url", "mmproj-url", "voiceover", "tts", "tts-url",
        "tts-api-key", "tts-model", "voices", "duck", "listen",
        "export-file", "export-audio", NULL
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
    sys->listen = var_InheritBool(filter, CFG_PREFIX "listen");
    sys->idle_gain = sys->listen ? 0.f : 1.f;
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
    }

    if (sys->voiceover || sys->listen)
    {
        float duck = var_InheritFloat(filter, CFG_PREFIX "duck");
        sys->duck = VLC_CLIP(duck, 0.f, 1.f);
        sys->gain = sys->idle_gain;
        vlc_mutex_init(&sys->mix_lock);
        sys->mix_lastp = &sys->mix_first;
    }

    OpenExports(filter, sys);

    vlc_queue_Init(&sys->queue, offsetof (block_t, p_next));
    sys->dead = false;

    var_Create(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR,
               VLC_VAR_STRING);

    if (vlc_clone(&sys->thread, Worker, filter))
    {
        var_Destroy(vlc_object_instance(filter), GEMMA_TRANSCRIPT_VAR);
        CloseExports(sys);
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

    msg_Dbg(filter, "transcribing %"PRId64" ms segments%s%s", segment_ms,
            sys->voiceover ? " with voice-over" : "",
            sys->listen ? " (listen mode)" : "");
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
    "model used by the local backend, e.g. gemma-4-E2B-it-Q4_K_M.gguf. " \
    "When left empty, a model bundled with VLC or previously downloaded " \
    "is used, or the model is downloaded automatically.")

#define DOWNLOAD_TEXT N_("Download the model automatically")
#define DOWNLOAD_LONGTEXT N_("When no model file is configured, bundled " \
    "with VLC or already downloaded, fetch it from the configured URL " \
    "into the VLC data directory on first use, showing the download " \
    "progress.")

#define MODEL_URL_TEXT N_("Model download URL")
#define MODEL_URL_LONGTEXT N_("HTTP(S) URL of the Gemma 4 audio-capable " \
    "GGUF model to download when it is not present locally.")

#define MMPROJ_URL_TEXT N_("Projector download URL")
#define MMPROJ_URL_LONGTEXT N_("HTTP(S) URL of the multimodal projector " \
    "(mmproj) GGUF matching the model, to download when it is not " \
    "present locally.")

#define MMPROJ_PATH_TEXT N_("Multimodal projector file")
#define MMPROJ_PATH_LONGTEXT N_("GGUF file of the multimodal projector " \
    "(mmproj) matching the model file, required for audio input with the " \
    "local backend.")

#define THREADS_TEXT N_("Inference threads")
#define THREADS_LONGTEXT N_("Number of CPU threads used by the local " \
    "backend (0 = automatic).")

#define GPU_TEXT N_("GPU acceleration (offloaded layers)")
#define GPU_LONGTEXT N_("How many model layers the local backend runs on " \
    "the GPU. -1 offloads the whole model to the GPU when a compatible one " \
    "is detected, which is far faster than the CPU and lets the subtitles " \
    "and voice-over keep up in real time; 0 forces CPU-only processing. " \
    "This needs a VLC build with GPU (Vulkan) support — the ready-to-run " \
    "release packages include it; otherwise processing stays on the CPU.")

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

#define LISTEN_TEXT N_("Listen mode (translate other applications)")
#define LISTEN_LONGTEXT N_("Never output the incoming sound itself, only " \
    "the subtitles and the synthesized voice-over. Use this with an " \
    "audio capture input that monitors the system output (for example " \
    "\"pulse://<sink>.monitor\" on Linux) to translate on the fly " \
    "whatever another application, such as a web browser, is playing.")

#define EXPORT_FILE_TEXT N_("Export subtitles to file")
#define EXPORT_FILE_LONGTEXT N_("Write the live transcription as a " \
    "SubRip (SRT) subtitle file with media timestamps.")

#define EXPORT_AUDIO_TEXT N_("Export voice-over to file")
#define EXPORT_AUDIO_LONGTEXT N_("Write the synthesized translation " \
    "track as a timeline-aligned mono WAV file (requires voice-over).")

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

    add_bool( CFG_PREFIX "listen", false, LISTEN_TEXT, LISTEN_LONGTEXT )

    set_section( N_("Local inference"), NULL )
    add_loadfile( CFG_PREFIX "model-path", NULL,
                  MODEL_PATH_TEXT, MODEL_PATH_LONGTEXT )
    add_loadfile( CFG_PREFIX "mmproj-path", NULL,
                  MMPROJ_PATH_TEXT, MMPROJ_PATH_LONGTEXT )
    add_integer_with_range( CFG_PREFIX "threads", 0, 0, 256,
                            THREADS_TEXT, THREADS_LONGTEXT )
    add_integer_with_range( CFG_PREFIX "gpu-layers", -1, -1, 1000,
                            GPU_TEXT, GPU_LONGTEXT )
    add_bool( CFG_PREFIX "download", true, DOWNLOAD_TEXT, DOWNLOAD_LONGTEXT )
    add_string( CFG_PREFIX "model-url",
                "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/"
                "resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
                MODEL_URL_TEXT, MODEL_URL_LONGTEXT )
    add_string( CFG_PREFIX "mmproj-url",
                "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/"
                "resolve/main/mmproj-F16.gguf",
                MMPROJ_URL_TEXT, MMPROJ_URL_LONGTEXT )

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

    set_section( N_("Export"), NULL )
    add_savefile( CFG_PREFIX "export-file", NULL,
                  EXPORT_FILE_TEXT, EXPORT_FILE_LONGTEXT )
    add_savefile( CFG_PREFIX "export-audio", NULL,
                  EXPORT_AUDIO_TEXT, EXPORT_AUDIO_LONGTEXT )

    add_shortcut( "gemma_transcribe", "gemma" )
    set_callback( Open )
vlc_module_end ()
