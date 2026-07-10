/*****************************************************************************
 * espeak.c : built-in espeak-ng synthesis for the transcription voice-over
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

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <vlc_common.h>
#include <vlc_block.h>
#include <vlc_filter.h>
#include <vlc_iso_lang.h>
#include <vlc_memstream.h>

#include <espeak-ng/speak_lib.h>

#include "transcribe.h"

/* Voice variants applied to the base language voice, in speaker order */
static const char *const espeak_variants[] = {
    "", "+m3", "+f2", "+m5", "+f4", "+m2", "+f3", "+m6",
};

struct espeak_sys
{
    char **voices;
    size_t voice_count;
    unsigned rate;
};

/* The espeak-ng engine is a process-wide singleton */
static vlc_mutex_t espeak_lock = VLC_STATIC_MUTEX;
static unsigned espeak_refs = 0;
static int espeak_rate = 0;
static struct vlc_memstream *espeak_sink = NULL;

static int SynthCallback(short *pcm, int count, espeak_EVENT *events)
{
    (void) events;
    if (espeak_sink != NULL && pcm != NULL && count > 0)
        vlc_memstream_write(espeak_sink, pcm, count * sizeof (short));
    return 0;
}

static block_t *EspeakSpeak(filter_t *filter, void *opaque, unsigned speaker,
                            const char *text, unsigned *restrict ratep)
{
    struct espeak_sys *sys = opaque;
    const char *voice = sys->voices[(speaker - 1) % sys->voice_count];
    struct vlc_memstream ms;

    if (vlc_memstream_open(&ms))
        return NULL;

    vlc_mutex_lock(&espeak_lock);
    if (espeak_SetVoiceByName(voice) != EE_OK)
    {
        msg_Warn(filter, "unknown espeak-ng voice \"%s\", using default",
                 voice);
        espeak_SetVoiceByName("default");
    }

    espeak_sink = &ms;
    espeak_ERROR err = espeak_Synth(text, strlen(text) + 1, 0, POS_CHARACTER,
                                    0, espeakCHARS_UTF8, NULL, NULL);
    espeak_sink = NULL;
    vlc_mutex_unlock(&espeak_lock);

    if (vlc_memstream_close(&ms))
        return NULL;

    if (err != EE_OK)
    {
        msg_Warn(filter, "espeak-ng synthesis failed (%d)", (int)err);
        free(ms.ptr);
        return NULL;
    }

    size_t frames = ms.length / sizeof (short);
    block_t *b = (frames > 0) ? block_Alloc(frames * sizeof (float)) : NULL;
    if (b != NULL)
    {
        const short *in = (const short *)ms.ptr;
        float *out = (float *)b->p_buffer;

        b->i_nb_samples = frames;
        for (size_t i = 0; i < frames; i++)
            out[i] = in[i] / 32768.f;
        *ratep = sys->rate;
    }
    free(ms.ptr);
    return b;
}

static void EspeakClose(void *opaque)
{
    struct espeak_sys *sys = opaque;

    vlc_mutex_lock(&espeak_lock);
    if (--espeak_refs == 0)
        espeak_Terminate();
    vlc_mutex_unlock(&espeak_lock);

    transcribe_FreeVoiceList(sys->voices, sys->voice_count);
    free(sys);
}

/**
 * Derives the espeak-ng base voice from the (free-form) target language,
 * e.g. "Russian" -> "ru".
 */
static const char *BaseVoice(filter_t *filter, const char *language)
{
    if (language == NULL || language[0] == '\0')
        return "en";

    const iso639_lang_t *lang = vlc_find_iso639(language, true);
    if (lang == NULL)
    {
        msg_Warn(filter, "unknown language \"%s\", using an English "
                 "voice-over voice (set %svoices to override)", language,
                 CFG_PREFIX);
        return "en";
    }
    return (lang->psz_iso639_1[0] != '\0') ? lang->psz_iso639_1
                                           : lang->psz_iso639_2T;
}

static int BuildVoices(filter_t *filter, struct espeak_sys *sys,
                       const char *voices, const char *language)
{
    if (voices != NULL && voices[0] != '\0')
        return transcribe_ParseVoiceList(voices, &sys->voices,
                                         &sys->voice_count);

    /* Derive one voice per speaker from the target language */
    const char *base = BaseVoice(filter, language);
    size_t count = ARRAY_SIZE(espeak_variants);
    char **list = vlc_alloc(count, sizeof (*list));
    if (unlikely(list == NULL))
        return VLC_ENOMEM;

    for (size_t i = 0; i < count; i++)
    {
        if (asprintf(&list[i], "%s%s", base, espeak_variants[i]) < 0)
        {
            transcribe_FreeVoiceList(list, i);
            return VLC_ENOMEM;
        }
    }

    sys->voices = list;
    sys->voice_count = count;
    return VLC_SUCCESS;
}

int transcribe_EspeakOpen(filter_t *filter, struct tts_backend *backend,
                          const char *voices, const char *language)
{
    struct espeak_sys *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
        return VLC_ENOMEM;

    int ret = BuildVoices(filter, sys, voices, language);
    if (ret != VLC_SUCCESS)
    {
        free(sys);
        return ret;
    }

    vlc_mutex_lock(&espeak_lock);
    if (espeak_refs == 0)
    {
        espeak_rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, NULL,
                                        espeakINITIALIZE_DONT_EXIT);
        if (espeak_rate > 0)
            espeak_SetSynthCallback(SynthCallback);
    }
    if (espeak_rate <= 0)
    {
        vlc_mutex_unlock(&espeak_lock);
        msg_Err(filter, "cannot initialize espeak-ng");
        transcribe_FreeVoiceList(sys->voices, sys->voice_count);
        free(sys);
        return VLC_EGENERIC;
    }
    espeak_refs++;
    vlc_mutex_unlock(&espeak_lock);

    sys->rate = espeak_rate;

    msg_Dbg(filter, "voice-over via espeak-ng at %u Hz (%zu voices, "
            "base \"%s\")", sys->rate, sys->voice_count, sys->voices[0]);

    backend->sys = sys;
    backend->speak = EspeakSpeak;
    backend->close = EspeakClose;
    return VLC_SUCCESS;
}
