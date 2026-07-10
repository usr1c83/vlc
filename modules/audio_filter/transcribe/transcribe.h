/*****************************************************************************
 * transcribe.h : live speech transcription filter interfaces
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

#ifndef VLC_TRANSCRIBE_H
#define VLC_TRANSCRIBE_H

#include <stdint.h>

#include <vlc_common.h>
#include <vlc_block.h>
#include <vlc_filter.h>

#define GEMMA_TRANSCRIPT_VAR "gemma-transcript-text"

#define CFG_PREFIX "gemma-transcribe-"

/* Gemma 4 ingests audio as 16 kHz mono PCM and accepts clips of at most
 * 30 seconds. */
#define TRANSCRIBE_RATE 16000

/**
 * Speech-to-text backend.
 * All callbacks are invoked from the filter worker thread only.
 */
struct transcribe_backend
{
    void *sys;
    /** Transcribes a 16 kHz mono PCM segment, returns heap-allocated text
     * (empty for silence) or NULL on error. */
    char *(*run)(filter_t *, void *sys, const int16_t *pcm, size_t frames);
    void (*close)(void *sys);
};

/**
 * Text-to-speech backend for the voice-over.
 * All callbacks are invoked from the filter worker thread only.
 */
struct tts_backend
{
    void *sys;
    /** Synthesizes one utterance. Returns a mono FL32 block and stores its
     * sample rate in *ratep, or returns NULL on error. Speakers are
     * numbered from 1 and mapped onto the backend voices. */
    block_t *(*speak)(filter_t *, void *sys, unsigned speaker,
                      const char *text, unsigned *ratep);
    void (*close)(void *sys);
};

/* Shared helpers (transcribe.c) */
char *transcribe_TrimDup(const char *str);
int transcribe_ParseVoiceList(const char *list, char ***voicesp,
                              size_t *countp);
void transcribe_FreeVoiceList(char **voices, size_t count);

/* HTTP backends (http.c) */
int transcribe_HttpOpen(filter_t *, struct transcribe_backend *,
                        const char *prompt, bool gemini);
int transcribe_TtsServerOpen(filter_t *, struct tts_backend *,
                             const char *voices);

#ifdef HAVE_LLAMACPP_MTMD
/* In-process Gemma 4 inference through llama.cpp libmtmd (llama.c) */
int transcribe_LocalOpen(filter_t *, struct transcribe_backend *,
                         const char *prompt);
#endif

#ifdef HAVE_ESPEAK_NG
/* Built-in speech synthesis through espeak-ng (espeak.c) */
int transcribe_EspeakOpen(filter_t *, struct tts_backend *,
                          const char *voices, const char *language);
#endif

#endif
