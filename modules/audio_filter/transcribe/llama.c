/*****************************************************************************
 * llama.c : in-process Gemma 4 inference through llama.cpp libmtmd
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
 * Runs a Gemma 4 audio-capable GGUF model directly inside VLC through
 * llama.cpp and its multimodal library (libmtmd), so live transcription
 * works without any external inference service. The model and its
 * multimodal projector (mmproj) files are configured with the
 * --gemma-transcribe-model-path and --gemma-transcribe-mmproj-path options.
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <vlc_common.h>
#include <vlc_block.h>
#include <vlc_configuration.h>
#include <vlc_filter.h>
#include <vlc_memstream.h>
#include <vlc_variables.h>

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#include "transcribe.h"

/* Upper bound on the generated answer, in tokens */
#define MAX_OUTPUT_TOKENS 512

struct llama_backend_sys
{
    struct llama_model *model;
    struct llama_context *lctx;
    const struct llama_vocab *vocab;
    mtmd_context *mctx;
    struct llama_sampler *sampler;
    char *prompt;   /* chat-templated prompt with the media marker */
    int32_t n_batch;
};

static void InitLlamaOnce(void *opaque)
{
    (void) opaque;
    llama_backend_init();
}

/**
 * Wraps the instruction into the model chat template, with the audio
 * marker placed before the instruction text as recommended for Gemma 4
 * multimodal prompts.
 */
static char *TemplatePrompt(filter_t *filter, const struct llama_model *model,
                            const char *prompt)
{
    char *user_msg;
    if (asprintf(&user_msg, "%s\n%s", mtmd_default_marker(), prompt) < 0)
        return NULL;

    const char *tmpl = llama_model_chat_template(model, NULL);
    char *full = NULL;

    if (tmpl != NULL)
    {
        const struct llama_chat_message msg = { "user", user_msg };
        int32_t size = llama_chat_apply_template(tmpl, &msg, 1, true,
                                                 NULL, 0);
        if (size > 0)
        {
            full = malloc(size + 1);
            if (full != NULL
             && llama_chat_apply_template(tmpl, &msg, 1, true, full,
                                          size + 1) == size)
                full[size] = '\0';
            else
            {
                free(full);
                full = NULL;
            }
        }
    }

    if (full == NULL)
    {   /* No or broken template metadata: use the Gemma chat format */
        msg_Dbg(filter, "using the built-in Gemma chat template");
        if (asprintf(&full, "<start_of_turn>user\n%s<end_of_turn>\n"
                     "<start_of_turn>model\n", user_msg) < 0)
            full = NULL;
    }

    free(user_msg);
    return full;
}

static char *LocalTranscribe(filter_t *filter, void *opaque,
                             const int16_t *pcm, size_t frames)
{
    struct llama_backend_sys *sys = opaque;

    float *samples = vlc_alloc(frames, sizeof (float));
    if (unlikely(samples == NULL))
        return NULL;
    for (size_t i = 0; i < frames; i++)
        samples[i] = pcm[i] / 32768.f;

    mtmd_bitmap *bitmap = mtmd_bitmap_init_from_audio(frames, samples);
    free(samples);
    if (bitmap == NULL)
        return NULL;

    char *text = NULL;
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (chunks == NULL)
        goto out;

    const mtmd_input_text input = {
        .text = sys->prompt,
        .add_special = true,
        .parse_special = true,
    };
    const mtmd_bitmap *bitmaps[1] = { bitmap };

    if (mtmd_tokenize(sys->mctx, chunks, &input, bitmaps, 1) != 0)
    {
        msg_Warn(filter, "cannot tokenize audio segment");
        goto out;
    }

    /* Each segment is transcribed in a fresh context */
    llama_memory_clear(llama_get_memory(sys->lctx), true);

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(sys->mctx, sys->lctx, chunks, 0, 0,
                                sys->n_batch, true, &n_past) != 0)
    {
        msg_Warn(filter, "cannot evaluate audio segment");
        goto out;
    }

    struct vlc_memstream ms;
    if (vlc_memstream_open(&ms))
        goto out;

    const vlc_tick_t start = vlc_tick_now();

    for (unsigned i = 0; i < MAX_OUTPUT_TOKENS; i++)
    {
        llama_token token = llama_sampler_sample(sys->sampler, sys->lctx, -1);
        if (llama_vocab_is_eog(sys->vocab, token))
            break;

        char piece[128];
        int32_t len = llama_token_to_piece(sys->vocab, token, piece,
                                           sizeof (piece), 0, false);
        if (len > 0)
            vlc_memstream_write(&ms, piece, len);

        struct llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(sys->lctx, batch) != 0)
            break;
    }

    if (vlc_memstream_close(&ms) == 0)
    {
        msg_Dbg(filter, "segment transcribed in %"PRId64" ms",
                MS_FROM_VLC_TICK(vlc_tick_now() - start));
        text = transcribe_TrimDup(ms.ptr);
        free(ms.ptr);
    }

out:
    if (chunks != NULL)
        mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    return text;
}

static void LocalClose(void *opaque)
{
    struct llama_backend_sys *sys = opaque;

    if (sys->sampler != NULL)
        llama_sampler_free(sys->sampler);
    if (sys->mctx != NULL)
        mtmd_free(sys->mctx);
    if (sys->lctx != NULL)
        llama_free(sys->lctx);
    if (sys->model != NULL)
        llama_model_free(sys->model);
    free(sys->prompt);
    free(sys);
}

int transcribe_LocalOpen(filter_t *filter,
                         struct transcribe_backend *backend,
                         const char *prompt)
{
    bool download = var_InheritBool(filter, CFG_PREFIX "download");
    char *model_path = transcribe_LocateModel(filter,
                                              CFG_PREFIX "model-path",
                                              CFG_PREFIX "model-url",
                                              download);
    char *mmproj_path = (model_path != NULL)
        ? transcribe_LocateModel(filter, CFG_PREFIX "mmproj-path",
                                 CFG_PREFIX "mmproj-url", download)
        : NULL;

    if (model_path == NULL || mmproj_path == NULL)
    {
        msg_Err(filter, "no Gemma 4 model available. The model is normally "
                "downloaded automatically on first use, or bundled with the "
                "installer. To use a model you already have, set "
                "%smodel-path and %smmproj-path to its GGUF files. To let "
                "VLC fetch it, enable %sdownload and check your network "
                "connection.", CFG_PREFIX, CFG_PREFIX, CFG_PREFIX);
        free(model_path);
        free(mmproj_path);
        return VLC_EGENERIC;
    }

    struct llama_backend_sys *sys = calloc(1, sizeof (*sys));
    if (unlikely(sys == NULL))
    {
        free(model_path);
        free(mmproj_path);
        return VLC_ENOMEM;
    }

    static vlc_once_t once = VLC_STATIC_ONCE;
    vlc_once(&once, InitLlamaOnce, NULL);

    int64_t threads = var_InheritInteger(filter, CFG_PREFIX "threads");
    if (threads <= 0)
        threads = vlc_GetCPUCount();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;

    msg_Info(filter, "loading Gemma model %s", model_path);
    sys->model = llama_model_load_from_file(model_path, mparams);
    if (sys->model == NULL)
    {
        msg_Err(filter, "cannot load model %s", model_path);
        goto error;
    }
    sys->vocab = llama_model_get_vocab(sys->model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    sys->n_batch = cparams.n_batch;

    sys->lctx = llama_init_from_model(sys->model, cparams);
    if (sys->lctx == NULL)
    {
        msg_Err(filter, "cannot create inference context");
        goto error;
    }

    struct mtmd_context_params mtparams = mtmd_context_params_default();
    mtparams.use_gpu = false;
    mtparams.print_timings = false;
    mtparams.n_threads = threads;
    mtparams.warmup = false;

    sys->mctx = mtmd_init_from_file(mmproj_path, sys->model, mtparams);
    if (sys->mctx == NULL)
    {
        msg_Err(filter, "cannot load multimodal projector %s", mmproj_path);
        goto error;
    }

    if (!mtmd_support_audio(sys->mctx))
    {
        msg_Err(filter, "this model does not support audio input: use a "
                "Gemma 4 audio-capable model (12B, E2B or E4B)");
        goto error;
    }

    int rate = mtmd_get_audio_sample_rate(sys->mctx);
    if (rate != TRANSCRIBE_RATE)
    {
        msg_Err(filter, "unsupported model audio sample rate %d Hz", rate);
        goto error;
    }

    sys->sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (sys->sampler == NULL)
        goto error;
    llama_sampler_chain_add(sys->sampler, llama_sampler_init_greedy());

    sys->prompt = TemplatePrompt(filter, sys->model, prompt);
    if (sys->prompt == NULL)
        goto error;

    msg_Dbg(filter, "local Gemma inference ready (%"PRId64" threads)",
            threads);
    free(model_path);
    free(mmproj_path);

    backend->sys = sys;
    backend->run = LocalTranscribe;
    backend->close = LocalClose;
    return VLC_SUCCESS;

error:
    LocalClose(sys);
    free(model_path);
    free(mmproj_path);
    return VLC_EGENERIC;
}
