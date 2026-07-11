/*****************************************************************************
 * download.c : Gemma model file lookup and download
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
 * Locates the GGUF files used by the local inference backend. Model files
 * are searched in this order:
 *   1. the explicit --gemma-transcribe-model-path/mmproj-path options,
 *   2. the VLC data directory (models bundled with the installation),
 *   3. the per-user VLC data directory (previously downloaded models),
 *   4. when downloading is enabled, the file is fetched from the configured
 *      URL (Hugging Face by default) into the per-user data directory,
 *      with a progress dialog in the interface.
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_dialog.h>
#include <vlc_filter.h>
#include <vlc_fs.h>
#include <vlc_stream.h>
#include <vlc_variables.h>

#include "transcribe.h"

#define MODEL_SUBDIR "gemma"

static bool FileExists(const char *path)
{
    struct stat st;
    return path != NULL && vlc_stat(path, &st) == 0 && st.st_size > 0;
}

static const char *UrlBasename(const char *url)
{
    const char *name = strrchr(url, '/');
    return (name != NULL && name[1] != '\0') ? name + 1 : NULL;
}

/* Interrupted downloads are retried this many times, resuming from the
 * last received byte. */
#define DOWNLOAD_TRIES 10

/**
 * Sleeps between download attempts, aborting early on cancellation.
 * Returns false when the download was cancelled.
 */
static bool RetryWait(filter_t *filter, vlc_dialog_id *dialog,
                      unsigned attempt)
{
    unsigned seconds = (attempt < 5) ? (2 * attempt) : 10;

    for (unsigned i = 0; i < seconds; i++)
    {
        if (dialog != NULL && vlc_dialog_is_cancelled(filter, dialog))
            return false;
        vlc_tick_sleep(VLC_TICK_FROM_SEC(1));
    }
    return true;
}

/**
 * Downloads url into dir/name, with a progress dialog, retries and
 * resumption of partial downloads. Returns the file path on success.
 */
static char *Fetch(filter_t *filter, const char *url, const char *dir,
                   const char *name)
{
    char *dest, *part = NULL;
    FILE *out = NULL;
    vlc_dialog_id *dialog = NULL;
    bool keep_part = false;

    if (asprintf(&dest, "%s" DIR_SEP "%s", dir, name) < 0)
        return NULL;
    if (asprintf(&part, "%s.part", dest) < 0)
        goto error;

    /* Resume a previously interrupted download */
    uint64_t done = 0;
    struct stat st;
    if (vlc_stat(part, &st) == 0 && st.st_size > 0)
    {
        done = st.st_size;
        msg_Info(filter, "resuming download of %s at %"PRIu64" MiB", name,
                 done >> 20);
    }

    out = vlc_fopen(part, (done > 0) ? "ab" : "wb");
    if (out == NULL)
    {
        msg_Err(filter, "cannot write %s: %s", part, vlc_strerror_c(errno));
        goto error;
    }

    msg_Info(filter, "downloading %s to %s", url, dest);
    dialog = vlc_dialog_display_progress(filter, false, 0.f, _("Cancel"),
                                         _("Downloading model"),
                                         _("Downloading the speech "
                                           "transcription model %s…"), name);

    uint64_t total = 0, logged = done;

    for (unsigned attempt = 1; attempt <= DOWNLOAD_TRIES; attempt++)
    {
        if (attempt > 1)
        {
            msg_Warn(filter, "download interrupted at %"PRIu64" MiB, "
                     "retrying (%u/%u)", done >> 20, attempt,
                     (unsigned)DOWNLOAD_TRIES);
            if (!RetryWait(filter, dialog, attempt))
            {
                msg_Warn(filter, "model download cancelled");
                keep_part = true;
                goto error;
            }
        }

        stream_t *stream = vlc_stream_NewURL(filter, url);
        if (stream == NULL)
            continue;

        uint64_t size;
        if (vlc_stream_GetSize(stream, &size) == 0 && size > 0)
            total = size;

        if (done > 0 && vlc_stream_Seek(stream, done))
        {   /* No resumption support: restart from scratch */
            msg_Warn(filter, "cannot resume download, restarting");
            if (fseek(out, 0, SEEK_SET) || ftruncate(fileno(out), 0))
            {
                vlc_stream_Delete(stream);
                goto error;
            }
            done = 0;
        }

        bool failed = false;
        while (total == 0 || done < total)
        {
            char buf[65536];
            ssize_t val = vlc_stream_Read(stream, buf, sizeof (buf));
            if (val < 0)
            {
                failed = true;
                break;
            }
            if (val == 0)
                break;

            if (fwrite(buf, 1, val, out) != (size_t)val)
            {
                msg_Err(filter, "cannot write %s: %s", part,
                        vlc_strerror_c(errno));
                vlc_stream_Delete(stream);
                keep_part = true;
                goto error;
            }
            done += val;

            if (dialog != NULL)
            {
                if (vlc_dialog_is_cancelled(filter, dialog))
                {
                    msg_Warn(filter, "model download cancelled");
                    vlc_stream_Delete(stream);
                    keep_part = true;
                    goto error;
                }
                if (total > 0)
                    vlc_dialog_update_progress(filter, dialog,
                                               (float)done / total);
            }
            if (done - logged >= (100u << 20))
            {   /* log every 100 MiB for console users */
                msg_Info(filter, "downloaded %"PRIu64" / %"PRIu64" MiB",
                         done >> 20, total >> 20);
                logged = done;
            }
        }
        vlc_stream_Delete(stream);

        if (!failed && (total == 0 || done >= total))
        {
            if (total == 0 && done == 0)
                continue; /* empty answer, try again */

            fflush(out);
            if (fclose(out))
            {
                out = NULL;
                keep_part = true;
                goto error;
            }
            out = NULL;

            if (vlc_rename(part, dest))
            {
                msg_Err(filter, "cannot rename %s: %s", part,
                        vlc_strerror_c(errno));
                keep_part = true;
                goto error;
            }

            msg_Info(filter, "model saved as %s", dest);
            if (dialog != NULL)
                vlc_dialog_release(filter, dialog);
            free(part);
            return dest;
        }

        fflush(out); /* keep the partial data for the next attempt */
    }

    msg_Err(filter, "download of %s failed after %u attempts", url,
            (unsigned)DOWNLOAD_TRIES);
    keep_part = true; /* resume on the next run */

error:
    if (dialog != NULL)
        vlc_dialog_release(filter, dialog);
    if (out != NULL)
        fclose(out);
    if (part != NULL && !keep_part)
        vlc_unlink(part);
    free(part);
    free(dest);
    return NULL;
}

char *transcribe_LocateModel(filter_t *filter, const char *path_option,
                             const char *url_option, bool download)
{
    /* 1. Explicit file path */
    char *path = var_InheritString(filter, path_option);
    if (path != NULL)
        return path;

    char *url = var_InheritString(filter, url_option);
    if (url == NULL)
        return NULL;

    const char *name = UrlBasename(url);
    if (name == NULL)
    {
        msg_Err(filter, "invalid model URL %s", url);
        free(url);
        return NULL;
    }

    /* 2. Model bundled with the VLC installation */
    char *sub;
    if (asprintf(&sub, MODEL_SUBDIR DIR_SEP "%s", name) >= 0)
    {
        path = config_GetSysPath(VLC_PKG_DATA_DIR, sub);
        free(sub);
        if (FileExists(path))
        {
            msg_Dbg(filter, "using bundled model %s", path);
            free(url);
            return path;
        }
        free(path);
    }

    /* 3. Previously downloaded model */
    char *datadir = config_GetUserDir(VLC_USERDATA_DIR);
    if (datadir == NULL)
    {
        free(url);
        return NULL;
    }

    char *dir;
    int ret = asprintf(&dir, "%s" DIR_SEP MODEL_SUBDIR, datadir);
    free(datadir);
    if (ret < 0)
    {
        free(url);
        return NULL;
    }

    if (asprintf(&path, "%s" DIR_SEP "%s", dir, name) >= 0)
    {
        if (FileExists(path))
        {
            msg_Dbg(filter, "using downloaded model %s", path);
            free(dir);
            free(url);
            return path;
        }
        free(path);
    }

    /* 4. Download */
    path = NULL;
    if (download)
    {
        if (vlc_mkdir_parent(dir, 0755) == 0 || errno == EEXIST)
            path = Fetch(filter, url, dir, name);
        else
            msg_Err(filter, "cannot create %s: %s", dir,
                    vlc_strerror_c(errno));
    }
    else
        msg_Err(filter, "model %s not found (automatic download disabled): "
                "set %s or enable %sdownload", name, path_option, CFG_PREFIX);

    free(dir);
    free(url);
    return path;
}
