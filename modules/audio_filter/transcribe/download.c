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

/**
 * Downloads url into dir/name, showing a progress dialog.
 * Returns the file path on success.
 */
static char *Fetch(filter_t *filter, const char *url, const char *dir,
                   const char *name)
{
    char *dest, *part = NULL;
    stream_t *stream = NULL;
    FILE *out = NULL;
    vlc_dialog_id *dialog = NULL;

    if (asprintf(&dest, "%s" DIR_SEP "%s", dir, name) < 0)
        return NULL;
    if (asprintf(&part, "%s.part", dest) < 0)
        goto error;

    stream = vlc_stream_NewURL(filter, url);
    if (stream == NULL)
    {
        msg_Err(filter, "cannot fetch model from %s", url);
        goto error;
    }

    uint64_t total = 0;
    vlc_stream_GetSize(stream, &total);

    out = vlc_fopen(part, "wb");
    if (out == NULL)
    {
        msg_Err(filter, "cannot write %s: %s", part, vlc_strerror_c(errno));
        goto error;
    }

    msg_Info(filter, "downloading %s (%.1f MiB) to %s", url,
             total / 1048576., dest);
    dialog = vlc_dialog_display_progress(filter, total == 0, 0.f,
                                         _("Cancel"),
                                         _("Downloading model"),
                                         _("Downloading the speech "
                                           "transcription model %s…"), name);

    uint64_t done = 0, logged = 0;
    for (;;)
    {
        char buf[65536];
        ssize_t val = vlc_stream_Read(stream, buf, sizeof (buf));
        if (val < 0)
        {
            msg_Err(filter, "download of %s failed", url);
            goto error;
        }
        if (val == 0)
            break;

        if (fwrite(buf, 1, val, out) != (size_t)val)
        {
            msg_Err(filter, "cannot write %s: %s", part,
                    vlc_strerror_c(errno));
            goto error;
        }
        done += val;

        if (dialog != NULL)
        {
            if (vlc_dialog_is_cancelled(filter, dialog))
            {
                msg_Warn(filter, "model download cancelled");
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

    if (total > 0 && done < total)
    {
        msg_Err(filter, "truncated download of %s (%"PRIu64" / %"PRIu64")",
                url, done, total);
        goto error;
    }

    if (fclose(out))
    {
        out = NULL;
        goto error;
    }
    out = NULL;

    if (vlc_rename(part, dest))
    {
        msg_Err(filter, "cannot rename %s: %s", part, vlc_strerror_c(errno));
        goto error;
    }

    msg_Info(filter, "model saved as %s", dest);
    if (dialog != NULL)
        vlc_dialog_release(filter, dialog);
    vlc_stream_Delete(stream);
    free(part);
    return dest;

error:
    if (dialog != NULL)
        vlc_dialog_release(filter, dialog);
    if (out != NULL)
        fclose(out);
    if (part != NULL)
        vlc_unlink(part);
    if (stream != NULL)
        vlc_stream_Delete(stream);
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
