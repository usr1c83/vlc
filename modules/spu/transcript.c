/*****************************************************************************
 * transcript.c : display live transcription subtitles
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
 * This sub source renders the live transcription text produced by the
 * "gemma_transcribe" audio filter (published on the libvlc instance through
 * the "gemma-transcript-text" string variable) as a subtitle overlay:
 *
 *   vlc --audio-filter=gemma_transcribe --sub-source=transcript input.mkv
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "common.h"

#include <vlc_common.h>
#include <vlc_configuration.h>
#include <vlc_plugin.h>
#include <vlc_filter.h>
#include <vlc_subpicture.h>
#include <vlc_variables.h>

#define GEMMA_TRANSCRIPT_VAR "gemma-transcript-text"

#define CFG_PREFIX "transcript-"

typedef struct
{
    vlc_mutex_t lock;

    char *message;   /* current transcription text */
    bool changed;
    bool shown;      /* a subpicture is (or was) on screen */

    int i_xoff, i_yoff;
    int i_pos;
    vlc_tick_t i_timeout;
    text_style_t *p_style;
} filter_sys_t;

/*****************************************************************************
 * Filter: emit a new subpicture whenever the transcription changes
 *****************************************************************************/
static subpicture_t *Filter( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    subpicture_t *p_spu = NULL;

    vlc_mutex_lock( &p_sys->lock );
    if( !p_sys->changed )
        goto out;

    bool b_empty = p_sys->message == NULL || p_sys->message[0] == '\0';
    if( b_empty && !p_sys->shown )
    {   /* nothing on screen to clear */
        p_sys->changed = false;
        goto out;
    }

    p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
        goto out;

    p_sys->changed = false;
    p_spu->i_start = date;
    p_spu->i_stop = p_sys->i_timeout == 0 ? VLC_TICK_INVALID
                                          : date + p_sys->i_timeout;
    p_spu->b_ephemer = true;

    if( b_empty )
    {   /* an empty subpicture replaces (clears) the previous text */
        p_sys->shown = false;
        goto out;
    }

    subpicture_region_t *p_region = subpicture_region_NewText();
    if( !p_region )
    {
        subpicture_Delete( p_spu );
        p_spu = NULL;
        goto out;
    }
    vlc_spu_regions_push( &p_spu->regions, p_region );
    p_region->fmt.i_sar_den = p_region->fmt.i_sar_num = 1;

    p_region->p_text = text_segment_New( p_sys->message );
    if( unlikely(p_region->p_text == NULL) )
    {
        subpicture_Delete( p_spu );
        p_spu = NULL;
        goto out;
    }
    p_region->p_text->style = text_style_Duplicate( p_sys->p_style );

    if( p_sys->i_pos < 0 )
    {
        p_region->i_align = SUBPICTURE_ALIGN_LEFT | SUBPICTURE_ALIGN_TOP;
        p_region->b_absolute = true;
    }
    else
    {
        p_region->i_align = p_sys->i_pos;
        p_region->b_absolute = false;
    }
    p_region->b_in_window = false;
    p_region->i_x = p_sys->i_xoff;
    p_region->i_y = p_sys->i_yoff;

    p_sys->shown = true;

out:
    vlc_mutex_unlock( &p_sys->lock );
    return p_spu;
}

/**********************************************************************
 * Callback invoked when the transcription filter publishes new text
 **********************************************************************/
static int TranscriptCallback( vlc_object_t *p_this, char const *psz_var,
                               vlc_value_t oldval, vlc_value_t newval,
                               void *p_data )
{
    filter_sys_t *p_sys = p_data;

    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);

    char *msg = (newval.psz_string != NULL) ? strdup(newval.psz_string)
                                            : NULL;

    vlc_mutex_lock( &p_sys->lock );
    free( p_sys->message );
    p_sys->message = msg;
    p_sys->changed = true;
    vlc_mutex_unlock( &p_sys->lock );

    return VLC_SUCCESS;
}

/*****************************************************************************
 * Open/Close
 *****************************************************************************/
static void DestroyFilter( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_object_t *vlc = VLC_OBJECT(vlc_object_instance(p_filter));

    var_DelCallback( vlc, GEMMA_TRANSCRIPT_VAR, TranscriptCallback, p_sys );
    var_Destroy( vlc, GEMMA_TRANSCRIPT_VAR );

    text_style_Delete( p_sys->p_style );
    free( p_sys->message );
    free( p_sys );
}

static const struct vlc_filter_operations filter_ops = {
    .source_sub = Filter, .close = DestroyFilter,
};

static int CreateFilter( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys = calloc( 1, sizeof( *p_sys ) );
    if( unlikely(p_sys == NULL) )
        return VLC_ENOMEM;

    p_sys->p_style = text_style_Create( STYLE_NO_DEFAULTS );
    if( unlikely(p_sys->p_style == NULL) )
    {
        free( p_sys );
        return VLC_ENOMEM;
    }
    vlc_mutex_init( &p_sys->lock );

    static const char *const options[] = {
        "x", "y", "position", "opacity", "color", "size", "timeout", NULL
    };
    config_ChainParse( p_filter, CFG_PREFIX, options, p_filter->p_cfg );

    p_sys->i_xoff = var_InheritInteger( p_filter, CFG_PREFIX "x" );
    p_sys->i_yoff = var_InheritInteger( p_filter, CFG_PREFIX "y" );
    p_sys->i_pos = var_InheritInteger( p_filter, CFG_PREFIX "position" );
    p_sys->i_timeout = VLC_TICK_FROM_MS(var_InheritInteger( p_filter,
                                                    CFG_PREFIX "timeout" ));

    p_sys->p_style->i_font_alpha = var_InheritInteger( p_filter,
                                                       CFG_PREFIX "opacity" );
    p_sys->p_style->i_features |= STYLE_HAS_FONT_ALPHA;
    p_sys->p_style->i_font_color = var_InheritInteger( p_filter,
                                                       CFG_PREFIX "color" );
    p_sys->p_style->i_features |= STYLE_HAS_FONT_COLOR;
    p_sys->p_style->i_font_size = var_InheritInteger( p_filter,
                                                      CFG_PREFIX "size" );

    /* Subscribe to the text published by the transcription audio filter */
    vlc_object_t *vlc = VLC_OBJECT(vlc_object_instance(p_filter));
    var_Create( vlc, GEMMA_TRANSCRIPT_VAR, VLC_VAR_STRING );
    var_AddCallback( vlc, GEMMA_TRANSCRIPT_VAR, TranscriptCallback, p_sys );

    char *msg = var_GetNonEmptyString( vlc, GEMMA_TRANSCRIPT_VAR );
    if( msg != NULL )
    {
        p_sys->message = msg;
        p_sys->changed = true;
    }

    p_filter->ops = &filter_ops;

    return VLC_SUCCESS;
}

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/

#define TIMEOUT_TEXT N_("Timeout (ms)")
#define TIMEOUT_LONGTEXT N_("Number of milliseconds a transcription line " \
    "stays on screen when no newer text arrives. 0 keeps the last line " \
    "displayed forever.")

#define SIZE_TEXT N_("Font size, pixels")
#define SIZE_LONGTEXT N_("Font size, in pixels. Default is 0 (use default " \
    "font size).")

#define COLOR_TEXT N_("Color")
#define COLOR_LONGTEXT N_("Color of the rendered text, as an hexadecimal " \
    "RGB value (like HTML colors).")

#define TRANSCRIPT_HELP N_("Display live transcription subtitles produced " \
    "by the Gemma transcription audio filter")

vlc_module_begin ()
    set_shortname( N_("Transcript") )
    set_description( N_("Live transcription subtitle display") )
    set_help( TRANSCRIPT_HELP )
    set_callback_sub_source( CreateFilter, 0 )
    set_subcategory( SUBCAT_VIDEO_SUBPIC )

    set_section( N_("Position"), NULL )
    add_integer( CFG_PREFIX "x", 0, POSX_TEXT, POSX_LONGTEXT )
    add_integer( CFG_PREFIX "y", 0, POSY_TEXT, POSY_LONGTEXT )
    add_integer( CFG_PREFIX "position", SUBPICTURE_ALIGN_BOTTOM,
                 POS_TEXT, POS_LONGTEXT )
        change_integer_list( pi_pos_values, ppsz_pos_descriptions )

    set_section( N_("Font"), NULL )
    add_integer_with_range( CFG_PREFIX "opacity", 255, 0, 255,
        OPACITY_TEXT, OPACITY_LONGTEXT )
    add_rgb( CFG_PREFIX "color", 0xFFFFFF, COLOR_TEXT, COLOR_LONGTEXT )
    add_integer( CFG_PREFIX "size", 0, SIZE_TEXT, SIZE_LONGTEXT )
        change_integer_range( 0, 4096 )

    set_section( N_("Misc"), NULL )
    add_integer( CFG_PREFIX "timeout", 12000, TIMEOUT_TEXT, TIMEOUT_LONGTEXT )

    add_shortcut( "gemma_transcript" )
vlc_module_end ()
