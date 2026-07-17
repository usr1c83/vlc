--[[
 gemma_translate.lua: friendly front-end for the Gemma 4 live translation
 audio filter.

 Adds a "Live Translation (Gemma 4)" entry to VLC's View menu. It shows a
 small dialog where the user picks a target language and whether they want
 subtitles and/or a spoken voice-over, then turns translation on for the
 video that is playing right now — no command line, no server, no digging
 through the advanced preferences.

 Under the hood it configures and enables the "gemma_transcribe" audio
 filter and the "transcript" sub source on the running output.

 Copyright (C) 2026 VLC authors and VideoLAN

 This program is free software; you can redistribute it and/or modify it
 under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software Foundation,
 Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
--]]

local dlg           -- the dialog, when open
local ui            -- table of widgets

-- Offered target languages. The first field is shown to the user, the
-- second is passed to the model (empty = transcribe in the original
-- language).
local languages = {
    { "English",              "English" },
    { "Español",              "Spanish" },
    { "Français",             "French" },
    { "Deutsch",              "German" },
    { "Italiano",             "Italian" },
    { "Português",            "Portuguese" },
    { "Русский",              "Russian" },
    { "Українська",           "Ukrainian" },
    { "Polski",               "Polish" },
    { "Türkçe",               "Turkish" },
    { "العربية",              "Arabic" },
    { "हिन्दी",                 "Hindi" },
    { "中文",                  "Chinese" },
    { "日本語",                "Japanese" },
    { "한국어",                "Korean" },
    { "— original (transcribe only) —", "" },
}

function descriptor()
    return {
        title = "Live Translation (Gemma 4)",
        version = "1.0",
        author = "VideoLAN",
        shortdesc = "Live Translation",
        description = "Translate the speech of the current video on the fly "
            .. "into live subtitles and an optional spoken voice-over, "
            .. "using the built-in Gemma 4 model.",
        capabilities = {},
    }
end

function activate()
    show_dialog()
end

function deactivate()
    if dlg then
        dlg:hide()
        dlg = nil
    end
end

function close()
    vlc.deactivate()
end

-- Whether our filter is currently active on the audio output.
local function is_running()
    local aout = vlc.object.aout()
    if not aout then
        return false
    end
    local af = vlc.var.get(aout, "audio-filter")
    return af ~= nil and string.find(af, "gemma_transcribe", 1, true) ~= nil
end

local function set_status(text)
    if ui and ui.status then
        ui.status:set_text("<i>" .. text .. "</i>")
    end
end

-- Enable the filter and sub source on the running output with the chosen
-- settings.
function start_translation()
    if not vlc.player or not vlc.player.item or not vlc.player.item() then
        set_status("Start playing a video first, then press Translate.")
        return
    end

    local lang = languages[ui.language:get_value()][2]
    local subs = ui.subs:get_checked()
    local voice = ui.voice:get_checked()
    local gpu = ui.gpu:get_checked()

    if not subs and not voice then
        set_status("Choose subtitles, voice-over, or both.")
        return
    end

    -- Configure the filter through the persistent settings; the filter
    -- reads them when it is created just below.
    vlc.config.set("gemma-transcribe-language", lang)
    vlc.config.set("gemma-transcribe-voiceover", voice)
    -- -1 offloads the whole model to the GPU (much faster, keeps up in
    -- real time); 0 keeps everything on the CPU.
    vlc.config.set("gemma-transcribe-gpu-layers", gpu and -1 or 0)

    -- Turn the audio filter on for the current output.
    local aout = vlc.object.aout()
    if aout then
        local af = vlc.var.get(aout, "audio-filter") or ""
        if not string.find(af, "gemma_transcribe", 1, true) then
            if af ~= "" then
                af = af .. ":gemma_transcribe"
            else
                af = "gemma_transcribe"
            end
            vlc.var.set(aout, "audio-filter", af)
        end
    end

    -- Turn the subtitle overlay on for the current video.
    if subs then
        local vout = vlc.object.vout()
        if vout then
            local ss = vlc.var.get(vout, "sub-source") or ""
            if not string.find(ss, "transcript", 1, true) then
                if ss ~= "" then
                    ss = ss .. ":transcript"
                else
                    ss = "transcript"
                end
                vlc.var.set(vout, "sub-source", ss)
            end
        end
    end

    local what = subs and voice and "subtitles and voice-over"
        or (voice and "voice-over" or "subtitles")
    set_status("Translating into " .. (lang ~= "" and lang or "the original "
        .. "language") .. " with " .. what .. ". The first line appears "
        .. "after a few seconds.")
end

-- Remove the filter and sub source again.
function stop_translation()
    local aout = vlc.object.aout()
    if aout then
        local af = vlc.var.get(aout, "audio-filter")
        if af then
            af = af:gsub("gemma_transcribe:?", ""):gsub(":$", "")
            vlc.var.set(aout, "audio-filter", af)
        end
    end

    local vout = vlc.object.vout()
    if vout then
        local ss = vlc.var.get(vout, "sub-source")
        if ss then
            ss = ss:gsub("transcript:?", ""):gsub(":$", "")
            vlc.var.set(vout, "sub-source", ss)
        end
    end

    set_status("Translation stopped.")
end

function show_dialog()
    dlg = vlc.dialog("Live Translation")
    ui = {}

    dlg:add_label("<b>Translate the speech of this video, live</b>",
                  1, 1, 4, 1)

    dlg:add_label("Translate speech to:", 1, 2, 1, 1)
    ui.language = dlg:add_dropdown(2, 2, 3, 1)
    for i, l in ipairs(languages) do
        ui.language:add_value(l[1], i)
    end

    ui.subs = dlg:add_check_box("Show subtitles", true, 1, 3, 2, 1)
    ui.voice = dlg:add_check_box("Speak a voice-over (dubbing)", false,
                                 3, 3, 2, 1)

    ui.gpu = dlg:add_check_box("Use the GPU (much faster — recommended)",
                               true, 1, 4, 4, 1)

    ui.status = dlg:add_label(" ", 1, 5, 4, 1)

    dlg:add_button("Translate", start_translation, 1, 6, 1, 1)
    dlg:add_button("Stop", stop_translation, 2, 6, 1, 1)
    dlg:add_button("Close", close, 4, 6, 1, 1)

    dlg:add_label("<small>Runs entirely on your computer with the bundled "
        .. "Gemma 4 model — no account and no server needed. If the "
        .. "subtitles lag on a slow computer, keep the GPU box ticked; "
        .. "without a GPU, translation stays on the CPU.</small>",
        1, 7, 4, 1)

    if is_running() then
        set_status("Translation is on.")
    end

    dlg:show()
end
