#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "live_session.h"
#include "options_table.h"
#include "renderer.h"

// Everything the viewer's menu edits: the playback and view settings, then
// the reconstruction options (options_table.h).
struct ViewerSettings {
    LiveSession::Settings session;
    RenderOptions render;
};

std::vector<SettingEntry> settingEntries(ViewerSettings& s);

// True for settings the view uses as they change (no restart needed).
bool appliesLive(const SettingEntry& e, const ViewerSettings& s);

// key=value lines; unknown keys are ignored, missing ones keep their value.
// `recent` holds the last opened videos (UTF-8, newest first).
void loadSettings(const std::filesystem::path& file, ViewerSettings& s, std::vector<std::string>& recent);
void saveSettings(const std::filesystem::path& file, ViewerSettings& s, const std::vector<std::string>& recent);
