#pragma once

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "live_session.h"
#include "renderer.h"

// Every tunable of a viewer run as one flat table, for the start menu and the
// settings file. Bounds only keep a value meaningful (a count cannot be
// negative, a field of view must be inside 0..180 degrees); otherwise any
// value, however extreme, can be set.
struct ViewerSettings {
    LiveSession::Settings session;
    RenderOptions render;
};

struct SettingEntry {
    const char* section;  // heading the entry is listed under
    const char* key;      // name in the settings file
    const char* label;
    const char* hint;     // short note shown next to the value ("" = none)
    std::variant<int*, float*, double*, bool*> value;
    double min = 0;
    double max = 0;   // +infinity = unbounded
    double step = 1;  // for the arrow keys
};

std::vector<SettingEntry> settingEntries(ViewerSettings& s);

bool isBool(const SettingEntry& e);
double settingValue(const SettingEntry& e);
// Clamps to the entry's bounds (and rounds integers); returns the value stored.
double setSettingValue(const SettingEntry& e, double v);
std::string formatSettingValue(const SettingEntry& e);
// Parses a typed number ("1e6", "-3", "0.25"); false if it is not one.
bool parseSettingValue(const std::string& text, double& v);

// key=value lines; unknown keys are ignored, missing ones keep their value.
// `recent` holds the last opened videos (UTF-8, newest first).
void loadSettings(const std::filesystem::path& file, ViewerSettings& s, std::vector<std::string>& recent);
void saveSettings(const std::filesystem::path& file, ViewerSettings& s, const std::vector<std::string>& recent);
