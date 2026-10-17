#pragma once

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "videogammetry/reconstructor.h"

// Every reconstruction option as one flat table, shared by the viewer's menu
// and settings file and by the CLI (--config). Bounds only keep a value
// meaningful (a count cannot be negative, a field of view must be inside
// 0..180 degrees); otherwise any value, however extreme, can be set.
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

std::vector<SettingEntry> optionEntries(vg::Reconstructor::Options& o);

// Full explanation of a setting (by key): what it does, what raising and
// lowering it changes, what extreme values do. "" for an unknown key.
const char* settingDetails(const char* key);

bool isBool(const SettingEntry& e);
double settingValue(const SettingEntry& e);
// Clamps to the entry's bounds (and rounds integers); returns the value stored.
double setSettingValue(const SettingEntry& e, double v);
std::string formatSettingValue(const SettingEntry& e);
// Parses a typed number ("1e6", "-3", "0.25"); false if it is not one.
bool parseSettingValue(const std::string& text, double& v);

// Applies one "key=value" line to the matching entry; false if no entry took
// it. `key` / `value` (optional) receive the split line ("" if it has no '=').
bool applySettingLine(const std::string& line, const std::vector<SettingEntry>& entries, std::string* key,
                      std::string* value);
// Reads a settings file (e.g. the viewer's viewer.ini); unknown keys are
// ignored, missing ones keep their value. False if it cannot be opened.
bool loadOptionsFile(const std::filesystem::path& file, vg::Reconstructor::Options& options);
