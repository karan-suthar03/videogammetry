// Live viewer for Windows: plays a video through the reconstruction and
// shows the point cloud, keyframes and current camera as they are built.
// Starts on a menu (open a video, recent videos, every setting); Esc returns
// to it. Plain Win32 + GDI; the 3D view is rendered on the CPU (renderer.h).

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <timeapi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "live_session.h"
#include "power.h"
#include "renderer.h"
#include "settings_table.h"
#include "videogammetry/version.h"

namespace {

constexpr UINT_PTR kTimer = 1;
constexpr wchar_t kClassName[] = L"VideogammetryViewer";

constexpr std::size_t kMaxRecent = 6;

// One selectable line of the start menu.
struct MenuItem {
    enum class Kind { Open, Resume, Restart, Recent, Reset, Quit, Setting } kind;
    int index = 0;  // into App::recent or App::entries
};

struct App {
    HWND hwnd = nullptr;
    ViewerSettings cfg;
    std::vector<SettingEntry> entries = settingEntries(cfg);
    std::vector<std::string> recent;  // UTF-8 paths, newest first
    std::filesystem::path settingsFile;
    std::unique_ptr<LiveSession> session;
    std::wstring videoPath;
    std::shared_ptr<const LiveSnapshot> scene = std::make_shared<LiveSnapshot>();

    OrbitView view;
    SceneRenderer renderer;
    bool dirty = true;
    std::wstring notice;  // last save result etc.

    // Back buffer: a DIB section selected into a memory DC.
    HDC memDC = nullptr;
    HBITMAP bitmap = nullptr;
    HGDIOBJ oldBitmap = nullptr;
    std::uint32_t* pixels = nullptr;
    int width = 0, height = 0;
    HFONT font = nullptr, titleFont = nullptr;
    int fontDpi = 0;

    // Start menu.
    bool menu = true;
    int selected = 0;      // into menuItems()
    int scroll = 0;        // first settings line shown
    bool editing = false;  // typing a value for the selected setting
    bool editFresh = false;  // the first key typed replaces the shown value
    std::wstring editText;
    std::vector<std::pair<RECT, int>> hits;  // clickable rows of the last paint -> item

    // Mouse drag.
    enum class Drag { None, Orbit, Pan } drag = Drag::None;
    POINT last{};
};

App g;

std::string toUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring fromUtf8(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

void updateTitle() {
    std::wstring title = L"Videogammetry viewer";
    if (!g.videoPath.empty()) title += L" - " + std::filesystem::path(g.videoPath).filename().wstring();
    SetWindowTextW(g.hwnd, title.c_str());
}

void saveConfig() {
    if (!g.settingsFile.empty()) saveSettings(g.settingsFile, g.cfg, g.recent);
}

void startSession(const std::wstring& path) {
    g.session.reset();  // stops the previous one
    g.menu = false;
    g.editing = false;
    const std::string utf8 = toUtf8(path);
    std::erase(g.recent, utf8);
    g.recent.insert(g.recent.begin(), utf8);
    if (g.recent.size() > kMaxRecent) g.recent.resize(kMaxRecent);
    saveConfig();
    g.videoPath = path;
    g.session = std::make_unique<LiveSession>(toUtf8(path), g.cfg.session);
    g.scene = std::make_shared<LiveSnapshot>();
    g.view.reset();
    g.notice.clear();
    g.dirty = true;
    updateTitle();
}

void openVideoDialog() {
    wchar_t file[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g.hwnd;
    ofn.lpstrFilter = L"Videos\0*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = L"Open a video";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&ofn)) startSession(file);
}

void savePlyDialog() {
    if (!g.session || !g.session->finished()) {
        g.notice = L"The point cloud can be saved once the video is done.";
        g.dirty = true;
        return;
    }
    const std::filesystem::path video(g.videoPath);
    std::wstring name = video.stem().wstring() + L"_cloud.ply";
    wchar_t file[MAX_PATH];
    wcsncpy_s(file, name.c_str(), _TRUNCATE);
    const std::wstring dir = video.parent_path().wstring();
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g.hwnd;
    ofn.lpstrFilter = L"PLY point cloud\0*.ply\0";
    ofn.lpstrDefExt = L"ply";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrInitialDir = dir.c_str();
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameW(&ofn)) return;
    std::size_t count = 0;
    const bool ok = g.session->savePly(std::filesystem::path(file).string(), count);
    g.notice = ok ? L"Saved " + std::to_wstring(count) + L" points to " + file : std::wstring(L"Could not write ") + file;
    g.dirty = true;
}

void ensureBackBuffer(int w, int h) {
    if (w == g.width && h == g.height && g.memDC) return;
    if (g.memDC) {
        SelectObject(g.memDC, g.oldBitmap);
        DeleteObject(g.bitmap);
        DeleteDC(g.memDC);
    }
    g.width = w;
    g.height = h;
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC screen = GetDC(g.hwnd);
    g.memDC = CreateCompatibleDC(screen);
    void* bits = nullptr;
    g.bitmap = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(g.hwnd, screen);
    g.pixels = static_cast<std::uint32_t*>(bits);
    g.oldBitmap = SelectObject(g.memDC, g.bitmap);
}

void ensureFont() {
    const int dpi = static_cast<int>(GetDpiForWindow(g.hwnd));
    if (g.font && dpi == g.fontDpi) return;
    if (g.font) DeleteObject(g.font);
    if (g.titleFont) DeleteObject(g.titleFont);
    g.fontDpi = dpi;
    g.font = CreateFontW(-MulDiv(10, dpi, 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                         OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
    g.titleFont = CreateFontW(-MulDiv(20, dpi, 72), 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
}

const wchar_t* stateName(vg::Reconstructor::State s) {
    switch (s) {
        case vg::Reconstructor::State::Initializing: return L"initializing";
        case vg::Reconstructor::State::Tracking: return L"tracking";
        case vg::Reconstructor::State::Lost: return L"LOST";
    }
    return L"?";
}

void drawText(HDC dc, int x, int y, const std::wstring& text, COLORREF color) {
    SetTextColor(dc, RGB(0, 0, 0));
    TextOutW(dc, x + 1, y + 1, text.c_str(), static_cast<int>(text.size()));
    SetTextColor(dc, color);
    TextOutW(dc, x, y, text.c_str(), static_cast<int>(text.size()));
}

void drawHud(HDC dc) {
    ensureFont();
    HGDIOBJ oldFont = SelectObject(dc, g.font);
    SetBkMode(dc, TRANSPARENT);
    TEXTMETRICW tm{};
    GetTextMetricsW(dc, &tm);
    const int lineH = tm.tmHeight + 2;
    const int margin = 12;
    const LiveSnapshot& s = *g.scene;
    const COLORREF white = RGB(235, 235, 235), dim = RGB(160, 165, 175);
    wchar_t buf[512];
    int y = margin;
    const auto line = [&](const std::wstring& text, COLORREF color) {
        drawText(dc, margin, y, text, color);
        y += lineH;
    };

    if (!g.session) {
        line(L"Press O to open a video.", white);
    } else {
        line(std::filesystem::path(g.videoPath).filename().wstring(), white);
        if (s.frameCount > 0) {
            swprintf(buf, 512, L"frame %ld / %ld   t = %.2f s   [%s]", s.frame, s.frameCount, s.timestamp,
                     stateName(s.state));
        } else {
            swprintf(buf, 512, L"frame %ld   t = %.2f s   [%s]", s.frame, s.timestamp, stateName(s.state));
        }
        line(buf, s.state == vg::Reconstructor::State::Lost ? RGB(255, 110, 110) : white);
        swprintf(buf, 512, L"map: %zu points   %zu keyframes   %d tracked in this frame%s", s.points.size(),
                 s.keyframes.size(), s.trackedCount,
                 s.maps > 1 ? (L"   (map " + std::to_wstring(s.maps) + L", tracking was lost)").c_str() : L"");
        line(buf, white);
        const wchar_t* playback = g.session->paused() ? L"PAUSED" : (g.session->realTime() ? L"real-time" : L"max speed");
        if (s.finished) playback = L"finished";
        swprintf(buf, 512, L"tracking + mapping: %.0f fps   playback: %s   features: %s", s.processingFps, playback,
                 fromUtf8(s.extractor).c_str());
        line(buf, dim);
        if (s.fovDeg > 0) {
            if (s.focalUpdates > 0) {
                swprintf(buf, 512, L"field of view: %.1f deg, lens distortion k1 %.3f (estimated, %d refinements; "
                         L"started from %.1f deg)", s.fovDeg, s.k1, s.focalUpdates, s.startFovDeg);
            } else {
                swprintf(buf, 512, L"field of view: %.1f deg (assumed; the motion has not determined it yet)", s.fovDeg);
            }
            line(buf, s.focalUpdates > 0 ? RGB(150, 230, 150) : dim);
        }
        if (!s.message.empty()) line(fromUtf8(s.message), RGB(255, 210, 120));
        if (!g.notice.empty()) line(g.notice, RGB(140, 220, 255));
    }

    const wchar_t* help[] = {
        L"left-drag orbit   right-drag pan   wheel zoom   C look through camera   F follow camera   R reset view",
        L"Space pause   M real-time/max speed   T highlight tracked   N noise colors   K keyframes   +/- point size   O open   S save .ply   Esc menu",
        L"red: current camera   blue: keyframes   orange: camera path   green: points tracked in this frame",
    };
    int hy = g.height - margin - lineH * 3;
    for (const wchar_t* h : help) {
        drawText(dc, margin, hy, h, dim);
        hy += lineH;
    }
    swprintf(buf, 512, L"view: %s", g.view.attached ? L"through the current camera (C or drag to leave)"
                                                      : (g.view.follow ? L"orbit, following the camera" : L"orbit"));
    drawText(dc, margin, hy - lineH * 4, buf, dim);
    SelectObject(dc, oldFont);
}

// ---- Start menu --------------------------------------------------------------

std::vector<MenuItem> menuItems() {
    using K = MenuItem::Kind;
    std::vector<MenuItem> items{{K::Open}};
    if (g.session) items.push_back({K::Resume});
    if (!g.videoPath.empty()) items.push_back({K::Restart});
    for (int i = 0; i < static_cast<int>(g.recent.size()); ++i) items.push_back({K::Recent, i});
    items.push_back({K::Reset});
    items.push_back({K::Quit});
    for (int i = 0; i < static_cast<int>(g.entries.size()); ++i) items.push_back({K::Setting, i});
    return items;
}

const SettingEntry* selectedSetting() {
    const std::vector<MenuItem> items = menuItems();
    if (g.selected < 0 || g.selected >= static_cast<int>(items.size())) return nullptr;
    const MenuItem& it = items[g.selected];
    return it.kind == MenuItem::Kind::Setting ? &g.entries[it.index] : nullptr;
}

void showMenu() {
    if (g.menu) return;
    g.menu = true;
    if (g.session) g.session->setPaused(true);  // Resume continues it
    g.selected = 0;
    g.dirty = true;
}

void beginEdit(const SettingEntry& e) {
    g.editing = true;
    g.editFresh = true;
    g.editText = fromUtf8(formatSettingValue(e));
}

void commitEdit() {
    if (!g.editing) return;
    g.editing = false;
    const SettingEntry* e = selectedSetting();
    if (!e) return;
    std::wstring text = g.editText;
    for (wchar_t& c : text) c = static_cast<wchar_t>(std::towlower(c));
    if (text == L"max") text = L"inf";
    if (text == L"min") text = L"-inf";
    double v;
    if (!parseSettingValue(toUtf8(text), v)) {
        g.notice = L"\"" + g.editText + L"\" is not a number.";
        return;
    }
    const double stored = setSettingValue(*e, v);
    g.notice = stored == v ? std::wstring()
                           : fromUtf8(std::string(e->label) + " set to " + formatSettingValue(*e) +
                                      " (the nearest allowed value)");
    saveConfig();
}

void adjustSetting(const SettingEntry& e, int dir) {
    const double v = settingValue(e);
    if (isBool(e)) {
        setSettingValue(e, v != 0 ? 0 : 1);
    } else if (GetKeyState(VK_CONTROL) < 0) {  // double / halve: reaches extremes fast
        setSettingValue(e, v == 0 ? dir * e.step : (dir > 0 ? v * 2 : v / 2));
    } else {
        setSettingValue(e, v + dir * e.step * (GetKeyState(VK_SHIFT) < 0 ? 10 : 1));
    }
    g.notice.clear();
    saveConfig();
}

void activate(const MenuItem& it) {
    using K = MenuItem::Kind;
    switch (it.kind) {
        case K::Open: openVideoDialog(); break;
        case K::Resume:
            g.menu = false;
            g.session->setPaused(false);
            break;
        case K::Restart: startSession(g.videoPath); break;
        case K::Recent: {
            const std::wstring path = fromUtf8(g.recent[it.index]);
            if (std::filesystem::exists(path)) {
                startSession(path);
            } else {
                g.notice = L"Not found: " + path;
                g.recent.erase(g.recent.begin() + it.index);
                saveConfig();
            }
            break;
        }
        case K::Reset:
            g.cfg = ViewerSettings{};  // entries point into g.cfg, so they stay valid
            g.notice = L"All settings reset to their defaults.";
            saveConfig();
            break;
        case K::Quit: DestroyWindow(g.hwnd); return;
        case K::Setting: {
            const SettingEntry& e = g.entries[it.index];
            if (isBool(e)) {
                adjustSetting(e, 1);
            } else {
                beginEdit(e);
            }
            break;
        }
    }
    g.dirty = true;
}

void onMenuKey(WPARAM key) {
    if (g.editing) {
        if (key == VK_RETURN) {
            commitEdit();
        } else if (key == VK_ESCAPE) {
            g.editing = false;
        } else if (key == VK_BACK) {
            if (g.editFresh) g.editText.clear();
            if (!g.editText.empty()) g.editText.pop_back();
            g.editFresh = false;
        } else {
            return;
        }
        g.dirty = true;
        return;
    }
    const std::vector<MenuItem> items = menuItems();
    const int last = static_cast<int>(items.size()) - 1;
    const SettingEntry* e = selectedSetting();
    switch (key) {
        case VK_UP: g.selected = std::max(g.selected - 1, 0); break;
        case VK_DOWN: g.selected = std::min(g.selected + 1, last); break;
        case VK_PRIOR: g.selected = std::max(g.selected - 10, 0); break;
        case VK_NEXT: g.selected = std::min(g.selected + 10, last); break;
        case VK_HOME: g.selected = 0; break;
        case VK_END: g.selected = last; break;
        case VK_LEFT:
        case VK_RIGHT:
            if (e) adjustSetting(*e, key == VK_RIGHT ? 1 : -1);
            break;
        case VK_RETURN:
        case VK_SPACE: activate(items[g.selected]); return;
        case VK_ESCAPE:
            if (g.session) {
                activate({MenuItem::Kind::Resume});
            } else {
                DestroyWindow(g.hwnd);
            }
            return;
        case 'O':
            if (GetKeyState(VK_CONTROL) < 0 || !e) openVideoDialog();
            break;
        default: return;
    }
    g.dirty = true;
}

void onMenuChar(wchar_t c) {
    if (c < 0x20 || c == 0x7f) return;  // Enter, Esc, Backspace: see onMenuKey
    if (!g.editing) {
        // Typing a number on a numeric setting starts entering a new value.
        const SettingEntry* e = selectedSetting();
        if (!e || isBool(*e) || !(std::iswdigit(c) || c == L'-' || c == L'.')) return;
        beginEdit(*e);
    }
    if (g.editFresh) g.editText.clear();
    g.editFresh = false;
    if (g.editText.size() < 40) g.editText += c;
    g.dirty = true;
}

void onMenuClick(POINT p) {
    for (const auto& [rc, item] : g.hits) {
        if (!PtInRect(&rc, p)) continue;
        const bool same = item == g.selected;
        commitEdit();
        g.selected = item;
        if (!(same && g.editing)) activate(menuItems()[item]);
        g.dirty = true;
        return;
    }
    commitEdit();
    g.dirty = true;
}

void drawMenu(HDC dc, int w, int h) {
    ensureFont();
    RECT all{0, 0, w, h};
    HBRUSH bg = CreateSolidBrush(RGB(22, 24, 28));
    FillRect(dc, &all, bg);
    DeleteObject(bg);
    SetBkMode(dc, TRANSPARENT);
    g.hits.clear();

    const COLORREF white = RGB(235, 235, 235), dim = RGB(150, 155, 165), accent = RGB(140, 220, 255),
                   changed = RGB(255, 200, 110);
    HGDIOBJ oldFont = SelectObject(dc, g.titleFont);
    const int margin = MulDiv(24, g.fontDpi, 96);
    SetTextColor(dc, white);
    TextOutW(dc, margin, margin, L"Videogammetry", 13);
    TEXTMETRICW tm{};
    GetTextMetricsW(dc, &tm);
    int top = margin + tm.tmHeight + margin / 2;

    SelectObject(dc, g.font);
    GetTextMetricsW(dc, &tm);
    const int lineH = tm.tmHeight + 2, rowH = lineH + MulDiv(6, g.fontDpi, 96), cw = tm.tmAveCharWidth;
    const int pad = cw;
    HBRUSH selBrush = CreateSolidBrush(RGB(48, 60, 84)), editBrush = CreateSolidBrush(RGB(70, 60, 30));
    const auto text = [&](int x, int y, const std::wstring& s, COLORREF c) {
        SetTextColor(dc, c);
        TextOutW(dc, x, y, s.c_str(), static_cast<int>(s.size()));
    };
    const auto row = [&](int item, RECT rc) {
        g.hits.push_back({rc, item});
        if (item == g.selected) FillRect(dc, &rc, g.editing ? editBrush : selBrush);
    };

    const std::vector<MenuItem> items = menuItems();
    g.selected = std::clamp(g.selected, 0, static_cast<int>(items.size()) - 1);
    const std::wstring version = L"version " + fromUtf8(vg::version());
    text(margin, top, version, dim);
    top += lineH + margin / 2;

    // Left column: actions and recent videos.
    const int colW = 52 * cw;
    const std::wstring current = std::filesystem::path(g.videoPath).filename().wstring();
    int y = top;
    bool recentHeader = false;
    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
        const MenuItem& it = items[i];
        if (it.kind == MenuItem::Kind::Setting) break;
        std::wstring label;
        switch (it.kind) {
            case MenuItem::Kind::Open: label = L"Open video...                 O"; break;
            case MenuItem::Kind::Resume: label = L"Resume " + current + L"         Esc"; break;
            case MenuItem::Kind::Restart: label = L"Restart " + current + L" with these settings"; break;
            case MenuItem::Kind::Recent: label = std::filesystem::path(fromUtf8(g.recent[it.index])).filename().wstring(); break;
            case MenuItem::Kind::Reset: label = L"Reset all settings to defaults"; break;
            case MenuItem::Kind::Quit: label = g.session ? L"Quit" : L"Quit                          Esc"; break;
            case MenuItem::Kind::Setting: break;
        }
        if (it.kind == MenuItem::Kind::Recent && !recentHeader) {
            y += lineH / 2;
            text(margin, y, L"Recent videos", accent);
            y += lineH + 2;
            recentHeader = true;
        }
        if (it.kind == MenuItem::Kind::Reset) y += lineH / 2;
        row(i, {margin, y, margin + colW, y + rowH});
        text(margin + pad, y + (rowH - lineH) / 2, label, white);
        y += rowH;
    }

    // Under the actions: the full explanation of the selected line.
    {
        static ViewerSettings explainDefaults;
        static const std::vector<SettingEntry> explainEntries = settingEntries(explainDefaults);
        const MenuItem& sel = items[g.selected];
        std::wstring title, body, facts;
        switch (sel.kind) {
            case MenuItem::Kind::Open:
                title = L"Open video";
                body = L"Pick a video file (mp4, mov, mkv, avi, webm, m4v) and reconstruct it with the settings on "
                       L"the right. The point cloud, keyframes and camera path are drawn as they are built; Esc "
                       L"comes back here.";
                break;
            case MenuItem::Kind::Resume:
                title = L"Resume";
                body = L"Back to the video that was playing, where it was paused. Setting changes made since do not "
                       L"apply to it, except the view settings (point size, highlighting, noise colors, keyframes); use Restart "
                       L"for the others.";
                break;
            case MenuItem::Kind::Restart:
                title = L"Restart with these settings";
                body = L"Reconstructs " + current + L" again from its first frame with the settings as they are now. "
                       L"The current reconstruction is discarded.";
                break;
            case MenuItem::Kind::Recent:
                title = L"Recent video";
                body = fromUtf8(g.recent[sel.index]) + L"\n\nOpens it again with the current settings.";
                break;
            case MenuItem::Kind::Reset:
                title = L"Reset all settings";
                body = L"Puts every setting on the right back to its default value (the values the benchmark is "
                       L"tuned with). Recent videos are kept.";
                break;
            case MenuItem::Kind::Quit:
                title = L"Quit";
                body = L"Closes the viewer. Settings and recent videos are already saved.";
                break;
            case MenuItem::Kind::Setting: {
                const SettingEntry& e = g.entries[sel.index];
                title = fromUtf8(e.label);
                body = fromUtf8(settingDetails(e.key));
                const std::string def = formatSettingValue(explainEntries[sel.index]);
                std::string range;
                if (!isBool(e)) {
                    char buf[96];
                    if (std::isinf(e.max)) {
                        std::snprintf(buf, sizeof(buf), "   Allowed: %.6g and up", e.min);
                    } else {
                        std::snprintf(buf, sizeof(buf), "   Allowed: %.6g to %.6g", e.min, e.max);
                    }
                    range = buf;
                }
                facts = fromUtf8("Now: " + formatSettingValue(e) + "   Default: " + def + range) + L"\n" +
                        (appliesLive(e, g.cfg) ? L"Takes effect immediately."
                                               : L"Takes effect when a video is opened or restarted.");
                break;
            }
        }
        y += lineH;
        RECT box{margin, y, margin + colW, h - margin - lineH * 5};
        if (box.bottom - box.top > lineH * 3) {
            HBRUSH panel = CreateSolidBrush(RGB(32, 35, 42));
            FillRect(dc, &box, panel);
            DeleteObject(panel);
            RECT r{box.left + pad, box.top + lineH / 2, box.right - pad, box.bottom - lineH / 2};
            text(r.left, r.top, title, accent);
            r.top += lineH + lineH / 2;
            if (!facts.empty()) {
                SetTextColor(dc, changed);
                RECT fr = r;
                r.top += DrawTextW(dc, facts.c_str(), -1, &fr, DT_WORDBREAK | DT_NOPREFIX) + lineH / 2;
            }
            SetTextColor(dc, white);
            DrawTextW(dc, body.c_str(), -1, &r, DT_WORDBREAK | DT_NOPREFIX | DT_EDITCONTROL);
        }
    }

    // Right column: every setting, grouped, scrolling with the selection.
    const int x1 = margin * 2 + colW, footerH = lineH * 4 + margin;
    const int visible = std::max(1, (h - top - footerH) / rowH);
    struct Line { const char* header; int item; };
    std::vector<Line> lines;
    const char* section = nullptr;
    int selLine = -1;
    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
        if (items[i].kind != MenuItem::Kind::Setting) continue;
        const SettingEntry& e = g.entries[items[i].index];
        if (!section || std::string(section) != e.section) {
            section = e.section;
            lines.push_back({section, -1});
        }
        if (i == g.selected) selLine = static_cast<int>(lines.size());
        lines.push_back({nullptr, i});
    }
    if (selLine >= 0) {
        if (selLine < g.scroll + 1) g.scroll = std::max(0, selLine - 1);  // keep its heading in view too
        if (selLine >= g.scroll + visible) g.scroll = selLine - visible + 1;
    }
    g.scroll = std::clamp(g.scroll, 0, std::max(0, static_cast<int>(lines.size()) - visible));

    static ViewerSettings defaults;
    static const std::vector<SettingEntry> defaultEntries = settingEntries(defaults);
    const int labelW = 34 * cw, valueW = 16 * cw;
    y = top;
    for (int l = g.scroll; l < std::min<int>(static_cast<int>(lines.size()), g.scroll + visible); ++l) {
        if (lines[l].header) {
            text(x1, y + (rowH - lineH) / 2, fromUtf8(lines[l].header), accent);
            y += rowH;
            continue;
        }
        const int item = lines[l].item;
        const int idx = items[item].index;
        const SettingEntry& e = g.entries[idx];
        row(item, {x1, y, w - margin, y + rowH});
        const int ty = y + (rowH - lineH) / 2;
        text(x1 + pad, ty, fromUtf8(e.label), white);
        const bool isChanged = settingValue(e) != settingValue(defaultEntries[idx]);
        std::wstring value = fromUtf8(formatSettingValue(e));
        if (item == g.selected && g.editing) value = g.editText + L"_";
        text(x1 + pad + labelW, ty, value, isChanged || (item == g.selected && g.editing) ? changed : white);
        if (*e.hint) text(x1 + pad + labelW + valueW, ty, fromUtf8(e.hint), dim);
        y += rowH;
    }
    if (g.scroll > 0) text(w - margin - 3 * cw, top - lineH, L"...", dim);
    if (g.scroll + visible < static_cast<int>(lines.size())) text(w - margin - 3 * cw, y, L"...", dim);

    // Footer.
    int fy = h - margin - lineH * 3;
    if (!g.notice.empty()) text(margin, fy - lineH, g.notice, accent);
    text(margin, fy, L"Up/Down select   Enter open / toggle / type a value   Esc " +
                         std::wstring(g.session ? L"back to the video" : L"quit"), dim);
    fy += lineH;
    text(margin, fy, L"Left/Right change a value   Shift x10 step   Ctrl double / halve   wheel scroll", dim);
    fy += lineH;
    text(margin, fy, L"Any number can be typed (1e6, 0.001, max, min); values only stop at what the setting allows. "
                     L"Changed values are orange.", dim);
    DeleteObject(selBrush);
    DeleteObject(editBrush);
    SelectObject(dc, oldFont);
}

// ---- Viewer ------------------------------------------------------------------

void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    const int w = std::max<int>(1, rc.right - rc.left), h = std::max<int>(1, rc.bottom - rc.top);
    ensureBackBuffer(w, h);
    if (g.menu) {
        drawMenu(g.memDC, w, h);
        BitBlt(dc, 0, 0, w, h, g.memDC, 0, 0, SRCCOPY);
        EndPaint(hwnd, &ps);
        g.dirty = false;
        return;
    }

    Framebuffer fb{g.pixels, w, h, w};
    g.renderer.render(*g.scene, g.view, g.cfg.render, fb);
    const RgbImage& thumb = g.scene->thumbnail;
    if (thumb.width > 0 && thumb.width + 24 < w) blitImage(thumb, w - thumb.width - 12, 12, fb);
    GdiFlush();
    drawHud(g.memDC);

    BitBlt(dc, 0, 0, w, h, g.memDC, 0, 0, SRCCOPY);
    EndPaint(hwnd, &ps);
    g.dirty = false;
}

void onTimer() {
    if (g.menu) {
        if (g.dirty) InvalidateRect(g.hwnd, nullptr, FALSE);
        return;
    }
    if (g.session) {
        std::shared_ptr<const LiveSnapshot> latest = g.session->latest();
        if (latest->version != g.scene->version) {
            g.scene = std::move(latest);
            g.dirty = true;
        }
    }
    if (g.view.follow && !g.scene->keyframes.empty()) {
        const vg::Vec3 before = g.view.target;
        g.view.update(*g.scene);
        if (vg::norm(g.view.target - before) > 1e-4 * g.scene->sceneScale) g.dirty = true;
    }
    if (g.dirty) InvalidateRect(g.hwnd, nullptr, FALSE);
}

void onKey(WPARAM key) {
    switch (key) {
        case VK_ESCAPE: showMenu(); return;
        case VK_SPACE:
            if (g.session) g.session->setPaused(!g.session->paused());
            break;
        case 'M':
            if (g.session) g.session->setRealTime(!g.session->realTime());
            break;
        case 'C': g.view.attached = !g.view.attached; break;
        case 'F': g.view.follow = !g.view.follow; break;
        case 'R': g.view.reset(); break;
        case 'T': g.cfg.render.highlightTracked = !g.cfg.render.highlightTracked; break;
        case 'N': g.cfg.render.colorByNoise = !g.cfg.render.colorByNoise; break;
        case 'K': g.cfg.render.showKeyframes = !g.cfg.render.showKeyframes; break;
        case VK_ADD:
        case VK_OEM_PLUS: g.cfg.render.pointSize = std::min(g.cfg.render.pointSize + 1, 100); break;
        case VK_SUBTRACT:
        case VK_OEM_MINUS: g.cfg.render.pointSize = std::max(g.cfg.render.pointSize - 1, 1); break;
        case 'O': openVideoDialog(); break;
        case 'S': savePlyDialog(); break;
        default: return;
    }
    g.dirty = true;
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_PAINT: paint(hwnd); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_SIZE: g.dirty = true; return 0;
        case WM_TIMER: onTimer(); return 0;
        case WM_KEYDOWN:
            if (g.menu) {
                onMenuKey(wp);
            } else {
                onKey(wp);
            }
            return 0;
        case WM_CHAR:
            if (g.menu) onMenuChar(static_cast<wchar_t>(wp));
            return 0;
        case WM_LBUTTONDOWN:
            if (g.menu) {
                onMenuClick({GET_X_LPARAM(lp), GET_Y_LPARAM(lp)});
                return 0;
            }
            [[fallthrough]];
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
            if (g.menu) return 0;
            g.drag = msg == WM_LBUTTONDOWN ? App::Drag::Orbit : App::Drag::Pan;
            g.last = {GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
            SetCapture(hwnd);
            return 0;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
            g.drag = App::Drag::None;
            ReleaseCapture();
            return 0;
        case WM_MOUSEMOVE: {
            if (g.drag == App::Drag::None) return 0;
            const POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
            const double dx = p.x - g.last.x, dy = p.y - g.last.y;
            g.last = p;
            g.view.attached = false;  // dragging takes the view back
            if (g.drag == App::Drag::Orbit) {
                g.view.orbit(dx, dy);
            } else {
                g.view.pan(dx, dy, g.scene->sceneScale, g.width);
            }
            g.dirty = true;
            return 0;
        }
        case WM_MOUSEWHEEL:
            if (g.menu) {
                g.scroll -= 3 * GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA;
                g.scroll = std::max(g.scroll, 0);
                g.dirty = true;
                return 0;
            }
            g.view.attached = false;
            g.view.zoomBy(GET_WHEEL_DELTA_WPARAM(wp) / double(WHEEL_DELTA));
            g.dirty = true;
            return 0;
        case WM_DPICHANGED: {
            const RECT* r = reinterpret_cast<const RECT*>(lp);
            SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            g.dirty = true;
            return 0;
        }
        case WM_DESTROY:
            KillTimer(hwnd, kTimer);
            saveConfig();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void showUsage() {
    MessageBoxW(nullptr,
                L"Usage: videogammetry_viewer [video] [options]\n\n"
                L"  --fov <deg>     assumed field of view across the long side (default 65)\n"
                L"  --size <px>     long side of the processed frame, 0 = full size (default)\n"
                L"  --center-boost <x>  extra share of features for the middle of the image (default 0)\n"
                L"  --fuse <n>      covisible keyframes to fuse points with, 0 = no fusion (default 10)\n"
                L"  --no-merge      fusion only links points to free keypoints, never merges duplicates\n"
                L"  --real-time     play at the video's frame rate instead of as fast as possible\n\n"
                L"Without a video, the start menu opens; it has every setting, and remembers them in\n"
                L"%APPDATA%\\videogammetry\\viewer.ini. Options given here override the saved ones.",
                L"Videogammetry viewer", MB_OK | MB_ICONINFORMATION);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCmd) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    disablePowerThrottling();
    timeBeginPeriod(1);  // fine-grained sleeps for real-time pacing

    wchar_t appData[MAX_PATH];
    const DWORD appDataLen = GetEnvironmentVariableW(L"APPDATA", appData, MAX_PATH);
    if (appDataLen > 0 && appDataLen < MAX_PATH) {
        g.settingsFile = std::filesystem::path(appData) / L"videogammetry" / L"viewer.ini";
        loadSettings(g.settingsFile, g.cfg, g.recent);
    }

    std::wstring video;
    for (int i = 1; i < __argc; ++i) {
        const std::wstring a = __wargv[i];
        if (a == L"--fov" && i + 1 < __argc) {
            g.cfg.session.options.fovDeg = _wtof(__wargv[++i]);
        } else if (a == L"--size" && i + 1 < __argc) {
            g.cfg.session.options.processSize = _wtoi(__wargv[++i]);
        } else if (a == L"--fuse" && i + 1 < __argc) {
            g.cfg.session.options.mapping.fuseNeighbours = _wtoi(__wargv[++i]);
        } else if (a == L"--no-merge") {
            g.cfg.session.options.mapping.fuseMerge = false;
        } else if (a == L"--center-boost" && i + 1 < __argc) {
            g.cfg.session.options.orb.centerBoost = static_cast<float>(_wtof(__wargv[++i]));
        } else if (a == L"--real-time") {
            g.cfg.session.realTime = true;
        } else if (a == L"-h" || a == L"--help" || a == L"--version") {
            showUsage();
            return 0;
        } else if (video.empty()) {
            video = a;
        } else {
            showUsage();
            return 1;
        }
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.lpszClassName = kClassName;
    RegisterClassExW(&wc);

    g.hwnd = CreateWindowExW(0, kClassName, L"Videogammetry viewer", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                             CW_USEDEFAULT, 1500, 900, nullptr, nullptr, instance, nullptr);
    if (!g.hwnd) return 1;
    ShowWindow(g.hwnd, showCmd);
    UpdateWindow(g.hwnd);
    SetTimer(g.hwnd, kTimer, 15, nullptr);

    if (!video.empty()) startSession(video);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    g.session.reset();
    timeEndPeriod(1);
    return 0;
}
