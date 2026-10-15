// Live viewer for Windows: plays a video through the reconstruction and
// shows the point cloud, keyframes and current camera as they are built.
// Plain Win32 + GDI; the 3D view is rendered on the CPU (renderer.h).

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <timeapi.h>

#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "live_session.h"
#include "power.h"
#include "renderer.h"
#include "videogammetry/version.h"

namespace {

constexpr UINT_PTR kTimer = 1;
constexpr wchar_t kClassName[] = L"VideogammetryViewer";

struct App {
    HWND hwnd = nullptr;
    LiveSession::Settings settings;
    std::unique_ptr<LiveSession> session;
    std::wstring videoPath;
    std::shared_ptr<const LiveSnapshot> scene = std::make_shared<LiveSnapshot>();

    OrbitView view;
    RenderOptions renderOptions;
    SceneRenderer renderer;
    bool dirty = true;
    std::wstring notice;  // last save result etc.

    // Back buffer: a DIB section selected into a memory DC.
    HDC memDC = nullptr;
    HBITMAP bitmap = nullptr;
    HGDIOBJ oldBitmap = nullptr;
    std::uint32_t* pixels = nullptr;
    int width = 0, height = 0;
    HFONT font = nullptr;
    int fontDpi = 0;

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

void startSession(const std::wstring& path) {
    g.session.reset();  // stops the previous one
    g.videoPath = path;
    g.session = std::make_unique<LiveSession>(toUtf8(path), g.settings);
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
    g.fontDpi = dpi;
    g.font = CreateFontW(-MulDiv(10, dpi, 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                         OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
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
        swprintf(buf, 512, L"tracking + mapping: %.0f fps   playback: %s", s.processingFps, playback);
        line(buf, dim);
        if (!s.message.empty()) line(fromUtf8(s.message), RGB(255, 210, 120));
        if (!g.notice.empty()) line(g.notice, RGB(140, 220, 255));
    }

    const wchar_t* help[] = {
        L"left-drag orbit   right-drag pan   wheel zoom   F follow camera   R reset view",
        L"Space pause   M real-time/max speed   T highlight tracked   K keyframes   +/- point size   O open   S save .ply",
        L"red: current camera   blue: keyframes   orange: camera path   green: points tracked in this frame",
    };
    int hy = g.height - margin - lineH * 3;
    for (const wchar_t* h : help) {
        drawText(dc, margin, hy, h, dim);
        hy += lineH;
    }
    swprintf(buf, 512, L"follow: %s", g.view.follow ? L"on" : L"off");
    drawText(dc, margin, hy - lineH * 4, buf, dim);
    SelectObject(dc, oldFont);
}

void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    const int w = std::max<int>(1, rc.right - rc.left), h = std::max<int>(1, rc.bottom - rc.top);
    ensureBackBuffer(w, h);

    Framebuffer fb{g.pixels, w, h, w};
    g.renderer.render(*g.scene, g.view, g.renderOptions, fb);
    const RgbImage& thumb = g.scene->thumbnail;
    if (thumb.width > 0 && thumb.width + 24 < w) blitImage(thumb, w - thumb.width - 12, 12, fb);
    GdiFlush();
    drawHud(g.memDC);

    BitBlt(dc, 0, 0, w, h, g.memDC, 0, 0, SRCCOPY);
    EndPaint(hwnd, &ps);
    g.dirty = false;
}

void onTimer() {
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
        case VK_ESCAPE: DestroyWindow(g.hwnd); return;
        case VK_SPACE:
            if (g.session) g.session->setPaused(!g.session->paused());
            break;
        case 'M':
            if (g.session) g.session->setRealTime(!g.session->realTime());
            break;
        case 'F': g.view.follow = !g.view.follow; break;
        case 'R': g.view.reset(); break;
        case 'T': g.renderOptions.highlightTracked = !g.renderOptions.highlightTracked; break;
        case 'K': g.renderOptions.showKeyframes = !g.renderOptions.showKeyframes; break;
        case VK_ADD:
        case VK_OEM_PLUS: g.renderOptions.pointSize = std::min(g.renderOptions.pointSize + 1, 8); break;
        case VK_SUBTRACT:
        case VK_OEM_MINUS: g.renderOptions.pointSize = std::max(g.renderOptions.pointSize - 1, 1); break;
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
        case WM_KEYDOWN: onKey(wp); return 0;
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
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
            if (g.drag == App::Drag::Orbit) {
                g.view.orbit(dx, dy);
            } else {
                g.view.pan(dx, dy, g.scene->sceneScale, g.width);
            }
            g.dirty = true;
            return 0;
        }
        case WM_MOUSEWHEEL:
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
                L"  --real-time     play at the video's frame rate instead of as fast as possible\n\n"
                L"Without a video, a file dialog opens.",
                L"Videogammetry viewer", MB_OK | MB_ICONINFORMATION);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCmd) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    disablePowerThrottling();
    timeBeginPeriod(1);  // fine-grained sleeps for real-time pacing

    std::wstring video;
    for (int i = 1; i < __argc; ++i) {
        const std::wstring a = __wargv[i];
        if (a == L"--fov" && i + 1 < __argc) {
            g.settings.options.fovDeg = _wtof(__wargv[++i]);
        } else if (a == L"--size" && i + 1 < __argc) {
            g.settings.options.processSize = _wtoi(__wargv[++i]);
        } else if (a == L"--center-boost" && i + 1 < __argc) {
            g.settings.options.orb.centerBoost = static_cast<float>(_wtof(__wargv[++i]));
        } else if (a == L"--real-time") {
            g.settings.realTime = true;
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

    if (!video.empty()) {
        startSession(video);
    } else {
        openVideoDialog();
    }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    g.session.reset();
    timeEndPeriod(1);
    return 0;
}
