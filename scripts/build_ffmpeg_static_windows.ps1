# Builds static FFmpeg libraries (LGPL, decoders only, /MT runtime) for Windows x64
# into third_party/ffmpeg-static, for self-contained exes (cmake --preset windows-static).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/build_ffmpeg_static_windows.ps1
#
# FFmpeg is compiled from source with vcpkg (pinned to one commit below), which
# fetches its own MSYS2 and NASM. Needs git and Visual Studio with the C++ tools.
# Takes a while the first time (about 15-30 minutes).
#
# Licensing: FFmpeg is LGPL. Shipping an exe with it linked in statically means
# also offering what is needed to relink it against another FFmpeg (e.g. this
# repository's source); the shared build (fetch_ffmpeg_windows.ps1) avoids that.

$ErrorActionPreference = "Stop"

$VcpkgCommit = "3cd21e48a75f67acfef289da52893ce79144aaa1"  # FFmpeg 9.0.2
$Triplet     = "x64-windows-static"
$Features    = "ffmpeg[core,avcodec,avformat,swscale]"

$Root    = Split-Path -Parent $PSScriptRoot
$Vcpkg   = Join-Path $Root "third_party/vcpkg"
$Install = Join-Path $Root "third_party/ffmpeg-static-install"
$Dest    = Join-Path $Root "third_party/ffmpeg-static"
$Stamp   = Join-Path $Dest ".version"
$Name    = "$VcpkgCommit $Features $Triplet"

if ((Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $Name)) {
    Write-Host "Static FFmpeg already present in $Dest"
    exit 0
}

if (-not (Test-Path (Join-Path $Vcpkg ".git"))) {
    git clone https://github.com/microsoft/vcpkg.git $Vcpkg
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
git -C $Vcpkg fetch --quiet origin
git -C $Vcpkg checkout --quiet $VcpkgCommit
if ($LASTEXITCODE -ne 0) { throw "could not check out vcpkg $VcpkgCommit" }

$Exe = Join-Path $Vcpkg "vcpkg.exe"
if (-not (Test-Path $Exe)) {
    & (Join-Path $Vcpkg "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed" }
}

& $Exe install "${Features}:$Triplet" "--x-install-root=$Install" --disable-metrics
if ($LASTEXITCODE -ne 0) { throw "vcpkg could not build FFmpeg" }

# Only the release headers and libraries are needed.
$Built = Join-Path $Install $Triplet
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item -Recurse (Join-Path $Built "include") $Dest
Copy-Item -Recurse (Join-Path $Built "lib") $Dest
Copy-Item -Recurse (Join-Path $Built "share/ffmpeg") (Join-Path $Dest "license")
Set-Content -Path $Stamp -Value $Name -Encoding ascii
Write-Host "Static FFmpeg installed to $Dest"
