# Downloads prebuilt FFmpeg (LGPL, shared) for Windows x64 into third_party/ffmpeg.
# Source: https://github.com/BtbN/FFmpeg-Builds
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/fetch_ffmpeg_windows.ps1

$ErrorActionPreference = "Stop"

$Version = "8.1"
$Name    = "ffmpeg-n$Version-latest-win64-lgpl-shared-$Version"
$Url     = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$Name.zip"

$Root    = Split-Path -Parent $PSScriptRoot
$Dest    = Join-Path $Root "third_party/ffmpeg"
$Zip     = Join-Path $env:TEMP "$Name.zip"
$Extract = Join-Path $env:TEMP "$Name-extract"

if (Test-Path (Join-Path $Dest "include/libavcodec/avcodec.h")) {
    Write-Host "FFmpeg already present in $Dest"
    exit 0
}

Write-Host "Downloading $Url"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing

if (Test-Path $Extract) { Remove-Item -Recurse -Force $Extract }
Expand-Archive -Path $Zip -DestinationPath $Extract

New-Item -ItemType Directory -Force (Split-Path -Parent $Dest) | Out-Null
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
Move-Item (Join-Path $Extract $Name) $Dest

Remove-Item -Force $Zip
Remove-Item -Recurse -Force $Extract
Write-Host "FFmpeg $Version installed to $Dest"
