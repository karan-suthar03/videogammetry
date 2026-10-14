# Downloads prebuilt FFmpeg (LGPL, shared) for Windows x64 into third_party/ffmpeg.
# Source: https://github.com/BtbN/FFmpeg-Builds
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/fetch_ffmpeg_windows.ps1
#
# The build is pinned to one dated release and checked against its SHA-256.
# BtbN prunes old autobuilds; if the download 404s, pick a newer
# autobuild-* release with the same FFmpeg major version, update the three
# values below, and recompute the hash with Get-FileHash.

$ErrorActionPreference = "Stop"

$Release = "autobuild-2026-10-13-13-05"
$Name    = "ffmpeg-n8.1.3-18-g46d899d7a0-win64-lgpl-shared-8.1"
$Sha256  = "19D92DFB1481E72E4E8F94844BF609418297A646F33DEB6864C409CA752469C4"

$Url     = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$Release/$Name.zip"
$Root    = Split-Path -Parent $PSScriptRoot
$Dest    = Join-Path $Root "third_party/ffmpeg"
$Stamp   = Join-Path $Dest ".version"
$Zip     = Join-Path $env:TEMP "$Name.zip"
$Extract = Join-Path $env:TEMP "$Name-extract"

if ((Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $Name)) {
    Write-Host "FFmpeg $Name already present in $Dest"
    exit 0
}

Write-Host "Downloading $Url"
$ProgressPreference = "SilentlyContinue"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing

$Actual = (Get-FileHash $Zip -Algorithm SHA256).Hash
if ($Actual -ne $Sha256) {
    Remove-Item -Force $Zip
    throw "SHA-256 mismatch for $Name.zip`n  expected $Sha256`n  actual   $Actual"
}

if (Test-Path $Extract) { Remove-Item -Recurse -Force $Extract }
Expand-Archive -Path $Zip -DestinationPath $Extract

New-Item -ItemType Directory -Force (Split-Path -Parent $Dest) | Out-Null
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
Move-Item (Join-Path $Extract $Name) $Dest
Set-Content -Path $Stamp -Value $Name -Encoding ascii

Remove-Item -Force $Zip
Remove-Item -Recurse -Force $Extract
Write-Host "FFmpeg $Name installed to $Dest"
