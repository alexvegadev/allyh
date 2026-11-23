<#
.SYNOPSIS
    Installs the AllyH CLI binary from GitHub Releases on Windows.

.DESCRIPTION
    Downloads the latest (or specified) AllyH release archive for Windows,
    verifies its SHA-256 checksum against SHA256SUMS, extracts allyh.exe into
    the installation directory, and adds that directory to the user PATH.

.EXAMPLE
    irm https://raw.githubusercontent.com/alexvegadev/allyh/main/scripts/install.ps1 | iex

.EXAMPLE
    .\scripts\install.ps1 -Version v0.2.0 -InstallDir "$env:LOCALAPPDATA\allyh\bin"
#>
[CmdletBinding()]
param(
    [string]$Version = $env:ALLYH_VERSION,
    [string]$InstallDir = $env:ALLYH_INSTALL_DIR,
    [string]$Repo = $env:ALLYH_REPO,
    [switch]$NoPathUpdate
)

$ErrorActionPreference = 'Stop'

# Ensure TLS 1.2+ on Windows PowerShell 5.1
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    # PowerShell 7+ manages TLS automatically
}

$BinName = 'allyh.exe'
if ([string]::IsNullOrWhiteSpace($Repo)) {
    $Repo = 'alexvegadev/allyh'
}
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:LOCALAPPDATA 'allyh\bin'
}

$AuthToken = if (-not [string]::IsNullOrWhiteSpace($env:GH_TOKEN)) {
    $env:GH_TOKEN.Trim()
} elseif (-not [string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
    $env:GITHUB_TOKEN.Trim()
} else {
    $null
}

function Write-Info {
    param([string]$Message)
    Write-Host "allyh-installer: $Message"
}

function Get-RequestHeaders {
    $headers = @{ 'User-Agent' = 'allyh-installer' }
    if ($AuthToken) {
        $headers['Authorization'] = "Bearer $AuthToken"
    }
    return $headers
}

function Get-WindowsTarget {
    $arch = $null
    try {
        $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToUpperInvariant()
    } catch {
        $raw = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
        $arch = if ($raw) { $raw.ToUpperInvariant() } else { '' }
    }

    switch ($arch) {
        { $_ -in @('X64', 'AMD64') } { return 'x86_64-pc-windows-msvc' }
        default {
            throw "Unsupported Windows architecture '$arch'. Build from source with: cargo install --git https://github.com/$Repo allyh-tui"
        }
    }
}

function Resolve-ReleaseTag {
    param([string]$RequestedVersion, [string]$Repository)

    if (-not [string]::IsNullOrWhiteSpace($RequestedVersion)) {
        $trimmed = $RequestedVersion.Trim()
        if ($trimmed.StartsWith('v')) {
            return $trimmed
        }
        return "v$trimmed"
    }

    $apiUrl = "https://api.github.com/repos/$Repository/releases/latest"
    try {
        $release = Invoke-RestMethod -Uri $apiUrl -Headers (Get-RequestHeaders)
    } catch {
        throw "Failed to query latest release for $Repository ($apiUrl): $($_.Exception.Message)"
    }

    if ([string]::IsNullOrWhiteSpace($release.tag_name)) {
        throw "Could not determine latest release tag from GitHub API response."
    }
    return $release.tag_name
}

function Test-ArchiveChecksum {
    param(
        [string]$ArchivePath,
        [string]$ArchiveName,
        [string]$SumsPath
    )

    $expected = $null
    foreach ($line in (Get-Content -Path $SumsPath)) {
        if ($line -match '^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$') {
            if ($Matches[2] -eq $ArchiveName) {
                $expected = $Matches[1].ToLowerInvariant()
                break
            }
        }
    }

    if (-not $expected) {
        throw "No checksum entry found for $ArchiveName in SHA256SUMS."
    }

    $actual = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expected -ne $actual) {
        throw "SHA-256 checksum mismatch for $ArchiveName (expected $expected, got $actual)."
    }
    Write-Info "verified SHA-256 checksum for $ArchiveName"
}

function Add-ToUserPath {
    param([string]$Directory)

    $normalizedTarget = $Directory.TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @()
    if (-not [string]::IsNullOrEmpty($userPath)) {
        $entries = $userPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    $alreadyPresent = $false
    foreach ($entry in $entries) {
        if ($entry.TrimEnd('\').Equals($normalizedTarget, [System.StringComparison]::OrdinalIgnoreCase)) {
            $alreadyPresent = $true
            break
        }
    }

    if (-not $alreadyPresent) {
        $newUserPath = ($entries + $Directory) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
        Write-Info "added $Directory to your user PATH"
    }

    # Also update the current PowerShell session so `allyh` works immediately
    $sessionEntries = ($env:Path -split ';') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    $inSession = $false
    foreach ($entry in $sessionEntries) {
        if ($entry.TrimEnd('\').Equals($normalizedTarget, [System.StringComparison]::OrdinalIgnoreCase)) {
            $inSession = $true
            break
        }
    }
    if (-not $inSession) {
        $env:Path = "$Directory;$env:Path"
    }
}

$target = Get-WindowsTarget
$tag = Resolve-ReleaseTag -RequestedVersion $Version -Repository $Repo
$archiveName = "allyh-$tag-$target.zip"
$baseUrl = "https://github.com/$Repo/releases/download/$tag"

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("allyh-install-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    $archivePath = Join-Path $tempDir $archiveName
    $sumsPath = Join-Path $tempDir 'SHA256SUMS'
    $headers = Get-RequestHeaders

    Write-Info "downloading allyh $tag ($target)..."
    Invoke-WebRequest -Uri "$baseUrl/$archiveName" -OutFile $archivePath -Headers $headers -UseBasicParsing

    $checksumDownloaded = $false
    try {
        Invoke-WebRequest -Uri "$baseUrl/SHA256SUMS" -OutFile $sumsPath -Headers $headers -UseBasicParsing
        $checksumDownloaded = $true
    } catch {
        Write-Info "warning: SHA256SUMS asset not found for $tag; skipping checksum verification"
    }

    if ($checksumDownloaded) {
        Test-ArchiveChecksum -ArchivePath $archivePath -ArchiveName $archiveName -SumsPath $sumsPath
    }

    $extractDir = Join-Path $tempDir 'extract'
    Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force

    $extractedBin = Join-Path $extractDir $BinName
    if (-not (Test-Path -Path $extractedBin -PathType Leaf)) {
        throw "Expected binary '$BinName' was not found inside $archiveName."
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $destPath = Join-Path $InstallDir $BinName
    Copy-Item -Path $extractedBin -Destination $destPath -Force

    if (-not $NoPathUpdate) {
        Add-ToUserPath -Directory $InstallDir
    }

    Write-Info "installed allyh to $destPath"
    try {
        $installedVersion = & $destPath --version
        if ($installedVersion) {
            Write-Info "version: $installedVersion"
        }
    } catch {
        # Non-fatal if version probe fails
    }
    Write-Info "run 'allyh --help' to get started (restart your terminal if another shell does not see it yet)."
} finally {
    if (Test-Path -Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
