<#
.SYNOPSIS
    Finds the MSVC toolchain (and CUDA, if present) on this machine and loads
    the environment variables (INCLUDE, LIB, LIBPATH, PATH, ...) that
    vcvars64.bat normally sets, without requiring a "Developer Command
    Prompt" to already be open.

.DESCRIPTION
    vcvarsall.bat / vcvars64.bat call `vswhere.exe` by bare name to locate the
    Visual Studio install. If the VS Installer directory isn't on PATH (common
    on a plain PowerShell prompt, vs. a VS Developer shell), that lookup fails
    with "'vswhere.exe' is not recognized..." and vcvars falls back to
    whatever it can guess -- or vcvars simply isn't run at all, so `cl.exe`
    can be found (ninja caches its absolute path) while `INCLUDE`/`LIB` stay
    empty and every compile fails with "cannot open include file: 'cstdint'".

    This script never assumes vswhere is already reachable: it locates it (or
    falls back to a directory scan if it truly is not installed), uses it to
    resolve the newest VS install with the C++ toolchain, runs vcvars64.bat in
    a throwaway cmd.exe, and imports the resulting environment into the
    CURRENT PowerShell session. It also detects an installed CUDA Toolkit and
    makes sure its bin directory is on PATH (needed for nvcc-based CMake CUDA
    builds).

.PARAMETER Arch
    Target architecture to pass to vcvars. Defaults to x64.

.PARAMETER Persist
    Also fixes the root cause for future plain shells: adds the VS Installer
    directory (where vswhere.exe lives) to the current user's PATH via
    [Environment]::SetEnvironmentVariable(..., 'User'), so a fresh
    "Developer Command Prompt" or a manual vcvars64.bat call finds vswhere
    without needing this script. Safe/idempotent: only appends if missing.

.PARAMETER Quiet
    Suppress the summary banner; only emit warnings/errors.

.EXAMPLE
    . .\scripts\setup-msvc-env.ps1
    Dot-source it so the environment variables land in your current shell,
    then run cmake/ninja/cl.exe directly.

.EXAMPLE
    . .\scripts\setup-msvc-env.ps1 -Persist
    Same, and also fixes vswhere resolution for future new shells.

.NOTES
    Must be dot-sourced ("." or "source") to affect your current session.
    Running it as `.\setup-msvc-env.ps1` only sets the variables in the
    script's own child process, which disappears when it exits.
#>
[CmdletBinding()]
param(
    [ValidateSet('x64', 'x86', 'x64_x86', 'x86_x64', 'x64_arm64', 'x86_arm')]
    [string]$Arch = 'x64',
    [switch]$Persist,
    [switch]$Quiet
)

function Write-Info {
    param([string]$Message)
    if (-not $Quiet) { Write-Host $Message -ForegroundColor Cyan }
}

function Write-Ok {
    param([string]$Message)
    if (-not $Quiet) { Write-Host $Message -ForegroundColor Green }
}

# ---------------------------------------------------------------------------
# 1. Locate vswhere.exe. This is the one artifact vcvars itself depends on
#    but cannot find unless it is already on PATH, so we resolve it by known
#    install location first and only fall back to a filesystem scan.
# ---------------------------------------------------------------------------
function Find-VsWhere {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    # Already on PATH under some other install layout?
    $onPath = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # Last resort: bounded scan of the two Program Files roots. This is a
    # scan, not a full-disk search, so it stays fast even without vswhere.
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $root -or -not (Test-Path $root)) { continue }
        $found = Get-ChildItem -Path $root -Filter 'vswhere.exe' -Recurse `
            -Depth 3 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

# ---------------------------------------------------------------------------
# 2. Resolve the newest VS install that actually has the C++ toolchain, and
#    the vcvars64.bat inside it. Prefer vswhere (arch/component-aware);
#    fall back to a directory scan under "Program Files\Microsoft Visual
#    Studio" for the rare case vswhere itself isn't installed at all (e.g. a
#    VS Build Tools install that predates it, or a stripped-down image).
# ---------------------------------------------------------------------------
function Get-VcVarsPath {
    param([string]$VsWherePath)

    if ($VsWherePath) {
        $installPath = & $VsWherePath -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($installPath) {
            $candidate = Join-Path $installPath 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $candidate) { return $candidate }
        }
    }

    Write-Info "vswhere didn't resolve an install (or wasn't found) -- scanning Program Files directly."
    $roots = @(
        "${env:ProgramFiles}\Microsoft Visual Studio",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio"
    ) | Where-Object { $_ -and (Test-Path $_) }

    $found = foreach ($root in $roots) {
        Get-ChildItem -Path $root -Filter 'vcvars64.bat' -Recurse `
            -Depth 5 -ErrorAction SilentlyContinue
    }
    # Prefer the newest by path (year/edition sort as a string works because
    # "2022" > "2019" and editions don't affect which toolchain is newest).
    $best = $found | Sort-Object FullName -Descending | Select-Object -First 1
    if ($best) { return $best.FullName }
    return $null
}

# ---------------------------------------------------------------------------
# 3. Run vcvars in a disposable cmd.exe and import every variable it set (or
#    changed) into this PowerShell process's environment.
# ---------------------------------------------------------------------------
function Import-VcVarsEnvironment {
    param([string]$VcVarsPath, [string]$Arch)

    $marker = '___VCVARS_ENV_START___'
    $output = & cmd.exe /c "`"$VcVarsPath`" $Arch >nul 2>&1 && echo $marker && set" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($output -match [regex]::Escape($marker))) {
        throw "vcvars64.bat failed to run (exit $LASTEXITCODE). Output:`n$($output -join "`n")"
    }

    $capture = $false
    $count = 0
    foreach ($line in $output) {
        if (-not $capture) {
            # cmd.exe's `echo X && next` keeps the space before `&&` as part
            # of X's output, so this arrives as "$marker " (trailing space) --
            # trim before comparing rather than matching exact equality.
            if ($line.TrimEnd() -eq $marker) { $capture = $true }
            continue
        }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { continue }
        $name = $line.Substring(0, $eq)
        $value = $line.Substring($eq + 1)
        Set-Item -Path "Env:$name" -Value $value
        $count++
    }
    return $count
}

# ---------------------------------------------------------------------------
# 4. CUDA Toolkit detection (this repo's CUDA build needs nvcc reachable).
#    CMake's CUDA language support finds it via CUDA_PATH or PATH; we surface
#    whichever the machine already has rather than guessing a version.
# ---------------------------------------------------------------------------
function Find-CudaBin {
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
        return "$env:CUDA_PATH\bin"
    }
    $root = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        $newest = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v\d' } |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($newest -and (Test-Path (Join-Path $newest.FullName 'bin\nvcc.exe'))) {
            return (Join-Path $newest.FullName 'bin')
        }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Info "Locating vswhere.exe..."
$vsWhere = Find-VsWhere
if ($vsWhere) {
    Write-Ok "  found: $vsWhere"
} else {
    Write-Warning "  vswhere.exe not found anywhere under Program Files; falling back to a directory scan for vcvars64.bat."
}

Write-Info "Resolving Visual Studio C++ toolchain..."
$vcvars = Get-VcVarsPath -VsWherePath $vsWhere
if (-not $vcvars) {
    throw "Could not find vcvars64.bat under any Visual Studio install on this machine. Install the 'Desktop development with C++' workload and re-run."
}
Write-Ok "  using: $vcvars"

Write-Info "Running vcvars64.bat ($Arch) and importing its environment into this session..."
$changed = Import-VcVarsEnvironment -VcVarsPath $vcvars -Arch $Arch
Write-Ok "  imported/updated $changed environment variables (INCLUDE, LIB, LIBPATH, PATH, ...)."

$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
    Write-Ok "  cl.exe now resolves to: $($cl.Source)"
} else {
    Write-Warning "  cl.exe still not on PATH after importing vcvars -- something is unusual about this install."
}

Write-Info "Checking for CUDA Toolkit..."
$cudaBin = Find-CudaBin
if ($cudaBin) {
    if (($env:PATH -split ';') -notcontains $cudaBin) {
        $env:PATH = "$cudaBin;$env:PATH"
    }
    Write-Ok "  nvcc found: $cudaBin\nvcc.exe"
} else {
    Write-Warning "  No CUDA Toolkit found (only relevant for -DCELEG_ENABLE_CUDA=ON builds)."
}

if ($Persist) {
    Write-Info "Persisting the vswhere fix for future shells (User PATH)..."
    if ($vsWhere) {
        $vsWhereDir = Split-Path $vsWhere -Parent
        $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        $userPathEntries = @()
        if ($userPath) { $userPathEntries = $userPath -split ';' }
        if ($userPathEntries -notcontains $vsWhereDir) {
            $newUserPath = if ($userPath) { "$userPath;$vsWhereDir" } else { $vsWhereDir }
            [Environment]::SetEnvironmentVariable('PATH', $newUserPath, 'User')
            Write-Ok "  added '$vsWhereDir' to your permanent User PATH -- new shells will resolve vswhere/vcvars on their own."
        } else {
            Write-Ok "  '$vsWhereDir' is already on your permanent User PATH."
        }
    } else {
        Write-Warning "  Skipped: vswhere.exe was never found, so there's no directory to persist."
    }
}

Write-Ok "`nDone. If this script was dot-sourced, cmake/ninja/cl.exe/nvcc are now usable directly in this shell."
