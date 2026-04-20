param(
    [switch]$SkipTests,
    [switch]$StartApp
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

if (Test-Path -LiteralPath $venvPath) {
    $resolvedVenv = (Resolve-Path $venvPath).Path
    if (-not $resolvedVenv.StartsWith($projectRoot)) {
        throw "Refusing to remove virtual environment outside project root: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

python -m venv .venv

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt
& $pythonExe -m compileall app src tests

if (-not $SkipTests) {
    & $pythonExe -m pytest -q
}

if ($StartApp) {
    & $pythonExe -m streamlit run app/streamlit_app.py
}
