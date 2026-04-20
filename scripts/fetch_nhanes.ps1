param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $projectRoot "data\\raw\\nhanes"
$baseUrl = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles"
$files = @(
    "DEMO_C.XPT",
    "DIQ_C.XPT",
    "L10_C.XPT",
    "LEXAB_C.XPT",
    "LEXPN_C.XPT",
    "SMQ_C.XPT"
)

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

foreach ($file in $files) {
    $target = Join-Path $outDir $file
    if ((-not $Force) -and (Test-Path -LiteralPath $target)) {
        Write-Host "Skipping $file (already exists)"
        continue
    }

    $url = "$baseUrl/$file"
    Write-Host "Downloading $file ..."
    Invoke-WebRequest -Uri $url -OutFile $target
}

Write-Host "Done. Files saved in $outDir"
