param(
    [string]$Plan = "configs\evaluation\final200-gplus-r3.json",
    [string]$OutputRoot = "outputs\evaluation\final200-gplus-r3"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$planPath = Join-Path $repoRoot $Plan
$outputPath = Join-Path $repoRoot $OutputRoot
$baseRubrics = Join-Path $repoRoot "tmp\final200-rubrics-human-reviewed-v1\rubrics.jsonl"
$corrections = Join-Path $repoRoot "data\evaluation\final200-minimal-rubric-corrections-v1.jsonl"
$correctedRubricRoot = Join-Path $repoRoot "tmp\final200-rubrics-minimal-v1"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $planPath)) {
    throw "Final-200 Judge plan not found: $planPath"
}
foreach ($required in @($baseRubrics, $corrections)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required Final-200 Rubric input not found: $required"
    }
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
& $python (Join-Path $repoRoot "scripts\apply_rubric_semantic_corrections.py") `
    --rubrics $baseRubrics `
    --corrections $corrections `
    --output-dir $correctedRubricRoot `
    --force
if ($LASTEXITCODE -ne 0) {
    throw "Failed to materialize the corrected Final-200 Rubrics"
}

$secureKey = Read-Host "Enter the API Key (kept in memory; never written to disk)" -AsSecureString
$keyPointer = [IntPtr]::Zero
try {
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "API Key cannot be empty"
    }
    if ($plainKey -ne $plainKey.Trim() -or $plainKey -match '[\x00-\x20]') {
        throw "API Key contains whitespace or control characters; paste the raw key only"
    }

    $env:OPENAI_API_KEY = $plainKey
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

    & $python (Join-Path $repoRoot "scripts\run_multiturn_panel_grid.py") `
        --plan $planPath `
        --output-root $outputPath `
        --comparison-output (Join-Path $outputPath "comparison-gplus.json") `
        --condition gap-ask-enabled `
        --allow-blind-final `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Final-200 Judge run failed; rerun this script to resume from completed records and cache entries"
    }
}
finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
