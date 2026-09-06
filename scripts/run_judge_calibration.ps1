param(
    [string]$BaseUrl = "https://ws-66q3vmu9ebhzahay.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    [string]$JudgeModel = "deepseek-v4-flash-0731"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$inputRoot = Join-Path $repoRoot "tmp\judge-calibration-v1-inputs-reviewed"
$outputRoot = Join-Path $repoRoot "outputs\evaluation\judge-calibration-v2"
$cacheRoot = Join-Path $outputRoot "semantic-judge-cache"
$priorRoot = Join-Path $repoRoot "outputs\evaluation\judge-calibration-v1"
$cases = Join-Path $repoRoot "data\evaluation\judge-calibration-review-v1\cases.jsonl"
$tasks = Join-Path $inputRoot "tasks.jsonl"
$rubrics = Join-Path $inputRoot "rubrics.jsonl"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
foreach ($required in @($cases, $tasks, $rubrics)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required calibration input not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot, $cacheRoot | Out-Null
if (Test-Path -LiteralPath (Join-Path $priorRoot "sft-325\judges.jsonl")) {
    $seedArguments = @(
        (Join-Path $repoRoot "scripts\seed_revalidated_judge_cache.py"),
        "--rubrics", $rubrics,
        "--source-run", "$(Join-Path $inputRoot 'sft-325\trajectories.jsonl')=$(Join-Path $priorRoot 'sft-325')",
        "--source-run", "$(Join-Path $inputRoot 'carl-bpo-v3-step200\trajectories.jsonl')=$(Join-Path $priorRoot 'carl-bpo-v3-step200')",
        "--output-cache", $cacheRoot,
        "--model", $JudgeModel,
        "--base-url", $BaseUrl
    )
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    & $python @seedArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to seed the revalidated Judge cache"
    }
}

$secureKey = Read-Host "请输入纯 API Key（不会写入磁盘）" -AsSecureString
$keyPointer = [IntPtr]::Zero
try {
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "API Key 不能为空"
    }
    $env:OPENAI_API_KEY = $plainKey
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $runs = @(
        @{ Label = "sft-325"; Directory = "sft-325" },
        @{ Label = "carl-bpo-v3-step200"; Directory = "carl-bpo-v3-step200" }
    )
    foreach ($run in $runs) {
        $trajectory = Join-Path $inputRoot "$($run.Directory)\trajectories.jsonl"
        $output = Join-Path $outputRoot $run.Directory
        $arguments = @(
            (Join-Path $repoRoot "scripts\evaluate_multiturn_panels.py"),
            "--expected-tasks", $tasks,
            "--trajectories", $trajectory,
            "--rubrics", $rubrics,
            "--output-dir", $output,
            "--judge-cache-dir", $cacheRoot,
            "--actor-label", $run.Label,
            "--condition", "gap-ask-enabled",
            "--judge-model", $JudgeModel,
            "--judge-base-url", $BaseUrl,
            "--allow-blind-final",
            "--resume"
        )
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Judge calibration failed for $($run.Label)"
        }
    }

    $auditArguments = @(
        (Join-Path $repoRoot "scripts\audit_judge_calibration.py"),
        "--cases", $cases,
        "--rubrics", $rubrics,
        "--run", "sft-325=$(Join-Path $outputRoot 'sft-325')",
        "--run", "carl-bpo-v3-step200=$(Join-Path $outputRoot 'carl-bpo-v3-step200')",
        "--output", (Join-Path $outputRoot "calibration-report.json"),
        "--force"
    )
    & $python @auditArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Judge outputs were generated, but the calibration gate failed; inspect calibration-report.json"
    }
}
finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
