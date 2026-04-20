param(
    [string]$DataFile = "data\longmemeval_s_cleaned.json",
    [string]$OutputDir = "experiments\public_results\longmemeval",
    [string]$SplitLabel = "longmemeval_s_cleaned",
    [string]$AnswerModel = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o-mini" }),
    [string]$ExtractionModel = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o-mini" }),
    [string]$JudgeModel = "gpt-4o",
    [ValidateSet("per_session", "record_batch")]
    [string]$ExtractionMode = "per_session",
    [int]$MaxSessions = 12,
    [int]$Limit = 0,
    [int]$StartIndex = 0,
    [int]$SampleSize = 0,
    [int]$SampleSeed = 0,
    [switch]$SkipEvaluation,
    [switch]$Trace,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataFile)) {
    Write-Host "Downloading LongMemEval-S cleaned data to $DataFile"
    python -m experiments.public_benchmarks.longmemeval download `
        --split s_cleaned `
        --output $DataFile
}

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required. Set it before running TruthGit on LongMemEval."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$SafeAnswerModel = $AnswerModel -replace '[^A-Za-z0-9_.-]', '_'
$RunLabel = if ($SampleSize -gt 0) { "random_$SampleSize`_seed_$SampleSeed" } elseif ($Limit -gt 0) { "shard_$StartIndex`_limit_$Limit" } elseif ($StartIndex -gt 0) { "from_$StartIndex" } else { "full" }
$BaseName = "$SplitLabel`_truthgit`_$SafeAnswerModel`_$ExtractionMode`_$RunLabel"
$Hypotheses = Join-Path $OutputDir "$BaseName.hypotheses.jsonl"
$EvalLog = Join-Path $OutputDir "$BaseName.eval-results-$JudgeModel.jsonl"
$SummaryJson = Join-Path $OutputDir "$BaseName.summary.json"
$TraceDir = Join-Path $OutputDir "$BaseName.traces"
$LimitArgs = @()
if ($SampleSize -gt 0 -and $Limit -gt 0) {
    throw "Use either -Limit or -SampleSize, not both."
}
if ($SampleSize -gt 0 -and $StartIndex -gt 0) {
    throw "Use -StartIndex only with deterministic -Limit shards, not random samples."
}
if ($SampleSize -gt 0) {
    $LimitArgs = @("--sample-size", "$SampleSize", "--sample-seed", "$SampleSeed")
} elseif ($Limit -gt 0) {
    $LimitArgs = @("--limit", "$Limit", "--start-index", "$StartIndex")
} elseif ($StartIndex -gt 0) {
    $LimitArgs = @("--start-index", "$StartIndex")
}
$TraceArgs = @()
if ($Trace) {
    $TraceArgs = @("--trace-dir", $TraceDir)
}
$ResumeArgs = @()
if ($NoResume) {
    $ResumeArgs = @("--no-resume")
}

python -m experiments.public_benchmarks.longmemeval_truthgit generate `
    --data $DataFile `
    --output-jsonl $Hypotheses `
    --answer-model $AnswerModel `
    --extraction-model $ExtractionModel `
    --extraction-mode $ExtractionMode `
    --max-sessions $MaxSessions `
    @LimitArgs `
    @TraceArgs `
    @ResumeArgs
if ($LASTEXITCODE -ne 0) {
    throw "TruthGit LongMemEval generation failed."
}

if (-not $SkipEvaluation) {
    python -m experiments.public_benchmarks.longmemeval evaluate `
        --data $DataFile `
        --hypotheses $Hypotheses `
        --output-log $EvalLog `
        --judge-model $JudgeModel `
        @LimitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "LongMemEval evaluation failed."
    }

    python -m experiments.public_benchmarks.longmemeval summarize `
        --data $DataFile `
        --eval-log $EvalLog `
        --output-json $SummaryJson `
        @LimitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "LongMemEval summarization failed."
    }
}

Write-Host ""
Write-Host "TruthGit LongMemEval run complete."
Write-Host "Hypotheses: $Hypotheses"
if (-not $SkipEvaluation) {
    Write-Host "Eval log:   $EvalLog"
    Write-Host "Summary:    $SummaryJson"
}
if ($Trace) {
    Write-Host "Traces:     $TraceDir"
}
