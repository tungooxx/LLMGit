param(
    [string]$DataFile = "data\longmemeval_s_cleaned.json",
    [string]$OutputDir = "experiments\public_results\longmemeval",
    [string]$SplitLabel = "longmemeval_s_cleaned",
    [string]$Model = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o-mini" }),
    [string]$JudgeModel = "gpt-4o",
    [int]$Limit = 0,
    [int]$SampleSize = 0,
    [int]$SampleSeed = 0,
    [string]$HistoryFormat = "json",
    [string]$ReaderMode = "con",
    [int]$MaxOutputTokens = 256,
    [int]$StartIndex = 0,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataFile)) {
    Write-Host "Downloading LongMemEval-S cleaned data to $DataFile"
    python -m experiments.public_benchmarks.longmemeval download `
        --split s_cleaned `
        --output $DataFile
}

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required for generation/evaluation. Set it in .env or the current shell."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$SafeModel = $Model -replace '[^A-Za-z0-9_.-]', '_'
$RunLabel = if ($SampleSize -gt 0) { "random_$SampleSize`_seed_$SampleSeed" } elseif ($Limit -gt 0) { "shard_$StartIndex`_limit_$Limit" } elseif ($StartIndex -gt 0) { "from_$StartIndex" } else { "full" }
$BaseName = "$SplitLabel`_$SafeModel`_$ReaderMode`_$RunLabel"
$PromptJsonl = Join-Path $OutputDir "$BaseName.prompts.jsonl"
$Hypotheses = Join-Path $OutputDir "$BaseName.hypotheses.jsonl"
$EvalLog = Join-Path $OutputDir "$BaseName.eval-results-$JudgeModel.jsonl"
$SummaryJson = Join-Path $OutputDir "$BaseName.summary.json"
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

python -m experiments.public_benchmarks.longmemeval inspect `
    --data $DataFile `
    --output-dir $OutputDir `
    --split-label $SplitLabel `
    @LimitArgs

python -m experiments.public_benchmarks.longmemeval make-prompts `
    --data $DataFile `
    --output-jsonl $PromptJsonl `
    --history-format $HistoryFormat `
    --reader-mode $ReaderMode `
    @LimitArgs

python -m experiments.public_benchmarks.longmemeval generate `
    --data $DataFile `
    --output-jsonl $Hypotheses `
    --model $Model `
    --history-format $HistoryFormat `
    --reader-mode $ReaderMode `
    --max-output-tokens $MaxOutputTokens `
    @LimitArgs

if (-not $SkipEvaluation) {
    python -m experiments.public_benchmarks.longmemeval evaluate `
        --data $DataFile `
        --hypotheses $Hypotheses `
        --output-log $EvalLog `
        --judge-model $JudgeModel `
        @LimitArgs

    python -m experiments.public_benchmarks.longmemeval summarize `
        --data $DataFile `
        --eval-log $EvalLog `
        --output-json $SummaryJson `
        @LimitArgs
}

Write-Host ""
Write-Host "LongMemEval run complete."
Write-Host "Prompts:    $PromptJsonl"
Write-Host "Hypotheses: $Hypotheses"
if (-not $SkipEvaluation) {
    Write-Host "Eval log:   $EvalLog"
    Write-Host "Summary:    $SummaryJson"
}
