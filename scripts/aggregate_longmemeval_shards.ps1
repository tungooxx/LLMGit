param(
    [string]$DataFile = "data\longmemeval_s_cleaned.json",
    [string]$OutputDir = "experiments\public_results\longmemeval",
    [string]$SplitLabel = "longmemeval_s_cleaned",
    [string]$SystemLabel = "truthgit_gpt-4o-mini_record_batch_ms12_beliefs_and_excerpts",
    [string]$JudgeModel = "gpt-4o",
    [switch]$AllowIncomplete
)

$ErrorActionPreference = "Stop"

$pattern = "$SplitLabel`_$SystemLabel`_shard_*_limit_*.eval-results-$JudgeModel.jsonl"
$evalLogs = Get-ChildItem -Path $OutputDir -Filter $pattern | Sort-Object Name
if (-not $evalLogs) {
    throw "No shard eval logs matched $pattern in $OutputDir"
}

$AggregateEvalLog = Join-Path $OutputDir "$SplitLabel`_$SystemLabel`_full_aggregated.eval-results-$JudgeModel.jsonl"
$SummaryJson = Join-Path $OutputDir "$SplitLabel`_$SystemLabel`_full_aggregated.summary.json"
$AllowArgs = @()
if ($AllowIncomplete) {
    $AllowArgs = @("--allow-incomplete")
}

$EvalLogArgs = @()
foreach ($item in $evalLogs) {
    $EvalLogArgs += @("--eval-log", $item.FullName)
}

python -m experiments.public_benchmarks.longmemeval aggregate `
    --data $DataFile `
    @EvalLogArgs `
    --output-log $AggregateEvalLog `
    --output-json $SummaryJson `
    @AllowArgs

Write-Host ""
Write-Host "Aggregated LongMemEval shards."
Write-Host "Shard count: $($evalLogs.Count)"
Write-Host "Eval log:    $AggregateEvalLog"
Write-Host "Summary:     $SummaryJson"
