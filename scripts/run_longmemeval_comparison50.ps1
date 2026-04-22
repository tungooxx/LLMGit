param(
    [string]$DataFile = "data\longmemeval_s_cleaned.json",
    [string]$OutputDir = "experiments\public_results\longmemeval",
    [string]$SplitLabel = "longmemeval_s_cleaned",
    [string]$AnswerModel = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o-mini" }),
    [string]$ExtractionModel = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o-mini" }),
    [string]$JudgeModel = "gpt-4o",
    [int]$SampleSize = 50,
    [int]$SampleSeed = 20260421,
    [int]$MaxSessions = 12,
    [switch]$SkipEvaluation,
    [switch]$IncludeTruthGitFullHistory
)

$ErrorActionPreference = "Stop"

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is required for LongMemEval comparison runs."
}

$FullHistoryScript = Join-Path $PSScriptRoot "run_longmemeval_full.ps1"
$TruthGitScript = Join-Path $PSScriptRoot "run_longmemeval_truthgit.ps1"

$FullArgs = @{
    DataFile = $DataFile
    OutputDir = $OutputDir
    SplitLabel = $SplitLabel
    Model = $AnswerModel
    JudgeModel = $JudgeModel
    SampleSize = $SampleSize
    SampleSeed = $SampleSeed
    ReaderMode = "con"
    HistoryFormat = "json"
    NoResume = $true
}
$TruthGitArgs = @{
    DataFile = $DataFile
    OutputDir = $OutputDir
    SplitLabel = $SplitLabel
    AnswerModel = $AnswerModel
    ExtractionModel = $ExtractionModel
    JudgeModel = $JudgeModel
    SampleSize = $SampleSize
    SampleSeed = $SampleSeed
    MaxSessions = $MaxSessions
    NoResume = $true
}
if ($SkipEvaluation) {
    $FullArgs.SkipEvaluation = $true
    $TruthGitArgs.SkipEvaluation = $true
}

Write-Host "Running LongMemEval random held-out comparison."
Write-Host "sample_size=$SampleSize sample_seed=$SampleSeed answer_model=$AnswerModel judge=$JudgeModel"

Write-Host ""
Write-Host "[1/4] Full-history chat baseline"
& $FullHistoryScript @FullArgs

Write-Host ""
Write-Host "[2/4] TruthGit selected sessions: record_batch + beliefs_and_excerpts"
& $TruthGitScript @TruthGitArgs -ExtractionMode record_batch -ContextMode beliefs_and_excerpts

Write-Host ""
Write-Host "[3/4] TruthGit selected sessions: per_session + beliefs_and_excerpts"
& $TruthGitScript @TruthGitArgs -ExtractionMode per_session -ContextMode beliefs_and_excerpts

Write-Host ""
Write-Host "[4/4] TruthGit selected sessions: record_batch + beliefs_only"
& $TruthGitScript @TruthGitArgs -ExtractionMode record_batch -ContextMode beliefs_only

if ($IncludeTruthGitFullHistory) {
    Write-Host ""
    Write-Host "[optional] TruthGit full history: record_batch + beliefs_and_excerpts"
    $FullTruthGitArgs = $TruthGitArgs.Clone()
    $FullTruthGitArgs["MaxSessions"] = 0
    & $TruthGitScript @FullTruthGitArgs -ExtractionMode record_batch -ContextMode beliefs_and_excerpts
}

Write-Host ""
Write-Host "Comparison run complete. Summaries are in $OutputDir."
