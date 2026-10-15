$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$PythonExe = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$MetricsDir = Join-Path $ProjectRoot 'implementations\wgan_gp'
$YoloScripts = Join-Path $ProjectRoot 'implementations\yolov8_det\scripts'
$LogRoot = Join-Path $ProjectRoot 'logs'
$PipelineLog = Join-Path $LogRoot 'q3_diffusion_multiseed_pipeline.log'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

function Invoke-Stage([string]$Name, [string]$WorkingDirectory, [string[]]$Arguments) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $Name" | Tee-Object -FilePath $PipelineLog -Append
    Push-Location -LiteralPath $WorkingDirectory
    try {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $LogRoot "$Name.log") -Append
        $ExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousPreference
        if ($ExitCode -ne 0) { throw "Stage failed: $Name (exit code $ExitCode)" }
    } finally {
        $ErrorActionPreference = 'Stop'
        Pop-Location
    }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE $Name" | Tee-Object -FilePath $PipelineLog -Append
}

Invoke-Stage 'q3_diffusion_multiseed_generative_metrics_retry' $MetricsDir @(
    'q3_generative_metrics.py',
    '--config', (Join-Path $MetricsDir 'q3_generative_metrics_config.json'),
    '--output', (Join-Path $ProjectRoot 'data\yolo_dataset\q3_generative_metrics.json')
)
Invoke-Stage 'q3_diffusion_multiseed_statistics' $YoloScripts @('analyze_q3_diffusion_multiseed.py')
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DIFFUSION MULTISEED PIPELINE COMPLETE" | Tee-Object -FilePath $PipelineLog -Append
