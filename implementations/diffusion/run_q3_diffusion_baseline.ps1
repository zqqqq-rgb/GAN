$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$PythonExe = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$WorkDir = Join-Path $ProjectRoot 'implementations\diffusion'
$LogRoot = Join-Path $ProjectRoot 'logs'
$PipelineLog = Join-Path $LogRoot 'q3_diffusion_pipeline.log'
$TrainLog = Join-Path $LogRoot 'q3_diffusion_train.log'
$TrainErr = Join-Path $LogRoot 'q3_diffusion_train.stderr.log'

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location -LiteralPath $WorkDir

function Invoke-Stage([string]$Name, [string[]]$Arguments) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $Name" | Tee-Object -FilePath $PipelineLog -Append
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name" }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE $Name" | Tee-Object -FilePath $PipelineLog -Append
}

Invoke-Stage 'latent_cache' @('conditional_latent_ddpm.py', 'cache')

$FinalCheckpoint = Join-Path $WorkDir 'checkpoints\cable_ldm_s42\final.pt'
if (-not (Test-Path -LiteralPath $FinalCheckpoint)) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START diffusion_train" | Tee-Object -FilePath $PipelineLog -Append
    $Process = Start-Process -FilePath $PythonExe `
        -ArgumentList @('conditional_latent_ddpm.py', 'train') `
        -WorkingDirectory $WorkDir -RedirectStandardOutput $TrainLog `
        -RedirectStandardError $TrainErr -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Diffusion training failed; see $TrainLog and $TrainErr" }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE diffusion_train" | Tee-Object -FilePath $PipelineLog -Append
} else {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP diffusion_train final checkpoint exists" | Tee-Object -FilePath $PipelineLog -Append
}

Invoke-Stage 'diffusion_sample_5k' @('conditional_latent_ddpm.py', 'sample')

$YoloScripts = Join-Path $ProjectRoot 'implementations\yolov8_det\scripts'
Set-Location -LiteralPath $YoloScripts
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.ultralytics'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
Invoke-Stage 'pseudo_label' @('leakfix_det_pipeline.py', 'pseudo', '--conf', '0.10', '--pseudo-groups', 'Diffusion_LDM')
Invoke-Stage 'fixed_q300_prepare' @('q3_fixedq_pipeline.py', 'prepare', '--q', '300', '--groups', 'Diffusion_LDM')
Invoke-Stage 'fixed_q300_detection' @(
    'q3_fixedq_pipeline.py', 'det', '--q', '300', '--seeds',
    '42', '1', '2', '3', '4', '5', '6', '7', '8', '9', '--groups', 'Diffusion_LDM'
)
Invoke-Stage 'updated_statistics' @('analyze_q3_fixedq_statistics.py')

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DIFFUSION BASELINE PIPELINE COMPLETE" | Tee-Object -FilePath $PipelineLog -Append
