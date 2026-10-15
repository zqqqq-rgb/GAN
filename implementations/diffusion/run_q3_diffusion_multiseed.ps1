$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$PythonExe = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$WorkDir = Join-Path $ProjectRoot 'implementations\diffusion'
$YoloScripts = Join-Path $ProjectRoot 'implementations\yolov8_det\scripts'
$MetricsDir = Join-Path $ProjectRoot 'implementations\wgan_gp'
$LogRoot = Join-Path $ProjectRoot 'logs'
$PipelineLog = Join-Path $LogRoot 'q3_diffusion_multiseed_pipeline.log'
$DetectorSeeds = @('42', '1', '2', '3', '4', '5', '6', '7', '8', '9')

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.ultralytics'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

function Write-Pipeline([string]$Message) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Tee-Object -FilePath $PipelineLog -Append
}

function Invoke-CheckedPython([string]$Stage, [string]$WorkingDirectory, [string[]]$Arguments) {
    Write-Pipeline "START $Stage"
    Push-Location -LiteralPath $WorkingDirectory
    try {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $LogRoot "$Stage.log") -Append
        $ExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousPreference
        if ($ExitCode -ne 0) { throw "Stage failed: $Stage (exit code $ExitCode)" }
    } finally {
        $ErrorActionPreference = 'Stop'
        Pop-Location
    }
    Write-Pipeline "COMPLETE $Stage"
}

foreach ($Seed in @(1, 2)) {
    $Group = "Diffusion_LDM_gs$Seed"
    $Cache = Join-Path $ProjectRoot "data\diffusion\cable_ldm_latents_s$Seed.pt"
    $CheckpointDir = Join-Path $WorkDir "checkpoints\cable_ldm_s$Seed"
    $FinalCheckpoint = Join-Path $CheckpointDir 'final.pt'
    $Output = Join-Path $ProjectRoot "data\yolo_dataset\images\leakfix\gen_Diffusion_LDM_s${Seed}_5k"
    $FlatOutput = Join-Path $ProjectRoot "data\yolo_dataset\images\leakfix\gen_Diffusion_LDM_s${Seed}_5k_flat"
    $Flat1k = Join-Path $ProjectRoot "data\yolo_dataset\images\leakfix\gen_Diffusion_LDM_s${Seed}_flat"
    $Protocol = Join-Path $ProjectRoot "data\diffusion\q3_diffusion_protocol_s$Seed.json"
    $Common = @(
        '--seed', "$Seed",
        '--sample-seed', '20260914',
        '--cache', $Cache,
        '--checkpoint-dir', $CheckpointDir,
        '--output', $Output,
        '--flat-output', $FlatOutput,
        '--flat-1k', $Flat1k,
        '--protocol', $Protocol
    )

    Invoke-CheckedPython "q3_diffusion_s${Seed}_cache" $WorkDir (@('conditional_latent_ddpm.py', 'cache') + $Common)
    if (-not (Test-Path -LiteralPath $FinalCheckpoint)) {
        Invoke-CheckedPython "q3_diffusion_s${Seed}_train" $WorkDir (@('conditional_latent_ddpm.py', 'train') + $Common)
    } else {
        Write-Pipeline "SKIP q3_diffusion_s${Seed}_train final checkpoint exists"
    }
    Invoke-CheckedPython "q3_diffusion_s${Seed}_sample" $WorkDir (@('conditional_latent_ddpm.py', 'sample') + $Common)
    Invoke-CheckedPython "q3_diffusion_s${Seed}_pseudo" $YoloScripts @(
        'leakfix_det_pipeline.py', 'pseudo', '--conf', '0.10', '--pseudo-groups', $Group
    )
    Invoke-CheckedPython "q3_diffusion_s${Seed}_q300_prepare" $YoloScripts @(
        'q3_fixedq_pipeline.py', 'prepare', '--q', '300', '--groups', $Group
    )
    $DetectionArgs = @('q3_fixedq_pipeline.py', 'det', '--q', '300', '--seeds') +
        $DetectorSeeds + @('--groups', $Group)
    Invoke-CheckedPython "q3_diffusion_s${Seed}_q300_detection" $YoloScripts $DetectionArgs
}

Invoke-CheckedPython 'q3_diffusion_multiseed_generative_metrics' $MetricsDir @(
    'q3_generative_metrics.py',
    '--config', (Join-Path $MetricsDir 'q3_generative_metrics_config.json'),
    '--output', (Join-Path $ProjectRoot 'data\yolo_dataset\q3_generative_metrics.json')
)
Invoke-CheckedPython 'q3_diffusion_multiseed_statistics' $YoloScripts @('analyze_q3_diffusion_multiseed.py')
Write-Pipeline 'DIFFUSION MULTISEED PIPELINE COMPLETE'
