$ErrorActionPreference = 'Stop'

$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$WorkDir = Join-Path $Root 'implementations\wgan_gp'
$Python = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$DataPath = Join-Path $Root 'data\neu_det\NEU-DET\train\images'
$Log = Join-Path $Root 'logs\q3_neu_comparators_pipeline.log'
$ErrLog = Join-Path $Root 'logs\q3_neu_comparators_pipeline.err.log'

function Invoke-Logged([string]$Program, [string[]]$Arguments) {
    # Windows PowerShell wraps native stderr (including harmless Python
    # warnings) as NativeCommandError when Stop is active. Preserve stderr in
    # the log while deciding success solely from the native exit code.
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Program @Arguments 2>> $ErrLog | Tee-Object -FilePath $Log -Append
    $NativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorAction
    if ($NativeExitCode -ne 0) { throw "Command failed ($NativeExitCode): $Program $Arguments" }
}

function Complete-GanGroup(
    [string]$Kind,
    [string]$Tag,
    [string]$Output,
    [string]$Checkpoint
) {
    Set-Location -LiteralPath $WorkDir
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAIN $Tag" | Tee-Object -FilePath $Log -Append
        $Latest = Join-Path $WorkDir "latest_checkpoint_${Tag}.pth"
        if ($Kind -eq 'text') {
            $TrainArgs = @('cwgan_gp_clip_proj.py', '--n_epochs', '500', '--batch_size', '8',
                '--img_size', '256', '--lambda_sp', '0', '--data_path', $DataPath,
                '--tag', $Tag, '--img_dir', (Join-Path 'images_neu_q3' $Tag), '--seed', '42')
        } else {
            $TrainArgs = @('cwgan_gp_proj.py', '--n_epochs', '500', '--batch_size', '8',
                '--img_size', '256', '--lambda_sp', '0', '--num_classes', '6', '--data_path', $DataPath,
                '--tag', $Tag, '--img_dir', (Join-Path 'images_neu_q3' $Tag), '--seed', '42')
        }
        if (Test-Path -LiteralPath $Latest) { $TrainArgs += @('--resume', $Latest) }
        Invoke-Logged -Program $Python -Arguments $TrainArgs
    } else {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP checkpoint $Tag" | Tee-Object -FilePath $Log -Append
    }

    $Count = 0
    if (Test-Path -LiteralPath $Output) {
        $Count = (Get-ChildItem -LiteralPath $Output -Recurse -File -Filter '*.png').Count
    }
    if ($Count -ne 3000) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] GENERATE $Tag" | Tee-Object -FilePath $Log -Append
        if ($Kind -eq 'text') {
            $GenArgs = @('cgenerate_images_clip.py', '--checkpoint', $Checkpoint,
                '--n_per_class', '500', '--output', $Output, '--img_size', '256',
                '--batch_size', '25', '--dataset', 'neu', '--by_class')
        } else {
            $GenArgs = @('cgenerate_images.py', '--checkpoint', $Checkpoint,
                '--n_per_class', '500', '--output', $Output, '--img_size', '256',
                '--batch_size', '25', '--num_classes', '6', '--dataset', 'neu')
        }
        Invoke-Logged -Program $Python -Arguments $GenArgs
    } else {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP generation $Tag" | Tee-Object -FilePath $Log -Append
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START NEU comparator pipeline" | Tee-Object -FilePath $Log -Append

$TextTag = 'e7proj_mses_nosp_neu_s42'
$TextCheckpoint = Join-Path $WorkDir "checkpoint_${TextTag}_final.pth"
$TextOutput = Join-Path $Root 'data\neu_det\yolo\images\gen_E7proj_mses_nosp_neu_s42'
Complete-GanGroup 'text' $TextTag $TextOutput $TextCheckpoint

$ClassTag = 'e6proj_mses_nosp_neu_s42'
$ClassCheckpoint = Join-Path $WorkDir "checkpoint_${ClassTag}_final.pth"
$ClassOutput = Join-Path $Root 'data\neu_det\yolo\images\gen_E6proj_mses_nosp_neu_s42'
Complete-GanGroup 'class' $ClassTag $ClassOutput $ClassCheckpoint

Set-Location -LiteralPath $Root
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PSEUDO/PREPARE/DETECT" | Tee-Object -FilePath $Log -Append
Invoke-Logged -Program $Python -Arguments @('-S', 'implementations\yolov8_det\scripts\q3_neu_comparators_pipeline.py', 'all')
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL NEU COMPARATORS COMPLETE" | Tee-Object -FilePath $Log -Append
