$ErrorActionPreference = 'Continue'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$WorkDir = Join-Path $ProjectRoot 'implementations\wgan_gp'
$PythonExe = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$DataDir = Join-Path $ProjectRoot 'data\cable_leakfix_train'
$GenRoot = Join-Path $ProjectRoot 'data\yolo_dataset\images\leakfix'
$LogRoot = Join-Path $ProjectRoot 'logs'

Set-Location -LiteralPath $WorkDir

function Invoke-CheckedPython {
    param(
        [string]$LogPath,
        [string[]]$Arguments
    )
    # Windows PowerShell wraps native stderr (including harmless PyTorch
    # warnings) in NativeCommandError records.  Continue here and rely on the
    # process exit code for the actual success/failure decision.
    & $PythonExe @Arguments *> $LogPath
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 80
        throw "Python command failed; see $LogPath"
    }
}

function Train-Model {
    param(
        [string]$Script,
        [string]$Tag,
        [double]$LambdaSp,
        [string]$ImageDir,
        [string]$LogName
    )
    $FinalCheckpoint = Join-Path $WorkDir ("checkpoint_{0}_final.pth" -f $Tag)
    if (Test-Path -LiteralPath $FinalCheckpoint) {
        Write-Output "[SKIP] $Tag final checkpoint already exists"
        return
    }
    Write-Output "[TRAIN] $Tag"
    Invoke-CheckedPython -LogPath (Join-Path $LogRoot $LogName) -Arguments @(
        $Script,
        '--n_epochs', '500',
        '--lambda_sp', $LambdaSp.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--lambda_fm', '50',
        '--batch_size', '4',
        '--data_path', $DataDir,
        '--tag', $Tag,
        '--img_dir', $ImageDir,
        '--seed', '42',
        '--resume', ("latest_checkpoint_{0}.pth" -f $Tag)
    )
}

function Generate-ConditionalImages {
    param(
        [string]$GeneratorScript,
        [string]$Checkpoint,
        [string]$OutputName,
        [string]$LogName
    )
    $OutputPath = Join-Path $GenRoot $OutputName
    $Existing = if (Test-Path -LiteralPath $OutputPath) {
        (Get-ChildItem -LiteralPath $OutputPath -Recurse -File -Filter '*.png').Count
    } else { 0 }
    if ($Existing -eq 5000) {
        Write-Output "[SKIP] $OutputName already has 5000 images"
        return
    }
    Write-Output "[GENERATE] $OutputName"
    $Arguments = @(
        $GeneratorScript,
        '--checkpoint', $Checkpoint,
        '--n_per_class', '1250',
        '--output', $OutputPath
    )
    if ($GeneratorScript -eq 'cgenerate_images_clip.py') {
        $Arguments += '--by_class'
    }
    Invoke-CheckedPython -LogPath (Join-Path $LogRoot $LogName) -Arguments $Arguments
}

function New-FlatDetectionSubset {
    param([string]$SourceName, [string]$DestinationName)
    $SourcePath = Join-Path $GenRoot $SourceName
    $DestinationPath = Join-Path $GenRoot $DestinationName
    New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
    Get-ChildItem -LiteralPath $DestinationPath -File | Remove-Item -Force
    foreach ($ClassDir in Get-ChildItem -LiteralPath $SourcePath -Directory) {
        Get-ChildItem -LiteralPath $ClassDir.FullName -File -Filter '*.png' |
            Sort-Object Name |
            Select-Object -First 250 |
            Copy-Item -Destination $DestinationPath
    }
    $Count = (Get-ChildItem -LiteralPath $DestinationPath -File -Filter '*.png').Count
    if ($Count -ne 1000) {
        throw "$DestinationName expected 1000 images, found $Count"
    }
    Write-Output "[FLAT] $DestinationName = $Count images"
}

Train-Model -Script 'cwgan_gp_clip_proj.py' -Tag 'e7proj_mses_sp_leak' -LambdaSp 5 `
    -ImageDir 'images_leakfix/E7proj_mses_sp' -LogName 'q3_train_E7proj_mses_sp.log'
Train-Model -Script 'cwgan_gp_clip_proj.py' -Tag 'e7proj_mses_nosp_leak' -LambdaSp 0 `
    -ImageDir 'images_leakfix/E7proj_mses_nosp' -LogName 'q3_train_E7proj_mses_nosp.log'
Train-Model -Script 'cwgan_gp_proj.py' -Tag 'e6proj_mses_sp_leak' -LambdaSp 5 `
    -ImageDir 'images_leakfix/E6proj_mses_sp' -LogName 'q3_train_E6proj_mses_sp.log'
Train-Model -Script 'cwgan_gp_proj.py' -Tag 'e6proj_mses_nosp_leak' -LambdaSp 0 `
    -ImageDir 'images_leakfix/E6proj_mses_nosp' -LogName 'q3_train_E6proj_mses_nosp.log'

Generate-ConditionalImages -GeneratorScript 'cgenerate_images_clip.py' `
    -Checkpoint 'checkpoint_e7proj_mses_sp_leak_final.pth' `
    -OutputName 'gen_E7proj_mses_sp_leak_5k' -LogName 'q3_gen_E7proj_mses_sp.log'
Generate-ConditionalImages -GeneratorScript 'cgenerate_images_clip.py' `
    -Checkpoint 'checkpoint_e7proj_mses_nosp_leak_final.pth' `
    -OutputName 'gen_E7proj_mses_nosp_leak_5k' -LogName 'q3_gen_E7proj_mses_nosp.log'
Generate-ConditionalImages -GeneratorScript 'cgenerate_images.py' `
    -Checkpoint 'checkpoint_e6proj_mses_sp_leak_final.pth' `
    -OutputName 'gen_E6proj_mses_sp_leak_5k' -LogName 'q3_gen_E6proj_mses_sp.log'
Generate-ConditionalImages -GeneratorScript 'cgenerate_images.py' `
    -Checkpoint 'checkpoint_e6proj_mses_nosp_leak_final.pth' `
    -OutputName 'gen_E6proj_mses_nosp_leak_5k' -LogName 'q3_gen_E6proj_mses_nosp.log'

New-FlatDetectionSubset -SourceName 'gen_E7proj_mses_sp_leak_5k' -DestinationName 'gen_E7proj_mses_sp_leak_flat'
New-FlatDetectionSubset -SourceName 'gen_E7proj_mses_nosp_leak_5k' -DestinationName 'gen_E7proj_mses_nosp_leak_flat'
New-FlatDetectionSubset -SourceName 'gen_E6proj_mses_sp_leak_5k' -DestinationName 'gen_E6proj_mses_sp_leak_flat'
New-FlatDetectionSubset -SourceName 'gen_E6proj_mses_nosp_leak_5k' -DestinationName 'gen_E6proj_mses_nosp_leak_flat'

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Invoke-CheckedPython -LogPath (Join-Path $LogRoot 'q3_generative_metrics.log') -Arguments @(
    'q3_generative_metrics.py',
    '--config', (Join-Path $WorkDir 'q3_generative_metrics_config.json'),
    '--output', (Join-Path $ProjectRoot 'data\yolo_dataset\q3_generative_metrics.json'),
    '--max_fake', '1000',
    '--seed', '20260910'
)

Write-Output '[COMPLETE] Projection models, permutation-invariant structure ablation, generation, and unified metrics finished.'
