$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$WorkDir = Join-Path $ProjectRoot 'implementations\wgan_gp'
$PythonExe = (Get-Command python -CommandType Application -ErrorAction Stop).Source
$DataDir = Join-Path $ProjectRoot 'data\cable_leakfix_train'
$GenRoot = Join-Path $ProjectRoot 'data\yolo_dataset\images\leakfix'
$LogRoot = Join-Path $ProjectRoot 'logs'

Set-Location -LiteralPath $WorkDir

function Invoke-LoggedPython {
    param([string]$LogPath, [string[]]$Arguments)
    $ErrorLog = "$LogPath.stderr"
    $QuotedArguments = $Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    $Process = Start-Process -FilePath $PythonExe -ArgumentList ($QuotedArguments -join ' ') `
        -WorkingDirectory $WorkDir -RedirectStandardOutput $LogPath `
        -RedirectStandardError $ErrorLog -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 80
        Get-Content -LiteralPath $ErrorLog -Tail 80
        throw "Python command failed with exit code $($Process.ExitCode); see $LogPath"
    }
}

function Run-One {
    param(
        [string]$Model,
        [string]$Script,
        [double]$LambdaSp,
        [int]$Seed,
        [string]$GeneratorScript
    )
    $Tag = ("{0}_s{1}" -f $Model, $Seed)
    $FinalCheckpoint = Join-Path $WorkDir ("checkpoint_{0}_final.pth" -f $Tag)
    $LatestCheckpoint = Join-Path $WorkDir ("latest_checkpoint_{0}.pth" -f $Tag)
    $TrainLog = Join-Path $LogRoot ("q3_multiseed_train_{0}.log" -f $Tag)

    if (-not (Test-Path -LiteralPath $FinalCheckpoint)) {
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAIN $Tag"
        $Args = @(
            $Script, '--n_epochs', '500', '--lambda_sp',
            $LambdaSp.ToString([Globalization.CultureInfo]::InvariantCulture),
            '--lambda_fm', '50', '--batch_size', '4', '--data_path', $DataDir,
            '--tag', $Tag, '--img_dir', ("images_leakfix/{0}" -f $Tag),
            '--seed', $Seed.ToString()
        )
        if (Test-Path -LiteralPath $LatestCheckpoint) {
            $Args += @('--resume', $LatestCheckpoint)
        }
        Invoke-LoggedPython -LogPath $TrainLog -Arguments $Args
    } else {
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP checkpoint $Tag"
    }

    $OutputName = "gen_{0}_5k" -f $Tag
    $OutputPath = Join-Path $GenRoot $OutputName
    $Count = if (Test-Path -LiteralPath $OutputPath) {
        (Get-ChildItem -LiteralPath $OutputPath -Recurse -File -Filter '*.png').Count
    } else { 0 }
    if ($Count -ne 5000) {
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] GENERATE $Tag"
        $GenArgs = @($GeneratorScript, '--checkpoint', $FinalCheckpoint,
                     '--n_per_class', '1250', '--output', $OutputPath)
        if ($GeneratorScript -eq 'cgenerate_images_clip.py') { $GenArgs += '--by_class' }
        Invoke-LoggedPython -LogPath (Join-Path $LogRoot ("q3_multiseed_gen_{0}.log" -f $Tag)) -Arguments $GenArgs
    } else {
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP generation $Tag"
    }

    $FlatPath = Join-Path $GenRoot ("gen_{0}_flat" -f $Tag)
    New-Item -ItemType Directory -Path $FlatPath -Force | Out-Null
    Get-ChildItem -LiteralPath $FlatPath -File -Filter '*.png' | Remove-Item -Force
    foreach ($ClassDir in Get-ChildItem -LiteralPath $OutputPath -Directory) {
        Get-ChildItem -LiteralPath $ClassDir.FullName -File -Filter '*.png' |
            Sort-Object Name | Select-Object -First 250 | Copy-Item -Destination $FlatPath
    }
    $FlatCount = (Get-ChildItem -LiteralPath $FlatPath -File -Filter '*.png').Count
    if ($FlatCount -ne 1000) { throw "$Tag flat subset expected 1000 images, found $FlatCount" }
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE $Tag"
}

# Seed 42 is already complete.  Additional seeds are run sequentially because
# the single 8 GB GPU cannot safely train multiple 512x512 GANs concurrently.
foreach ($Seed in @(1, 2)) {
    Run-One -Model 'e7proj_mses_nosp_leak' -Script 'cwgan_gp_clip_proj.py' `
        -LambdaSp 0 -Seed $Seed -GeneratorScript 'cgenerate_images_clip.py'
}
foreach ($Seed in @(1, 2)) {
    Run-One -Model 'e6proj_mses_nosp_leak' -Script 'cwgan_gp_proj.py' `
        -LambdaSp 0 -Seed $Seed -GeneratorScript 'cgenerate_images.py'
}
foreach ($Seed in @(1, 2)) {
    Run-One -Model 'e7proj_mses_sp_leak' -Script 'cwgan_gp_clip_proj.py' `
        -LambdaSp 5 -Seed $Seed -GeneratorScript 'cgenerate_images_clip.py'
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL MULTI-SEED GAN RUNS COMPLETE"
