$ErrorActionPreference = "Stop"

$Device = "cuda:0"
$Epochs = 60
$Horizons = @(96)
$Seeds = @(0)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$baseArgs = @(
    "--model_name", "KAN_TQNet",
    "--device", $Device,
    "--epochs", "$Epochs"
)

$experiments = @(
    @{
        Tag = "core_full"
        Desc = "Full wavelet-main + FFT-aux KAN_TQNet"
        Args = @()
    },
    @{
        Tag = "core_no_kan"
        Desc = "Disable KAN backbone"
        Args = @("--use_kan", "0")
    },
    @{
        Tag = "core_no_tq"
        Desc = "Disable Temporal Query"
        Args = @("--use_tq", "0")
    },
    @{
        Tag = "core_no_freq"
        Desc = "Disable spectral branch"
        Args = @("--use_freq_branch", "0")
    },
    @{
        Tag = "core_no_trend"
        Desc = "Disable trend residual decomposition"
        Args = @("--use_trend_residual", "0")
    },
    @{
        Tag = "core_no_hda"
        Desc = "Disable HDA routing"
        Args = @("--use_hda", "0")
    },
    @{
        Tag = "core_no_timemixer"
        Desc = "Disable TimeMixer refiner"
        Args = @("--use_timemixer", "0")
    }
)

foreach ($seed in $Seeds) {
    foreach ($horizon in $Horizons) {
        foreach ($exp in $experiments) {
            $tag = "{0}_s{1}" -f $exp.Tag, $seed
            $cmdArgs = @(
                "--pred_len_override", "$horizon",
                "--seed", "$seed",
                "--exp_tag", $tag
            ) + $baseArgs + $exp.Args

            Write-Host ""
            Write-Host "========== Running $($exp.Tag) | horizon=$horizon | seed=$seed =========="
            Write-Host $exp.Desc
            Write-Host ("python .\train.py " + ($cmdArgs -join " "))
            & python .\train.py @cmdArgs
        }
    }
}

Write-Host ""
Write-Host "Core ablation runs finished."
