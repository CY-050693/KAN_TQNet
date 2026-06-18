$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$fullArgs = @(
    "--model_name", "KAN_TQNet",
    "--use_tq", "1",
    "--use_kan", "1",
    "--use_freq_branch", "1",
    "--use_hda", "1",
    "--use_multi_scale", "1",
    "--use_timemixer", "1"
)

$experiments = @(
    @{
        Tag = "AFull_Model"
        Desc = "Full baseline"
        Args = @("--exp_tag", "AFull_Model")
    },
    @{
        Tag = "B_no_KAN"
        Desc = "KAN -> MLP"
        Args = @("--exp_tag", "B_no_KAN", "--use_kan", "0")
    },
    @{
        Tag = "C_no_TQ"
        Desc = "Disable Temporal Query"
        Args = @("--exp_tag", "C_no_TQ", "--use_tq", "0")
    },
    @{
        Tag = "D_no_Freq"
        Desc = "Disable frequency branch"
        Args = @("--exp_tag", "D_no_Freq", "--use_freq_branch", "0")
    },
    @{
        Tag = "E_no_HDA"
        Desc = "HDA routing -> single branch"
        Args = @("--exp_tag", "E_no_HDA", "--use_hda", "0")
    },
    @{
        Tag = "F_no_MultiScale"
        Desc = "Disable multi-scale branch"
        Args = @("--exp_tag", "F_no_MultiScale", "--use_multi_scale", "0")
    },
    @{
        Tag = "G_no_TimeMixer"
        Desc = "Disable TimeMixer refiner"
        Args = @("--exp_tag", "G_no_TimeMixer", "--use_timemixer", "0")
    }
)

foreach ($exp in $experiments) {
    $cmd = @("python", ".\train.py") + $fullArgs + $exp.Args
    Write-Host ""
    Write-Host "========== Running $($exp.Tag) =========="
    Write-Host $exp.Desc
    Write-Host ($cmd -join " ")
    & python .\train.py @($fullArgs + $exp.Args)
}

Write-Host ""
Write-Host "Core ablations finished."
