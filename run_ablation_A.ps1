$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$experiments = @(
    @{
        Tag = "A0_full"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A0_full")
    },
    @{
        Tag = "A1_no_heads"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A1_no_heads", "--use_channel_heads", "0")
    },
    @{
        Tag = "A2_no_adapter"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A2_no_adapter", "--use_channel_adapter", "0")
    },
    @{
        Tag = "A3_no_ele_refine"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A3_no_ele_refine", "--use_electricity_refine", "0")
    },
    @{
        Tag = "A4_no_hda"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A4_no_hda", "--use_hda", "0")
    },
    @{
        Tag = "A5_no_multiscale"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A5_no_multiscale", "--use_multi_scale", "0")
    },
    @{
        Tag = "A6_no_freq"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A6_no_freq", "--use_freq_branch", "0")
    },
    @{
        Tag = "A7_no_trend"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A7_no_trend", "--use_trend_residual", "0")
    },
    @{
        Tag = "A8_no_timemixer"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A8_no_timemixer", "--use_timemixer", "0")
    },
    @{
        Tag = "A9_no_patchms"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A9_no_patchms", "--use_patch_multiscale", "0")
    },
    @{
        Tag = "A10_no_localconv"
        Args = @("--model_name", "KAN_TQNet", "--exp_tag", "A10_no_localconv", "--use_local_conv", "0")
    }
)

foreach ($exp in $experiments) {
    $cmd = @("python", ".\train.py") + $exp.Args
    Write-Host ""
    Write-Host "========== Running $($exp.Tag) =========="
    Write-Host ($cmd -join " ")
    & python .\train.py @($exp.Args)
}

Write-Host ""
Write-Host "All A-group ablations finished."
