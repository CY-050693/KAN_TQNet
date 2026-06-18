$ErrorActionPreference = "Stop"


$Device = "cuda:0"
$Epochs = 60
$Horizons = @(48, 72, 96)
$Seeds = @(0)  # 如果你需要多种子，改成 @(0,1,2)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$experiments = @(
    @{
        Tag = "P0_backbone_mlp"
        Desc = "Minimal backbone: remove KAN/TQ/channel decoupling/frequency/trend/HDA/refinement/TimeMixer"
        Args = @(
            "--use_kan", "0",
            "--use_tq", "0",
            "--use_channel_heads", "0",
            "--use_channel_adapter", "0",
            "--use_freq_branch", "0",
            "--use_trend_residual", "0",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P1_plus_kan"
        Desc = "Add KAN backbone"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "0",
            "--use_channel_heads", "0",
            "--use_channel_adapter", "0",
            "--use_freq_branch", "0",
            "--use_trend_residual", "0",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P2_plus_channel"
        Desc = "Add channel heads and channel adapter"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "0",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "0",
            "--use_trend_residual", "0",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P3_plus_tq"
        Desc = "Add Temporal Query"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "0",
            "--use_trend_residual", "0",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P4_plus_freq"
        Desc = "Add frequency branch"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "1",
            "--use_trend_residual", "0",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P5_plus_trend"
        Desc = "Add trend residual"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "1",
            "--use_trend_residual", "1",
            "--use_hda", "0",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P6_plus_hda"
        Desc = "Add HDA routing"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "1",
            "--use_trend_residual", "1",
            "--use_hda", "1",
            "--use_electricity_refine", "0",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P7_plus_ele_refine"
        Desc = "Add electricity refinement head"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "1",
            "--use_trend_residual", "1",
            "--use_hda", "1",
            "--use_electricity_refine", "1",
            "--use_timemixer", "0"
        )
    },
    @{
        Tag = "P8_plus_timemixer"
        Desc = "Add TimeMixer refiner (full progressive endpoint)"
        Args = @(
            "--use_kan", "1",
            "--use_tq", "1",
            "--use_channel_heads", "1",
            "--use_channel_adapter", "1",
            "--use_freq_branch", "1",
            "--use_trend_residual", "1",
            "--use_hda", "1",
            "--use_electricity_refine", "1",
            "--use_timemixer", "1"
        )
    }
)

foreach ($seed in $Seeds) {
    foreach ($horizon in $Horizons) {
        foreach ($exp in $experiments) {
            $tag = "{0}_s{1}" -f $exp.Tag, $seed
            $cmdArgs = @(
                "--model_name", "KAN_TQNet",
                "--device", $Device,
                "--epochs", "$Epochs",
                "--pred_len_override", "$horizon",
                "--seed", "$seed",
                "--exp_tag", $tag
            ) + $exp.Args

            Write-Host ""
            Write-Host "========== Running $($exp.Tag) | horizon=$horizon | seed=$seed =========="
            Write-Host $exp.Desc
            Write-Host ("python .\\train.py " + ($cmdArgs -join " "))
            & python .\train.py @cmdArgs
        }
    }
}

Write-Host ""
Write-Host "Progressive ablation runs finished."