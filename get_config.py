import argparse


class BaseConfig:
    """基类，包含所有模型共享的超参数"""

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 48
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.dropout = 0.1
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.moving_avg = 24
        self.factor = 1


class PatchTSTConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 128
        self.enc_in = 12

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class TimeMixerConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast'、'short_term_forecast'、'imputation' 等
        self.seq_len = 168  # 输入序列长度 T
        # 目标序列长度 L
        self.pred_in_len = 168  # 预测时间步长

        self.pred_len = 24
        self.individual = True

        self.down_sampling_window = 2  # 下采样窗口大小
        self.channel_independence = False  # 是否进行通道独立处理
        self.e_layers = 3  # 编码器层数
        self.down_sampling_layers = 2  # 下采样层数
        self.enc_in = 12  # 输入通道数
        self.embed = 64  # 嵌入层维度
        self.d_model = 64  # 模型内部表示的维度
        self.freq = 'h'  # 数据的频率（如：'h'表示小时数据）
        self.dropout = 0.1  # Dropout比率
        self.use_norm = 1  # 是否使用归一化（1表示使用）
        self.c_out = 12  # 输出通道数（通常用于预测任务）
        self.num_class = 10  # 分类任务的类别数
        self.moving_avg = 3  # 时间序列分解中的移动平均窗口大小
        self.down_sampling_method = 'avg'  # 下采样方法（'max'、'avg'、'conv'）
        self.d_ff = 64

        self.lr = 0.00016
        self.batchsize = 256
        self.epochs = 60
        self.weightdecay = 0.8
        self.decaypatience = 6


class TimesNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Task-related configurations
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'
        self.seq_len = 168  # 输入序列长度 T (例如，168表示一周的小时数据)
        self.label_len = 96  # 目标序列长度 L (用于预测的时间步数)
        self.pred_len = 96  # 预测时间步长
        self.e_layers = 2  # 编码器层数
        self.d_model = 32  # 模型的维度（例如：每个时间步的特征数量）
        self.dropout = 0.1  # Dropout比率
        self.freq = 'h'  # 数据频率（例如：'h'表示小时数据）

        # Embedding-related configurations
        self.enc_in = 12  # 输入特征的数量
        self.embed = 16  # 嵌入层的维度
        self.use_norm = 1  # 是否使用归一化（1表示使用）

        # Output-related configurations
        self.c_out = 3  # 输出通道数（通常用于回归任务）
        # self.num_class = 10  # 分类任务的类别数（如果是分类任务使用）
        self.num_kernels = 4
        # Time series specific configurations
        self.moving_avg = 3  # 时间序列分解中的移动平均窗口大小
        self.down_sampling_window = 2  # 下采样窗口大小
        self.channel_independence = False  # 是否进行通道独立处理
        self.down_sampling_layers = 2  # 下采样层数
        self.down_sampling_method = 'avg'  # 下采样方法（'max', 'avg', 'conv'）
        self.top_k = 5

        # Feed-forward layer dimension
        self.d_ff = 16  # Feed-forward层的维度

        self.lr = 0.00008
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.8
        self.decaypatience = 5


class TSMixerConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Task-related configurations
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'
        self.seq_len = 168  # 输入序列长度 T (例如，168表示一周的小时数据)
        self.label_len = 96  # 目标序列长度 L (用于预测的时间步数)
        self.pred_len = 96  # 预测时间步长
        self.e_layers = 2  # 编码器层数
        self.d_model = 64  # 模型的维度（例如：每个时间步的特征数量）
        self.dropout = 0.2  # Dropout比率
        self.freq = 'h'  # 数据频率（例如：'h'表示小时数据）

        # 输入和输出的通道数，通常在时间序列预测任务中，这会是特征的数量
        self.enc_in = 12  # 输入特征维度 (例如：7个通道或7种特征)
        self.dec_in = 12  # 如果有解码器输入，通常与编码器相同

        self.lr = 0.0008
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class FITSConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.enc_in = 12
        self.seq_len = 168
        self.pred_len = 96
        self.individual = True

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.5
        self.decaypatience = 5


class CFCConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.in_features = 12
        self.hidden_size = 64
        self.out_feature = 3
        self.seq_in_len = 168
        self.seq_out_len =96
        self.individual = True
        self.batch = 256
        self.pred_len = self.seq_out_len

        self.lr = 0.001
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.8
        self.decaypatience = 5


class KAN_TQNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.task_name = 'long_term_forecast'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 24
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 3
        self.d_model = 64
        self.dropout = 0.05

        self.cycle = 24
        self.model_type = 'KAN_TQNet'
        self.use_revin = 1
        self.use_hda = 1
        self.hda_consistency_prefix_len = 24
        self.use_patch_multiscale = 0
        self.patch_fine_len = 8
        self.patch_fine_stride = 4
        self.patch_coarse_len = 16
        self.patch_coarse_stride = 8
        self.use_uncertainty = 0
        self.use_kan = 1
        self.kan_grid_size = 4
        self.kan_hidden_dim = 64
        self.use_trend_residual = 1
        self.trend_kernel_size = 25
        self.trend_hidden_dim = 64
        self.use_freq_branch = 1
        self.freq_topk = 16
        self.freq_hidden_dim = 64
        self.use_complex_freq = 0
        self.use_freq_weight = 1
        self.wavelet_levels = 3
        self.wavelet_pool_len = 4
        self.use_fft_aux = 1
        self.fft_aux_scale = 0.35
        self.use_multi_scale = 0
        self.multi_scale_factor = 2
        self.low_branch_hidden_dim = 64
        self.use_local_conv = 0
        self.use_stat_gate = 1
        self.use_cooling_refine = 0
        self.cooling_refine_hidden_dim = 64
        self.cooling_refine_scale = 0.1
        self.use_timemixer = 1
        self.timemixer_e_layers = 1
        self.timemixer_down_sampling_layers = 2
        self.timemixer_down_sampling_window = 2
        self.timemixer_moving_avg = 25
        self.timemixer_d_ff = 64
        self.use_tq = 1
        self.use_kan = 1
        self.target_channels = 3
        self.use_channel_heads = 1
        self.use_channel_adapter = 1
        self.channel_adapter_hidden_dim = 64
        self.channel_loss_weights = (1.7, 1.2, 1.1)
        self.balanced_channel_priority = (1.6, 1.0, 1.1)
        self.mixed_mae_ratio_weight = 0.2
        self.use_electricity_refine = 1
        self.electricity_refine_hidden_dim = 64
        self.electricity_refine_scale = 0.08
        self.electricity_peak_weight = 0.25
        self.electricity_peak_quantile = 0.85
        self.cooling_peak_weight = 0.0
        self.cooling_peak_quantile = 0.75
        self.use_balanced_checkpoint = 1
        self.balanced_guard_relax = 0.03
        self.balanced_guard_abs = 0.2

        self.lr = 0.0005
        self.batchsize = 256
        self.epochs = 60
        self.weightdecay = 0.8
        self.decaypatience = 6


class TimeProConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.task_name = 'long_term_forecast'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 24
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 3
        self.d_model = 96
        self.d_ff = 192
        self.n_heads = 4
        self.e_layers = 2
        self.dropout = 0.1
        self.moving_avg = 25
        self.freq = 'h'
        self.timepro_num_prototypes = 16
        self.timepro_kernel_sizes = (3, 7, 15)
        self.target_channels = 3
        self.use_balanced_checkpoint = 1

        self.lr = 0.0005
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.8
        self.decaypatience = 5


class KARMAConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.task_name = 'long_term_forecast'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 24
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 3
        self.d_model = 96
        self.d_ff = 192
        self.e_layers = 2
        self.dropout = 0.1
        self.moving_avg = 25
        self.freq = 'h'
        self.target_channels = 3
        self.use_balanced_checkpoint = 1

        self.lr = 0.0005
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.8
        self.decaypatience = 5


def get_config(model_name: str):
    """根据模型名称获取特定的超参数配置"""
    if model_name == 'PatchTST':
        return PatchTSTConfig()
    if model_name == 'TimeMixer':
        return TimeMixerConfig()
    if model_name == 'TimesNet':
        return TimesNetConfig()
    if model_name == 'TSMixer':
        return TSMixerConfig()
    if model_name == 'FITS':
        return FITSConfig()
    if model_name == 'CFC':
        return CFCConfig()
    if model_name == 'KAN_TQNet':
        return KAN_TQNetConfig()
    if model_name == 'TimePro':
        return TimeProConfig()
    if model_name == 'KARMA':
        return KARMAConfig()



    else:
        raise ValueError(f"Unsupported model name: {model_name}")
