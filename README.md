KAN-TQNet: KAN 增强时频双流多能负荷预测网络
Project Introduction
This repository provides the complete open-source implementation of KAN-TQNet, the model proposed in the paper KAN-Enhanced Time-Frequency Dual-Stream Network for Multi-Energy Load Forecasting, which has been submitted to the journal Sustainable Energy Technologies and Assessments.
KAN-TQNet designs a time-frequency dual-stream architecture for joint electric, cooling and heating load forecasting in integrated energy systems (IES):
Time stream: Temporal-query-guided KAN encoder to mine complex nonlinear coupling and periodic patterns of multi-energy loads;
Frequency stream: Wavelet-decomposition-dominated & FFT-assisted spectral modeling module to extract local non-stationary spectral features and global periodic information;
Multi-scale refinement + Hierarchical Dynamic Aggregation (HDA): Adaptively allocate dominant temporal features for 24/48/72/96 multi-step forecasting tasks.
Comprehensive experiments on the real ASU IES dataset prove that our model outperforms mainstream state-of-the-art time series models (DLinear, iTransformer, Mamba, TimesNet, etc.) in prediction accuracy, missing-data robustness and model interpretability.
