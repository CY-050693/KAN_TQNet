import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# 设置SCI风格
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.unicode_minus': False,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 数据输入
data = {
    'Model': ['DLinear', 'NLinear', 'TimeMixer', 'TimeXer', 'TSMixer', 'TimesNet',
              'FITS', 'FiLM', 'iTransformer', 'Crossformer', 'Amplifier', 'SegRNN',
              'Mamba', 'PatchTST', 'xPatch', 'LSTM-Transformer', 'KAN_TQNet(Ours)'],
    'Horizons': [24, 48, 72, 96],
    'Elec_MAE': [[857.04, 1049.02, 1266.63, 1285.19],
                 [853.64, 1048.70, 1128.00, 1283.76],
                 [838.57, 1067.27, 1176.56, 1246.54],
                 [978.19, 1099.70, 1170.41, 1188.23],
                 [1097.30, 1388.23, 1295.20, 1020.39],
                 [1170.53, 1212.77, 1217.08, 1449.15],
                 [448.60, 1034.16, 1284.45, 1184.14],
                 [941.00, 1026.02, 1232.36, 1262.36],
                 [825.19, 1067.68, 1197.59, 1261.68],
                 [867.64, 1107.44, 1186.94, 1312.74],
                 [823.30, 984.28, 1170.51, 1179.12],
                 [891.37, 1101.04, 1335.59, 1373.79],
                 [869.11, 1120.56, 1285.78, 818.75],
                 [846.75, 1047.33, 1168.40, 1284.44],
                 [787.45, 986.75, 1170.35, 1312.31],
                 [842.64, 1109.45, 1369.87, 1439.07],
                 [782.22, 999.94, 1133.81, 1210.48]],
    'Elec_RMSE': [[1251.96, 1550.51, 1814.01, 1826.66],
                  [1261.01, 1542.58, 1717.18, 1827.92],
                  [1254.92, 1570.27, 1715.21, 1792.54],
                  [1204.70, 1647.79, 1747.17, 1347.54],
                  [1564.24, 1988.04, 1820.96, 1439.22],
                  [1645.12, 1747.38, 1735.17, 1926.91],
                  [669.19, 1554.44, 1786.13, 1852.51],
                  [1379.59, 1508.45, 1757.08, 1803.59],
                  [1231.95, 1579.47, 1746.58, 1810.24],
                  [1261.45, 1571.25, 1680.31, 1889.65],
                  [1225.28, 1433.30, 1612.99, 1686.63],
                  [1309.03, 1607.89, 1889.09, 1897.89],
                  [1261.38, 1581.82, 1767.71, 1087.90],
                  [1240.96, 1544.45, 1740.83, 1846.91],
                  [1204.65, 1450.04, 1787.11, 1878.46],
                  [1251.65, 1606.31, 1900.47, 1847.46],
                  [1203.79, 1506.02, 1679.85, 1778.53]],
    'Cool_MAE': [[1066.52, 1468.36, 1704.49, 2046.82],
                 [1079.42, 1408.49, 1793.60, 2093.98],
                 [1043.30, 1429.26, 1758.73, 1940.26],
                 [1198.84, 1619.10, 2116.44, 2166.08],
                 [1167.08, 1638.53, 2113.54, 1181.33],
                 [1186.74, 1787.78, 2234.78, 1007.02],
                 [706.42, 1462.09, 1709.17, 2017.05],
                 [1093.02, 1474.37, 2029.74, 1874.29],
                 [1155.42, 1600.38, 1937.43, 2117.99],
                 [1135.99, 1635.99, 1764.32, 2257.17],
                 [1025.07, 1379.90, 1826.46, 1862.26],
                 [1047.06, 1403.60, 2037.39, 2037.39],
                 [1214.87, 1583.50, 1983.60, 2226.28],
                 [1136.25, 1413.06, 1819.85, 2062.07],
                 [1319.07, 1419.95, 1780.36, 1970.83],
                 [1247.21, 1812.84, 2182.12, 1892.77],
                 [1007.04, 1357.69, 1591.39, 1849.14]],
    'Cool_RMSE': [[1480.41, 2032.83, 2470.18, 2702.01],
                  [1473.38, 2033.94, 2430.46, 2707.39],
                  [1471.12, 2082.94, 2390.82, 2580.94],
                  [1667.18, 2323.30, 2690.18, 2900.18],
                  [1542.80, 2364.83, 2744.07, 1705.67],
                  [2194.27, 2471.57, 2162.35, 1494.91],
                  [1041.96, 2037.57, 2411.67, 2681.68],
                  [1527.44, 2020.81, 2684.98, 2628.08],
                  [1562.49, 2236.06, 2576.40, 2862.13],
                  [1510.31, 2178.73, 2381.72, 3072.06],
                  [1446.08, 1859.96, 2456.49, 2461.26],
                  [1480.50, 2031.82, 2716.26, 2716.26],
                  [1608.70, 2159.96, 2602.24, 2893.94],
                  [1553.76, 2105.04, 2433.53, 2907.30],
                  [1447.53, 1951.35, 2608.06, 2680.96],
                  [1677.37, 2713.55, 3088.78, 2688.58],
                  [1365.43, 1858.22, 2144.61, 2485.88]],
    'Heat_MAE': [[85.96, 104.18, 134.32, 135.59],
                 [86.34, 108.89, 115.37, 126.26],
                 [92.16, 106.03, 121.21, 121.65],
                 [86.43, 109.78, 104.48, 134.48],
                 [108.49, 139.05, 123.80, 123.99],
                 [110.81, 118.05, 139.03, 85.04],
                 [185.30, 107.71, 114.57, 125.48],
                 [87.98, 103.16, 135.32, 135.92],
                 [94.37, 107.68, 129.57, 129.20],
                 [89.58, 111.42, 109.18, 147.65],
                 [92.49, 107.13, 117.75, 137.99],
                 [87.67, 110.49, 131.95, 131.95],
                 [91.64, 111.77, 113.37, 137.46],
                 [85.59, 121.70, 133.91, 143.37],
                 [43.37, 104.94, 122.65, 129.79],
                 [94.06, 114.03, 135.95, 154.31],
                 [83.69, 101.41, 111.49, 122.59]],
    'Heat_RMSE': [[120.17, 149.36, 170.42, 135.14],
                  [114.42, 157.08, 179.85, 125.53],
                  [129.87, 131.92, 160.81, 164.85],
                  [110.91, 135.81, 137.70, 117.25],
                  [146.00, 183.08, 162.06, 123.95],
                  [154.43, 154.34, 182.70, 121.31],
                  [267.21, 143.38, 159.38, 161.05],
                  [122.59, 135.10, 183.03, 183.03],
                  [132.92, 173.86, 175.15, 167.70],
                  [104.30, 139.95, 139.54, 189.87],
                  [128.94, 136.52, 162.10, 179.18],
                  [110.15, 135.47, 166.37, 166.37],
                  [116.92, 158.37, 168.09, 184.35],
                  [118.58, 153.83, 166.36, 117.46],
                  [187.18, 139.39, 164.06, 171.49],
                  [119.06, 165.36, 178.03, 178.03],
                  [112.61, 136.31, 147.59, 159.38]]
}

models = data['Model']
horizons = data['Horizons']

# SCI常用配色方案 (色盲友好)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
          '#c49c94', '#dbdb8d']

# 创建图形 - 6列2行布局，6*9英寸
fig = plt.figure(figsize=(9, 6))

# 为每个horizon创建子图
horizon_labels = ['Horizon 24', 'Horizon 48', 'Horizon 72', 'Horizon 96']

for idx, h in enumerate(horizons):
    ax1 = plt.subplot(2, 4, idx + 1)
    ax2 = plt.subplot(2, 4, idx + 5)

    # 提取当前horizon的数据
    h_idx = idx

    # MAE数据
    elec_mae = [data['Elec_MAE'][i][h_idx] for i in range(len(models))]
    cool_mae = [data['Cool_MAE'][i][h_idx] for i in range(len(models))]
    heat_mae = [data['Heat_MAE'][i][h_idx] for i in range(len(models))]

    # RMSE数据
    elec_rmse = [data['Elec_RMSE'][i][h_idx] for i in range(len(models))]
    cool_rmse = [data['Cool_RMSE'][i][h_idx] for i in range(len(models))]
    heat_rmse = [data['Heat_RMSE'][i][h_idx] for i in range(len(models))]

    x = np.arange(len(models))
    width = 0.25

    # 绘制MAE (上排)
    bars1 = ax1.bar(x - width, elec_mae, width, label='Electrical', color='#E64B35', alpha=0.85, edgecolor='black',
                    linewidth=0.5)
    bars2 = ax1.bar(x, cool_mae, width, label='Cooling', color='#4DBBD5', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x + width, heat_mae, width, label='Heating', color='#00A087', alpha=0.85, edgecolor='black',
                    linewidth=0.5)

    ax1.set_ylabel('MAE', fontsize=10, fontweight='bold')
    ax1.set_title(f'{horizon_labels[idx]}', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # 绘制RMSE (下排)
    bars4 = ax2.bar(x - width, elec_rmse, width, label='Electrical', color='#E64B35', alpha=0.85, edgecolor='black',
                    linewidth=0.5)
    bars5 = ax2.bar(x, cool_rmse, width, label='Cooling', color='#4DBBD5', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars6 = ax2.bar(x + width, heat_rmse, width, label='Heating', color='#00A087', alpha=0.85, edgecolor='black',
                    linewidth=0.5)

    ax2.set_ylabel('RMSE', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)

# 添加统一的图例
handles = [mpatches.Patch(color='#E64B35', alpha=0.85, label='Electrical Load'),
           mpatches.Patch(color='#4DBBD5', alpha=0.85, label='Cooling Load'),
           mpatches.Patch(color='#00A087', alpha=0.85, label='Heating Load')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)

# 保存图像
plt.savefig('model_comparison_SCI.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('model_comparison_SCI.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("图像已保存为 'model_comparison_SCI.png' 和 'model_comparison_SCI.pdf'")