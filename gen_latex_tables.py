import csv

rows = []
with open('./output/noise_experiment_all.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows.append(r)

noise_label = {'gaussian': 'Gaussian', 'impulse': 'Impulse', 'missing': 'Missing'}
levels      = ['0.0', '0.01', '0.02', '0.05', '0.1', '0.2']
pred_lens   = [24, 48, 72, 96]
col_levels  = ['0.01', '0.02', '0.05', '0.1', '0.2']

def get(ntype, lv, pl, col):
    r = next(x for x in rows if x['noise_type']==ntype and x['noise_level']==lv and int(x['pred_len'])==pl)
    return float(r[col])

L = []

# ── Table 1: Overall MAPE ─────────────────────────────────────────────────────
L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{Overall MAPE (\%) of KAN-TQNet under Different Noise Types and Levels}')
L.append(r'\label{tab:noise_overall_mape}')
L.append(r'\begin{tabular}{llrrrr}')
L.append(r'\toprule')
L.append(r'Noise Type & Level & $T{=}24$ & $T{=}48$ & $T{=}72$ & $T{=}96$ \\')
L.append(r'\midrule')
for ntype, nlabel in noise_label.items():
    for i, lv in enumerate(levels):
        label = '\\multirow{6}{*}{' + nlabel + '}' if i == 0 else ''
        v = [get(ntype, lv, pl, 'overall_mape') for pl in pred_lens]
        L.append('{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(label, lv, *v))
    L.append(r'\midrule')
L[-1] = r'\bottomrule'
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')

# ── Table 2: Relative MAPE Degradation ───────────────────────────────────────
L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{Relative MAPE Degradation (\%) of KAN-TQNet under Noise}')
L.append(r'\label{tab:noise_degradation}')
L.append(r'\begin{tabular}{llrrrrr}')
L.append(r'\toprule')
L.append(r'Noise Type & $T$ & $\sigma{=}0.01$ & $\sigma{=}0.02$ & $\sigma{=}0.05$ & $\sigma{=}0.10$ & $\sigma{=}0.20$ \\')
L.append(r'\midrule')
for ntype, nlabel in noise_label.items():
    for j, pl in enumerate(pred_lens):
        label = '\\multirow{4}{*}{' + nlabel + '}' if j == 0 else ''
        base  = get(ntype, '0.0', pl, 'overall_mape')
        degs  = [(get(ntype, lv, pl, 'overall_mape') - base) / base * 100 for lv in col_levels]
        L.append('{} & {} & {:+.2f} & {:+.2f} & {:+.2f} & {:+.2f} & {:+.2f} \\\\'.format(label, pl, *degs))
    L.append(r'\midrule')
L[-1] = r'\bottomrule'
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')

# ── Table 3: Per-channel MAPE at level 0.2 ───────────────────────────────────
L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{Per-channel MAPE (\%) at $\sigma{=}0.2$ vs.\ Clean Baseline}')
L.append(r'\label{tab:noise_channel}')
L.append(r'\begin{tabular}{llrrrrrrrr}')
L.append(r'\toprule')
L.append(r'Noise & $T$ & Elec$_0$ & Elec$_{.2}$ & $\Delta$Elec & Cool$_0$ & Cool$_{.2}$ & $\Delta$Cool & Heat$_0$ & Heat$_{.2}$ \\')
L.append(r'\midrule')
for ntype, nlabel in noise_label.items():
    for j, pl in enumerate(pred_lens):
        label = '\\multirow{4}{*}{' + nlabel + '}' if j == 0 else ''
        e0 = get(ntype,'0.0',pl,'electricity_mape'); e2 = get(ntype,'0.2',pl,'electricity_mape')
        c0 = get(ntype,'0.0',pl,'cooling_mape');     c2 = get(ntype,'0.2',pl,'cooling_mape')
        h0 = get(ntype,'0.0',pl,'heating_mape');     h2 = get(ntype,'0.2',pl,'heating_mape')
        L.append('{} & {} & {:.4f} & {:.4f} & {:+.4f} & {:.4f} & {:.4f} & {:+.4f} & {:.4f} & {:.4f} \\\\'.format(
            label, pl, e0, e2, e2-e0, c0, c2, c2-c0, h0, h2))
    L.append(r'\midrule')
L[-1] = r'\bottomrule'
L.append(r'\end{tabular}')
L.append(r'\end{table}')

text = '\n'.join(L)
out  = './output/noise_experiment_latex.tex'
with open(out, 'w', encoding='utf-8') as f:
    f.write(text)
print('saved', out)
print()
print(text)
