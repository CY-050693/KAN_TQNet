# Feature Evidence Summary: KW

- Data file: `data\dataset_input_jiuzheng.csv`
- Target column: `KW`
- Time index source: `synthetic hourly index from row order`
- Sampling interval (hours): `1.0`

## Temporal Evidence
- Rolling window used: `168`
- STL seasonal period: `24`
- Peak hour by group mean: `14`
- Peak weekday by group mean: `4`
- Peak month by group mean: `6`
- Ljung-Box p-value @ lag 24: `0`
- ADF p-value: `0.000119769`
- KPSS p-value: `0.01`

## Frequency Evidence
- Top period 1: `7935.67` hours (`330.65` days), power `1.42376e+07`
- Top period 2: `3967.83` hours (`165.33` days), power `1.42607e+06`
- Top period 3: `23807.00` hours (`991.96` days), power `987093`
- Top period 4: `11903.50` hours (`495.98` days), power `541784`
- Top period 5: `24.00` hours (`1.00` days), power `362711`
- Top period 6: `881.74` hours (`36.74` days), power `220277`
- Top period 7: `2645.22` hours (`110.22` days), power `196142`
- Top period 8: `167.65` hours (`6.99` days), power `183311`

## Nonlinear Evidence
- `GHG`: mutual_info=`2.24487`, pearson=`0.854353`
- `CHWTON`: mutual_info=`0.777589`, pearson=`0.849616`
- `DayOfYear_cos`: mutual_info=`0.687272`, pearson=`-0.703583`
- `wet_bulb_temperature`: mutual_info=`0.616037`, pearson=`0.813752`
- `HTmmBTU`: mutual_info=`0.468639`, pearson=`-0.65383`
- `dew_point_temperature`: mutual_info=`0.41096`, pearson=`0.682072`
- `temperature`: mutual_info=`0.405054`, pearson=`0.690347`
- `Combined mmBTU`: mutual_info=`0.32318`, pearson=`-0.510303`
- `sea_level_pressure`: mutual_info=`0.221437`, pearson=`-0.452858`
- `altimeter`: mutual_info=`0.1829`, pearson=`-0.388826`

- `temperature` quadratic improvement ratio over linear fit: `0.4102%`
- `wet_bulb_temperature` quadratic improvement ratio over linear fit: `15.0820%`
- `dew_point_temperature` quadratic improvement ratio over linear fit: `11.1539%`
