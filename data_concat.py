#concat consenlist csv
from glob import glob

import numpy as np
import pandas as pd


consenlist = glob('./data/consenlist*.csv')
for idx, file in enumerate(consenlist):
    if idx == 0:
        df = pd.read_csv(file)
    else:
        df = pd.concat([df, pd.read_csv(file)], ignore_index=True, axis=0)
df = df.dropna(subset=['BPS', 'E_EPS(지배)', 'A_EPS(지배)'])

gdp = pd.read_excel('./data/QGDP.xlsx', sheet_name='Sheet1', header=13, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
gdp.index = gdp.index + pd.DateOffset(months=2)
gdp_roll = gdp.rolling(4).mean().dropna()
#gdp_roll.index to daily
gdp_roll = gdp_roll.resample('D').ffill()

#eps_data = df.drop_duplicates(subset=['Code','FY','A_EPS(지배)'])
#df['A_EPS_1Y'] = df.apply(lambda x: find_eps(x, eps_data), axis=1)
df['Year'] = df.FY.str.extract(r'(\d{4})')
df['CutDate'] = (df.Year.astype(int) - 1).astype(str) + '-03-31'
df = df[df.Date > df.CutDate]
#df['VolSlope'] = df.Date.map(vol_slope.iloc[:,0])
df['Gdp'] = df.Date.map(gdp_roll.iloc[:,0])

df.to_csv('./data/total.csv', index=False, encoding='utf-8-sig')
