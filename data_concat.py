#concat consenlist csv
import numpy as np
import pandas as pd

df1 = pd.read_csv('./data/consenlist.csv')
df2 = pd.read_csv('./data/consenlist2.csv')
vol = pd.read_excel('./data/Kvol.xlsx', sheet_name='Sheet1', header=13, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
gdp = pd.read_excel('./data/QGDP.xlsx', sheet_name='Sheet1', header=13, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
gdp.index = gdp.index + pd.DateOffset(months=2)
gdp_roll = gdp.rolling(4).mean().dropna()
#gdp_roll.index to daily
gdp_roll = gdp_roll.resample('D').ffill()

df = pd.concat([df1, df2], ignore_index=True, axis=0).dropna(subset=['BPS', 'E_EPS(지배)', 'A_EPS(지배)'])
df['Year'] = df.FY.str.extract(r'(\d{4})')
df['CutDate'] = (df.Year.astype(int) - 1).astype(str) + '-03-31'
df = df[df.Date > df.CutDate]
# apply vol by date to df, date can be not match
vol = vol * np.sqrt(252)
df['Vol'] = df.Date.map(vol.iloc[:,0])
#vol_slope = vol.diff(60).dropna()
#df['VolSlope'] = df.Date.map(vol_slope.iloc[:,0])
df['Gdp'] = df.Date.map(gdp_roll.iloc[:,0])

df.to_csv('./data/total.csv', index=False, encoding='utf-8-sig')
