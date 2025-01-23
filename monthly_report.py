from configparser import ConfigParser
from glob import glob
import refinitiv.data as rd
from calc_Enhanced_EPS import get_sector_growth_rate, groupby_sector
import datetime as dt

import pandas as pd

year = 2025
country = 'kr'
postdate = '2025-03-01'
resultpath_list = glob(f'result/{country}/IMC_adp_{year}_*.csv')
resultpath_list.sort()
resultpath = resultpath_list[-1]
result = pd.read_csv(resultpath)

codes = result['Code'].unique().tolist()

'''config = ConfigParser()
config.read('./src/config.ini')
app_key = config['main']['api_key']

session = rd.session.desktop.Definition(app_key=app_key).get_session()
rd.session.set_default(session)
session.open()

df = rd.get_data(
    universe=codes,
    fields=['TR.ExpectedReportDate(Period=FQ1)/*Next EPS Report Date*/']
)
df = df.set_index('Instrument')
result = result.set_index('Code')
result['annDate'] = df['Expected Report Date']'''
#result = result[(result.annDate < postdate) & (result.annDate >= dt.datetime.today().strftime('%Y-%m-%d'))]
result = result.set_index('Code')
sector_name = pd.read_csv(f'data/{country}/industry_map.csv', encoding='utf-8-sig', dtype=str).set_index('Code')
sector_name.index = sector_name.index.astype(str)
shares = pd.read_excel(f'data/{country}/shares.xlsx', sheet_name='share', index_col=0)
shares = shares.loc[pd.to_numeric(shares.iloc[:, 0], errors='coerce').dropna().index]

result['shares'] = shares

result_sector = groupby_sector(result, sector_name, 'EQBtw', True,'C:/Users/NHWM/PycharmProjects/SmartConsensus/result/kr/IMC_adp_2024_2025-01-15.csv')

result.to_csv(f'result/{country}/IMSE_adp_annDate_{year}_{dt.datetime.today().strftime("%Y%m%d")}.csv', encoding='utf-8-sig')
result_sector.to_csv(f'result/{country}/IMSE_adp_Sector_annDate_{year}_{dt.datetime.today().strftime("%Y%m%d")}.csv', encoding='utf-8-sig')
c = 1