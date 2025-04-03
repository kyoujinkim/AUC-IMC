import warnings

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas._libs.tslibs.offsets import MonthEnd
from glob import glob
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from src.funs import build_data, filter_guided_stock

# turn off warnings
warnings.filterwarnings('ignore')


def getConsensusDate(x):
    cdate = dt.datetime.strptime(x.PeriodEndDate, '%Y-%m-%d') + relativedelta(months=3) - relativedelta(months=x.QBtw * 3) - dt.timedelta(days=1)

    return cdate


def getMatchPrice(x, rtn, distance=0):
    distCDate = x.CDate + MonthEnd(n=distance)
    try:
        return rtn[x.Code][rtn.index<=distCDate][-1]
    except:
        return None

def calc_shock(model):
    model_result = pd.read_csv(model, index_col=0).set_index(['Code','QBtw', 'FY']).dropna()
    model_name = model.split('\\')[-1].split('.')[0].split('/')[-1].split('.')[0]

    global total_data
    model_result['EPS_former'] = total_data['EPS_former']
    model_result['PeriodEndDate'] = total_data.PeriodEndDate

    if model_name == '':
        model_name = model.split('/')[-1].split('.')[0]

    diff = model_result.Error - EW.Error
    diff = diff[diff!=0]
    diff = pd.DataFrame(diff)
    diff['Est'] = model_result.Est
    diff['Act_Prev'] = model_result.EPS_former
    diff['PeriodEndDate'] = model_result.PeriodEndDate
    diff = diff.dropna().reset_index()

    diff['CDate'] = diff.apply(getConsensusDate, axis=1)
    for idx in tqdm(range(0, 28)):
        diff[f'RtnT{idx}'] = diff.apply(getMatchPrice, rtn=rtn_abs, distance=idx, axis=1)

    for idx in range(1, 10):
        diff[f'{idx*3}M_Rtn'] = (diff[diff.columns[8+(idx-1)*3:8+(idx)*3]] + 1).prod(axis=1)
        #diff[f'{idx * 3}M_Rtn'] = (diff[diff.columns[8:8 + (idx) * 3]] + 1).prod(axis=1)

    diff = diff.set_index(['Code', 'QBtw', 'FY'])
    diff['Model_Error'] = model_result.Error
    diff['BM_Error'] = diff.Model_Error - diff.Error
    diff['GEst'] = diff.Est - diff.Act_Prev
    diff['EWEst'] = EW.Est
    diff['GEWEst'] = diff.EWEst - diff.Act_Prev
    diff = diff.reset_index()

    diff['QGEst'] = diff.groupby(['FY', 'EQBtw']).GEst.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    diff['QGEWEst'] = diff.groupby(['FY', 'EQBtw']).GEWEst.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    diff['Shock'] = diff.groupby(['FY', 'EQBtw']).Error.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))

    result = diff.groupby(['EQBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].mean().dropna(how='all')
    result_ew = diff.groupby(['EQBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].mean().dropna(how='all')
    result_sk = diff.groupby(['EQBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].mean().dropna(how='all')
    result_std = diff.groupby(['EQBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].std().dropna(how='all')
    result_ew_std = diff.groupby(['EQBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].std().dropna(how='all')
    result_sk_std = diff.groupby(['EQBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].std().dropna(how='all')
    result_count = diff.groupby(['FY', 'EQBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].count().dropna(how='all')

    result_ts = diff.groupby(['FY', 'EQBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].mean().dropna(how='all')
    result_ew_ts = diff.groupby(['FY', 'EQBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].mean().dropna(how='all')
    result_sk_ts = diff.groupby(['FY', 'EQBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].mean().dropna(how='all')

    for idx in range(9, 0, -1):
        result[f'{idx*3}M_Rtn'] = result[result.columns[:idx]].prod(axis=1)
        result_ew[f'{idx * 3}M_Rtn'] = result_ew[result_ew.columns[:idx]].prod(axis=1)
        result_sk[f'{idx * 3}M_Rtn'] = result_sk[result_sk.columns[:idx]].prod(axis=1)
        result_ts[f'{idx*3}M_Rtn'] = result_ts[result_ts.columns[:idx]].prod(axis=1)
        result_ew_ts[f'{idx * 3}M_Rtn'] = result_ew_ts[result_ew_ts.columns[:idx]].prod(axis=1)
        result_sk_ts[f'{idx * 3}M_Rtn'] = result_sk_ts[result_sk_ts.columns[:idx]].prod(axis=1)

    result.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_Model.csv', encoding='utf-8-sig')
    result_ew.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_EW.csv', encoding='utf-8-sig')
    result_sk.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_Shock.csv', encoding='utf-8-sig')
    result_ts.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_TS.csv', encoding='utf-8-sig')
    result_ew_ts.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_EW_TS.csv', encoding='utf-8-sig')
    result_sk_ts.to_csv(f'./result/{country}/return_analysis/rtn{model_name}_Shock_TS.csv', encoding='utf-8-sig')
    result_std.to_csv(f'./result/{country}/return_analysis/std{model_name}_Model.csv', encoding='utf-8-sig')
    result_ew_std.to_csv(f'./result/{country}/return_analysis/std{model_name}_EW.csv', encoding='utf-8-sig')
    result_sk_std.to_csv(f'./result/{country}/return_analysis/std{model_name}_Shock.csv', encoding='utf-8-sig')
    result_count.to_csv(f'./result/{country}/return_analysis/count{model_name}_Model.csv', encoding='utf-8-sig')

country = 'us' # if case is us, model is 'IMSE', else if korea model is 'IMC'

model_list = glob(f'result/{country}/IMSE_adp_g.csv')
model_list = [model for model in model_list if '_MSFE' not in model and 'EW.csv' not in model]

# weekly price data of stocks
if country == 'us':
    price = pd.read_excel(f'./data/{country}/price.xlsx', sheet_name='price', header=1, index_col=0, parse_dates=True).astype(float, errors='ignore')
    price.index = pd.to_datetime(price.index)
    # which not contain 'Unnamed'
    price = price[price.columns[~price.columns.str.contains('Unnamed')]]

    iprice = pd.read_excel(f'./data/{country}/price.xlsx', sheet_name='iprice', header=1, index_col=0, parse_dates=True).astype(float, errors='ignore')
    iprice = iprice.dropna(how='all',axis=1)
    iprice.index = pd.to_datetime(iprice.index)
else:
    price = pd.read_excel(f'./data/{country}/price.xlsx', sheet_name='price', header=7, index_col=0, parse_dates=True).iloc[6:].astype(float)
    price.index = pd.to_datetime(price.index)

    iprice = pd.read_excel(f'./data/{country}/price.xlsx', sheet_name='iprice', header=7, index_col=0, parse_dates=True).iloc[6:].astype(float)
    iprice = iprice.dropna(how='all',axis=1)
    iprice.index = pd.to_datetime(iprice.index)

# weekly return of stocks
rtn = price.pct_change().iloc[1:]
irtn = iprice.pct_change().iloc[1:]
rtn_abs = rtn - irtn.values
rtn_abs = rtn_abs.where(rtn.notnull(), np.nan)

country = 'us'
if country == 'us':
    use_gdp = False
    gdp_path = f'data/{country}/PR.xlsx'
    gdp_header = 8
    gdp_lag = 0
    rolling = 1
    ts_length = -1
    sector_len = 2
elif country == 'kr':
    use_gdp = True
    gdp_path = f'data/{country}/QGDP.xlsx'
    gdp_header = 13
    gdp_lag = 2
    rolling = 4
    ts_length = -1
    sector_len = 3

print('build train data')
train = build_data(f'data/{country}/consenlist/*.csv'
                   , use_gdp=use_gdp
                   , gdp_path=gdp_path
                   , gdp_header=gdp_header
                   , gdp_lag=gdp_lag
                   , rolling=rolling
                   , ts_length=ts_length
                   , sector_len=sector_len
                   , country=country
                   , use_cache=True)

total_data = train.dropna(subset=['E_ROE', 'A_ROE'])

if country == 'us':
    total_data = total_data[train.A_EPS_1.abs() / train.BPS < 1]
    # if previous year's error is less than 0.5%, remove the stock from the list
    total_data = filter_guided_stock(total_data, 'Code', 'Error', 0.001)
    # retain only guidance given stock
    #new_train = train[train.Guidance == 1]
    #UniqueSymbol = new_train.UniqueSymbol.unique()
    UniqueSymbol = total_data.UniqueSymbol.unique()
    #UniqueSymbol = train[~train.UniqueSymbol.isin(UniqueSymbol_total)].UniqueSymbol.unique()
else:
    UniqueSymbol = total_data.UniqueSymbol.unique()

# total data
total_data['EPS_former'] = total_data['A_EPS_1'] / total_data['BPS'].values
total_data = total_data.drop_duplicates(subset=['Code', 'QBtw', 'FY'])
total_data = total_data[total_data.FY>='2005AS']
total_data = total_data.set_index(['Code', 'QBtw', 'FY'])

EW = pd.read_csv(f'result/{country}/EW.csv', index_col=0).set_index(['Code', 'QBtw', 'FY'])

if __name__ == '__main__':
    # multiprocessing
    for model in model_list:
        calc_shock(model)
    #process_map(calc_shock, model_list, max_workers=17)
