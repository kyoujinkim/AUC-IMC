import warnings

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas._libs.tslibs.offsets import MonthEnd
from glob import glob
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

# turn off warnings
warnings.filterwarnings('ignore')


def getConsensusDate(x):
    year = x.FY.replace('AS','')
    cdate = dt.datetime(int(year)+1, 4, 1) - relativedelta(months=x.QBtw * 3) - dt.timedelta(days=1)

    return cdate


def getMatchPrice(x, rtn, distance=0):
    distCDate = x.CDate + MonthEnd(n=distance)
    try:
        return rtn[x.Code][rtn.index<=distCDate][-1]
    except:
        return None

def calc_shock(model):
    model_result = pd.read_csv(model, index_col=0).set_index(['Code','QBtw', 'FY']).dropna()
    model_name = model.split('\\')[-1].split('.')[0]

    global total_data
    model_result['EPS_former'] = total_data['EPS_former']

    if model_name == '':
        model_name = model.split('/')[-1].split('.')[0]

    diff = model_result.Error - EW.Error
    diff = diff[diff!=0]
    diff = pd.DataFrame(diff)
    diff['Est'] = model_result.Est
    diff['Act_Prev'] = model_result.EPS_former
    diff = diff.dropna().reset_index()

    diff['CDate'] = diff.apply(getConsensusDate, axis=1)
    for idx in tqdm(range(0, 28)):
        diff[f'RtnT{idx}'] = diff.apply(getMatchPrice, rtn=rtn, distance=idx, axis=1)

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

    #c = diff.groupby(['Code', 'FY']).Error.std()
    c = diff.groupby(['Code', 'FY']).Error.max() - diff.groupby(['Code', 'FY']).Error.min().values
    #d = diff.groupby(['Code', 'FY']).Error.mean()
    d = diff.set_index(['Code', 'FY']).loc[diff.groupby(['Code', 'FY']).QBtw.min().index].Error
    #d = diff[diff.QBtw==0].set_index(['Code', 'FY']).Error
    e = pd.concat([c, d], axis=1)
    e.columns = ['std', 'mean']
    e['Shock'] = e.groupby(['FY'])['mean'].transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    e['Vol'] = e.groupby(['FY', 'Shock'])['std'].transform(
        lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    overlist = e[(e.Shock == '10') & (e.Vol == '10')].index.drop_duplicates()
    smalllist = e[(e.Shock == '80') & (e.Vol == '10')].index.drop_duplicates()

    overlist_rtn = diff.set_index(['Code', 'FY']).loc[overlist]
    overlist_rtn = overlist_rtn[overlist_rtn.QBtw==0].reset_index()
    smalllist_rtn = diff.set_index(['Code', 'FY']).loc[smalllist]
    smalllist_rtn = smalllist_rtn[smalllist_rtn.QBtw==0].reset_index()
    LS_rtn = smalllist_rtn.groupby(['FY'])['6M_Rtn'].mean() - overlist_rtn.groupby(['FY'])['6M_Rtn'].mean()
    LS_rtn

    #diff['Shock'] = pd.cut(diff.Error, bins=[-100,-0.1,-0.05,-0.03,-0.01,0.01,0.03,0.05,100], labels=['-10<','-5<','-3<','-1<','1<','3<','5<','5>'])
    #diff['Shock'] = pd.cut(diff.Error, bins=[-100, -0.03, 0, 0.03, 100], labels=['--', '-', '+', '++'])
    #diff['Shock'] = diff.groupby('QBtw').Error.transform(lambda x: pd.qcut(x, 10, labels=[str(x) for x in range(10,101, 10)]))
    #diff['Shock'] = pd.qcut(diff.Error, 20, labels=[str(x) for x in range(10, 201, 10)])

    #diff = diff_orig.copy()
    #diff = diff[(diff.Est < 1.0)]# & (diff.Est > 0)]
    #diff['Shock'] = pd.cut(diff.Error, bins=[-100, -0.03, 0, 100], labels=['--', '-', '+'])
    diff['QGEst'] = diff.groupby(['FY', 'QBtw']).GEst.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    diff['QGEWEst'] = diff.groupby(['FY', 'QBtw']).GEWEst.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    diff['Shock'] = diff.groupby(['FY', 'QBtw']).Error.transform(lambda x: pd.qcut(x, 8, labels=[str(x) for x in range(10, 81, 10)]))
    #diff['QEst'] = pd.qcut(diff.Est, 2, labels=[str(x) for x in range(10, 21, 10)])

    result = diff.groupby(['QBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].mean().dropna(how='all')
    result_ew = diff.groupby(['QBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].mean().dropna(how='all')
    result_sk = diff.groupby(['QBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].mean().dropna(how='all')
    result_std = diff.groupby(['QBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].std().dropna(how='all')
    result_ew_std = diff.groupby(['QBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].std().dropna(how='all')
    result_sk_std = diff.groupby(['QBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].std().dropna(how='all')
    result_count = diff.groupby(['FY', 'QBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].count().dropna(how='all')

    result_ts = diff.groupby(['FY', 'QBtw', 'QGEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Est']].mean().dropna(how='all')
    result_ew_ts = diff.groupby(['FY', 'QBtw', 'QGEWEst'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'EWEst']].mean().dropna(how='all')
    result_sk_ts = diff.groupby(['FY', 'QBtw', 'Shock'])[
        [str(x)+'M_Rtn' for x in range(3,28,3)] + ['Model_Error', 'BM_Error', 'Error']].mean().dropna(how='all')

    for idx in range(9, 0, -1):
        result[f'{idx*3}M_Rtn'] = result[result.columns[:idx]].prod(axis=1)
        result_ew[f'{idx * 3}M_Rtn'] = result_ew[result_ew.columns[:idx]].prod(axis=1)
        result_sk[f'{idx * 3}M_Rtn'] = result_sk[result_sk.columns[:idx]].prod(axis=1)
        result_ts[f'{idx*3}M_Rtn'] = result_ts[result_ts.columns[:idx]].prod(axis=1)
        result_ew_ts[f'{idx * 3}M_Rtn'] = result_ew_ts[result_ew_ts.columns[:idx]].prod(axis=1)
        result_sk_ts[f'{idx * 3}M_Rtn'] = result_sk_ts[result_sk_ts.columns[:idx]].prod(axis=1)

    result.to_csv(f'./result/return_analysis/rtn{model_name}_IMC.csv', encoding='utf-8-sig')
    result_ew.to_csv(f'./result/return_analysis/rtn{model_name}_EW.csv', encoding='utf-8-sig')
    result_sk.to_csv(f'./result/return_analysis/rtn{model_name}_Shock.csv', encoding='utf-8-sig')
    result_ts.to_csv(f'./result/return_analysis/rtn{model_name}_TS.csv', encoding='utf-8-sig')
    result_ew_ts.to_csv(f'./result/return_analysis/rtn{model_name}_EW_TS.csv', encoding='utf-8-sig')
    result_sk_ts.to_csv(f'./result/return_analysis/rtn{model_name}_Shock_TS.csv', encoding='utf-8-sig')
    result_std.to_csv(f'./result/return_analysis/std{model_name}_IMC.csv', encoding='utf-8-sig')
    result_ew_std.to_csv(f'./result/return_analysis/std{model_name}_EW.csv', encoding='utf-8-sig')
    result_sk_std.to_csv(f'./result/return_analysis/std{model_name}_Shock.csv', encoding='utf-8-sig')
    result_count.to_csv(f'./result/return_analysis/count{model_name}_IMC.csv', encoding='utf-8-sig')

    LS_rtn.to_csv(f'./result/return_analysis/LS_rtn{model_name}.csv', encoding='utf-8-sig')

model_list = glob('./result/IMC_adp_10y_5c_10y.csv')
model_list = [model for model in model_list if '_MSFE' not in model and 'EW.csv' not in model]

# weekly price data of stocks
price = pd.read_excel('./data/price_qw.xlsx', sheet_name='price', header=7, index_col=0, parse_dates=True).iloc[6:].astype(float)
price.index = pd.to_datetime(price.index)

kprice = pd.read_excel('./data/price_qw.xlsx', sheet_name='kprice', header=7, index_col=0, parse_dates=True).iloc[6:].astype(float)
kprice = kprice.dropna(how='all',axis=1)
kprice.index = pd.to_datetime(kprice.index)

# weekly return of stocks
rtn = price.pct_change().iloc[1:]
krtn = kprice.pct_change().iloc[1:]
rtn_abs = rtn - krtn.values
rtn_abs = rtn_abs.where(rtn.notnull(), np.nan)

# total data
total_data = pd.read_csv('./data/total.csv')
total_data['EPS_former'] = total_data['A_EPS_1(지배)'] / total_data['BPS'].values
total_data['Year'] = total_data.FY.str.extract(r'(\d{4})')
total_data['EDate'] = pd.to_datetime((total_data.Year.astype(int) + 1).astype(str) + '-03-31')
total_data['YearDiff'] = total_data.EDate.dt.year - pd.to_datetime(total_data.Date).dt.year
total_data['MonthDiff'] = total_data.EDate.dt.month - pd.to_datetime(total_data.Date).dt.month
total_data['totalDiff'] = total_data['YearDiff'] * 12 + total_data['MonthDiff']
total_data['QBtw'] = (total_data['totalDiff'] / 3).astype(int)
total_data = total_data.drop_duplicates(subset=['Code', 'QBtw', 'FY'])
total_data = total_data.set_index(['Code', 'QBtw', 'FY'])

EW = pd.read_csv('./result/EW.csv', index_col=0).set_index(['Code','QBtw', 'FY'])

if __name__ == '__main__':
    # multiprocessing
    for model in model_list:
        calc_shock(model)
    #process_map(calc_shock, model_list, max_workers=17)
