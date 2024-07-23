import warnings
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
    if model_name == '':
        model_name = model.split('/')[-1].split('.')[0]

    diff = model_result.Error - EW.Error
    diff = diff[diff!=0]
    diff = diff.reset_index()

    diff['CDate'] = diff.apply(getConsensusDate, axis=1)
    for idx in range(0, 25):
        diff[f'RtnT{idx}'] = diff.apply(getMatchPrice, rtn=rtn, distance=idx, axis=1)

    diff['Shock'] = pd.cut(diff.Error, bins=[-100,-0.1,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.1,100], labels=['-10<','-5<','-3<','-1<','1<','3<','5<','10<','10>'])

    diff = diff.set_index(['Code', 'QBtw', 'FY'])
    diff['Model_Error'] = model_result.Error
    diff['BM_Error'] = diff.Model_Error - diff.Error
    diff = diff.reset_index()

    result = diff.groupby(['QBtw', 'Shock'])[['RtnT' + str(x) for x in range(25)] + ['Model_Error', 'BM_Error']].mean().dropna(how='all')
    result_ts = diff.groupby(['QBtw', 'Shock', 'FY'])[['RtnT' + str(x) for x in range(25)] + ['Model_Error', 'BM_Error']].mean().dropna(how='all')

    result['3M_Rtn'] = (result[result.columns[1:4]] + 1).prod(axis=1)
    result['6M_Rtn'] = (result[result.columns[1:7]] + 1).prod(axis=1)
    result['9M_Rtn'] = (result[result.columns[1:10]] + 1).prod(axis=1)
    result['12M_Rtn'] = (result[result.columns[1:13]] + 1).prod(axis=1)
    result['15M_Rtn'] = (result[result.columns[1:16]] + 1).prod(axis=1)
    result['18M_Rtn'] = (result[result.columns[1:19]] + 1).prod(axis=1)
    result['21M_Rtn'] = (result[result.columns[1:22]] + 1).prod(axis=1)
    result['24M_Rtn'] = (result[result.columns[1:25]] + 1).prod(axis=1)

    result_ts['3M_Rtn'] = (result_ts[result_ts.columns[1:4]] + 1).prod(axis=1)
    result_ts['6M_Rtn'] = (result_ts[result_ts.columns[1:7]] + 1).prod(axis=1)
    result_ts['9M_Rtn'] = (result_ts[result_ts.columns[1:10]] + 1).prod(axis=1)
    result_ts['12M_Rtn'] = (result_ts[result_ts.columns[1:13]] + 1).prod(axis=1)

    result.to_csv(f'./result/return_analysis/rtn{model_name}.csv', encoding='utf-8-sig')
    result_ts.to_csv(f'./result/return_analysis/rtn{model_name}_ts.csv', encoding='utf-8-sig')

model_list = glob('./result/EW_adp_10y_all.csv')
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
rtn = rtn - krtn.values

EW = pd.read_csv('./result/EW.csv', index_col=0).set_index(['Code','QBtw', 'FY'])

if __name__ == '__main__':
    # multiprocessing
    #for model in model_list:
    #    calc_shock(model)
    process_map(calc_shock, model_list, max_workers=17)
