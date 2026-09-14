import configparser
import json
import os
from glob import glob
from itertools import product

import datetime as dt

import duckdb
import numpy as np
from numpy import nan
import pandas as pd
from pandas._libs.tslibs.offsets import MonthEnd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#disable warnings
import warnings
warnings.filterwarnings('ignore')


def save_db(db):
    config = configparser.ConfigParser()
    config.read('./src/config.ini')

    MYSQL_HOSTNAME = config['DB']['MYSQL_HOSTNAME']  # you probably don't need to change this
    MYSQL_USER = config['DB']['MYSQL_USER']
    MYSQL_PASSWORD = config['DB']['MYSQL_PASSWORD']
    MYSQL_DATABASE = config['DB']['MYSQL_DATABASE']

    connection_string = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOSTNAME}/{MYSQL_DATABASE}'
    engine = create_engine(connection_string)
    db.to_sql(con=engine, name='cache_enh_eps', if_exists='replace', index=False)


def load_duckdb(code):
    query = f"""SELECT * 
        FROM read_parquet('./cache/cache.parquet')
        WHERE Code='{code}'"""

    data = duckdb.sql(query).to_df()

    return data


def load_db(code):
    config = configparser.ConfigParser()
    config.read('./src/config.ini')

    MYSQL_HOSTNAME = config['DB']['MYSQL_HOSTNAME']  # you probably don't need to change this
    MYSQL_USER = config['DB']['MYSQL_USER']
    MYSQL_PASSWORD = config['DB']['MYSQL_PASSWORD']
    MYSQL_DATABASE = config['DB']['MYSQL_DATABASE']

    connection_string = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOSTNAME}/{MYSQL_DATABASE}'
    engine = create_engine(connection_string)
    # set data type when read from db
    db = pd.read_sql(f"SELECT * FROM cache_enh_eps WHERE Code = '{code}'", con=engine)
    db.Year = db.Year.astype(str)

    return db


def term_spread(x, b0, c, b1, b2, lam):
    theta = x / 365 / lam
    return b0 + b1 * np.exp(-theta) + b2 * theta * np.exp(-theta)


def avg_by_unique_sec(df, column='E_ROE', weight_column=None, method='simple'):
    result = []
    prev = None
    for Q in df.QBtw.unique():
        temp = df[df.QBtw >= Q].groupby('Security')
        if len(temp) > 0:
            if method == 'simple':
                prev = temp[column].mean()
            elif method == 'weighted':
                prev = (temp[column] * (10 - temp.QBtw)).sum() / (10 - temp.QBtw).sum()
            elif method == 'custom_weighted':
                prev = (temp[column] * temp[weight_column]).sum() / temp[weight_column].sum()
            result.append([Q, prev])
        else:
            result.append([Q, prev])

    return pd.DataFrame(result, columns=['QBtw', 'E_ROE']).set_index('QBtw')


def search_prev(x, df):
    try:
        if df.loc[x.QBtw][-1] == 1:
            return 1
        else:
            return df[x.Security][x.QBtw]
    except:
        return None


def apply_bam(x, tempset):
    try:
        bam = (x * tempset.loc[x.name, 'Slope'] + tempset.loc[x.name, 'Intercept']).values[0]
        return bam
    except:
        return x.values[0]


def apply_imc(x, tempset):
    try:
        imc = x.E_ROE * tempset.loc[(x.SecAnl, x.QBtw), 'Slope'] + tempset.loc[(x.SecAnl, x.QBtw), 'Intercept']
        return imc
    except:
        return x.E_ROE


def term_spread_adj(sector, year, train):

    if 'Q' in year:
        currYear = int('20'+year[2:4])
        currQ = year[:2]
    else:
        currYear = int(year[:4])
        currQ = None

    # ----------------------------------------------------------------
    # OPTIMIZATION 1: 데이터 필터링 선행 (전체 데이터 dropna 금지)
    # ----------------------------------------------------------------
    # Sector 필터링 1차 적용 (전체 스캔 방지)
    if not pd.isna(sector):
        sector_df = train[train['SectorClass'] == sector]
    else:
        sector_df = train

    # 과거 연도 경계값 문자열 미리 생성 (루프 내 반복 방지)
    str_bf_min, str_bf_max = str(currYear - 11), str(currYear - 2)
    str_af_min, str_af_max = str(currYear - 10), str(currYear - 1)

    # 데이터 서브셋 슬라이싱
    if currQ:
        sector_df = sector_df[sector_df['Q']==currQ]
    prev_data_bf = sector_df[(sector_df['Year'] >= str_bf_min) & (sector_df['Year'] <= str_bf_max)]
    prev_data_af = sector_df[(sector_df['Year'] >= str_af_min) & (sector_df['Year'] <= str_af_max)]

    # ----------------------------------------------------------------
    # OPTIMIZATION 2: 아주 작아진 데이터프레임에만 dropna 적용 (속도 혁명)
    # ----------------------------------------------------------------
    prev_data_bf = prev_data_bf.dropna(subset=['Error'])
    prev_data_af = prev_data_af.dropna(subset=['Error'])
    num_of_obs = 5

    fit_kwargs = {
        'method': 'trf',
        'p0': [0, 0.01, 0.01, 0.01, 1],  # b0, c, b1, b2, lam
        'bounds': ((-1, -np.inf, -np.inf, -np.inf, -np.inf), (1, np.inf, np.inf, np.inf, np.inf))
    }
    # --- Case 1: Before Previous Year's Actual Data ---
    if len(prev_data_bf) < 20:
        popt_bf = np.full(num_of_obs, np.nan)
    else:
        try:
            x_matrix = prev_data_bf['DBtw'].to_numpy(dtype=np.float64)
            y_vector = prev_data_bf['Error'].to_numpy(dtype=np.float64)
            popt_bf, pcov_bf = curve_fit(term_spread, x_matrix, y_vector, **fit_kwargs)
            if pcov_bf[0, 0] == np.inf:
                raise Exception
        except:
            popt_bf = np.full(num_of_obs, np.nan)

    # --- Case 2: After Previous Year's Actual Data ---
    if len(prev_data_af) < 20:
        popt_af = np.full(num_of_obs, np.nan)
    else:
        try:
            # BUG FIXED: xdata와 ydata를 prev_data_af 기준으로 올바르게 변경
            x_matrix = prev_data_af['DBtw'].to_numpy(dtype=np.float64)
            y_vector = prev_data_af['Error'].to_numpy(dtype=np.float64)
            popt_af, pcov_af = curve_fit(term_spread, x_matrix, y_vector, **fit_kwargs)
            if pcov_af[0, 0] == np.inf:
                raise Exception
        except:
            popt_af = np.full(num_of_obs, np.nan)

    return {'popt_bf':popt_bf, 'popt_af':popt_af}


def get_quarter(x, lag:int=1):
    if x.month <= 3 + lag and x.month > 0 + lag:
        return f'1Q{str(x.year)[-2:]}AS'
    elif x.month > 3 + lag and x.month <= 6 + lag:
        return f'2Q{str(x.year)[-2:]}AS'
    elif x.month > 6 + lag and x.month <= 9 + lag:
        return f'3Q{str(x.year)[-2:]}AS'
    elif x.month > 9 + lag and x.month <= 12 + lag:
        return f'4Q{str(x.year)[-2:]}AS'
    else:
        return f'4Q{str(x.year-1)[-2:]}AS'

def shift_period(period, lag=1):
    if 'Q' in period:
        fy = int(period[-4:-2])
        q = period[:2]
        return f'{q}{fy - lag}AS'
    else:
        fy = int(period[:4])
        return f'{fy - lag}AS'

def safe_read_csv(file, **kwargs)->pd.DataFrame:
    try: df = pd.read_csv(file, **kwargs)
    except:
        try: df = pd.read_csv(file, encoding='cp949', **kwargs)
        except: df = pd.read_csv(file, encoding='utf-8-sig', **kwargs)

    return df

def build_data(path: str = './data/consenlist/*.csv'
               , period: str = 'Y'
               , ts_length: int = 10
               , sector_len: int = 2
               , country: str = 'kr'
               , prddate: str = None):
    '''
    build dataset for calculate Smart Consensus
    :param path: estimation data path
    :param period: period type - 'Y' for year, 'Q' for quarter
    :param rolling: rolling window for economic data
    :param ts_length: year length to use
    :param country: country('kr' or 'us')
    :return: dataset
    '''
    if country not in ['kr', 'us']:
        raise ValueError('country should be either kr or us')

    print('Building dataset...')

    consenlist = glob(path)
    if period == 'Y':
        consenlist = [file for file in consenlist if 'FQ' not in file]
    elif period == 'Q':
        consenlist = [file for file in consenlist if 'FQ' in file]
    else:
        raise ValueError('period should be either Y or Q')

    df_list = []
    for idx, file in enumerate(consenlist):
        tmp_df = safe_read_csv(file)
        tmp_df['NomialFY'] = file.split('_')[-1].split('.')[0]
        df_list.append(tmp_df)
    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True, axis=0).dropna(how='all')

    if country == 'us':
        df = df.rename(columns={'Instrument': 'Code'
            , 'Analyst Name': 'Analyst'
            , 'Broker Name': 'Security'
            , 'Period Year': 'Year'
            , 'Earnings Per Share - Broker Estimate': 'E_EPS'
            , 'EPS': 'A_EPS'
            , 'GICS': 'Sector'})
        df = df.dropna(subset=['Year'])
        # in PeriodEndDate, some data has 'YYYY-MM-DD' format, and some has 'YYYY-MM-DD HH:MM:SS' format.
        df['PeriodEndDate'] = pd.to_datetime(df['PeriodEndDate'].str.slice(0, 10))
        df['Year'] = np.where(df['PeriodEndDate'].dt.month < 7, df['Year'].astype(int) - 1, df['Year'].astype(int)).astype(str)
    else:
        row_fq = df['NomialFY'].str[0]
        row_fy = df['NomialFY'].str[-4:]
        fq_map = {'1': '03-31', '2': '06-30', '3': '09-30', '4': '12-31'}
        df['PeriodEndDate'] = pd.to_datetime(row_fy + '-' + row_fq.map(fq_map))

    df['Date'] = pd.to_datetime(df['Date'].str.slice(0, 10))

    if period == 'Q':
        df['FY'] = pd.to_datetime(df.PeriodEndDate).apply(lambda x: get_quarter(x, lag=1))
        df['Year']= '20' + df['FY'].str[-4:-2]
        df['Q'] =  df['FY'].str[:2]
    else:
        df['FY'] = df.Year.astype(str) + 'AS'
        df['Q'] = '4Q'

    df['Security'] = df.Security.replace(r'\([^)]*\)','', regex=True)
    df['FilingDeadline'] = df.PeriodEndDate - MonthEnd(9)
    df['A_EPS_1'] = np.where(df['Date'] < df['FilingDeadline'], df['EPS_2Y'], df['EPS_1Y'])

    df = df.dropna(subset=['BPS', 'E_EPS'])
    df = df.drop_duplicates()
    df.Sector = df.Sector.astype(int, errors='ignore')

    unique_sector_map = df.dropna(subset='Sector').groupby('Code')[['Sector']].last()
    # if sector is na, fill with unique_sector_map
    df['Sector'] = df['Sector'].fillna(df['Code'].map(unique_sector_map['Sector'])).astype(str)
    df['SectorClass'] = df.Sector.str[:sector_len]
    date_max = df.Date.max()

    # fill missed BPS data with previous fiscal year's BPS
    df['prev_FY'] = df['FY'].apply(lambda x: shift_period(x, lag=1))
    unique_bps_map = df.dropna(subset='BPS').groupby(['Code', 'FY'])['BPS'].last()
    unique_bps_map.index = unique_bps_map.index.set_names(['Code', 'prev_FY'])
    unique_bps_map = unique_bps_map.rename('BPS_fill')

    df_idx = pd.MultiIndex.from_arrays([df['Code'], df['prev_FY']])
    bps_values = unique_bps_map.reindex(df_idx).values

    df['BPS'] = df['BPS'].where(~df['BPS'].isna(), bps_values).astype(float)

    # drop BPS if it is less than 0
    df = df[df.BPS > 0]
    # Since previously downloaded data do not contain A_EPS data, fill missed A_EPS data with newly updated A_EPS
    df_eps = df[['Code', 'FY', 'A_EPS']].dropna(subset=['A_EPS']).groupby(['Code', 'FY'])['A_EPS'].last()
    df_eps.index = df_eps.index.set_names(['Code', 'FY'])
    df_idx = pd.MultiIndex.from_arrays([df['Code'], df['FY']])
    eps_values = df_eps.reindex(df_idx).values
    df['A_EPS'] = df['A_EPS'].where(~df['A_EPS'].isna(), eps_values).astype(float)

    df['UniqueSymbol'] = df['Code'] + df['FY']
    df['E_ROE'] = df['E_EPS'] / df.BPS
    df['A_ROE'] = df['A_EPS'] / df.BPS
    df['Error'] = df['E_ROE'] - df['A_ROE']

    # since information about earnings change as time, seperate date window as 90 days
    df['EDate'] = df['PeriodEndDate'] + MonthEnd(3)
    df['DBtw'] = (df.EDate - df.Date).dt.days
    df['YearDiff'] = df.EDate.dt.year - df.Date.dt.year
    df['MonthDiff'] = df.EDate.dt.month - df.Date.dt.month
    df['totalDiff'] = df['YearDiff'] * 12 + df['MonthDiff']
    df['QBtw'] = (df['totalDiff'] / 3).astype(int)
    df['QBtw'] = df['QBtw'].apply(lambda x: 7 if x > 7 else x)

    if period=='Q':
        df['equalEDate'] = np.where(df.Q=='1Q',
            pd.to_datetime((df.Year.astype(int)).astype(str) + '-04-30'),
            np.where(df.Q=='2Q',
                     pd.to_datetime((df.Year.astype(int)).astype(str) + '-07-31'),
                     np.where(df.Q=='3Q',
                              pd.to_datetime((df.Year.astype(int)).astype(str) + '-10-31'),
                              pd.to_datetime((df.Year.astype(int)+1).astype(str) + '-01-31')
                              )
                     )
                                    )
    else:
        df['equalEDate'] = pd.to_datetime((df.Year.astype(int) + 1).astype(str) + '-03-31')
    df['YearDiff'] = df.equalEDate.dt.year - df.Date.dt.year
    df['MonthDiff'] = df.equalEDate.dt.month - df.Date.dt.month
    df['totalDiff'] = df['YearDiff'] * 12 + df['MonthDiff']
    df['EQBtw'] = (df['totalDiff'] / 3).astype(int)
    df['Year'] = df.Year.astype(str)

    if prddate:
        temp_today = pd.to_datetime(prddate)
    else:
        temp_today = dt.datetime.today()
    df['totalDiff'] = (temp_today.year - df.Date.dt.year) * 12 + (temp_today.month - df.Date.dt.month)
    df['CQBtw'] = (df['totalDiff'] / 3).astype(int)

    df = df.drop(['YearDiff', 'MonthDiff', 'totalDiff', 'equalEDate'], axis=1)
    df['CutDate'] = df['FilingDeadline']

    if ts_length == -1:
        pass
    else:
        df = df[df.Year.astype(int) >= int(df.Year.max()) - (ts_length+5)] # add some margin on length of years

    df.to_parquet('./cache/cache.parquet', engine="pyarrow", compression="snappy")

    return df


def filter_guided_stock(df, codecol, errorcol, error_rate=0.01):
    '''
    if the stock is guided stock, remove the stock from the list
    we consider guided stock as the stock's error rate is under 1%
    :param df: dataset
    :param codecol: code column name
    :param errorcol: error column name
    :return: filtered new dataset
    '''
    new_train = []
    for year in df.Year.unique()[1:]:
        temp = df[df.Year == str(int(year) - 1)]
        temp_grouped = temp.groupby(codecol)[errorcol].mean()
        for idx in range(2, 6):
            temp = df[df.Year == str(int(year) - idx)]
            temp_grouped = temp_grouped.fillna(temp.groupby(codecol)[errorcol].mean())
        temp_list = temp_grouped[temp_grouped > error_rate].index
        temp = df[(df.Year == year) & (df.Code.isin(temp_list))]
        new_train.append(temp)
    new_train = pd.concat(new_train)

    return new_train

def eps_growth(x, col_name='EPS_Est', caption=False):
    # if eps_1y is nan, use eps_2y
    if x[col_name] > 0:
        if not(pd.isna(x.EPS_1Y)) and x.EPS_1Y>0:
            if caption:
                return '1Y'
            else:
                return ((x[col_name] / x.EPS_1Y) - 1) * 100
        elif not(pd.isna(x.EPS_2Y)) and x.EPS_2Y>0:
            if caption:
                return '2Y'
            else:
                return ((x[col_name] / x.EPS_2Y) ** (1/2) - 1) * 100
        else:
            return np.nan
    else:
        return np.nan


def result_formatter(data, code, df, popt_bf, popt_af):
    data['Code'] = code
    data['Sector'] = df.Sector.iloc[-1]
    data['Popt'] = [[popt_bf, popt_af]] * len(data)
    data['BPS'] = df.BPS.iloc[0]
    data['PeriodEndDate'] = df.PeriodEndDate.iloc[0]
    data['EPS_Actual'] = df.A_EPS.iloc[0]
    data['EPS_1Y'] = df['EPS_1Y'].mean()
    data['EPS_2Y'] = df['EPS_2Y'].mean()

    return data

def result_formatter_calc_growth(data):
    data['EPS_Est'] = data.Est * data.BPS
    data.EPS_Est = data.apply(lambda x: x.EPS_Actual if (~pd.isna(x.EPS_Actual)) & (x.QBtw == 0) else x.EPS_Est, axis=1)
    data['EPS_EW'] = data.EW * data.BPS
    data.EPS_EW = data.apply(lambda x: x.EPS_Actual if (~pd.isna(x.EPS_Actual)) & (x.QBtw == 0) else x.EPS_EW, axis=1)

    data['Est'] = data['EPS_Est'] / data['BPS']
    data['GEst'] = data['Est'] - data['EW_prev']

    data['Shock'] = data.Est - data.EW
    data['GEst'] = data.Est - data.EW_prev

    # if eps_1y is nan, use eps_2y
    data['EPS_G'] = data.apply(lambda x: eps_growth(x, col_name='EPS_Est'), axis=1)
    data['EPS_EW_G'] = data.apply(lambda x: eps_growth(x, col_name='EPS_EW'), axis=1)

    data['EPS_G_caption'] = data.apply(lambda x: eps_growth(x, caption=True), axis=1)

    return data

def generate_financial_periods(start_prd, suffix='AS'):
    # '2Q26' 또는 '2Q26AS'에서 접미사 제거 후 분기/연도 추출
    clean_prd = start_prd.replace(suffix, '')
    quarter = int(clean_prd[0])
    year = int(clean_prd[2:]) + 2000  # 26 -> 2026

    # pandas Period 객체로 변환 (예: 2026Q2)
    base_period = pd.Period(f"{year}Q{quarter}", freq='Q')

    # 1. prdFY 생성 (기준 분기 포함 향후 4개 분기)
    prd_periods = [base_period + i for i in range(4)]
    prdFY = [f"{p.quarter}Q{str(p.year)[2:]}{suffix}" for p in prd_periods]

    # 2. curveFY 생성 (과거 3개년 = 12개 분기 전부터 prdFY의 마지막 분기까지)
    # 총 개수: 과거 12개 분기 + prdFY 4개 분기 = 16개 분기
    start_curve_period = base_period - 12
    curve_periods = [start_curve_period + i for i in range(16)]
    curveFY = [f"{p.quarter}Q{str(p.year)[2:]}{suffix}" for p in curve_periods]

    return prdFY, curveFY


def term_spread_now(x, gdp, b0, c, b1, b2, lam):
    try:
        theta = x / 365 / lam
        return b0 + c * gdp / 100 + b1 * np.exp(-theta) + b2 * theta * np.exp(-theta)

    except:
        return np.full_like(x, np.nan)


def merged_ts(total_ts, fy:str, prddate:str='2024-11-11'):
    '''
    merge total term spread data by gdp senarios
    :param total_ts: ts per gdp senarios
    :param year: prediected eps year
    :return: merged ts
    '''
    year = int(fy[:4])
    ts = total_ts.filter(regex=(f"{year}AS"))
    ts.index = [dt.datetime(year+1, 3, 31) - dt.timedelta(t) for t in ts.index]

    # split ()_date to ()
    ts_sector = ts.columns.str.split('_').str[0].unique()
    tmp_sector_ts = []
    for sector in ts_sector:
        sector_tmp = ts.filter(regex=(sector))
        sector_tmp.columns = [x.split('_')[1] for x in sector_tmp.columns]
        # if date of index is smaller than column name, fill nan for those index's data
        for column in sector_tmp.columns:
            column_date = pd.to_datetime(column)
            sector_tmp.loc[sector_tmp.index < column_date, column] = np.nan

        try:
            basecolumn = sector + '_' + sector_tmp.columns[sector_tmp.columns > prddate].min()
            sector_tmp.columns = sector + '_' + sector_tmp.columns
            columns = sector_tmp.columns[::-1]

            # Create a new column with the merged data
            sector_tmp[sector] = sector_tmp[columns].bfill(axis=1).iloc[:, 0]
            tmp_sector_ts.append(sector_tmp[[sector, basecolumn]])
        except:
            pass

    if len(tmp_sector_ts) > 0:
        ts = pd.concat(tmp_sector_ts, axis=1).dropna().sort_index()
    else:
        ts = pd.DataFrame()

    return ts


def build_gdp_scenario(model):

    keys = list(model.ucurve.keys())
    ts = np.linspace(1, 365 * 2)
    arr = np.column_stack([
        term_spread_now(ts, 0, *model.ucurve[key]['popt_af'])
        for key in tqdm(keys, desc='build_gdp_scenario')
    ])
    total_ts_pd = pd.DataFrame(arr, index=ts, columns=keys)

    return total_ts_pd