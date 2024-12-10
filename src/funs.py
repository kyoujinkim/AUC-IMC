import configparser
import json
import os
from glob import glob
from itertools import product

import datetime as dt
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


def load_db(code):
    config = configparser.ConfigParser()
    config.read('./src/config.ini')

    MYSQL_HOSTNAME = config['DB']['MYSQL_HOSTNAME']  # you probably don't need to change this
    MYSQL_USER = config['DB']['MYSQL_USER']
    MYSQL_PASSWORD = config['DB']['MYSQL_PASSWORD']
    MYSQL_DATABASE = config['DB']['MYSQL_DATABASE']

    connection_string = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOSTNAME}/{MYSQL_DATABASE}'
    engine = create_engine(connection_string)
    db = pd.read_sql(f"SELECT * FROM cache_enh_eps WHERE Code = '{code}'", con=engine)

    return db


def term_spread(x, b0, c, b1, b2, lam):
    theta = x.DBtw / 365 / lam
    return b0 + c*x.Gdp/100 + b1 * np.exp(-theta) + b2 * theta * np.exp(-theta)


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

    train = train.dropna(subset=['Error'])
    currYear = int(year)
    sector = sector
    # if sector is na
    if pd.isna(sector):
        prev_data_bf = train[(train.Year <= str(currYear - 2))  & (train.Year >= str(currYear - 11))]
        prev_data_af = train[(train.Year <= str(currYear - 1))  & (train.Year >= str(currYear - 10))]
    else:
        prev_data_bf = train[(train.SectorClass == sector) & (train.Year <= str(currYear - 2)) & (train.Year >= str(currYear - 11))]
        prev_data_af = train[(train.SectorClass == sector) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 10))]

    num_of_obs = 5

    # case for data which announced before previous year's actual data
    if len(prev_data_bf) < 20:
        popt_bf = np.array([np.nan] * num_of_obs)
    else:
        try:
            popt_bf, pcov_bf = curve_fit(term_spread
                , prev_data_bf
                , prev_data_bf.Error
                , method='trf'
                , p0=[0, 0.01, 0.01, 0.01, 1]#b0, c, b1, b2, lam
                , bounds=((-1, -np.inf, -np.inf, -np.inf, -np.inf),(1, np.inf, np.inf, np.inf, np.inf))
            )
            if pcov_bf[0,0] == np.inf:
                raise Exception
        except:
            popt_bf = np.array([np.nan] * num_of_obs)

    # case for data which announced after previous year's actual data
    if len(prev_data_af) < 20:
        popt_af = np.array([np.nan] * num_of_obs)
    else:
        try:
            popt_af, pcov_af = curve_fit(
                term_spread
                , xdata=prev_data_bf
                , ydata=prev_data_bf.Error
                , method='trf'
                , p0=[0, 0.01, 0.01, 0.01, 1]#b0, c, b1, b2, lam
                , bounds=((-1, -np.inf, -np.inf, -np.inf, -np.inf),(1, np.inf, np.inf, np.inf, np.inf))
            )
            if pcov_af[0,0] == np.inf:
                raise Exception
        except:
            popt_af = np.array([np.nan] * num_of_obs)

    return {'popt_bf':popt_bf, 'popt_af':popt_af}


def EW(x, train):

    symbol = x
    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    EW = df.groupby('QBtw')['E_ROE'].mean()
    data = EW - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Est': EW.values, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data), 'FY': [symbol[7:]] * len(data)}
    )

    return fulldata

def build_data(path: str = './data/consenlist/*.csv'
               , use_gdp: bool = True
               , gdp_path: str = './data/QGDP.xlsx'
               , gdp_header: int = 0
               , gdp_lag: int = 0
               , rolling: int = 0
               , ts_length: int = 10
               , sector_len: int = 2
               , country: str = 'kr'
               , use_cache: bool = False):
    '''
    build dataset for calculate Smart Consensus
    :param path: estimation data path
    :param gdppath: economic data path
    :param gdp_header: header row - 13 for QuantiWise data, 0 for Refinitiv data
    :param gdp_lag: lagged month for economic data
    :param rolling: rolling window for economic data
    :param ts_length: year length to use
    :param country: country('kr' or 'us')
    :return: dataset
    '''
    if country not in ['kr', 'us']:
        raise ValueError('country should be either kr or us')

    if use_cache:
        if os.path.exists('./cache/cache.csv'):
            return pd.read_csv('./cache/cache.csv', encoding='utf-8-sig', index_col=0
                               , dtype={'Year':str, 'Sector':str, 'SectorClass':str})

    consenlist = glob(path)
    for idx, file in enumerate(consenlist):
        if idx == 0:
            df = pd.read_csv(file)
        else:
            df = pd.concat([df, pd.read_csv(file)], ignore_index=True, axis=0)

    if country == 'us':
        df = df.rename(columns={'Instrument': 'Code'
            , 'Analyst Name': 'Analyst'
            , 'Broker Name': 'Security'
            , 'Period Year': 'Year'
            , 'Earnings Per Share - Broker Estimate': 'E_EPS'
            , 'EPS': 'A_EPS'
            , 'GICS': 'Sector'})
        df = df.dropna(subset='Year')
        df.PeriodEndDate = pd.to_datetime(df.PeriodEndDate)
        # if PeriodEndDate is less than july, use next year. Else, use current year
        df['Year'] = df.apply(lambda x: x.Year - 1 if x.PeriodEndDate.month < 7 else x.Year, axis=1)
        df.Year = df.Year.astype(int).astype(str)
    else:
        df['PeriodEndDate'] = pd.to_datetime(df.Year.astype(str) + '-12-31')

    df.Date = pd.to_datetime(df.Date)

    df['FY'] = df.Year.astype(str) + 'AS'
    df['Security'] = df.Security.replace(r'\([^)]*\)','', regex=True)
    df['FilingDeadline'] = df.PeriodEndDate - MonthEnd(9)
    df['A_EPS_1'] = df.apply(lambda x: x.EPS_1Y if x.Date > x.FilingDeadline else x.EPS_2Y, axis=1)

    df = df.dropna(subset=['BPS', 'E_EPS'])
    df = df.drop_duplicates()
    df.Sector = df.Sector.astype(int, errors='ignore').astype(str, errors='ignore')
    df['SectorClass'] = df.Sector.str[:sector_len]
    date_max = df.Date.max()

    df.BPS = df.BPS.astype(float)
    # droprow if BPS is less than 0
    df = df[df.BPS > 0]
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

    df['equalEDate'] = pd.to_datetime((df.Year.astype(int) + 1).astype(str) + '-03-31')
    df['YearDiff'] = df.equalEDate.dt.year - df.Date.dt.year
    df['MonthDiff'] = df.equalEDate.dt.month - df.Date.dt.month
    df['totalDiff'] = df['YearDiff'] * 12 + df['MonthDiff']
    df['EQBtw'] = (df['totalDiff'] / 3).astype(int)

    df = df.drop(['YearDiff', 'MonthDiff', 'totalDiff', 'equalEDate'], axis=1)
    df['CutDate'] = df['FilingDeadline']

    if use_gdp:
        gdp = pd.read_excel(gdp_path, sheet_name='act', header=gdp_header, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
        if gdp_lag > 0:
            gdp.index = gdp.index + pd.DateOffset(months=gdp_lag)
        gdp_roll = gdp.rolling(rolling).mean().dropna()
        # extend index of gdp_roll to date_max
        gdp_roll = gdp_roll.reindex(pd.date_range(gdp_roll.index[0], date_max, freq='D'))
        gdp_roll = gdp_roll.ffill()

        df['Gdp'] = df.Date.map(gdp_roll.iloc[:,0])
    else:
        df['Gdp'] = 0

    if ts_length == -1:
        pass
    else:
        df = df[df.Year.astype(int) >= int(df.Year.max()) - (ts_length+5)] # add some margin on length of years

    df.to_csv('./cache/cache.csv', encoding='utf-8-sig')

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


def result_formatter(data, code, df, df_copy, popt_bf, popt_af):
    data['Code'] = code
    data['Sector'] = df.Sector.iloc[-1]
    data['Popt'] = [[popt_bf, popt_af]] * len(data)
    data['BPS'] = df.BPS.iloc[0]
    data['PeriodEndDate'] = df.PeriodEndDate.iloc[0]
    data['EPS_Actual'] = df.A_EPS.iloc[0]
    data['EPS_1Y'] = df_copy['EPS_1Y'].mean()
    data['EPS_2Y'] = df_copy['EPS_2Y'].mean()

    return data

def result_formatter_calc_growth(data):
    data['EPS_Est'] = data.Est * data.BPS
    data.EPS_Est = data.apply(lambda x: x.EPS_Actual if (~pd.isna(x.EPS_Actual)) & (x.QBtw == 0) else x.EPS_Est, axis=1)
    data['EPS_EW'] = data.EW * data.BPS

    data['Est'] = data['EPS_Est'] / data['BPS']
    data['GEst'] = data['Est'] - data['EW_prev']

    data['Shock'] = data.Est - data.EW
    data['GEst'] = data.Est - data.EW_prev

    # if eps_1y is nan, use eps_2y
    data['EPS_G'] = data.apply(lambda x: eps_growth(x, col_name='EPS_Est'), axis=1)
    data['EPS_EW_G'] = data.apply(lambda x: eps_growth(x, col_name='EPS_EW'), axis=1)

    data['EPS_G_caption'] = data.apply(lambda x: eps_growth(x, caption=True), axis=1)

    return data

def IMC_adp_cache(x):

    symbol = x[0]
    code = symbol[:-6]
    train = load_db(code)

    min_count = x[1]
    year_range = x[2]
    ucurve = x[3]

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    year = symbol[-6:-2]
    sector = df.SectorClass.iloc[-1]
    popt = ucurve[sector, int(year)]
    popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
    popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x, *popt_bf)
                              if x.Date <= x.CutDate
                              else term_spread(x, *popt_af)
                              , axis=1).fillna(0))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df_copy = df_copy.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
    df['SecAnl'] = df['Security'] + df['CoreAnalyst']
    Q_result = []
    S_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - year_range))
                             ]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - year_range - 1))
                             ]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if pd.isna(popt_af[0]):
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
        else:
            if pd.isna(popt_bf[0]):
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
        tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

        if len(tempdata) > 0:
            tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
            tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

            # polyfit E_ROE with A_ROE per analyst
            tempset = []
            for S in df.SecAnl.unique():
                temp = tempdata[tempdata.SecAnl == S]
                if len(temp) >= 10 and len(temp.Year.unique()) >= min_count:
                    try:
                        lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),
                                                                              tempdata['A_ROE'])
                        slope = lr_result.coef_[0]
                        intercept = lr_result.intercept_
                    except:
                        slope = 1
                        intercept = 0
                else:
                    slope = 1
                    intercept = 0
                tempset.append([S, Q, slope, intercept])

            # get S by S data
            SQ_df = pd.DataFrame(tempset, columns=['S', 'Q', 'Slope', 'Intercept']).set_index(['S', 'Q'])
            S_result.append(SQ_df)

            # polyfit E_ROE with A_ROE per company
            tempdata.E_ROE = tempdata.apply(lambda x: apply_imc(x, SQ_df), axis=1)

        # list to append previous year's error rate by analyst
        lenYear = len(tempdata.Year.unique())
        if lenYear >= min_count:
            # Linear Regression between E_ROE and A_ROE
            try:
                lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),
                                                                      tempdata['A_ROE'])
                slope = lr_result.coef_[0]
                intercept = lr_result.intercept_
            except:
                slope = 1
                intercept = 0
        elif lenYear == 0:
            slope = 1
            intercept = 0
        else:
            slope = 1
            intercept = 0

        Q_result.append([Q, slope, intercept])

    if len(S_result) > 0:
        Scoeffset = pd.concat(S_result)
        df['E_ROE'] = df.apply(lambda x: apply_imc(x, Scoeffset), axis=1)

        Scoeffset_mean = Scoeffset.reset_index().groupby('Q').Slope.mean()
    else:
        Scoeffset_mean = pd.Series([1] * len(df.QBtw.unique()), index=df.QBtw.unique())

    Qcoeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
    # with slope and intercept, calculate BAM
    estIMC_step1 = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
    estIMC = estIMC_step1.apply(lambda x: apply_bam(x, Qcoeffset), axis=1)

    estEW = pd.DataFrame(df_copy.groupby('QBtw')['E_ROE'].mean())
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1'].last() / df_copy.groupby('QBtw')['BPS'].last())

    eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

    data = pd.concat([estIMC, estEW, estEW_prev, Scoeffset_mean, Qcoeffset.Slope, eqbtw], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev', 'SCoeff', 'QCoeff', 'EQBtw']

    data = result_formatter(data, code, df, df_copy, popt_bf, popt_af)

    return data


def IMSE_adp_cache(x):

    symbol = x[0]
    code = symbol[:-6]
    train = load_db(code)

    min_count = x[1]
    year_range = x[2]
    ucurve = x[3]

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    year = symbol[-6:-2]
    sector = df.SectorClass.iloc[-1]
    popt = ucurve[sector, int(year)]
    popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
    popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x, *popt_bf)
                              if x.Date <= x.CutDate
                              else term_spread(x, *popt_af)
                              , axis=1).fillna(0))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df_copy = df_copy.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
        tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

        # list to append previous year's error rate by analyst
        tempset = []

        unique_sec = df.Security.unique()
        for sec in unique_sec:
            df_sec = tempdata[tempdata.Security == sec]
            if len(df_sec) > 0:
                df_sec_error = df_sec['Error'].abs().mean()
                tempset.append([sec, df_sec_error])

        # if previous year's data exist, calculate smart consensus
        if len(tempset) > 0:
            prev_error = pd.DataFrame(tempset, columns=['Security', 'Error']).set_index('Security')
            # if prev_year's anaylst data is not enough(less than 5 data point), append all
            check_prev_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.index))]
            if len(check_prev_count) < min_count:
                check_prev_count = df[df.QBtw == Q]
                check_prev_count['PrevError'] = 1
                Q_result.append(check_prev_count)
            else:
                check_prev_count['PrevError'] = check_prev_count.apply(lambda x: prev_error.loc[x.Security].values[0], axis=1)
                Q_result.append(check_prev_count)
        else:
            check_prev_count = df[df.QBtw == Q]
            check_prev_count['PrevError'] = 1
            Q_result.append(check_prev_count)

    if len(Q_result) > 0:
        df = pd.concat(Q_result)
        df.PrevError += 0.01

    df['I_PrevError'] = df['PrevError'].pow(-1)
    # limit upper and lower bound of I_PrevError as +- 2 stdev
    df_mean = df['I_PrevError'].mean()
    df_std = df['I_PrevError'].std()
    df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 2 * df_std, upper=df_mean + 2 * df_std)
    df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']
    estIMSE = df.groupby('QBtw')['W_E_ROE'].sum() / df.groupby('QBtw')['I_PrevError'].sum()

    estEW = pd.DataFrame(df_copy.groupby('QBtw')['E_ROE'].mean())
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1'].last() / df_copy.groupby('QBtw')['BPS'].last())

    eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

    data = pd.concat([estIMSE, estEW, estEW_prev, eqbtw], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

    data = result_formatter(data, code, df, df_copy, popt_bf, popt_af)

    return data

def EW_adp_cache(x):

    symbol = x[0]
    code = symbol[:-6]
    train = load_db(code)

    min_count = x[1]
    year_range = x[2]
    ucurve = x[3]

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    year = symbol[-6:-2]
    sector = df.SectorClass.iloc[-1]
    popt = ucurve[sector, int(year)]
    popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
    popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x, *popt_bf)
                              if x.Date <= x.CutDate
                              else term_spread(x, *popt_af)
                              , axis=1).fillna(0))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])

    estAdpEW = df.groupby('QBtw')['E_ROE'].mean()

    estEW = pd.DataFrame(df_copy.groupby('QBtw')['E_ROE'].mean())
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1'].last() / df_copy.groupby('QBtw')['BPS'].last())

    eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

    data = pd.concat([estAdpEW, estEW, estEW_prev, eqbtw], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

    data = result_formatter(data, code, df, df_copy, popt_bf, popt_af)

    return data


def EW_cache(x):

    symbol = x[0]
    code = symbol[:-6]
    train = load_db(code)

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])

    estEW = df.groupby('QBtw')['E_ROE'].mean()

    # estEW = pd.DataFrame(df_copy.groupby('QBtw')['E_ROE'].mean())
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1'].last() / df_copy.groupby('QBtw')['BPS'].last())

    eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

    data = pd.concat([estEW, estEW, estEW_prev, eqbtw], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

    data = result_formatter(data, code, df, df_copy, 0, 0)

    return data


def term_spread_now(x, gdp, b0, c, b1, b2, lam):
    theta = x / 365 / lam
    return b0 + c * gdp / 100 + b1 * np.exp(-theta) + b2 * theta * np.exp(-theta)


def merged_ts(total_ts, year:int, prddate:str='2024-11-11'):
    '''
    merge total term spread data by gdp senarios
    :param total_ts: ts per gdp senarios
    :param year: prediected eps year
    :return: merged ts
    '''
    ts = total_ts.filter(regex=(f", {year}"))
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
        basecolumn = sector + '_' + sector_tmp.columns[sector_tmp.columns < prddate][-1]
        sector_tmp.columns = sector + '_' + sector_tmp.columns
        columns = sector_tmp.columns[::-1]

        # Create a new column with the merged data
        sector_tmp[sector] = sector_tmp[columns].bfill(axis=1).iloc[:, 0]
        tmp_sector_ts.append(sector_tmp[[sector, basecolumn]])

    ts = pd.concat(tmp_sector_ts, axis=1).dropna().sort_index()

    return ts


def build_gdp_scenario(model, gdp_data_path, use_prd:bool=True, use_gdp:bool=True):

    if use_gdp:
        # save total term spread by gdp and time delta to csv
        gdp = pd.read_excel(gdp_data_path, sheet_name='act', header=13, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
        gdp.index = gdp.index + pd.DateOffset(months=2)
        gdp.columns = ['gdp']
        if use_prd:
            gdp_prd = pd.read_excel(gdp_data_path, sheet_name='prd', header=0, index_col=0, parse_dates=True)
            # append gdp_prd to gdp's row
            gdp = pd.concat([gdp, gdp_prd], axis=0)
        gdp_roll = gdp.rolling(4).mean().dropna()
        gdp_roll = gdp_roll.iloc[-7:]

        total_ts = []
        for key in tqdm(model.ucurve.keys()):

            popt_af = model.ucurve[key]['popt_af']

            ts = np.linspace(1, 365*2)

            for gdpidx, gdp in gdp_roll.iterrows():
                tempts = term_spread_now(ts, gdp.values, *popt_af)
                total_ts.append(pd.Series(tempts, name=f'{key}_{gdpidx.strftime("%Y-%m-%d")}', index=ts))

        total_ts_pd = pd.concat(total_ts, axis=1)

    else:
        total_ts = []
        for key in tqdm(model.ucurve.keys()):

            popt_af = model.ucurve[key]['popt_af']

            ts = np.linspace(1, 365 * 2)

            tempts = term_spread_now(ts, 0, *popt_af)
            total_ts.append(pd.Series(tempts, name=f'{key}', index=ts))

        total_ts_pd = pd.concat(total_ts, axis=1)

    return total_ts_pd