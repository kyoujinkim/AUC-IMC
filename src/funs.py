import json
import os
from glob import glob
from itertools import product

import numpy as np
from numpy import nan
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#disable warnings
import warnings
warnings.filterwarnings('ignore')


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
        prev_data_bf = train[(train.Sector == sector) & (train.Year <= str(currYear - 2)) & (train.Year >= str(currYear - 11))]
        prev_data_af = train[(train.Sector == sector) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 10))]

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

def build_data(path:str='./data/consenlist*.csv', gdppath:str='./data/QGDP.xlsx', ts_length:int=10, country:str='kr'):
    if country not in ['kr', 'us']:
        raise ValueError('country should be either kr or us')

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
        df.Year = df.Year.astype(int).astype(str)
        df['FY'] = df.Year + 'AS'
        df['Security'] = df.Security.replace(r'\([^)]*\)','', regex=True)
        df['FilingDeadline'] = (df.Year.astype(int) - 1).astype(str) + '-04-15'
        df['A_EPS_1'] = df.apply(lambda x: x.EPS_1Y if x.Date > x.FilingDeadline else x.EPS_2Y, axis=1)

    df = df.dropna(subset=['BPS', 'E_EPS'])
    df = df.drop_duplicates()
    df.Date = pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    date_max = df.Date.max()

    gdp = pd.read_excel(gdppath, sheet_name='Sheet1', header=13, index_col=0, parse_dates=True).dropna(how='all', axis=0).dropna(axis=1)
    gdp.index = gdp.index + pd.DateOffset(months=2)
    gdp_roll = gdp.rolling(4).mean().dropna()
    # extend index of gdp_roll to date_max
    gdp_roll = gdp_roll.reindex(pd.date_range(gdp_roll.index[0], date_max, freq='D'))
    gdp_roll = gdp_roll.ffill()

    df['Year'] = df.FY.str.extract(r'(\d{4})')
    df['CutDate'] = (df.Year.astype(int) - 1).astype(str) + '-03-31'
    df = df[df.Date > df.CutDate]
    df['Gdp'] = df.Date.map(gdp_roll.iloc[:,0])

    if ts_length == -1:
        pass
    else:
        df = df[df.Year >= str(int(df.Year.max()) - (ts_length+5))] # add some margin on length of years

    return df