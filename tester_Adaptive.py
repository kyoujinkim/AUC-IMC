import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#disable warnings
import warnings
warnings.filterwarnings('ignore')


def term_spread(x, k, t):
    return k * (1 / (1 + np.exp(-x/t)) - 0.5)


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
        return (x * tempset.loc[x.name, 'Slope'] + tempset.loc[x.name, 'Intercept']).values[0]
    except:
        return x.values[0]


def apply_imc(x, tempset):
    try:
        return x.E_ROE * tempset.loc[(x.SecAnl, x.QBtw), 'Slope'] + tempset.loc[(x.SecAnl, x.QBtw), 'Intercept']
    except:
        return x.E_ROE


def term_spread_adj(x):

    currYear = int(x[-6:-2])
    code = str(x[:7])
    prev_data_bf = train[(train.Code == code) & (train.Year <= str(currYear - 2)) & (train.Year >= str(currYear - 3))]
    prev_data_af = train[(train.Code == code) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 3))]

    # case for data which announced before previous year's actual data
    if len(prev_data_bf) < 10:
        popt_bf = np.array([0, 1])
    else:
        popt_bf, _ = curve_fit(term_spread, prev_data_bf.DBtw, prev_data_bf.Error)

    # case for data which announced after previous year's actual data
    if len(prev_data_af) < 10:
        popt_af = np.array([0, 1])
    else:
        popt_af, _ = curve_fit(term_spread, prev_data_af.DBtw, prev_data_af.Error)

    return popt_bf, popt_af

def EW_adp(x):

    symbol = x
    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])

    data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()

    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def PBest_adp(x):

    symbol = x[0]
    star_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata[~(tempdata.Year == str(int(df.Year.iloc[0]) - 1))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_af), axis=1))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_bf), axis=1))
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
            check_star_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.nsmallest(star_count, 'Error').index))]
            if len(check_star_count) < 5:
                Q_result.append(df[df.QBtw == Q])
            else:
                Q_result.append(check_star_count)
        else:
            Q_result.append(df[df.QBtw == Q])

    if len(Q_result) > 0:
        df = pd.concat(Q_result)

    data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata

def IMSE_adp(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
            tempdata = tempdata[~(tempdata.Year == str(int(df.Year.iloc[0]) - 1))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_af), axis=1))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_bf), axis=1))
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
    df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']
    data = ((df.groupby('QBtw')['W_E_ROE'].sum() / df.groupby('QBtw')['I_PrevError'].sum())
            - df.groupby('QBtw')['A_ROE'].mean())
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def BAM_adp(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata[~(tempdata.Year == str(int(df.Year.iloc[0]) - 1))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_af), axis=1))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_bf), axis=1))
        tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

        # list to append previous year's error rate by analyst
        tempset = tempdata.groupby(['Year'])[['E_ROE', 'A_ROE']].mean()
        if len(tempset) >= min_count:
            # Linear Regression between E_ROE and A_ROE
            slope, intercept = np.polyfit(tempset['E_ROE'], tempset['A_ROE'], 1)
        elif len(tempset) == 0:
            slope = 1
            intercept = 0
        else:
            slope = 1
            intercept = 0

        Q_result.append([Q, slope, intercept])

    coeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
    # with slope and intercept, calculate BAM
    estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
    estBAM = estEW.apply(lambda x: apply_bam(x, coeffset), axis=1)

    data = estBAM - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def BAM_adj_adp(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata[~(tempdata.Year == str(int(df.Year.iloc[0]) - 1))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_af), axis=1))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_bf), axis=1))
        tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

        # list to append previous year's error rate by analyst
        lenYear = len(tempdata.Year.unique())
        if lenYear >= min_count:
            # Linear Regression between E_ROE and A_ROE
            randarr = np.random.randint(low=-20, high=20, size=len(tempdata)) / 10000
            slope, intercept = np.polyfit(tempdata['E_ROE'], tempdata['A_ROE'] + randarr, 1)
        elif lenYear == 0:
            slope = 1
            intercept = 0
        else:
            slope = 1
            intercept = 0

        Q_result.append([Q, slope, intercept])

    coeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
    # with slope and intercept, calculate BAM
    estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
    estBAM = estEW.apply(lambda x: apply_bam(x, coeffset), axis=1)

    data = estBAM - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def IMC_adp(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = (df['E_ROE']
                   - df.apply(lambda x:
                              term_spread(x.DBtw, *popt_bf)
                              if x.Date<=x.CutDate
                              else term_spread(x.DBtw, *popt_af)
                              , axis=1))
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
    df['SecAnl'] = df['Security'] + df['CoreAnalyst']
    Q_result = []
    S_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))]
            tempdata = tempdata[~(tempdata.Year == str(int(df.Year.iloc[0]) - 1))]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if popt_af[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_af), axis=1))
        else:
            if popt_bf[0] == 0:
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x.DBtw, *popt_bf), axis=1))
        tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

        if len(tempdata) > min_count:
            tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
            tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

            # polyfit E_ROE with A_ROE per analyst
            tempset = []
            for S in df.SecAnl.unique():
                temp = tempdata[tempdata.SecAnl == S]
                if len(temp) >= min_count and len(temp.Year.unique())>1:
                    randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                    slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] + randarr, 1)
                elif len(temp) == 0:
                    slope = 1
                    intercept = 0
                else:
                    slope = 1
                    intercept = (temp.A_ROE - temp.E_ROE).mean()
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
            randarr = np.random.randint(low=-20, high=20, size=len(tempdata)) / 10000
            slope, intercept = np.polyfit(tempdata['E_ROE'], tempdata['A_ROE'] + randarr, 1)
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

    Qcoeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
    # with slope and intercept, calculate BAM
    estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
    estIMC = estEW.apply(lambda x: apply_bam(x, Qcoeffset), axis=1)

    data = estIMC - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


train = pd.read_csv('./data/train.csv', encoding='utf-8-sig')
train['UniqueSymbol'] = train['Code'] + train['FY']
train['E_ROE'] = train['E_EPS(지배)'] / train.BPS
train['A_ROE'] = train['A_EPS(지배)'] / train.BPS
train['Error'] = train['E_ROE'] - train['A_ROE']

# since information about earnings change as time, seperate date window as 90 days
train['Year'] = train.FY.str.extract(r'(\d{4})')
train['EDate'] = pd.to_datetime((train.Year.astype(int) + 1).astype(str) + '-03-31')
train['DBtw'] = (train.EDate - pd.to_datetime(train.Date)).dt.days
train['YearDiff'] = train.EDate.dt.year - pd.to_datetime(train.Date).dt.year
train['MonthDiff'] = train.EDate.dt.month - pd.to_datetime(train.Date).dt.month
train['totalDiff'] = train['YearDiff'] * 12 + train['MonthDiff']
train['QBtw'] = (train['totalDiff'] / 3).astype(int)

train = train.drop(['YearDiff','MonthDiff','totalDiff'], axis=1)
train['CutDate'] = train.Year + '-03-31'

if __name__ == '__main__':
    UniqueSymbol = train.UniqueSymbol.unique()

    # (1) simple average
    dataset = []

    dataset = process_map(EW_adp, UniqueSymbol, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (2) smart consensus
    # measure analyst's error rate by year
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [star_count]))

    dataset = process_map(PBest_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (3) Inverse MSE (IMSE)
    min_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(IMSE_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (4) Bias-Adjusted Mean (BAM)
    min_count = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(BAM_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (5) Bias-Adjusted Mean Adjusted (BAM_adj)
    min_count = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(BAM_adj_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (6) Iterated Mean Combination (IMC)
    min_count = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(IMC_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


# equation should be y = AVG( x * q(t) ) + b
# where q(t) = k / ( 1 + exp(-(t - t0)) )
#train['q'] = (1 / (1 + np.exp(-train.DBtw)) - 0.5) * 2 # set range as 0 to 1
