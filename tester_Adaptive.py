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


def search_prev(x, df):
    try:
        if df.loc[x.QBtw][-1] == 1:
            return 1
        else:
            return df[x.Security][x.QBtw]
    except:
        return None


def avg_by_unique_sec(df, column='E_ROE', weight_column=None, method='simple'):
    result = []
    prev = None
    for Q in df.QBtw.unique():
        temp = df[df.QBtw >= Q]
        temp = temp.drop_duplicates(subset=['Security'], keep='last')
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


def apply_inv(x):
    try:
        return x**(-1)
    except:
        return x


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
    prev_data_af = train[(train.Code == code) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 2))]

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
    df['E_ROE'] = df['E_ROE'] - df.apply(lambda x: term_spread(x.DBtw, *popt_bf) if x.Date<=x.CutDate else term_spread(x.DBtw, *popt_af), axis=1)
    dfexp = avg_by_unique_sec(df, method='weighted')

    data = dfexp - df['A_ROE'].iloc[0]
    data_std = df.groupby('QBtw')['E_ROE'].std()

    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.E_ROE.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def PBest_adp(x):

    symbol = x[0]
    star_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    popt_bf, popt_af = term_spread_adj(symbol)
    df['E_ROE'] = df['E_ROE'] - df.apply(lambda x: term_spread(x.DBtw, *popt_bf) if x.Date<=x.CutDate else term_spread(x.DBtw, *popt_af), axis=1)

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
    tempdata['E_ROE'] = tempdata['E_ROE'] - term_spread(tempdata['DBtw'], *popt)
    tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

    prev_error = tempdata.groupby('Security')['Error'].apply(lambda x: x.abs().mean())
    prev_error.name = 'Smart'

    # if previous year's data exist, calculate smart consensus
    if len(prev_error) > 0:
        # if prev_year's anaylst data is not enough(less than 5 data point), fill with 1
        prev_error = prev_error.sort_values().iloc[:star_count]
        df_smart = df.join(prev_error, on='Security')
        df_smart = df_smart.dropna(subset='Smart')

        # if there is no Smart Consensus available
        if len(df_smart) > 0:
            dfexp = avg_by_unique_sec(df_smart, method='weighted')

            data = dfexp - df_smart['A_ROE'].iloc[0]
            data_std = df_smart.groupby('QBtw')['E_ROE'].std()

            fulldata = pd.DataFrame(
                {'QBtw': data.index, 'Error': data.E_ROE.values, 'Std': data_std.values,
                 'Code': [symbol[:7]] * len(data)}
            )
        else:
            data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
            data_std = df.groupby('QBtw')['E_ROE'].std()

            fulldata = pd.DataFrame(
                {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
            )

    # else use simple average of all
    else:
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
    popt = term_spread_adj(symbol)
    df['E_ROE'] = df['E_ROE'] - term_spread(df['DBtw'], *popt)

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year == str(int(df.Year.iloc[0]) - 1))]
    tempdata['E_ROE'] = tempdata['E_ROE'] - term_spread(tempdata['DBtw'], *popt)
    tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

    prev_error = tempdata.groupby('Security')['Error'].apply(lambda x: x.abs().mean())
    prev_error.name = 'Smart'

    # if previous year's data exist, calculate smart consensus
    if len(prev_error) > 0:
        df = df.join(prev_error, on='Security')
        df_smart = df.dropna(subset='Smart')

        # calculate inverse MSE
        df_smart['ISmart'] = df_smart['Smart'].apply(lambda x: apply_inv(x))
        df_smart['CustomWeight'] = df_smart['QBtw'] * df_smart['ISmart']
        estIMSE = avg_by_unique_sec(df_smart, column='E_ROE', weight_column='CustomWeight', method='custom_weighted')

        data = estIMSE - df_smart['A_ROE'].iloc[0]
        data_std = df_smart.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.E_ROE.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    # else use simple average of all
    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    return fulldata


def BAM_adp(x):

    symbol = x[0]
    min_count = x[1]
    min_year = x[2]

    df = train[train.UniqueSymbol == symbol]
    popt = term_spread_adj(symbol)
    df['E_ROE'] = df['E_ROE'] - term_spread(df['DBtw'], *popt)

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata['E_ROE'] = tempdata['E_ROE'] - term_spread(tempdata['DBtw'], *popt)
    tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

    if len(tempdata) > min_count:

        tempset = tempdata.groupby(['Year'])[['E_ROE', 'A_ROE']].mean()

        if len(tempset) >= min_count:
            #Linear Regression between E_ROE and A_ROE
            randarr = np.random.randint(low=-20, high=20, size=len(tempset)) / 10000
            slope, intercept = np.polyfit(tempset['E_ROE'], tempset['A_ROE'] - randarr, 1)
        elif len(tempset) == 0:
            slope = 1
            intercept = 0
        else:
            #If there is less than three data points, slope is 1 and intercept is the difference between E_ROE and A_ROE
            slope = 1
            intercept = (tempset.A_ROE - tempset.E_ROE).mean()

        tempset.loc['Slope'] = slope
        tempset.loc['Intercept'] = intercept

        # get Q by Q data
        tempset = tempset.groupby('QBtw').mean()
        # with slope and intercept, calculate BAM
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
        estBAM = estEW.apply(lambda x: apply_bam(x, tempset), axis=1)

        data = estBAM - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    return fulldata


def BAM_adj_adp(x):
    '''
    same as BAM, but with individual's forecast
    :param x: index
    :return:
    '''

    symbol = x[0]
    min_count = x[1]
    min_year = x[2]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['E_ROE'] = term_spread_adj(symbol)

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year','QBtw'])

    if len(tempdata) > min_count:

        Q_set = []
        for Q in df.QBtw.unique():
            temp = tempdata[tempdata.QBtw == Q]
            if len(temp) >= min_count and len(temp.Year.unique())>1:
                #Linear Regression between E_ROE and A_ROE
                randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] - randarr, 1)
            elif len(temp) == 0:
                slope = 1
                intercept = 0
            else:
                #If there is less than three data points, slope is 1 and intercept is the difference between E_ROE and A_ROE
                slope = 1
                intercept = (temp.A_ROE - temp.E_ROE).mean()

            Q_set.append([Q, slope, intercept])

        # get Q by Q data
        Q_df = pd.DataFrame(Q_set).set_index(0)
        Q_df.columns = ['Slope', 'Intercept']
        # with slope and intercept, calculate BAM
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
        estBAM = estEW.apply(lambda x: apply_bam(x, Q_df), axis=1)

        data = estBAM - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    return fulldata

def IMC_adp(x):
    '''
    Iterated Mean Combination
    :param x:
    :return:
    '''

    symbol = x[0]
    min_count = x[1]
    min_year = x[2]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
    df['SecAnl'] = df['Security'] + df['CoreAnalyst']
    df['E_ROE'] = term_spread_adj(symbol)

    tempdata = train[(train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['Code', 'E_ROE', 'Security', 'Year','QBtw'])

    # calculate error rate by analyst
    if len(tempdata) > min_count:

        tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
        tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

        SQ_set = []
        for S, Q in df[['SecAnl','QBtw']].drop_duplicates().values:
            temp = tempdata[(tempdata.SecAnl == S) & (tempdata.QBtw == Q)]
            if len(temp) >= min_count and len(temp.Year.unique())>1:
                #Linear Regression between E_ROE and A_ROE
                randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] - randarr, 1)
            elif len(temp) == 0:
                slope = 1
                intercept = 0
            else:
                #If there is less than three data points, slope is 1 and intercept is the difference between E_ROE and A_ROE
                slope = 1
                intercept = (temp.A_ROE - temp.E_ROE).mean()

            SQ_set.append([S, Q, slope, intercept])

        # get S by S data
        SQ_df = pd.DataFrame(SQ_set).set_index([0, 1])
        SQ_df.columns = ['Slope', 'Intercept']
        # with slope and intercept, calculate BAM

    else:
        SQ_df = pd.DataFrame()

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

    # calculate error rate by company
    if len(tempdata) > min_count:

        tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
        tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

        tempdata.E_ROE = tempdata.apply(lambda x: apply_imc(x, SQ_df), axis=1)

        Q_set = []
        for Q in df.QBtw.unique():
            temp = tempdata[tempdata.QBtw == Q]
            if len(temp) >= min_count and len(temp.Year.unique())>1:
                randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] - randarr, 1)
            elif len(temp) == 0:
                slope = 1
                intercept = 0
            else:
                slope = 1
                intercept = (temp.A_ROE - temp.E_ROE).mean()

            Q_set.append([Q, slope, intercept])

        Q_df = pd.DataFrame(Q_set).set_index(0)
        Q_df.columns = ['Slope', 'Intercept']

        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
        estIMC = estEW.apply(lambda x: apply_bam(x, Q_df), axis=1)

        data = estIMC - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    return fulldata


def PBIMC_adp(x):
    '''
    Iterated Mean Combination
    :param x:
    :return:
    '''

    symbol = x[0]
    min_count = x[1]
    min_year = x[2]
    star_count = x[3]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
    df['SecAnl'] = df['Security'] + df['CoreAnalyst']
    df['E_ROE'] = term_spread_adj(symbol)

    tempdata = train[(train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['Code', 'E_ROE', 'Security', 'Year','QBtw'])

    # calculate error rate by analyst
    if len(tempdata) > min_count:

        tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
        tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

        SQ_set = []
        for S, Q in df[['SecAnl','QBtw']].drop_duplicates().values:
            temp = tempdata[(tempdata.SecAnl == S) & (tempdata.QBtw == Q)]
            if len(temp) >= min_count and len(temp.Year.unique())>1:
                #Linear Regression between E_ROE and A_ROE
                randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] - randarr, 1)
            elif len(temp) == 0:
                slope = 1
                intercept = 0
            else:
                #If there is less than three data points, slope is 1 and intercept is the difference between E_ROE and A_ROE
                slope = 1
                intercept = (temp.A_ROE - temp.E_ROE).mean()

            SQ_set.append([S, Q, slope, intercept])

        # get S by S data
        SQ_df = pd.DataFrame(SQ_set).set_index([0, 1])
        SQ_df.columns = ['Slope', 'Intercept']
        # with slope and intercept, calculate BAM

    else:
        SQ_df = pd.DataFrame()

    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

    # find best previous forecast
    tempset = []
    unique_sec = df.Security.unique()
    # calculate previous year's error rate by analyst(generalized as house)
    for sec in unique_sec:
        df_sec = tempdata[tempdata.Security == sec]
        if len(df_sec) > 0:
            df_sec_error = df_sec.groupby('QBtw')['Error'].apply(lambda x: x.abs().mean())
            df_sec_error.name = sec
            tempset.append(df_sec_error)
        else:
            df_sec_error = pd.Series(index=df.QBtw.unique())
            df_sec_error.name = sec
            tempset.append(df_sec_error)

    # if previous year's data exist, calculate smart consensus
    if len(tempset) > 0:
        prev_error = pd.concat(tempset, axis=1)
        # if prev_year's anaylst data is not enough(less than 5 data point), fill with 1
        for row in prev_error.iterrows():
            if row[1].count() < 3:
                prev_error.loc[row[0]] = 1
            else:
                # remain only few(star_count) data point from smallest
                prev_error.loc[row[0]] = prev_error.loc[row[0]].abs().sort_values().iloc[:star_count]

        df['Smart'] = df.apply(lambda x: search_prev(x, prev_error), axis=1)
        df = df.dropna(subset='Smart')

    # calculate error rate by company
    if len(tempdata) > min_count:

        tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
        tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

        tempdata.E_ROE = tempdata.apply(lambda x: apply_imc(x, SQ_df), axis=1)

        Q_set = []
        for Q in df.QBtw.unique():
            temp = tempdata[tempdata.QBtw == Q]
            if len(temp) >= min_count and len(temp.Year.unique())>1:
                randarr = np.random.randint(low=-20, high=20, size=len(temp)) / 10000
                slope, intercept = np.polyfit(temp['E_ROE'], temp['A_ROE'] - randarr, 1)
            elif len(temp) == 0:
                slope = 1
                intercept = 0
            else:
                slope = 1
                intercept = (temp.A_ROE - temp.E_ROE).mean()

            Q_set.append([Q, slope, intercept])

        Q_df = pd.DataFrame(Q_set).set_index(0)
        Q_df.columns = ['Slope', 'Intercept']

        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE'].mean())
        estPBIMC = estEW.apply(lambda x: apply_bam(x, Q_df), axis=1)

        data = estPBIMC - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()
        fulldata = pd.DataFrame(
            {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
        )

    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
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
train['CutDate'] = train.Year + '-03-31'

#train['theta'] = (1 / (1 + np.exp(-train.DBtw/90)) - 0.5)

if __name__ == '__main__':
    UniqueSymbol = train.UniqueSymbol.unique()

    # (1) simple average
    dataset = []

    dataset = process_map(EW_adp, UniqueSymbol, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('EW', '\n', MSFE_result)


    '''# (2) smart consensus
    # measure analyst's error rate by year
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [star_count]))

    #for arg in tqdm(multi_arg):
    #    dataset.append(PBest_adp(arg))
    dataset = process_map(PBest_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('PBest', '\n', MSFE_result)


    # (3) Inverse MSE (IMSE)
    min_count = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    for arg in tqdm(multi_arg):
        dataset.append(IMSE_adp(arg))

    dataset = process_map(IMSE_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('IMSE', '\n', MSFE_result)'''


    ''''# (4) Bias-Adjusted Mean (BAM)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    for arg in tqdm(multi_arg):
        dataset.append(BAM_adp(arg))
    dataset = process_map(BAM_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('BAM', '\n', MSFE_result)


    # (5) Bias-Adjusted Mean Adjusted (BAM_adj)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    dataset = process_map(BAM_adj_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('BAM_adj', '\n', MSFE_result)


    # (6) Iterated Mean Combination (IMC)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    dataset = process_map(IMC_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('IMC', '\n', MSFE_result)


    # (7) Privious Best Iterated Mean Combination (PBIMC)
    min_count = 3
    min_year = 3
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year], [star_count]))

    dataset = process_map(PBIMC_adp, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print('PBIMC_adp', '\n', MSFE_result)'''

# equation should be y = AVG( x * q(t) ) + b
# where q(t) = k / ( 1 + exp(-(t - t0)) )
#train['q'] = (1 / (1 + np.exp(-train.DBtw)) - 0.5) * 2 # set range as 0 to 1
