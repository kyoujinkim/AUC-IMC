import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#disable warnings
import warnings
warnings.filterwarnings('ignore')


def search_prev(x, df):
    try:
        if df.loc[x.QBtw][-1] == 1:
            return 1
        else:
            return df[x.Security][x.QBtw]
    except:
        return None


def apply_inv(x):
    try:
        return x.pow(-1)
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


def EW_adp(x):

    symbol = x
    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])

    currYear = int(x[-6:-2])
    code = str(x[:7])
    prev_data = train[(train.Code == code) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 3))]
    if len(prev_data.Year.unique()) < 3:
        slope, intercept = None, None
        #slope = None
    else:
        slope, intercept = np.polyfit(prev_data.theta, prev_data.Error, 1)
        #slope = np.linalg.lstsq(prev_data.theta.values.reshape(-1,1), prev_data.Error.values, rcond=None)[0][0]

    if slope is not None:
        df['E_ROE_adj'] = df.E_ROE - (df['theta'] * slope + intercept)
        #df['E_ROE_adj'] = df.E_ROE - df['theta'] * slope
        data = df.groupby('QBtw')['E_ROE_adj'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE_adj'].std()
    else:
        data = df.groupby('QBtw')['E_ROE'].mean() - df.groupby('QBtw')['A_ROE'].mean()
        data_std = df.groupby('QBtw')['E_ROE'].std()

    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:7]] * len(data)}
    )

    return fulldata


def PBest(x):

    symbol = x[0]
    star_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year','QBtw'])
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


def IMSE(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    unique_sec = df.Security.unique()
    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year == str(int(df.Year.iloc[0]) - 1))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year','QBtw'])
    tempset = []

    # calculate previous year's error rate by analyst(generalized as house)
    for sec in unique_sec:
        df_sec = tempdata[tempdata.Security == sec]
        if len(df_sec) > 0:
            df_sec_error = df_sec.groupby('QBtw')['Error'].apply(lambda x: x.pow(2).mean())
            df_sec_error.name = sec
            tempset.append(df_sec_error)

    # if previous year's data exist, calculate smart consensus
    if len(tempset) > 0:
        prev_error = pd.concat(tempset, axis=1)
        # if prev_year's anaylst data is not enough(less than 5 data point), fill with 1
        for row in prev_error.iterrows():
            if row[1].count() < min_count:
                prev_error.loc[row[0]] = 1
            else:
                # remain only few(star_count) data point from smallest
                prev_error.loc[row[0]] = prev_error.loc[row[0]].abs().sort_values()
        df['Smart'] = df.apply(lambda x: search_prev(x, prev_error), axis=1)
        df = df.dropna(subset='Smart')

        # calculate inverse MSE
        df['ISmart'] = df['Smart'].apply(lambda x: apply_inv(x))
        df['ME_ROE'] = df['E_ROE'] * df['ISmart']
        estIMSE = df.groupby(['QBtw'])['ME_ROE'].sum() / df.groupby(['QBtw'])['ISmart'].sum()

        data = estIMSE - df.groupby('QBtw')['A_ROE'].mean()
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


def BAM(x):

    symbol = x[0]
    min_count = x[1]
    min_year = x[2]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    tempdata = train[(train.Code == df.Code.iloc[0]) & (train.Year <= str(int(df.Year.iloc[0]) - 1)) & (train.Year >= str(int(df.Year.iloc[0]) - min_year))]
    tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year','QBtw'])

    if len(tempdata) > min_count:

        tempset = tempdata.groupby(['QBtw', 'Year'])[['E_ROE', 'A_ROE']].mean()

        for Q in df.QBtw.unique():
            temp = tempset[tempset.index.get_level_values('QBtw') == Q]
            if len(temp) >= min_count:
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

            tempset.loc[Q, 'Slope'] = slope
            tempset.loc[Q, 'Intercept'] = intercept

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


def BAM_adj(x):
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

def IMC(x):
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


def PBIMC(x):
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
train['QBtw'] = (train.DBtw / 90).astype(int)

#train['theta'] = (1 / (1 + np.exp(-train.DBtw/90)) - 0.5)

def term_spread(x, k, t):
    return k / (1 + np.exp(-x/t)) - 0.5

if __name__ == '__main__':
    UniqueSymbol = train.UniqueSymbol.unique()

    # (1) simple average
    dataset = []

    #for symbol in tqdm(UniqueSymbol):
    #    dataset.append(EW_adp(symbol))
    dataset = process_map(EW_adp, UniqueSymbol, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    '''# (2) smart consensus
    # measure analyst's error rate by year
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [star_count]))

    dataset = process_map(PBest, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)'''


    '''# (3) Inverse MSE (IMSE)
    min_count = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(IMSE, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (4) Bias-Adjusted Mean (BAM)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    dataset = process_map(BAM, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (5) Bias-Adjusted Mean Adjusted (BAM_adj)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    dataset = process_map(BAM_adj, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (6) Iterated Mean Combination (IMC)
    min_count = 3
    min_year = 3
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year]))

    dataset = process_map(IMC, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)


    # (7) Privious Best Iterated Mean Combination (PBIMC)
    min_count = 3
    min_year = 3
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count], [min_year], [star_count]))

    dataset = process_map(PBIMC, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MSFE'] = dataset_pd.Error ** 2
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MSFE', 'Std']].mean()
    print(MSFE_result)'''

# equation should be y = AVG( x * q(t) ) + b
# where q(t) = k / ( 1 + exp(-(t - t0)) )
#train['q'] = (1 / (1 + np.exp(-train.DBtw)) - 0.5) * 2 # set range as 0 to 1
