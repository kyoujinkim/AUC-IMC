import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#disable warnings
import warnings
from src.funs import *
warnings.filterwarnings('ignore')


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


def EW(x):

    symbol = x[0]
    code = symbol[:-6]
    train = load_db(code)

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    EW = df.groupby('QBtw')['E_ROE'].mean()
    data = EW - df.groupby('QBtw')['A_ROE'].mean()
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Est': EW.values, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data)}
    )

    return fulldata


def PBest(x):

    symbol = x[0]
    star_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

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
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data)}
    )

    return fulldata

def IMSE(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

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
    df_mean = df['I_PrevError'].mean()
    df_std = df['I_PrevError'].std()
    df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 3 * df_std, upper=df_mean + 3 * df_std)
    df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']
    data = ((df.groupby('QBtw')['W_E_ROE'].sum() / df.groupby('QBtw')['I_PrevError'].sum())
            - df.groupby('QBtw')['A_ROE'].mean())
    data_std = df.groupby('QBtw')['E_ROE'].std()
    fulldata = pd.DataFrame(
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data)}
    )

    return fulldata


def BAM(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 5))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

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
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data)}#, , 'Coeff': [coeffset]*len(data)}
    )

    return fulldata


def BAM_adj(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    Q_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        # list to append previous year's error rate by analyst
        lenYear = len(tempdata.Year.unique())
        if lenYear >= min_count:
            # Linear Regression between E_ROE and A_ROE
            try:
                lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']), tempdata['A_ROE'])
                slope = lr_result.coef_[0]
                intercept = lr_result.intercept_
            except:
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
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data), 'QCoeff': coeffset.Slope}
    )

    return fulldata


def IMC(x):

    symbol = x[0]
    min_count = x[1]

    df = train[train.UniqueSymbol == symbol]
    df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
    df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
    df['SecAnl'] = df['Security'] + df['CoreAnalyst']
    Q_result = []
    S_result = []

    for Q in df.QBtw.unique():
        if Q < 4:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
        else:
            tempdata = train[(train.Code == df.Code.iloc[0])
                             & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                             & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                             & (train.QBtw == Q)]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if len(tempdata) > 0:
            tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
            tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

            # polyfit E_ROE with A_ROE per analyst
            tempset = []
            for S in df.SecAnl.unique():
                temp = tempdata[tempdata.SecAnl == S]
                if len(temp) >= 10 and len(temp.Year.unique()) >= min_count:
                    try:
                        lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),tempdata['A_ROE'])
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
                lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']), tempdata['A_ROE'])
                slope = lr_result.coef_[0]
                intercept = lr_result.intercept_
            except:
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
        {'QBtw': data.index, 'Error': data.values, 'Std': data_std.values, 'Code': [symbol[:-6]] * len(data), 'FY': [symbol[-6:]] * len(data), 'QCoeff': Qcoeffset.Slope}
    )

    return fulldata

country = 'us'
if country == 'us':
    use_gdp = False
    period = 'Q'
    gdp_path = f'data/{country}/PR.xlsx'
    gdp_header = 8
    gdp_lag = 0
    rolling = 1
    ts_length = -1
    sector_len = 2
elif country == 'kr':
    use_gdp = True
    period = 'Y'
    gdp_path = f'data/{country}/QGDP.xlsx'
    gdp_header = 13
    gdp_lag = 2
    rolling = 4
    ts_length = -1
    sector_len = 3


if __name__ == '__main__':

    print('build train data')
    # if os.path.exists(f'./cache/cache.parquet'):
    #    # remove cache
    #    os.remove(f'./cache/cache.parquet')

    train = build_data(f'data/{country}/consenlist/*.csv'
                       , period=period
                       , use_gdp=use_gdp
                       , gdp_path=gdp_path
                       , gdp_header=gdp_header
                       , gdp_lag=gdp_lag
                       , rolling=rolling
                       , ts_length=ts_length
                       , sector_len=sector_len
                       , country=country
                       , use_cache=True)

    if country == 'us':
        new_train = train[train.A_EPS_1.abs() / train.BPS < 3]
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)
        # retain only guidance given stock
        if period == 'Y':
            new_train = new_train[new_train.Guidance == 1]
        UniqueSymbol = new_train.UniqueSymbol.unique()
    else:
        new_train = train[train.A_EPS_1.abs() / train.BPS < 3]
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)
        UniqueSymbol = new_train.UniqueSymbol.unique()

    # (1) simple average
    dataset = []

    dataset = process_map(EW_cache, UniqueSymbol, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MAFE'] = dataset_pd.Error.abs()
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MAFE', 'Std']].mean()
    print(MSFE_result)
    dataset_pd.to_csv(f'./result/{country}/EW.csv', encoding='utf-8-sig')
    MSFE_result.to_csv(f'./result/{country}/EW_MSFE.csv', encoding='utf-8-sig')


    '''# (2) smart consensus
    # measure analyst's error rate by year
    star_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [star_count]))

    dataset = process_map(PBest, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MAFE'] = dataset_pd.Error.abs()
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MAFE', 'Std']].mean()
    print(MSFE_result)
    dataset_pd.to_csv(f'./result/{country}/PBest.csv', encoding='utf-8-sig')
    MSFE_result.to_csv(f'./result/{country}/PBest_MSFE.csv', encoding='utf-8-sig')


    # (3) Inverse MSE (IMSE)
    min_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(IMSE, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MAFE'] = dataset_pd.Error.abs()
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MAFE', 'Std']].mean()
    print(MSFE_result)
    dataset_pd.to_csv(f'./result/{country}/IMSE.csv', encoding='utf-8-sig')
    MSFE_result.to_csv(f'./result/{country}/IMSE_MSFE.csv', encoding='utf-8-sig')'''


    '''# (5) Bias-Adjusted Mean Adjusted (BAM_adj)
    min_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(BAM_adj, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MAFE'] = dataset_pd.Error.abs()
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MAFE', 'Std']].mean()
    print(MSFE_result)
    dataset_pd.to_csv(f'./result/{country}/BAM_adj.csv', encoding='utf-8-sig')
    MSFE_result.to_csv(f'./result/{country}/BAM_adj_MSFE.csv', encoding='utf-8-sig')


    # (6) Iterated Mean Combination (IMC)
    min_count = 5
    dataset = []
    multi_arg = list(product(UniqueSymbol, [min_count]))

    dataset = process_map(IMC, multi_arg, max_workers=os.cpu_count()-1)

    dataset_pd = pd.concat(dataset)
    dataset_pd['MAFE'] = dataset_pd. Error.abs()
    MSFE_result = dataset_pd.groupby(['QBtw'])[['MAFE', 'Std']].mean()
    print(MSFE_result)
    dataset_pd.to_csv(f'./result/{country}/IMC.csv', encoding='utf-8-sig')
    MSFE_result.to_csv(f'./result/{country}/IMC_MSFE.csv', encoding='utf-8-sig')'''

    '''#draw 3d surface plot with dataset_pd
    byFY = dataset_pd.groupby(['FY', 'QBtw'])['MSFE'].mean()
    byFY_r = byFY.reset_index()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(byFY_r.QBtw, byFY_r.FY.str[:4].astype(float), byFY_r.MSFE, cmap='viridis')
    ax.set_xlabel('QBtw')
    ax.set_ylabel('FY')
    ax.set_zlabel('Error')
    plt.show()'''


# equation should be y = AVG( x * q(t) ) + b
# where q(t) = k / ( 1 + exp(-(t - t0)) )
#train['q'] = (1 / (1 + np.exp(-train.DBtw)) - 0.5) * 2 # set range as 0 to 1
