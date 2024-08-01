import json
import os
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
from src.funs import *
import warnings
warnings.filterwarnings('ignore')


class Enhanced_EPS(object):
    def __init__(self):
        self.ucurve = dict()
        self.min_count = 5
        self.year_range = 10

    def calc_ucurve(self, prdYear, train):
        listoftable = list(product(train.Sector.unique(), prdYear))
        for table in tqdm(listoftable):
            self.ucurve[table] = term_spread_adj(*table, train=train)

        pd.DataFrame(self.ucurve).to_json('./result/ucurve.json')

    def calc_IMC(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        dataset = process_map(IMC_adp, multi_arg, max_workers=os.cpu_count() - 1)

        dataset_pd = pd.concat(dataset)
        dataset_pd.index.name = 'QBtw'
        dataset_pd = dataset_pd[dataset_pd.Shock != 0]
        dataset_pd = dataset_pd.reset_index()
        dataset_pd = dataset_pd.set_index('Code')
        #get quantile of shock
        dataset_pd['MinMax'] = dataset_pd.groupby(['Code']).Shock.max() - dataset_pd.groupby(['Code']).Shock.min().values
        counter = 8
        while True:
            try:
                dataset_pd['Q_GEst'] = dataset_pd.groupby(['QBtw']).GEst.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1
        counter = 8
        while True:
            try:
                dataset_pd['Q_Shock'] = dataset_pd.groupby(['QBtw']).Shock.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1
        counter = 8
        while True:
            try:
                dataset_pd['Q_Vol'] = dataset_pd.groupby(['QBtw', 'Q_Shock'])['MinMax'].transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1

        return dataset_pd

    def calc_adpEW(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        dataset = process_map(EW_adp, multi_arg, max_workers=os.cpu_count() - 1)

        dataset_pd = pd.concat(dataset)
        dataset_pd.index.name = 'QBtw'
        dataset_pd = dataset_pd[dataset_pd.Shock != 0]
        dataset_pd = dataset_pd.reset_index()
        dataset_pd = dataset_pd.set_index('Code')
        #get quantile of shock
        dataset_pd['MinMax'] = dataset_pd.groupby(['Code']).Shock.max() - dataset_pd.groupby(['Code']).Shock.min().values
        counter = 8
        while True:
            try:
                dataset_pd['Q_GEst'] = dataset_pd.groupby(['QBtw']).GEst.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1
        counter = 8
        while True:
            try:
                dataset_pd['Q_Shock'] = dataset_pd.groupby(['QBtw']).Shock.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1
        counter = 8
        while True:
            try:
                dataset_pd['Q_Vol'] = dataset_pd.groupby(['QBtw', 'Q_Shock'])['MinMax'].transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1

        return dataset_pd



def IMC_adp(x):

    global train

    symbol = x[0]
    min_count = x[1]
    year_range = x[2]
    ucurve = x[3]

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    year = symbol[-6:-2]
    sector = df.Sector.iloc[-1]
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
                             & (train.Year >= str(int(df.Year.iloc[0]) - year_range -1))
                            ]
            tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

        if Q < 4:
            if pd.isna(popt_af[0]):
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1))
        else:
            if pd.isna(popt_bf[0]):
                tempdata = tempdata[(tempdata.QBtw == Q)]
            tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1))
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
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1(지배)'].last()/df_copy.groupby('QBtw')['BPS'].last())

    data = pd.concat([estIMC, estEW, estEW_prev, Scoeffset_mean, Qcoeffset.Slope], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev', 'SCoeff', 'QCoeff']
    data['Code'] = symbol[:7]
    data['Popt'] = [[popt_bf, popt_af]]*len(data)
    data['Shock'] = data.Est - data.EW
    data['GEst'] = data.Est - data.EW_prev
    data['BPS'] = df.BPS.iloc[0]
    data['EPS_Est'] = data.Est * data.BPS

    return data


def EW_adp(x):

    symbol = x[0]
    min_count = x[1]
    year_range = x[2]
    ucurve = x[3]

    df = train[train.UniqueSymbol == symbol]
    df_copy = df.copy()

    year = symbol[-6:-2]
    sector = df.Sector.iloc[-1]
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
    estEW_prev = pd.DataFrame(df_copy.groupby('QBtw')['A_EPS_1(지배)'].last()/df_copy.groupby('QBtw')['BPS'].last())

    data = pd.concat([estAdpEW, estEW, estEW_prev], axis=1)
    data.columns = ['Est', 'EW', 'EW_prev']
    data['Code'] = symbol[:7]
    data['Popt'] = [[popt_bf, popt_af]]*len(data)
    data['Shock'] = data.Est - data.EW
    data['GEst'] = data.Est - data.EW_prev
    data['BPS'] = df.BPS.iloc[0]
    data['EPS_Est'] = data.Est * data.BPS

    return data


train = build_data('./data/consenlist*.csv', './data/QGDP.xlsx', 10)
train.BPS = train.BPS.astype(float)
#droprow if BPS is less than 0
train = train[train.BPS > 0]
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
    prdYears = [2023, 2024, 2025]
    model = Enhanced_EPS()
    model.calc_ucurve(prdYears, train)
    for prdyear in prdYears:
        UniqueSymbol = train[(train.Year==str(prdyear))].UniqueSymbol.unique()
        result = model.calc_IMC(UniqueSymbol)
        result.to_csv(f'./result/IMC_adp{prdyear}.csv', encoding='utf-8-sig')
        result = model.calc_adpEW(UniqueSymbol)
        result.to_csv(f'./result/EW_adp{prdyear}.csv', encoding='utf-8-sig')
