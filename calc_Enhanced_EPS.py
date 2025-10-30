import itertools
import os

import matplotlib
import numpy as np
import datetime as dt

import pandas as pd
from tqdm import tqdm

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import build_data, build_gdp_scenario, merged_ts, filter_guided_stock
from src.enh_EPS import Enhanced_EPS
import warnings
warnings.filterwarnings('ignore')

def get_sector_growth_rate(x, est_name='earning_Est', act_name='earning_1Y', cap_name='earning_1Y_caption'):
    # if col earning_1Y_caption is smaller than 0, **1/2 to growth rate
    # and if earning or earning_1Y is negative, return None
    if x[act_name] > 0 and x[est_name] > 0:
        return (x[est_name] / x[act_name] - 1) * 100 if x[cap_name] > 0 else (x[est_name] / x[act_name] ** 0.5 - 1) * 100
    elif x[act_name] > 0 and x[est_name] < 0:
        return 'T/L'
    elif x[act_name] < 0 and x[est_name] < 0:
        return 'R/L'
    elif x[act_name] < 0 and x[est_name] > 0:
        return 'T/P'
    else:
        return None

def groupby_func(df, cols):
    result = df.groupby(cols)[['Est', 'EW', 'EW_prev', 'EPS_G', 'EPS_EW_G', 'EPS_G_bld', 'EPS_EW_G_bld']].median()
    result_total = df.groupby(cols)[['earning_Est', 'earning_EW', 'earning_Est_bld', 'earning_EW_bld', 'earning_1Y', 'earning_1Y_caption', 'earning_1Y_EW',
                                     'earning_1Y_EW_caption']].sum()
    result['earning_total'] = result_total['earning_Est']
    result['earning_EW_total'] = result_total['earning_EW']
    result['earning_total_bld'] = result_total['earning_Est_bld']
    result['earning_EW_total_bld'] = result_total['earning_EW_bld']
    result['earning_prev_total'] = result_total['earning_1Y']
    # if earning_1Y_catpion smaller than 0, **1/2 to growth rate
    result['earning_G'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_Est', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result['earning_EW_G'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_EW', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result['earning_G_bld'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_Est_bld', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result['earning_EW_G_bld'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_EW_bld', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result_count = df.groupby(cols)[['Est', 'EPS_Actual']].count()
    result_gcount = df.groupby(cols)[['G']].sum()
    result['count'] = result_count['Est']
    result['gcount'] = result_gcount['G']
    result['acount'] = result_count['EPS_Actual']

    return result.reset_index()

def groupby_sector(df, df_sector, col_name):
    '''
    Sector별로 groupby하여 median값을 구함
    :param df: 데이터를 포함한 DataFrame
    :param df_sector: Sector name을 지닌 DataFrame
    :param col_name: df에서 groupby할 column name
    :return:
    '''
    # 섹터 단위 계산을 위해 net earning 계산
    df = df.groupby(['Code','FY',col_name]).first().reset_index()

    df['earning_Est'] = df['shares'] * df['EPS_Est']
    df['earning_EW'] = df['shares'] * df['EPS_EW']

    df['earning_Est_bld'] = df['shares'] * df['EPS_Est_bld']
    df['earning_EW_bld'] = df['shares'] * df['EPS_EW_bld']

    df['earning_1Y'] = df['shares'] * df.apply(lambda x: x['EPS_2Y'] if pd.isna(x['EPS_1Y_Est']) else x['EPS_1Y_Est'], axis=1)
    df['earning_1Y_EW'] = df['shares'] * df.apply(lambda x: x['EPS_2Y'] if pd.isna(x['EPS_1Y_EW']) else x['EPS_1Y_EW'], axis=1)
    df['earning_1Y_caption'] = df.apply(lambda x: -1 if pd.isna(x['EPS_1Y_Est']) else 1, axis=1)
    df['earning_1Y_EW_caption'] = df.apply(lambda x: -1 if pd.isna(x['EPS_1Y_EW']) else 1, axis=1)

    result_all = groupby_func(df, ['FY', col_name])
    # make index as (All, 0)
    result_all['Sector_name'] = 'All'
    result_all['Sector'] = '00'
    #result_all = result_all.set_index(['Sector_name', col_name])

    result_sector = groupby_func(df, ['FY', 'Sector', col_name])
    result_Lsector = groupby_func(df, ['FY', 'LSector', col_name])
    result_Lsector = result_Lsector.rename(columns={'LSector': 'Sector'})

    # Sector별로 median값을 구한 결과를 result_sector에 추가
    result_sector = pd.concat([result_all, result_sector, result_Lsector], axis=0)
    result_sector = result_sector.reset_index()
    result_sector = result_sector.set_index(result_sector.columns[0])

    result_sector = result_sector.set_index('Sector')
    # Sector별로 median값을 구한 결과에 Sector_name을 추가
    result_sector['Sector_name'] = df_sector.Sector

    return result_sector

def eps_growth(x, est_name, y1_name, caption=False):
    # if eps_1y is nan, use eps_2y
    if x[est_name] > 0:
        if not(pd.isna(x.EPS_1Y)) and x.EPS_1Y>0:
            if caption:
                return '1Y'
            else:
                return ((x[est_name] / x.EPS_1Y) - 1) * 100
        elif not(pd.isna(x[y1_name])) and  x[y1_name] > 0:
            if caption:
                return '1Y_Est'
            else:
                return ((x[est_name] / x[y1_name]) - 1) * 100
        elif not(pd.isna(x.EPS_2Y)) and x.EPS_2Y>0:
            if caption:
                return '2Y'
            else:
                return ((x[est_name] / x.EPS_2Y) ** (1/2) - 1) * 100
        else:
            return np.nan
    else:
        return np.nan

def get_shares(x, shares):
    # get shares by Code and PeriodEndDate
    try:
        shares = shares[x.Code].loc[:x.PeriodEndDate].values[0]
        return shares
    except:
        return np.nan


setting = True

if __name__ == '__main__':

    calc_est = True
    calc_gdp_effect = True
    reuse_ucurve = True
    reuse_cache = False
    reuse_result = False

    #prddate as today
    prddate = dt.datetime.today().strftime('%Y-%m-%d')
    setting = 'US_Q'  # 'US_Q' 'US_Y' 'KR_Q' 'KR_Y'

    if setting == 'KR_Y':
        country = 'kr'
        period = 'Y' # '2024-07-18' '2024-08-19'
        prdFY = ['2024AS','2025AS']
        curveFY = ['2022AS', '2023AS'] + prdFY
        model_name = 'mixed_model'
    elif setting == 'US_Q':
        country = 'us'
        period = 'Q'
        prdFY = ['3Q25AS', '4Q25AS', '1Q26AS', '2Q26AS']
        curveFY = ['3Q22AS', '4Q22AS', '1Q23AS', '2Q23AS', '3Q23AS', '4Q23AS', '1Q24AS', '2Q24AS', '3Q24AS', '4Q24AS', '1Q25AS', '2Q25AS'] + prdFY
        model_name = 'mixed_model'

    if country == 'us':
        use_gdp = False
        gdp_path = None
        use_custom_gdp = False
        gdp_header = None
        gdp_lag = None
        rolling = None
        use_prd = False
        ts_length = 10
        sector_len = 2
        sector_groupby_len = 6
    elif country == 'kr':
        use_gdp = True
        gdp_path = f'data/{country}/QGDP.xlsx'
        use_custom_gdp = True
        gdp_header = 13
        gdp_lag = 2
        rolling = 4
        use_prd = True
        ts_length = 10
        sector_len = 3
        sector_groupby_len = 7

    print('build train data')

    if not reuse_cache:
        if os.path.exists(f'./cache/cache.parquet'):
        #    # remove cache
            os.remove(f'./cache/cache.parquet')

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

    sector_name = pd.read_csv(f'data/{country}/industry_map.csv', encoding='utf-8-sig', dtype=str).set_index('Code')
    shares = pd.read_excel(f'data/{country}/shares.xlsx', sheet_name='share', index_col=0) * 1000
    shares = shares[pd.to_numeric(shares.iloc[-1,:], errors='coerce').dropna().index]
    shares.columns = [code.split('(')[0] for code in shares.columns]
    comp_name = pd.read_csv(f'data/{country}/name.csv', encoding='utf-8-sig', dtype=str).set_index('instrument')

    new_train = train[train.A_EPS_1.abs() / train.BPS < 10]
    if country == 'us':
        # if previous year's error is less than 0.5%, remove the stock from the list
        # retain only guidance given stock
        if period == 'Y':
            new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)
            new_train = new_train[new_train.Guidance == 1]
    else:
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)

    UniqueSymbol_model = new_train[new_train.FY.isin(prdFY)].UniqueSymbol.unique()
    UniqueSymbol_EW = train[train.FY.isin(prdFY)][~train.UniqueSymbol.isin(UniqueSymbol_model)].UniqueSymbol.unique()

    # make some changes(sector classification, prediction date) to the train data
    train = train[train.Date <= prddate]

    model = Enhanced_EPS()
    model.set_data(train)
    model.calc_ucurve(country, curveFY, train, reuse=reuse_ucurve)

    if calc_est:
        # step1: save IMC and EW Enhanced EPS to csv
        if reuse_result:
            result = pd.read_parquet('totalresult.parquet', engine='pyarrow')
        else:
            if model_name == 'mixed_model':
                result = model.calc_mixed_model(UniqueSymbol_model
                                                , ['EW', 'EW_adp', 'PBest', 'PBest_adp', 'IMSE', 'IMSE_adp', 'BAM',
                                                   'BAM_adp', 'IMC', 'IMC_adp']
                                                , country)
            else:
                result = model.calc_model(UniqueSymbol_model, model_name)
            resultEW = model.calc_model(UniqueSymbol_EW, 'EW')

            result = pd.concat([result, resultEW]).reset_index()

            result.to_parquet('totalresult.parquet', engine='pyarrow', index=False)

        # 만약 실적이 발표되었으나 0 QBtw가 없다면, 0 QBtw 추가
        result_tmp = []
        for code in tqdm(result.Code.unique(), desc='Adding 0 QBtw'):
            if not(0 in result[result.Code == code].QBtw.unique()) and (result[result.Code == code].EPS_Actual.notnull().all()):
                tmprow = result[result.Code == code].copy().iloc[0:1]
                tmprow.EQBtw -= tmprow.QBtw
                tmprow.QBtw = 0
                result_tmp.append(tmprow)
        if len(result_tmp) > 0:
            result_tmp = pd.concat(result_tmp).groupby(['Code', 'FY', 'EQBtw']).last().reset_index()
            result = pd.concat([result, result_tmp])

        # 만약 Code별로 없는 EQBtw가 있다면, 이전 EQBtw값으로 EQBtw 추가
        result_tmp = []
        for fy, result_fy in tqdm(result.groupby('FY'), desc='Adding EQBtw by FY'):
            EQBtw_unique = np.sort(result_fy.EQBtw.unique())[::-1]
            for code, result_code in result_fy.groupby('Code'):
                result_code_EQBtw = result_code.EQBtw.unique()
                for EQBtw in EQBtw_unique:
                    if EQBtw not in result_code_EQBtw:
                        EQBtw_list = result_code[result_code.EQBtw > EQBtw].EQBtw.unique()
                        if EQBtw_list.size > 0:
                            tmprow = result_code[result_code.EQBtw == min(EQBtw_list)].copy()
                            tmprow.EQBtw = EQBtw
                            result_tmp.append(tmprow)
        if len(result_tmp) > 0:
            result_tmp = pd.concat(result_tmp).groupby(['Code', 'FY', 'EQBtw']).last().reset_index()
            result = pd.concat([result, result_tmp])

        # get bluffer
        result['G'] = result.apply(lambda x: 0 if x.model in ['EW','PBest','IMSE'] else 1, axis=1)

        result['Over'] = result['Est'] - result['EW']
        # 각 FY에 EQBtw별로 아웃라이어는 EW로 전환
        std = result.groupby(['FY', 'EQBtw'])['Over'].std()
        mean = result.groupby(['FY', 'EQBtw'])['Over'].mean()
        result['Over'] = result.apply(lambda x: 0 if (abs(x.Over - mean[x.FY][x.EQBtw]) < 5 * std[x.FY][x.EQBtw]) or x.G==0 else 1, axis=1)

        # change Over into EW
        result.loc[result.Code == 'US1266501006', 'Est'] = result.loc[result.Code == 'US1266501006', 'EW']
        result.loc[result.Code == 'US1266501006', 'model'] = 'EW'
        #result['Est'] = result.apply(lambda x: x.EW if x.Over == 1 else x.Est, axis=1)
        #result['model'] = result.apply(lambda x: 'EW' if x.Over == 1 else x.model, axis=1)

        # filter out outlier
        result = result[result.Est<10]

        # EPS_1Y가 없는 경우에는 지난년도 동분기 실적 전망치를 가져옴
        # 현분기 실적 전망치 제작
        result['EPS_Est'] = result.Est * result.BPS
        result['EPS_EW'] = result.EW * result.BPS
        result['EPS_Est_bld'] = result.apply(lambda x: x.EPS_Actual if (~pd.isna(x.EPS_Actual)) & (x.QBtw == 0) else x.Est * x.BPS, axis=1)
        result['EPS_EW_bld'] = result.apply(lambda x: x.EPS_Actual if (~pd.isna(x.EPS_Actual)) & (x.QBtw == 0) else x.EW * x.BPS, axis=1)
        # get min EQBtw's EPS estimation
        result['FY_prev'] = result.FY.apply(model.shift_period)
        result_minEQBtw = result.groupby(['Code','FY']).EQBtw.idxmin()
        result_minEQBtw = result[['Code','FY','EPS_Est_bld', 'EPS_EW_bld']].loc[result_minEQBtw.values]
        result_minEQBtw.columns = ['Code','FY_prev','EPS_1Y_Est', 'EPS_1Y_EW']
        # assign EPS_1Y_Est', 'EPS_1Y_EW to the result based on Code and FY_prev
        result = result.merge(result_minEQBtw, how='left', on=['Code','FY_prev'])
        result['EPS_1Y_Est'] = result.apply(lambda x: x['EPS_1Y_Est'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)
        result['EPS_1Y_EW'] = result.apply(lambda x: x['EPS_1Y_EW'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)

        result['EPS_G'] = result.apply(lambda x: eps_growth(x, est_name='EPS_Est', y1_name='EPS_1Y_EW'), axis=1)
        result['EPS_EW_G'] = result.apply(lambda x: eps_growth(x, est_name='EPS_EW', y1_name='EPS_1Y_EW'), axis=1)
        result['EPS_G_bld'] = result.apply(lambda x: eps_growth(x, est_name='EPS_Est_bld', y1_name='EPS_1Y_EW'), axis=1)
        result['EPS_EW_G_bld'] = result.apply(lambda x: eps_growth(x, est_name='EPS_EW_bld', y1_name='EPS_1Y_EW'), axis=1)
        result['EPS_G_caption'] = result.apply(lambda x: eps_growth(x, est_name='EPS_Est', y1_name='EPS_1Y_EW', caption=True), axis=1)

        # 이미 실적이 발표되었으나, QBtw가 0이 아닌 경우, 값을 NA로 변경
        result.loc[(result.EPS_Actual.notnull()) & (result.QBtw != 0), 'EPS_Actual'] = np.nan

        # Sector별로 median값을 구하기 위해 Sector를 sector_groupby_len만큼 자름
        result['Sector'] = result['Sector'].str[:sector_groupby_len]
        result['LSector'] = result['Sector'].str[:sector_len]

        # get data by Code and PeriodEndDate
        result['shares'] = result.apply(lambda x:get_shares(x, shares=shares), axis=1)
        result['name'] = result.apply(lambda x: comp_name.loc[x.Code] if x.Code in comp_name.index else np.nan, axis=1)
        result = result.sort_values(['Code','FY','EQBtw','QBtw'])
        # data 저장
        result.groupby(['Code','FY','EQBtw']).first().reset_index().sort_values(['Code','FY','EQBtw'], ascending=[True,True,False]).to_csv(f'./result/{country}/{model_name}_{period}_{prddate}.csv', encoding='utf-8-sig')

        result_sector = groupby_sector(result, sector_name, 'EQBtw')
        result_sector.index = result_sector.index.astype(int)
        result_sector.sort_values(['Sector','FY','EQBtw'], ascending=[True,True,False]).to_csv(f'./result/{country}/{model_name}_EQBtw_{period}_sector_{prddate}.csv', encoding='utf-8-sig')

        # 최근 QBtw에 대한 snapshot 형태 data 제작
        result_snapshot = result.reset_index().set_index(['Code', 'QBtw']).loc[result.groupby('Code').QBtw.min().reset_index().set_index(['Code','QBtw']).index]
        result_sector = groupby_sector(result_snapshot, sector_name, 'QBtw')
        result_sector.to_csv(f'./result/{country}/{model_name}_QBtw_{period}_{prddate}.csv', encoding='utf-8-sig')

        print('Step1 Finished')

    if calc_gdp_effect:

        # step2: save total term spread by gdp and time delta to csv
        total_ts_pd = build_gdp_scenario(model
                                         , gdp_data_path=gdp_path
                                         , use_prd=use_prd
                                         , use_gdp=use_gdp
                                         , gdp_lag=gdp_lag
                                         , rolling=rolling
                                         )
        total_ts_pd.to_csv(f'./result/{country}/total_ts.csv', encoding='utf-8-sig')

        cg_list = []
        if use_gdp:
            for prdyear in prdFY:
                ts = merged_ts(total_ts_pd, prdyear, prddate)
                if len(ts) > 0:
                    tmp_ts = ts.filter(regex='nan').copy()
                    cg_list.append(tmp_ts)
                    ts.to_csv(f'./result/{country}/ts_{prdyear}.csv', encoding='utf-8-sig')

        if use_custom_gdp:
            custom_gdp = pd.read_excel(f'data/{country}/QGDP.xlsx', sheet_name='prd_con', index_col=0)
            for cg in custom_gdp:
                total_ts_pd = build_gdp_scenario(model
                                                 , gdp_data_path=gdp_path
                                                 , use_prd=use_prd
                                                 , use_gdp=use_gdp
                                                 , gdp_lag=gdp_lag
                                                 , rolling=rolling
                                                 , custom_data=custom_gdp[[cg]]
                                                 )
                total_ts_pd.to_csv(f'./result/{country}/total_ts_{cg.replace(" ","")}.csv', encoding='utf-8-sig')

                if use_gdp:
                    for prdyear in prdFY:
                        ts = merged_ts(total_ts_pd, prdyear, prddate)
                        if len(ts) > 0:
                            tmp_ts = ts.filter(regex='nan').copy()
                            tmp_ts.columns = tmp_ts.columns + cg
                            cg_list.append(tmp_ts)
                            ts.to_csv(f'./result/{country}/ts_{prdyear}_{cg.replace(" ","")}.csv', encoding='utf-8-sig')

            pd.concat(cg_list,axis=1).to_csv(f'./result/{country}/total_ts_gdp.csv', encoding='utf-8-sig')

        print('Step2 Finished')

        # step3: save ucurve coefficients to csv
        ucurve_df = pd.DataFrame.from_dict(model.ucurve, orient='index')
        ucurve_df = ucurve_df.explode(['popt_bf', 'popt_af'])
        ucurve_df['coeff'] = ['b0', 'c', 'b1', 'b2', 'lam'] * (len(ucurve_df) // 5)
        ucurve_df = ucurve_df.set_index([ucurve_df.index, 'coeff'])
        ucurve_df.unstack(level=1).to_csv(f'./result/{country}/coeff.csv', encoding='utf-8-sig')
        print('Step3 Finished')