import itertools

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import *
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


def groupby_sector(df, df_sector, col_name, extend, filename):
    '''
    Sector별로 groupby하여 median값을 구함
    :param df: 데이터를 포함한 DataFrame
    :param df_sector: Sector name을 지닌 DataFrame
    :param col_name: df에서 groupby할 column name
    :param extend: True if the data is for the next year, False if the data is for the current year
    :param filename: 저장할 파일 이름
    :return:
    '''

    if extend:
        df_prevyear = pd.read_csv(filename, encoding='utf-8-sig', index_col=0)
        df_prevyear = df_prevyear[df_prevyear.EQBtw == df_prevyear.EQBtw.min()]
        df['EPS_1YE'] = df_prevyear['EPS_Est']
        df['EPS_1YEW'] = df_prevyear['EPS_EW']
        df['EPS_1Y'] = df.apply(lambda x: x['EPS_1YE'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)
        df['EPS_1YEW'] = df.apply(lambda x: x['EPS_1YEW'] if pd.isna(x['EPS_1YEW']) else x['EPS_1YEW'], axis=1)
    else:
        df['EPS_1YEW'] = df['EPS_1Y']
    # 섹터 단위 계산을 위해 net earning 계산
    df['earning_Est'] = df['shares'] * df['EPS_Est']
    df['earning_EW'] = df['shares'] * df['EPS_EW']
    df['earning_1Y'] = df['shares'] * df.apply(lambda x: x['EPS_2Y'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)
    df['earning_1YEW'] = df['shares'] * df.apply(lambda x: x['EPS_2Y'] if pd.isna(x['EPS_1YEW']) else x['EPS_1YEW'], axis=1)
    df['earning_1Y_caption'] = df.apply(lambda x: -1 if pd.isna(x['EPS_1Y']) else 1, axis=1)
    df['earning_1YEW_caption'] = df.apply(lambda x: -1 if pd.isna(x['EPS_1YEW']) else 1, axis=1)

    result_all = df.groupby(col_name)[['Est', 'EW', 'EW_prev', 'Shock', 'GEst', 'EPS_G', 'EPS_EW_G']].median()
    result_total_all = df.groupby(col_name)[['earning_Est', 'earning_EW', 'earning_1Y', 'earning_1Y_caption', 'earning_1YEW', 'earning_1YEW_caption']].sum()
    # if earning_1Y_catpion smaller than 0, **1/2 to growth rate
    result_all['earning_G'] = result_total_all.apply(lambda x: get_sector_growth_rate(x, 'earning_Est', 'earning_1Y', 'earning_1Y_caption'), axis=1)
    result_all['earning_EW_G'] = result_total_all.apply(lambda x: get_sector_growth_rate(x, 'earning_EW', 'earning_1YEW', 'earning_1YEW_caption'), axis=1)
    result_all_count = df.groupby(col_name)[['Est', 'EPS_Actual']].count()
    result_all_gcount = df.groupby(col_name)[['G']].sum()
    result_all['count'] = result_all_count['Est']
    result_all['gcount'] = result_all_gcount['G']
    result_all['acount'] = result_all_count['EPS_Actual']
    # make index as (All, 0)
    result_all = result_all.reset_index()
    result_all['Sector_name'] = 'All'
    result_all = result_all.set_index(['Sector_name', col_name])

    result_sector = df.groupby(['Sector', col_name])[['Est', 'EW', 'EW_prev', 'Shock', 'GEst', 'EPS_G', 'EPS_EW_G']].median()
    result_total_sector = df.groupby(['Sector', col_name])[['earning_Est', 'earning_EW', 'earning_1Y', 'earning_1Y_caption', 'earning_1YEW', 'earning_1YEW_caption']].sum()
    # if earning_1Y_catpion smaller than 0, **1/2 to growth rate
    result_sector['earning_G'] = result_total_sector.apply(lambda x: get_sector_growth_rate(x, 'earning_Est', 'earning_1Y', 'earning_1Y_caption'), axis=1)
    result_sector['earning_EW_G'] = result_total_sector.apply(lambda x: get_sector_growth_rate(x, 'earning_EW', 'earning_1YEW', 'earning_1YEW_caption'), axis=1)
    result_sector_count = df.groupby(['Sector', col_name])[['Est', 'EPS_Actual']].count()
    result_sector_gcount = df.groupby(['Sector', col_name])[['G']].sum()
    result_sector['count'] = result_sector_count['Est']
    result_sector['gcount'] = result_sector_gcount['G']
    result_sector['acount'] = result_sector_count['EPS_Actual']

    # Lsector별로 median값을 구함
    result_Lsector = df.groupby(['LSector', col_name])[['Est', 'EW', 'EW_prev', 'Shock', 'GEst', 'EPS_G', 'EPS_EW_G']].median()
    result_total_Lsector = df.groupby(['LSector', col_name])[['earning_Est', 'earning_EW', 'earning_1Y', 'earning_1Y_caption', 'earning_1YEW', 'earning_1YEW_caption']].sum()
    result_Lsector['earning_G'] = result_total_Lsector.apply(lambda x: get_sector_growth_rate(x, 'earning_Est', 'earning_1Y', 'earning_1Y_caption'), axis=1)
    result_Lsector['earning_EW_G'] = result_total_Lsector.apply(lambda x: get_sector_growth_rate(x, 'earning_EW', 'earning_1YEW', 'earning_1YEW_caption'), axis=1)
    result_Lsector_count = df.groupby(['LSector', col_name])[['Est', 'EPS_Actual']].count()
    result_Lsector_gcount = df.groupby(['LSector', col_name])[['G']].sum()
    result_Lsector['count'] = result_Lsector_count['Est']
    result_Lsector['gcount'] = result_Lsector_gcount['G']
    result_Lsector['acount'] = result_Lsector_count['EPS_Actual']

    # Sector별로 median값을 구한 결과를 result_sector에 추가
    result_sector = pd.concat([result_all, result_sector, result_Lsector], axis=0)
    result_sector = result_sector.reset_index()
    result_sector = result_sector.set_index(result_sector.columns[0])

    # Sector별로 median값을 구한 결과에 Sector_name을 추가
    result_sector['Sector_name'] = df_sector.Sector

    return result_sector


if __name__ == '__main__':

    country = 'kr'
    prddate = '2025-01-15'  # '2024-07-18' '2024-08-19'

    prdYears = [2024,2025]
    extend = [False,True]

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
        model_list = ['IMSE_adp']
        sector_groupby_len = 6
    elif country == 'kr':
        use_gdp = True
        gdp_path = f'data/{country}/QGDP.xlsx'
        use_custom_gdp = False
        gdp_header = 13
        gdp_lag = 2
        rolling = 4
        use_prd = True
        ts_length = 10
        sector_len = 3
        model_list = ['IMC_adp']
        sector_groupby_len = 7
    elif country == 'kr2':
        use_gdp = True
        gdp_path = f'data/{country}/FX.xlsx'
        use_custom_gdp = False
        gdp_header = 13
        gdp_lag = 0
        rolling = 1
        use_prd = True
        ts_length = 10
        sector_len = 3
        model_list = ['IMC_adp']
        sector_groupby_len = 5


    print('build train data')
    if os.path.exists(f'./cache/cache.csv'):
        # remove cache
        os.remove(f'./cache/cache.csv')

    train = build_data(f'data/{country}/consenlist/*.csv'
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
    shares = pd.read_excel(f'data/{country}/shares.xlsx', sheet_name='share', index_col=0)
    shares = shares.loc[pd.to_numeric(shares.iloc[:,0], errors='coerce').dropna().index]

    if country == 'us':
        new_train = train[train.A_EPS_1.abs() / train.BPS < 3]
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)
        # retain only guidance given stock
        new_train = new_train[new_train.Guidance == 1]
        UniqueSymbol_total = new_train.UniqueSymbol.unique()
    else:
        new_train = train[train.A_EPS_1.abs() / train.BPS < 3]
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(new_train, 'Code', 'Error', 0.001)
        UniqueSymbol_total = new_train.UniqueSymbol.unique()

    # make some changes(sector classification, prediction date) to the train data
    train = train[train.Date <= prddate]

    print('save train data to db')
    save_db(train)

    model = Enhanced_EPS()
    model.calc_ucurve(country, prdYears, train)

    # concat prdYears with extend[False, True]
    prdYears_list = [[prdYears[i], extend[i]] for i in range(len(prdYears))]

    # step1: save IMC and EW Enhanced EPS to csv
    for prdyear, extend in prdYears_list:
        UniqueSymbol = train[(train.UniqueSymbol.isin(UniqueSymbol_total)) & (train.Year==str(prdyear))].UniqueSymbol.unique()
        UniqueSymbol_sub = train[(~train.UniqueSymbol.isin(UniqueSymbol_total)) & (train.Year == str(prdyear))].UniqueSymbol.unique()

        for model_name in model_list:
            result = model.calc(UniqueSymbol, model_name)
            result['G'] = 1
            if len(UniqueSymbol_sub) > 0:
                result_sub = model.calc(UniqueSymbol_sub, 'EW')
                result = pd.concat([result, result_sub])

            # 만약 실적이 발표되었으나 0 QBtw가 없다면, 0 QBtw 추가
            result_tmp = []
            for code in result.index.unique():
                if not(0 in result[result.index == code].QBtw.unique()) and (result[result.index == code].EPS_Actual.notnull().all()):
                    tmprow = result[result.index == code].copy().iloc[0:1]
                    tmprow.EQBtw -= tmprow.QBtw
                    tmprow.QBtw = 0
                    result_tmp.append(tmprow)
            if len(result_tmp) > 0:
                result_tmp = pd.concat(result_tmp).reset_index().groupby(['Code', 'EQBtw']).last().reset_index().set_index('Code')
                result = pd.concat([result, result_tmp])

            # 만약 Code별로 없는 EQBtw가 있다면, 이전 EQBtw값으로 EQBtw 추가
            result_tmp = []
            for code in result.index.unique():
                for EQBtw in result.EQBtw.unique()[::-1]:
                    EQBtw_list = result[(result.index == code) & (result.EQBtw > EQBtw)].EQBtw.unique()
                    lastEQBtw = min(EQBtw_list, default=EQBtw)
                    if not(EQBtw in result[result.index == code].EQBtw.unique()) and (EQBtw < lastEQBtw):
                        tmprow = result[(result.index == code) & (result.EQBtw == lastEQBtw)].copy()
                        tmprow.EQBtw = EQBtw
                        result_tmp.append(tmprow)
            if len(result_tmp) > 0:
                result_tmp = pd.concat(result_tmp).reset_index().groupby(['Code', 'EQBtw']).last().reset_index().set_index('Code')
                result = pd.concat([result, result_tmp])

            result = result_formatter_calc_growth(result)

            result = model.calc_quantile(result, model_name)

            # 이미 실적이 발표되었고, QBtw가 0인 경우, G를 0으로 변경
            result.loc[(result.EPS_Actual.notnull()) & (result.QBtw == 0), 'G'] = 0
            # 그 외의 경우에는 EPS_Actual을 NA로 변경
            result.loc[(result.QBtw != 0), 'EPS_Actual'] = np.nan

            # Sector별로 median값을 구하기 위해 Sector를 sector_groupby_len만큼 자름
            result['Sector'] = result['Sector'].str[:sector_groupby_len]
            result['LSector'] = result['Sector'].str[:sector_groupby_len-2]

            # data 저장
            result.to_csv(f'./result/{country}/{model_name}_{prdyear}_{prddate}.csv', encoding='utf-8-sig')

            result['shares'] = shares

            # Sector별로 median값을 구함
            result_sector = groupby_sector(result, sector_name, 'EQBtw', extend, f'./result/{country}/{model_name}_{prdyear-1}_{prddate}.csv')
            # Sector별로 median값을 구한 결과를 csv로 저장
            result_sector.to_csv(f'./result/{country}/{model_name}_EQBtw_{prdyear}_sector_{prddate}.csv', encoding='utf-8-sig')

            # 최근 QBtw에 대한 snapshot 형태 data 제작
            result_snapshot = result.reset_index().set_index(['Code', 'QBtw']).loc[result.groupby('Code').QBtw.min().reset_index().set_index(['Code','QBtw']).index]
            # Sector별로 median값을 구함
            result_sector = groupby_sector(result_snapshot, sector_name, 'QBtw', extend, f'./result/{country}/{model_name}_{prdyear-1}_{prddate}.csv')
            # result를 최근 'EQBtw'에 대하여 snapshot 형태로 groupby하여 저장
            result_sector.to_csv(f'./result/{country}/{model_name}_QBtw_{prdyear}_{prddate}.csv', encoding='utf-8-sig')

    print('Step1 Finished')

    # step2: save total term spread by gdp and time delta to csv
    total_ts_pd = build_gdp_scenario(model
                                     , gdp_data_path=gdp_path
                                     , use_prd=use_prd
                                     , use_gdp=use_gdp
                                     , gdp_lag=gdp_lag
                                     , rolling=rolling
                                     )
    total_ts_pd.to_csv(f'./result/{country}/total_ts.csv', encoding='utf-8-sig')

    if use_gdp:
        for prdyear in prdYears:
            ts = merged_ts(total_ts_pd, prdyear, prddate)
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
                for prdyear in prdYears:
                    ts = merged_ts(total_ts_pd, prdyear, prddate)
                    ts.to_csv(f'./result/{country}/ts_{prdyear}_{cg.replace(" ","")}.csv', encoding='utf-8-sig')

    print('Step2 Finished')

    # step3: save ucurve coefficients to csv
    ucurve_df = pd.DataFrame.from_dict(model.ucurve, orient='index')
    ucurve_df = ucurve_df.explode(['popt_bf', 'popt_af'])
    ucurve_df['coeff'] = ['b0', 'c', 'b1', 'b2', 'lam'] * (len(ucurve_df) // 5)
    ucurve_df = ucurve_df.set_index([ucurve_df.index, 'coeff'])
    ucurve_df.unstack(level=2).to_csv(f'./result/{country}/coeff.csv', encoding='utf-8-sig')
    print('Step3 Finished')