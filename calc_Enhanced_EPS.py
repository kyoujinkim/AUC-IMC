import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import *
from src.enh_EPS import Enhanced_EPS
import warnings
warnings.filterwarnings('ignore')


def groupby_sector(df, df_sector, col_name):
    '''
    Sector별로 groupby하여 median값을 구함
    :param df: 데이터를 포함한 DataFrame
    :param df_sector: Sector name을 지닌 DataFrame
    :param col_name: df에서 groupby할 column name
    :return:
    '''

    result_sector = df.groupby(['Sector', col_name])[['Est', 'EW', 'EW_prev', 'Shock', 'GEst', 'EPS_G', 'EPS_EW_G']].median()
    result_sector_count = df.groupby(['Sector', col_name])[['Est', 'EPS_Actual']].count()
    result_sector_gcount = df.groupby(['Sector', col_name])[['G']].sum()
    result_sector['count'] = result_sector_count['Est']
    result_sector['gcount'] = result_sector_gcount['G']
    result_sector['acount'] = result_sector_count['EPS_Actual']

    # Lsector별로 median값을 구함
    result_Lsector = df.groupby(['LSector', col_name])[['Est', 'EW', 'EW_prev', 'Shock', 'GEst', 'EPS_G', 'EPS_EW_G']].median()
    result_Lsector_count = df.groupby(['LSector', col_name])[['Est', 'EPS_Actual']].count()
    result_Lsector_gcount = df.groupby(['LSector', col_name])[['G']].sum()
    result_Lsector['count'] = result_Lsector_count['Est']
    result_Lsector['gcount'] = result_Lsector_gcount['G']
    result_Lsector['acount'] = result_Lsector_count['EPS_Actual']

    # Sector별로 median값을 구한 결과를 result_sector에 추가
    result_sector = pd.concat([result_sector, result_Lsector], axis=0)
    result_sector = result_sector.reset_index()
    result_sector = result_sector.set_index(result_sector.columns[0])

    # Sector별로 median값을 구한 결과에 Sector_name을 추가
    result_sector['Sector_name'] = df_sector.Sector

    return result_sector


if __name__ == '__main__':

    country = 'us'
    prddate = '2024-12-06'  # '2024-07-18' '2024-08-19'
    if country == 'us':
        use_gdp = False
        gdp_path = None
        gdp_header = None
        gdp_lag = None
        rolling = None
        use_prd = False
        ts_length = 10
        sector_len = 2
        model_list = ['IMSE_adp']
        sector_groupby_len = 4
    elif country == 'kr':
        use_gdp = True
        gdp_path = f'data/{country}/QGDP.xlsx'
        gdp_header = 13
        gdp_lag = 2
        rolling = 4
        use_prd = True
        ts_length = 10
        sector_len = 3
        model_list = ['IMC_adp', 'EW']
        sector_groupby_len = 5


    print('build train data')
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

    if country == 'us':
        train = train[train.A_EPS_1.abs() / train.BPS < 1]
        # if previous year's error is less than 0.5%, remove the stock from the list
        new_train = filter_guided_stock(train, 'Code', 'Error', 0.001)
        # retain only guidance given stock
        new_train = new_train[new_train.Guidance == 1]
        UniqueSymbol_total = new_train.UniqueSymbol.unique()
    else:
        UniqueSymbol_total = train.UniqueSymbol.unique()

    # make some changes(sector classification, prediction date) to the train data
    train = train[train.Date <= prddate]

    print('save train data to db')
    save_db(train)

    prdYears = [2024, 2025]

    model = Enhanced_EPS()
    model.calc_ucurve(country, prdYears, train)

    # step1: save IMC and EW Enhanced EPS to csv
    for prdyear in prdYears:
        UniqueSymbol = train[(train.UniqueSymbol.isin(UniqueSymbol_total)) & (train.Year==str(prdyear))].UniqueSymbol.unique()
        UniqueSymbol_sub = train[(~train.UniqueSymbol.isin(UniqueSymbol_total)) & (train.Year == str(prdyear))].UniqueSymbol.unique()

        for model_name in model_list:
            result = model.calc(UniqueSymbol, model_name)
            result['G'] = 1
            if country == 'us':
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

            # Sector별로 median값을 구함
            result_sector = groupby_sector(result, sector_name, 'EQBtw')
            # Sector별로 median값을 구한 결과를 csv로 저장
            result_sector.to_csv(f'./result/{country}/{model_name}_EQBtw_{prdyear}_sector_{prddate}.csv', encoding='utf-8-sig')

            # 최근 QBtw에 대한 snapshot 형태 data 제작
            result_snapshot = result.reset_index().set_index(['Code', 'QBtw']).loc[result.groupby('Code').QBtw.min().reset_index().set_index(['Code','QBtw']).index]
            # Sector별로 median값을 구함
            result_sector = groupby_sector(result_snapshot, sector_name, 'QBtw')
            # result를 최근 'EQBtw'에 대하여 snapshot 형태로 groupby하여 저장
            result_sector.to_csv(f'./result/{country}/{model_name}_QBtw_{prdyear}_{prddate}.csv', encoding='utf-8-sig')

    print('Step1 Finished')

    # step2: save total term spread by gdp and time delta to csv
    total_ts_pd = build_gdp_scenario(model, f'./data/{country}/QGDP.xlsx', use_prd=use_prd, use_gdp=use_gdp)
    total_ts_pd.to_csv(f'./result/{country}/total_ts.csv', encoding='utf-8-sig')

    if use_gdp:
        for prdyear in prdYears:
            ts = merged_ts(total_ts_pd, prdyear, prddate)
            ts.to_csv(f'./result/{country}/ts_{prdyear}.csv', encoding='utf-8-sig')
    print('Step2 Finished')

    # step3: save ucurve coefficients to csv
    ucurve_df = pd.DataFrame.from_dict(model.ucurve, orient='index')
    ucurve_df = ucurve_df.explode(['popt_bf', 'popt_af'])
    ucurve_df['coeff'] = ['b0', 'c', 'b1', 'b2', 'lam'] * (len(ucurve_df) // 5)
    ucurve_df = ucurve_df.set_index([ucurve_df.index, 'coeff'])
    ucurve_df.unstack(level=2).to_csv(f'./result/{country}/coeff.csv', encoding='utf-8-sig')
    print('Step3 Finished')