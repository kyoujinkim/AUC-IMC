import numpy as np
import pandas as pd
from tqdm import tqdm
from calc_Enhanced_EPS import get_shares, eps_growth, get_sector_growth_rate

def groupby_func(df, cols):
    result = df.groupby(cols)[['Est', 'EW', 'EW_prev', 'EPS_G', 'EPS_EW_G']].median()
    result_total = df.groupby(cols)[['earning_Est', 'earning_EW', 'earning_Act', 'earning_1Y', 'earning_1Y_caption', 'earning_1Y_EW',
                                     'earning_1Y_EW_caption']].sum()
    # if earning_1Y_catpion smaller than 0, **1/2 to growth rate
    result['earning_G'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_Est', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result['earning_EW_G'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_EW', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
    result['earning_Act_G'] = result_total.apply(
        lambda x: get_sector_growth_rate(x, 'earning_Act', 'earning_1Y_EW', 'earning_1Y_EW_caption'), axis=1)
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
    df['earning_Act'] = df['shares'] * df.apply(lambda x: x['EPS_EW'] if pd.isna(x['EPS_Actual']) else x['EPS_Actual'], axis=1)
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

def shift_period(period, lag=1):
    if 'Q' in period:
        fy = int(period[-4:-2])
        q = period[:2]
        return f'{q}{fy - lag}AS'
    else:
        fy = int(period[:4])
        return f'{fy - lag}AS'

result = pd.read_parquet('./totalresult.parquet', engine='pyarrow')
result = result[result.FY == '1Q25AS']
sector_name = pd.read_csv(f'data/us/industry_map.csv', encoding='utf-8-sig', dtype=str).set_index('Code')
shares = pd.read_excel(f'data/us/shares.xlsx', sheet_name='share', index_col=0) * 1000
shares = shares[pd.to_numeric(shares.iloc[-1, :], errors='coerce').dropna().index]
shares.columns = [code.split('(')[0] for code in shares.columns]

# 만약 실적이 발표되었으나 0 QBtw가 없다면, 0 QBtw 추가
result_tmp = []
for code in tqdm(result.Code.unique(), desc='Adding 0 QBtw'):
    if not (0 in result[result.Code == code].QBtw.unique()) and (
    result[result.Code == code].EPS_Actual.notnull().all()):
        tmprow = result[result.Code == code].copy().iloc[0:1]
        tmprow.EQBtw -= tmprow.QBtw
        tmprow.QBtw = 0
        result_tmp.append(tmprow)
if len(result_tmp) > 0:
    result_tmp = pd.concat(result_tmp).groupby(['Code', 'FY', 'EQBtw']).last().reset_index()
    result = pd.concat([result, result_tmp])

# get bluffer
result['G'] = result.apply(lambda x: 0 if x.model in ['EW', 'PBest', 'IMSE'] else 1, axis=1)

result['Over'] = result['Est'] - result['EW']
# 각 FY에 EQBtw별로 아웃라이어는 EW로 전환
std = result.groupby(['FY', 'EQBtw'])['Over'].std()
mean = result.groupby(['FY', 'EQBtw'])['Over'].mean()
result['Over'] = result.apply(
    lambda x: 0 if (abs(x.Over - mean[x.FY][x.EQBtw]) < 5 * std[x.FY][x.EQBtw]) or x.G == 0 else 1, axis=1)

# change Over into EW
result['Est'] = result.apply(lambda x: x.EW if x.Over == 1 else x.Est, axis=1)
result['model'] = result.apply(lambda x: 'EW' if x.Over == 1 else x.model, axis=1)

# filter out outlier
result = result[result.Est < 10]

# EPS_1Y가 없는 경우에는 지난년도 동분기 실적 전망치를 가져옴
# 현분기 실적 전망치 제작
result['EPS_Est'] = result.Est * result.BPS
result['EPS_EW'] = result.EW * result.BPS
# get min EQBtw's EPS estimation
result['FY_prev'] = result.FY.apply(shift_period)
# assgin EPS_1Y_Est', 'EPS_1Y_EW to the result based on Code and FY_prev
result['EPS_1Y_Est'] = result.apply(lambda x: x['EPS_1Y_EW'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)
result['EPS_1Y_EW'] = result.apply(lambda x: x['EPS_1Y_EW'] if pd.isna(x['EPS_1Y']) else x['EPS_1Y'], axis=1)

result['EPS_G'] = result.apply(lambda x: eps_growth(x, est_name='EPS_Est', y1_name='EPS_1Y_EW'), axis=1)
result['EPS_EW_G'] = result.apply(lambda x: eps_growth(x, est_name='EPS_EW', y1_name='EPS_1Y_EW'), axis=1)
result['EPS_G_caption'] = result.apply(lambda x: eps_growth(x, est_name='EPS_Est', y1_name='EPS_1Y_EW', caption=True), axis=1)

# QBtw가 현 분기인 경우만 필터링
result = result[result.QBtw == 0]

# Sector별로 median값을 구하기 위해 Sector를 sector_groupby_len만큼 자름
result['Sector'] = result['Sector'].str[:6]
result['LSector'] = result['Sector'].str[:2]

# get data by Code and PeriodEndDate
result['shares'] = result.apply(lambda x: get_shares(x, shares=shares), axis=1)
result = result.sort_values(['Code', 'FY', 'EQBtw', 'QBtw'])

result.to_csv(f'./result/us/mixed_model_Q_Surprise.csv', encoding='utf-8-sig', index=False)

# 최근 QBtw에 대한 snapshot 형태 data 제작
result_snapshot = result.reset_index().set_index(['Code', 'QBtw']).loc[
    result.groupby('Code').QBtw.min().reset_index().set_index(['Code', 'QBtw']).index]
result_sector = groupby_sector(result_snapshot, sector_name, 'QBtw')
result_sector.to_csv(f'./result/us/mixed_model_QBtw_Q_Surprise.csv', encoding='utf-8-sig')

print('Step1 Finished')