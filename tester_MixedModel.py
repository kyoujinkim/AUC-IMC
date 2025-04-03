# test mixed model's average MAFE
# import all test result
# for each quarter, select the best model. Best model is the model with the lowest MAFE at the last year's same quarter
from glob import glob

import pandas as pd
from tqdm import tqdm

dataset_list = glob('./result/us/*.csv')
# except MSFE is in text or - in text
dataset_list = [dataset for dataset in dataset_list if '_MSFE' not in dataset and '-' not in dataset and 'coeff' not in dataset and 'total' not in dataset and '_a.' not in dataset  and '_ag' not in dataset]

col = 'MAFE'
temp_result = []
for result in dataset_list:
    method = result.split('\\')[-1].split('.')[0]
    temp_pd = pd.read_csv(result, index_col=0)
    temp_pd['method'] = method
    temp_result.append(temp_pd)

data = pd.concat(temp_result, axis=0)
data = data[data.QBtw < 5]
data_group = data.groupby(['Code','method','FY']).MAFE.mean().reset_index()
data_group_best = data_group.groupby(['Code', 'FY']).MAFE.idxmin().dropna()
data_group_best = pd.DataFrame({'method':data_group.method[data_group_best].values}, index=data_group_best.index)

uniquecode = (data.Code + data.FY).drop_duplicates()

total_result = []
for uc in tqdm(uniquecode):
    fy = uc[-6:]
    code = uc[:-6]
    fy_former = fy[:2] + f'{int(fy[2:4]) - 1:02d}' + fy[4:]

    try:
        method = data_group_best.loc[code,fy_former].method
    except:
        method = 'EW'

    total_result.append([code, fy, method])


dt = pd.read_parquet('./cache/cache.parquet', engine='pyarrow')
dt_tmp = dt.reset_index().drop_duplicates(subset='Code')

data = data.set_index('Code')
data['Sector'] = dt_tmp.set_index('Code').Sector
data['BSec'] = data.Sector.str[:6]

dataew = data.reset_index()
dataew = dataew[dataew['method'] == 'EW']
ews = dataew.groupby(['QBtw']).MAFE.mean().reset_index()
ewrs = dataew.groupby(['BSec', 'QBtw']).MAFE.mean().reset_index()
ewrs = pd.pivot(ewrs, columns='BSec', index='QBtw', values='MAFE')

data = data.reset_index().set_index(['Code','FY','method'])

mmbest = data.loc[total_result]

mmbest = mmbest.dropna(subset='MAFE').reset_index()
mmbest['FQ'] = mmbest.FY.str[:2]

s = mmbest.groupby(['QBtw']).MAFE.mean().reset_index()

rs = mmbest.groupby(['method', 'BSec']).MAFE.count().reset_index()
rs = pd.pivot(rs, columns='BSec', index='method', values='MAFE')
rsc = rs / rs.sum(axis=0)

rs = mmbest.groupby(['QBtw', 'BSec']).MAFE.mean().reset_index()
rs = pd.pivot(rs, columns='BSec', index='QBtw', values='MAFE')

c = 1
