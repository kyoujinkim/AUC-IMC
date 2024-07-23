from glob import glob

import pandas as pd

result_list = glob('./result/*.csv')
result_list_exMSFE = [result for result in result_list if 'MSFE' not in result and 'cache' not in result]

temp_result = []
for result in result_list_exMSFE:
    temp_pd = pd.read_csv(result, index_col=0)
    temp_pd['MAFE'] = temp_pd.Error.abs()
    group_temp = temp_pd.groupby(['QBtw'])[['Error', 'MAFE', 'Std']].mean()
    group_temp['Class'] = result.split('\\')[-1].split('.')[0]
    temp_result.append(group_temp)

result = pd.concat(temp_result, axis=0)
result.Class = result.Class.str.replace('_adj', 'adj')
result.Class = result.Class.str.replace('_slope', 'slope')
result['SubClass'] = result.Class.str.split('_', expand=True, n=1)[1]
result['Class'] = result.Class.str.split('_', expand=True, n=1)[0]
result = result.reset_index()

result_pv = result.pivot(index=['Class', 'QBtw'], columns=['SubClass'], values=['Error','MAFE' , 'Std'])
result_pv.to_csv('./result/result_MSFE/result_pv.csv', encoding='utf-8-sig')
print('done')