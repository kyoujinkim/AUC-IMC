import numpy as np
import pandas as pd
import sys

abs_path = 'D:/Factor_DB/us'

mktval = pd.read_parquet(f'./amktcap.parquet', engine='pyarrow')
listequ = pd.read_parquet(f'D:\Factor_DB\members/master_RAY.parquet', engine='pyarrow')

list_total = []

date_tmp = pd.DataFrame({'date':listequ.index.values, 'year':listequ.index.year.values}).drop_duplicates(subset='year',keep='last')
for date in date_tmp.date:
    temp_row = listequ.loc[date]
    temp_row = temp_row[temp_row==1]
    if len(mktval.index[mktval.index <= date])==0:
        continue
    nearest_index = mktval.index[mktval.index <= date][-1]
    match_code = list(set(mktval.loc[nearest_index].index) & set(temp_row.index))
    mktval_filtered = mktval.loc[nearest_index,match_code].sort_values(ascending=False).iloc[:500]

    list_row = pd.DataFrame({'Code': mktval_filtered.index,
                             'Date': [date.strftime('%Y-%m-%d')]*len(mktval_filtered)})

    list_total.append(list_row)

# append fy1 and fy2 data
last_index = list_total[-1]
for delta in [1]:
    list_row = pd.DataFrame({'Code': last_index.Code,
                             'Date': (pd.to_datetime(last_index.Date)+pd.DateOffset(years=delta)).dt.strftime('%Y-%m-%d')})

    list_total.append(list_row)

list_total.append(list_row)

list_total_df = pd.concat(list_total, ignore_index=True, axis=0).drop_duplicates()

list_total_df.to_csv('./data/us/list_total.csv', index=False, encoding='utf-8-sig')

c = 1