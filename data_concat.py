#concat consenlist csv
import pandas as pd

df1 = pd.read_csv('consenlist.csv')
df2 = pd.read_csv('consenlist2.csv')

df = pd.concat([df1, df2], ignore_index=True, axis=0).dropna()
df['Year'] = df.FY.str.extract(r'(\d{4})')
df['CutDate'] = (df.Year.astype(int) - 1).astype(str) + '-03-31'
df = df[df.Date > df.CutDate]

df_train = df[df.FY < '2015AS']
df_test = df[df.FY >= '2015AS']

df_train.to_csv('./data/train.csv', index=False, encoding='utf-8-sig')
df_test.to_csv('./data/test.csv', index=False, encoding='utf-8-sig')
