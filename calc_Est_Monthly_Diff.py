import pandas as pd

prdDates = ['2024-07-18', '2024-08-19']

prdyear = 2025
beforeEst = pd.read_csv(f'./result/IMC_adp{prdyear}_{prdDates[0]}.csv')
afterEst = pd.read_csv(f'./result/IMC_adp{prdyear}_{prdDates[1]}.csv')
currQ = beforeEst.QBtw.min()
beforeEst = beforeEst[beforeEst.QBtw == currQ]
afterEst = afterEst[afterEst.QBtw == currQ]
beforeEst = beforeEst.set_index('Code')
afterEst = afterEst.set_index('Code')
afterEst['diffEst'] = afterEst['Est'] - beforeEst['Est']
afterEst['diffEWEst'] = afterEst['EW'] - beforeEst['EW']
afterEst['UCallev'] = afterEst['diffEst'] - afterEst['diffEWEst']
afterEst = afterEst.sort_values(by='diffEst', ascending=False)
afterEst.to_csv(f'./result/IMC_adp{prdyear}_diff.csv')

prdyear = 2024
beforeEst = pd.read_csv(f'./result/EW_adp{prdyear}_{prdDates[0]}.csv')
afterEst = pd.read_csv(f'./result/EW_adp{prdyear}_{prdDates[1]}.csv')
currQ = beforeEst.QBtw.min()
beforeEst = beforeEst[beforeEst.QBtw == currQ]
afterEst = afterEst[afterEst.QBtw == currQ]
beforeEst = beforeEst.set_index('Code')
afterEst = afterEst.set_index('Code')
afterEst['diffEst'] = afterEst['Est'] - beforeEst['Est']
afterEst['diffEWEst'] = afterEst['EW'] - beforeEst['EW']
afterEst['UCallev'] = afterEst['diffEst'] - afterEst['diffEWEst']
afterEst = afterEst.sort_values(by='diffEst', ascending=False)
afterEst.to_csv(f'./result/EW_adp{prdyear}_diff.csv')
c = 1