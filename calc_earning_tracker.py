from configparser import ConfigParser
from glob import glob
import datetime as dt

import pandas as pd

def get_earning_prev(sector_code, curr_FY, curr_EQBtw):
    recent_q_sector = recent_q[(recent_q['Sector'] == sector_code) &
                               (recent_q['FY'] == curr_FY) &
                               (recent_q['EQBtw'] == curr_EQBtw + 1)]

    if len(recent_q_sector) > 0:
        return recent_q_sector['earning_G_bld'].iloc[-1]
    else:
        return None

def sub_with_string(col1, col2):
    try:
        return float(col1) - float(col2)
    except:
        return None

def add_quarter(curr_fq, adder):
    while adder > 0:
        q = curr_fq[:1]
        y = curr_fq[2:4]
        if q == '4' :
            curr_fq = f"1Q{int(y)+1:02d}AS"
        else:
            curr_fq = f"{int(q)+1}Q{y}AS"
        adder -= 1
    return curr_fq

import refinitiv.data as rd

def open_session(app_key: str):
    # Open Refinitiv Session with loaded app key
    # Refinitiv Workspace/Eikon Desktop application must be on session
    session = rd.session.desktop.Definition(app_key=app_key).get_session()
    rd.session.set_default(session)
    session.open()

config = ConfigParser()
config.read('D:/config.ini')
app_key = config['main']['api_key']

open_session(app_key)

country = 'us'
current_quarter = "2Q26AS"

# if sector is '0', change sector name as All(Top Free-Float Mkt Cap 500)
if country == 'us':
    mktname = 'All(Top Free-Float Mkt Cap 500)'
    sector_len = 2
elif country == 'kr':
    mktname = 'All(Coverage Analyst >= 3)'
    sector_len = 3
else:
    raise ValueError("Unsupported country code")

result_list = glob(f"result/{country}/mixed_model_EQBtw_Q_sector_*.csv")
result_list.sort()

recent_q = pd.read_csv(result_list[-3])

recent_q.loc[recent_q['Sector'] == 0, 'Sector_name'] = mktname
recent_q.loc[recent_q['Sector'] == '00', 'Sector_name'] = mktname

# calc change between EQBtw == 0 and EQBtw == 1
recent_q['earning_G_bld_prev'] = recent_q.apply(
    lambda row: get_earning_prev(row['Sector'], row['FY'], row['EQBtw']),
    axis=1
)

recent_q['chg'] = recent_q.apply(
    lambda row: sub_with_string(row['earning_G_bld'], row['earning_G_bld_prev']),
    axis=1
)
recent_q['surp'] = recent_q.apply(
    lambda row: sub_with_string(row['earning_G_bld'], row['earning_EW_G_bld']),
    axis=1
)

print('Preparing Earning Tracker Report...')
final_result = []
for fq in range(0, 4):
    curr_fy = add_quarter(current_quarter, fq)

    df_fq = recent_q[(recent_q['FY']==curr_fy)]
    min_eqbtw = df_fq['EQBtw'].min()
    df_fq_min_eqbtw = df_fq[df_fq['EQBtw']==min_eqbtw]

    df_fq_result = df_fq_min_eqbtw[['Sector_name', 'Sector', 'FY', 'EQBtw', 'earning_G_bld_prev', 'earning_G_bld', 'earning_EW_G_bld', 'surp', 'chg']]
    df_fq_result = df_fq_result.astype({
        'Sector': pd.StringDtype(),
        'earning_G_bld': float,
        'earning_G_bld_prev': float,
        'earning_EW_G_bld': float,
        'chg': float,
        'surp': float},
        errors='ignore')

    df_fq_result['Sector_len'] = df_fq_result['Sector'].str.len()
    df_fq_result = df_fq_result.sort_values(by=['Sector_len','Sector']).drop('Sector_len', axis=1)

    final_result.append(df_fq_result)

    print(f'Processing done for {curr_fy}...')

print('Preparing Expected Report Equity Data Sheet...')
# make expected report date sheet
equity_data_list = glob(f"result/{country}/mixed_model_Q_*.csv")
equity_data_list.sort()
equity_data = pd.read_csv(equity_data_list[-1], index_col=0).astype({
    'Sector': pd.StringDtype(),
})

# load industry data
ind = pd.read_excel(f'data/{country}/infos.xlsx', sheet_name='industry_map', dtype=str).set_index('Code')
equity_data['SectorCode'] = equity_data['Sector'].str[:sector_len]
equity_data['GICS Sector'] = equity_data['SectorCode'].map(ind['Sector'])
equity_data['GICS Industry'] = equity_data['Sector'].map(ind['Sector'])

codes = equity_data['Code'].unique().tolist()
if country == 'kr':
    isin = pd.read_excel(f'data/{country}/infos.xlsx', sheet_name='isin', index_col=0, dtype=str)
    ds_codes = [isin.loc[k][0] for k in codes]
elif country == 'us':
    ds_codes = codes
else:
    raise ValueError('Invalid country code')

print('Fetching Expected Report Dates and Ticker Symbols from Refinitiv...')
exprepdate = rd.get_data(
    universe=ds_codes,
    fields=['TR.ExpectedReportDate', 'TR.ExpectedReportDate.periodenddate', 'TR.TickerSymbol'],
    parameters={'period':'FQ1'}
).set_index('Instrument').rename(columns={'Period End Date': 'Expected Period End Date'})
exprepdate['Expected Period End Date'] = exprepdate.apply(
    lambda x: x['Expected Period End Date'] + pd.Timedelta(days=90) if x['Expected Report Date'] - x['Expected Period End Date'] > pd.Timedelta(days=90) else x['Expected Period End Date']
    , axis=1)
exprepdate.index = codes

def get_FQFY(x):
    month = x.month
    year = x.year % 100
    if month in [1]:
        return f"4Q{year-1:02d}AS"
    elif month in [2,3,4]:
        return f"1Q{year:02d}AS"
    elif month in [5,6,7]:
        return f"2Q{year:02d}AS"
    elif month in [8,9,10]:
        return f"3Q{year:02d}AS"
    elif month in [11, 12]:
        return f"4Q{year:02d}AS"
    else:
        return None

def get_next_FQFY(x):
    month = x.month
    year = x.year % 100
    if month in [1]:
        return f"1Q{year:02d}AS"
    elif month in [2,3,4]:
        return f"2Q{year:02d}AS"
    elif month in [5,6,7]:
        return f"3Q{year:02d}AS"
    elif month in [8,9,10]:
        return f"4Q{year:02d}AS"
    elif month in [11, 12]:
        return f"1Q{year+1:02d}AS"
    else:
        return None

exprepdate['Expected FY'] = pd.to_datetime(exprepdate['Expected Period End Date']).apply(get_FQFY)
exprepdate['Expected Next FY'] = pd.to_datetime(exprepdate['Expected Period End Date']).apply(get_next_FQFY)
expreplist = exprepdate[(exprepdate['Expected Report Date'].dt.date>=dt.date.today())&(exprepdate['Expected Report Date'].dt.date<=dt.date.today() + dt.timedelta(days=61))].index.tolist()

exprep_df = pd.DataFrame(index=expreplist)
exprep_df['Expected Report Date'] = exprepdate.loc[expreplist, 'Expected Report Date']
exprep_df['Ticker'] = exprepdate.loc[expreplist, 'Ticker Symbol']
exprep_df['Expected FY'] = exprepdate.loc[expreplist, 'Expected FY']
exprep_df['Expected Next FY'] = exprepdate.loc[expreplist, 'Expected Next FY']
exprep_df = exprep_df.reset_index().set_index(['index','Expected FY'])

equity_data = equity_data.sort_values(by=['Code','FY','EQBtw'])
equity_data['EPS_EW_prev'] = equity_data['EPS_EW'].shift(-1)

# get only row with current quarter and min EQBtw
equity_data_min = equity_data.groupby(['Code','FY'])[['name', 'EQBtw', 'GICS Sector', 'GICS Industry', 'model', 'EPS_Est', 'EPS_EW', 'EPS_EW_prev']].first()

exprep_df['Name'] = equity_data_min['name']
exprep_df['GICS Sector'] = equity_data_min['GICS Sector']
exprep_df['GICS Industry'] = equity_data_min['GICS Industry']
exprep_df['Model'] = equity_data_min['model']
exprep_df['EPS_Est'] = equity_data_min['EPS_Est']
exprep_df['EPS_EW'] = equity_data_min['EPS_EW']
exprep_df['EPS_EW_prev'] = equity_data_min['EPS_EW_prev']
exprep_df = exprep_df.rename(columns={'Model':'FQ1 Model', 'EPS_Est': 'FQ1 Model EPS', 'EPS_EW': 'FQ1 EW EPS', 'EPS_EW_prev': 'FQ1 EW Prev EPS'})

exprep_df = exprep_df.reset_index().set_index(['index', 'Expected Next FY'])
exprep_df['Model'] = equity_data_min['model']
exprep_df['EPS_Est'] = equity_data_min['EPS_Est']
exprep_df['EPS_EW'] = equity_data_min['EPS_EW']
exprep_df['EPS_EW_prev'] = equity_data_min['EPS_EW_prev']
exprep_df = exprep_df.rename(columns={'Model':'FQ2 Model', 'EPS_Est': 'FQ2 Model EPS', 'EPS_EW': 'FQ2 EW EPS', 'EPS_EW_prev': 'FQ2 EW Prev EPS'})

exprep_df = exprep_df.reset_index().sort_values('Expected Report Date')
exprep_df = exprep_df[['index','Ticker','Name','GICS Sector','GICS Industry','Expected FY','Expected Report Date','FQ1 Model','FQ1 Model EPS','FQ1 EW EPS','FQ1 EW Prev EPS','FQ2 Model','FQ2 Model EPS','FQ2 EW EPS','FQ2 EW Prev EPS']]

print('Writing to Excel File...')
# to excel file, assign each excel sheet as each quarter
with pd.ExcelWriter(f'./result/earning_tracker_{country}/earning_tracker.xlsx') as writer:
    for i in range(0, 4):
        sheet_name = f"Q{i+1}"
        final_result[i].to_excel(writer, sheet_name=sheet_name, index=False)

    sheet_name = "실적 발표 예정 종목"
    exprep_df.to_excel(writer, sheet_name=sheet_name, index=False)