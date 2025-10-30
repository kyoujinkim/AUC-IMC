import os
import warnings

import numpy as np
import pandas as pd
import refinitiv.data as rd
from configparser import ConfigParser
from datetime import datetime as dt

from pandas._libs.tslibs.offsets import MonthEnd
from tqdm import tqdm
warnings.filterwarnings('ignore')


class DataDownloader:
    @staticmethod
    def open_session(app_key: str):
        # Open Refinitiv Session with loaded app key
        # Refinitiv Workspace/Eikon Desktop application must be on session
        session = rd.session.desktop.Definition(app_key=app_key).get_session()
        rd.session.set_default(session)
        session.open()

    def read_list(self, path:str):
        '''
        Read list of US stocks which listed at proc_USListSetup.py
        :param path:
        :return:
        '''
        pd_const = pd.read_csv(path)

        pd_const['fy'] = pd_const.Date.apply(lambda x: dt.strptime(x,'%Y-%m-%d').year).astype(int)
        self.const = pd_const
        self.fy = pd_const['fy'].unique()
        self.fq = list([f'{q}FQ{fy}' for fy in self.fy for q in range(1, 5)])

    def set_period(self, period: str, startyear: int, endyear: int, startquarter: int = None, endquarter: int = None):
        '''
        Set period base
        :param period: Y or Q
        :return:
        '''
        self.period = period
        if period == 'Q':
            self.q_window = 4
            # 8*
            down_list = []
            for fy in range(startyear, endyear + 1):
                if fy < startyear or fy > endyear:
                    continue
                if fy == startyear:
                    start_q = startquarter
                else:
                    start_q = 1
                if fy == endyear:
                    end_q = endquarter
                else:
                    end_q = 4
                down_list += [f'{q}FQ{fy}' for q in range(start_q, end_q+1)]

            self.methodology = ''
        else:
            self.q_window = 7
            down_list = [f'{fy}' for fy in range(startyear, endyear + 1)]
            self.methodology = 'InterimSum'

        self.down_list = down_list

    @staticmethod
    def prev_period(period):
        if 'FQ' in period:
            fy = int(period[-4:])
            q = int(period[0])
            return f'{q}FQ{fy-1}'
        else:
            return str(int(period)-1)

    def run(self, skip_existing=True):
        # if self.period is not set, raise error
        if not hasattr(self, 'period'):
            raise ValueError('set_period method is not called')

        # run process
        with tqdm(total=len(self.down_list), desc='Downloading EPS data') as pbar:
            for fdate in self.down_list:
                if skip_existing:
                    if os.path.exists(f'./data/us/consenlist/eps_{fdate}.csv'):
                        pbar.update(1)
                        continue

                if self.period == 'Q':
                    fy = int(fdate[-4:])
                    fdate_m1 = self.prev_period(fdate)
                    fdate_m2 = self.prev_period(self.prev_period(fdate))
                else:
                    fy = fdate
                    fdate_m1 = self.prev_period(fdate)
                    fdate_m2 = self.prev_period(self.prev_period(fdate))

                if fy > self.const['fy'].max():
                    fy_for_code = self.const.fy.max()
                else:
                    fy_for_code = fy
                codes = self.const[self.const['fy'] == fy_for_code].Code.to_list()

                df_date = rd.get_data(
                    universe=codes,
                    fields=['TR.EPSMean.periodenddate'],
                    parameters={
                        'Period': str(fdate)
                    }
                ).set_index('Instrument').rename(columns={'Period End Date': 'PeriodEndDate'})
                df_date = df_date.dropna(subset=['PeriodEndDate'])

                df_total = []
                for dates in df_date['PeriodEndDate'].unique():
                    codes = df_date[df_date['PeriodEndDate'] == dates].index.values
                    df_list = []
                    for q in range(0, self.q_window+1):
                        edate = dates + MonthEnd(3 - 3*q)
                        try:
                            pbar.set_description(f'Downloading EPS data {int(fy)} Q{q + 1} : {edate.strftime("%Y-%m-%d")}')
                            df = rd.get_data(
                                universe = codes,
                                fields = ['TR.EPSEstValue.date','TR.EPSEstValue.analystname','TR.EPSEstValue.brokername','TR.EPSEstValue.periodyear',f'TR.EPSEstValue(Methodology={self.methodology})'],
                                parameters = {
                                    'SDate': edate.strftime('%Y-%m-%d'),
                                    'Frq': 'D',
                                    'Period': str(fdate)
                                }
                            )
                            df_list.append(df)

                        except:
                            pbar.set_description(f'Downloading EPS data {str(fdate)} Q{q + 1} : failed')

                    df_tmp = pd.concat(df_list, axis=0).drop_duplicates().dropna(subset='Earnings Per Share - Broker Estimate')
                    if df_tmp.empty:
                        pbar.update(1)
                        pbar.set_description(f'Failed to download EPS data {str(fdate)}')
                        continue

                    pbar.set_description(f'Downloading Sector, EPS data')
                    try:
                        df_cross1 = rd.get_data(
                            universe=codes,
                            fields=['TR.GICSSubIndustryCode', 'TR.EPSActValue'],
                            parameters={
                                'SDate': edate.strftime('%Y-%m-%d'),
                                'Period': str(fdate)
                            }
                        ).set_index('Instrument')
                    except:
                        df_cross1 = pd.DataFrame(np.nan, index=codes, columns=['GICS', 'EPS'], dtype='float64')

                    pbar.set_description(f'Downloading EPS_FY-1 data')
                    try:
                        df_cross2 = rd.get_data(
                            universe=codes,
                            fields=['TR.EPSActValue'],
                            parameters={
                                'SDate': edate.strftime('%Y-%m-%d'),
                                'Period': str(fdate_m1)
                            }
                        ).set_index('Instrument')
                    except:
                        df_cross2 = pd.DataFrame(np.nan, index=codes, columns=['EPS_1Y'], dtype='float64')

                    pbar.set_description(f'Downloading EPS_FY-2 data')
                    try:
                        df_cross3 = rd.get_data(
                            universe=codes,
                            fields=['TR.EPSActValue', 'TR.F.BVperShrIssue(InstrumentType=Primary)'],
                            parameters={
                                'SDate': edate.strftime('%Y-%m-%d'),
                                'Period': str(fdate_m2)
                            }
                        ).set_index('Instrument')
                    except:
                        df_cross3 = pd.DataFrame(np.nan, index=codes, columns=['EPS_2Y', 'BPS'], dtype='float64')

                    try:
                        pbar.set_description(f'Downloading Guidance data')
                        df_cross4 = rd.get_data(
                            universe=codes,
                            fields=['TR.GuidanceDate'],
                            parameters={
                                'Frq': 'D',
                                'SDate': "1D",
                                'Period': str(fdate)
                            }
                        ).set_index('Instrument')
                        # get the most fastest guidance date
                        df_cross4_min = df_cross4.groupby('Instrument')['Activation Date'].min()
                    except:
                        print('No Guidance Date')
                        df_cross4_min = pd.DataFrame(np.nan, index=codes, columns=['GuidanceDate'], dtype='datetime64[ns]')

                    df_cross1 = df_cross1[~df_cross1.index.duplicated(keep='first')]
                    df_cross2 = df_cross2[~df_cross2.index.duplicated(keep='first')]
                    df_cross3 = df_cross3[~df_cross3.index.duplicated(keep='first')]
                    df_cross = pd.concat([df_cross1, df_cross2, df_cross3, df_cross4_min], axis=1)
                    df_cross.columns = ['GICS', 'EPS', 'EPS_1Y', 'EPS_2Y', 'BPS', 'GuidanceDate']

                    df_tmp = df_tmp.join(df_cross, on='Instrument')

                    # if GuidanceDate is faster than Date, give 1 else 0
                    df_tmp['Guidance'] = (df_tmp['Date'] > df_tmp['GuidanceDate']).astype(int)

                    df_total.append(df_tmp)

                df_total = pd.concat(df_total, axis=0)
                df_total = df_total.join(df_date, on='Instrument')

                if os.path.exists(f'./data/us/consenlist/eps_{fdate}.csv'):
                    # concat and drop duplicates
                    prev_file = pd.read_csv(f'./data/us/consenlist/eps_{fdate}.csv', encoding='utf-8-sig', dtype={'Date': str, 'PeriodEndDate': str})
                    df_total = pd.concat([prev_file, df_total], axis=0).drop_duplicates()
                df_total.to_csv(f'./data/us/consenlist/eps_{fdate}.csv', index=False, encoding='utf-8-sig')

                pbar.update(1)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('D:/config.ini')
    app_key = config['main']['api_key']

    dl = DataDownloader()
    dl.open_session(app_key)
    dl.read_list('./data/us/list_total.csv')
    dl.set_period('Q', 2025, 2025, startquarter=1, endquarter=4)
    #dl.set_period('Y', 2024, 2026, startquarter=None, endquarter=None)

    dl.run(skip_existing=False)