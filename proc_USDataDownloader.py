import os
import warnings
from concurrent.futures import ThreadPoolExecutor

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

    def _fetch(self, fields, params, codes):
        """Helper: call rd.get_data and return indexed DataFrame, or NaN fallback."""
        try:
            return rd.get_data(
                universe=codes,
                fields=fields,
                parameters=params
            ).set_index('Instrument')
        except Exception:
            return pd.DataFrame()

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

                    fetch_args = []
                    for q in range(0, self.q_window+1):
                        edate = dates + MonthEnd(3 - 3*q)
                        fetch_args.append(
                            (['TR.EPSEstValue.date','TR.EPSEstValue.analystname','TR.EPSEstValue.brokername','TR.EPSEstValue.periodyear',f'TR.EPSEstValue(Methodology={self.methodology})'],
                             {'SDate': edate.strftime('%Y-%m-%d'), 'Period': str(fdate), 'Frq': 'D'})
                        )

                    pbar.set_description(f'Downloading EPS estimate data for {str(dates)} with {len(codes)} codes')
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [
                            executor.submit(self._fetch, fields, params, codes)
                            for fields, params in fetch_args
                        ]
                        results = [f.result() for f in futures]  # preserves order

                    df_tmp = pd.concat(results, axis=0).drop_duplicates().dropna(subset='Earnings Per Share - Broker Estimate')

                    if df_tmp.empty:
                        pbar.update(1)
                        pbar.set_description(f'Failed to download EPS data {str(fdate)}')
                        continue

                    fetch_args = [
                        # (fields, parameters)
                        (
                            ['TR.GICSSubIndustryCode', 'TR.EPSActValue'],
                            {'SDate': edate.strftime('%Y-%m-%d'), 'Period': str(fdate)}
                        ),
                        (
                            ['TR.EPSActValue'],
                            {'SDate': edate.strftime('%Y-%m-%d'), 'Period': str(fdate_m1)}
                        ),
                        (
                            ['TR.EPSActValue', 'TR.F.BVperShrIssue(InstrumentType=Primary)'],
                            {'SDate': edate.strftime('%Y-%m-%d'), 'Period': str(fdate_m2)}
                        ),
                        (
                            ['TR.GuidanceDate'],
                            {'Frq': 'D', 'SDate': '1D', 'Period': str(fdate)}
                        ),
                    ]

                    pbar.set_description(f'Downloading Sector, EPS data')

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [
                            executor.submit(self._fetch, fields, params, codes)
                            for fields, params in fetch_args
                        ]
                        results = [f.result() for f in futures]  # preserves order

                    df_cross1, df_cross2, df_cross3, df_cross4 = results

                    if not df_cross4.empty:
                        df_cross4_min = df_cross4.groupby('Instrument')['Activation Date'].min()
                    else:
                        df_cross4_min = pd.DataFrame(np.nan, index=codes, columns=['GuidanceDate'], dtype='datetime64[ns]')

                    df_cross1 = df_cross1[~df_cross1.index.duplicated(keep='first')]
                    if len(df_cross1.columns) != 2:
                        df_cross1 = df_cross1.reindex(columns=['GICS Sub-Industry Code', 'Earnings Per Share - Actual'])
                    df_cross2 = df_cross2[~df_cross2.index.duplicated(keep='first')]
                    if len(df_cross2.columns) != 1:
                        df_cross2 = df_cross2.reindex(columns=['Earnings Per Share - Actual'])
                    df_cross3 = df_cross3[~df_cross3.index.duplicated(keep='first')]
                    if len(df_cross3.columns) != 2:
                        df_cross3 = df_cross3.reindex(columns=['Earnings Per Share - Actual', 'Book Value Per Share - Issue'])
                    df_cross = pd.concat([df_cross1, df_cross2, df_cross3, df_cross4_min], axis=1)
                    df_cross.columns = ['GICS', 'EPS', 'EPS_1Y', 'EPS_2Y', 'BPS', 'GuidanceDate']

                    df_tmp = df_tmp.join(df_cross, on='Instrument')

                    # if GuidanceDate is faster than Date, give 1 else 0
                    df_tmp['Guidance'] = (df_tmp['Date'] > df_tmp['GuidanceDate']).astype(int)

                    df_tmp = df_tmp.reset_index()

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
    dl.set_period('Q', 2026, 2028, startquarter=2, endquarter=3)
    #dl.set_period('Y', 2024, 2026, startquarter=None, endquarter=None)

    dl.run(skip_existing=False)