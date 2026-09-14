import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import List

from dateutil.relativedelta import relativedelta
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import *
import warnings
warnings.filterwarnings('ignore')


class Enhanced_EPS(object):
    def __init__(self, q_basis='QBtw'):
        '''
        Dividend Calculation Class
        :param q_basis: Either 'QBtw' or 'CQBtw'. QBtw is for quarterly distance between estimated report date. CQBtw is for Calendar distance from now.
        '''
        self.ucurve = dict()
        self.min_count = 2
        self.year_range = 10
        if q_basis in ['QBtw', 'CQBtw']:
            self.q_basis = q_basis
        else:
            raise ValueError("q_basis should be 'QBtw' or 'CQBtw'")

    #set base data for calculation
    def set_data(self, train):
        self.train = train

    def __save_memory__(self):
        datadf = self.train
        datapk = pickle.dumps(datadf)
        # Create shared memory
        shm_train = SharedMemory(create=True, size=len(datapk))
        setattr(self, 'data', shm_train)

        # Create a NumPy array from shared memory buffer
        buf_array = np.frombuffer(shm_train.buf, dtype=np.uint8)
        #setattr(self, path + '_buf', buf_array)

        # Copy serialized data into shared memory
        buf_array[:] = np.frombuffer(datapk, dtype=np.uint8)

        ucurve = self.ucurve
        ucurvepk = pickle.dumps(ucurve)
        # Create shared memory
        shm_ucurve = SharedMemory(create=True, size=len(ucurvepk))
        setattr(self, 'data_uc', shm_ucurve)

        # Create a NumPy array from shared memory buffer
        buf_array_uc = np.frombuffer(shm_ucurve.buf, dtype=np.uint8)
        #setattr(self, path + '_buf', buf_array)

        # Copy serialized data into shared memory
        buf_array_uc[:] = np.frombuffer(ucurvepk, dtype=np.uint8)

        del buf_array, buf_array_uc

        self.shm_train = shm_train.name
        self.shm_ucurve = shm_ucurve.name

        return True

    @staticmethod
    def __load_memory__(name):
        # Retrieve data from shared memory (for validation)
        shm = SharedMemory(name=name)  # Attach to existing shared memory
        datadfex = pickle.loads(shm.buf[:])
        # Close shared memory
        shm.close()# Corrected syntax
        shm.unlink()

        return datadfex

    @staticmethod
    def __filter_outlier__(df, codecol, errorcol):
        df_grouped = df.groupby(codecol)[errorcol].mean()
        meanval = df_grouped.mean()
        stdval = df_grouped.std()
        df_list = df_grouped[(df_grouped < meanval + 2 * stdval) & (df_grouped > meanval - 2 * stdval)].index

        return df[df[codecol].isin(df_list)]

    def calc_ucurve(self, country, prdFY, train, reuse:bool=False):
        if reuse:
            self.ucurve = pd.read_json(f'result/{country}/ucurve.json').T.to_dict(orient='index')
            return True

        # add np.nan to train.SectorClass.unique
        unique_sectors = set(train.SectorClass.unique().tolist() + [np.nan])

        listoftable = list(product(unique_sectors, prdFY))

        def _fit(table):
            return str(table[0]) + str(table[1]), term_spread_adj(*table, train=train)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            for key, val in tqdm(pool.map(_fit, listoftable), total=len(listoftable), desc='calc_ucurve'):
                self.ucurve[key] = val

        pd.DataFrame(self.ucurve).to_json(f'result/{country}/ucurve.json')

        return True

    @staticmethod
    def shift_period(period, lag=1):
        if 'Q' in period:
            fy = int(period[-4:-2])
            q = period[:2]
            return f'{q}{fy - lag}AS'
        else:
            fy = int(period[:4])
            return f'{fy - lag}AS'

    def get_presym(self, sym, bestmodel, depth):
        # if depth is too deep, return
        if depth > 2:
            return ['EW', sym, depth]
        else:
            best = bestmodel.get(sym, None)
            if best is None:
                presym = sym[:-6] + self.shift_period(sym[-6:], 1)
                depth += 1
                aeps_na = self.train[self.train.UniqueSymbol == presym].A_EPS.isna().all()
                # if previous year's actual data is not yet announced, get in to depth
                if aeps_na:
                    return self.get_presym(presym, bestmodel, depth)
                # if previous year's actual data is already announced, return
                else:
                    return [None, presym, depth]
            # if best is not None
            else:
                return [best, sym, depth]

    def calc_mixed_model(self, UniqueSymbol:List, model_list:List, country:str):
        if len(UniqueSymbol)==0:
            return pd.DataFrame()

        bestmodel_path = f'result/{country}/bestmodel_history/bestmodel.json'
        if os.path.exists(bestmodel_path):
            with open(bestmodel_path, 'r') as f:
                bestmodel = json.load(f)
        else:
            bestmodel = {}
        bestmodel_tosave = bestmodel.copy()

        # set shared memory
        _ = self.__save_memory__()
        # calculate symbol's last fy perfomance by model within model_list
        # convert unique symbol(code(12) + fy(12:) to last fy
        ncal_presym = []
        for sym in UniqueSymbol:
            best = bestmodel.get(sym, None)
            if best is None:
                presym = sym[:-6] + self.shift_period(sym[-6:], 1)
                ncal_presym.append(presym)

        ncal_presym_list = {}
        for sym in UniqueSymbol:
            ncal_presym_list[sym] = self.get_presym(sym, bestmodel, 0)

        # investigate which to calculate
        ncal_presym = []
        ncal_presym_dict = {}
        for sym in ncal_presym_list.keys():
            best, presym, depth = ncal_presym_list[sym]
            if best is None:
                ncal_presym.append(presym)
                ncal_presym_dict[presym] = sym
            else:
                bestmodel[sym] = best

        # get pre error of each model
        if len(ncal_presym) > 0:
            multiproclist = list(product(ncal_presym, model_list, [self.shm_train], [self.shm_ucurve], [self.q_basis]))
            pre_result = process_map(Enhanced_EPS.__calc__, multiproclist, max_workers=os.cpu_count()-1)
            pre_result = pd.concat(pre_result)
            pre_result.to_parquet(f'result/{country}/bestmodel_history/pre_result.parquet', engine='pyarrow', index=False)

            pre_result = pre_result[~((pre_result.Sector.str[:2]=='35') & (pre_result.model.str[-3:]=='adp'))]
            pre_result['MAFE'] = (pre_result['Est'] - (pre_result['EPS_Actual'] / pre_result['BPS'])).abs()
            data_group = pre_result.groupby(['Code', 'model', 'FY'])[['MAFE', 'EPS_Actual']].mean().reset_index()
            data_group_best = data_group.groupby(['Code', 'FY']).MAFE.idxmin().dropna()
            data_group_best = pd.DataFrame({'model': data_group.model[data_group_best].values,'EPS_Actual': data_group.EPS_Actual[data_group_best].isna().values}, index=data_group_best.index).reset_index()

            for _, row in data_group_best.iterrows():
                presym = row['Code'] + row['FY']
                # code which save in bestmodel database
                code = presym[:-6] + self.shift_period(presym[-6:], -1)
                # code which save in bestmodel cache
                sym = ncal_presym_dict[presym]

                model = row['model']
                epsact = row['EPS_Actual']
                bestmodel[sym] = model
                if not epsact:
                    bestmodel_tosave[code] = model

        # save only EPS actual exist
        with open(bestmodel_path, 'w') as f:
            json.dump(bestmodel_tosave, f)

        # calculate with best model
        symwithbm = []
        for sym in UniqueSymbol:
            symwithbm.append([sym, bestmodel.get(sym, 'EW'), self.shm_train, self.shm_ucurve, self.q_basis])
        result = process_map(Enhanced_EPS.__calc__, symwithbm, max_workers=os.cpu_count()-1)

        return pd.concat(result)

    def calc_model(self, UniqueSymbol:List, model_name:str):
        if len(UniqueSymbol)==0:
            return pd.DataFrame()

        # set shared memory
        _ = self.__save_memory__()

        symwithbm = []
        for sym in UniqueSymbol:
            symwithbm.append([sym, model_name, self.shm_train, self.shm_ucurve, self.q_basis])
        result = process_map(Enhanced_EPS.__calc__, symwithbm, max_workers=os.cpu_count()-1)

        return pd.concat(result)

    @staticmethod
    def __calc__(x):
        model_name = x[1]

        if model_name == 'EW':
            dataset = Enhanced_EPS.__EW__(x)
        elif model_name == 'PBest':
            dataset = Enhanced_EPS.__PBest__(x)
        elif model_name == 'IMSE':
            dataset = Enhanced_EPS.__IMSE__(x)
        elif model_name == 'BAM':
            dataset = Enhanced_EPS.__BAM__(x)
        elif model_name == 'IMC':
            dataset = Enhanced_EPS.__IMC__(x)
        elif model_name == 'EW_adp':
            dataset = Enhanced_EPS.__EW_adp__(x)
        elif model_name == 'PBest_adp':
            dataset = Enhanced_EPS.__PBest_adp__(x)
        elif model_name == 'IMSE_adp':
            dataset = Enhanced_EPS.__IMSE_adp__(x)
        elif model_name == 'BAM_adp':
            dataset = Enhanced_EPS.__BAM_adp__(x)
        elif model_name == 'IMC_adp':
            dataset = Enhanced_EPS.__IMC_adp__(x)
        else:
            raise('Invalid model name')

        if len(dataset) == 0:
            return pd.DataFrame()
        else:
            dataset['model'] = model_name
            return dataset

    @staticmethod
    def __fill_missing_est__(df, q_basis):
        df = df.sort_values('Date')
        df_pivot = pd.pivot_table(df, values='E_EPS', columns=['Security'], index=[q_basis], aggfunc='last').ffill()
        # 최솟값부터 최댓값까지 빠짐없이 있어야 할 풀 세트 생성: {1, 2, 3, 4, 5}
        full_set = set(range(df_pivot.index[0], df_pivot.index[-1] + 1))
        missing_elements = full_set - set(df_pivot.index)

        # fill up missing rows
        for me in missing_elements:
            df_pivot.loc[me] = np.nan
        df_pivot = df_pivot.sort_index().ffill()

        df_ravel = df_pivot.melt(ignore_index=False).reset_index()
        df_ravel = df_ravel[df_ravel.value.isna()]

        if not df_ravel.empty:
            # 3. 'r['QBtw']+1'에 해당하는 데이터를 한 번에 가져오기 위해 조인 키 생성
            df_ravel['target_QBtw'] = df_ravel[q_basis] + 1

            # 4. 원본 df에서 필요한 컬럼만 추출하여 병합 (Security와 QBtw를 기준으로 매칭)
            # iloc[-1]의 효과를 내기 위해 drop_duplicates로 마지막 값만 남긴 df_target 사용
            df_target = df.drop_duplicates(subset=['Security', q_basis], keep='last').sort_values(q_basis)
            df_target['target_QBtw'] = df_target[q_basis]

            # Type Competence 확보
            df_ravel['target_QBtw'] = df_ravel['target_QBtw'].astype(int)
            df_target['target_QBtw'] = df_target['target_QBtw'].astype(int)
            df_ravel = df_ravel.sort_values('target_QBtw', ignore_index=True)
            df_target = df_target.sort_values('target_QBtw', ignore_index=True)

            '''# 2. merge_asof 실행
            res = pd.merge_asof(
                df_ravel,
                df_target,
                on='target_QBtw',  # 기준이 되는 주기 컬럼
                by='Security',  # 종목별로 그룹을 묶어서 매칭
                direction='forward',  # ★ 핵심: 내 주기보다 '크거나 같은' 값 중 가장 가까운 것 선택
                suffixes=('_main', '')
            )'''

            res = df_ravel.merge(
                df_target,
                left_on=['Security', 'target_QBtw'],
                right_on=['Security', 'target_QBtw'],
                suffixes=('_main', '')
            )

            res = res.dropna(subset=['E_EPS'])

            res['Q_diff'] = res[q_basis] - res[f'{q_basis}_main']
            res['Date'] += pd.to_timedelta(res['Q_diff'] * 90, unit='D')
            #res['DBtw'] -= res['Q_diff'] * 90 # We need to fix DBtw to adapt term spread, since term spread function adjusting based on DBtw
            res['QBtw'] -= res['Q_diff']
            res['EQBtw'] -= res['Q_diff']

            # Change main with q_basis
            res[q_basis] = res[f'{q_basis}_main']

            # 불필요한 임시 컬럼 제거 및 결과 리스트화
            res = res.drop(columns=['value', 'target_QBtw', f'{q_basis}_main'])
            return res
        else:
            return pd.DataFrame()

    @staticmethod
    def __resform__(data, code, df):
        data['Code'] = code
        data['Sector'] = df.Sector.iloc[-1]
        data['BPS'] = df.BPS.iloc[0]
        data['PeriodEndDate'] = df.PeriodEndDate.iloc[0]
        data['EPS_Actual'] = df.A_EPS.iloc[0]
        data['EPS_1Y'] = df['EPS_1Y'].mean()
        data['EPS_2Y'] = df['EPS_2Y'].mean()

        return data

    @staticmethod
    def __EW__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        #shm_ucurve = x[3]
        q_basis = x[4]

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol==symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        # --- OPTIMIZATION: 단 한 번의 Groupby로 모든 집계 연산 처리 ---
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

            # 결합 연산을 Vectorized 연산으로 한 번에 DataFrame 구축
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE']
        data['EW'] = agg_df['E_ROE']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __PBest__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        #shm_ucurve = x[3]
        q_basis = x[4]

        star_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol==symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 4)
        year_max = str(target_year - 1)

        # 전체 train 스캔 최소화
        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        # 미리 중복 제거 및 절댓값 에러 컬럼 생성 (벡터화 준비)
        hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
        hist_train['abs_Error'] = hist_train['Error'].abs()

        # 모든 Q와 Security별 평균 에러율을 한 번에 계산
        grouped_errors = hist_train.groupby([q_basis, 'Security'])['abs_Error'].mean()
        available_Qs = set(grouped_errors.index.get_level_values(0))

        unique_securities = df['Security'].unique()
        Q_result = []

        # Q에 대해서만 루프 수행 (Security 루프는 완전히 증발함)
        for Q in df[q_basis].unique():
            df_Q = df[df[q_basis] == Q]

            if Q in available_Qs:
                # 해당 Q의 애널리스트 에러 정보 가져오기
                q_errors = grouped_errors.xs(Q, level=0)
                # 현재 분석 중인 애널리스트(unique_securities)만 필터링
                q_errors = q_errors[q_errors.index.isin(unique_securities)]

                if not q_errors.empty:
                    # 상위 5명(star_count)의 애널리스트 추출 (nsmallest 활용)
                    top_secs = q_errors.nsmallest(star_count).index
                    check_star_count = df_Q[df_Q['Security'].isin(top_secs)]

                    if len(check_star_count) >= 2:
                        Q_result.append(check_star_count)
                        continue

            # 데이터가 없거나 조건 미달 시 원본 유지
            Q_result.append(df_Q)

        estEW = pd.DataFrame(df.groupby(q_basis)['E_ROE_o'].mean())
        if Q_result:
            df = pd.concat(Q_result)

        # 단 1번의 groupby로 나머지 연산 일괄 처리
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 가로 병합(concat) 대신 딕셔너리 스타일 매핑으로 오버헤드 방지
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE']
        data['EW'] = estEW  # 인덱스(q_basis) 기준으로 자동 정렬 및 매핑됨
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 결과 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __IMSE__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        #shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 3)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
        hist_train['abs_Error'] = hist_train['Error'].abs()

        grouped_errors = hist_train.groupby([q_basis, 'Security'])['abs_Error'].mean()
        available_Qs = set(grouped_errors.index.get_level_values(0))

        Q_result = []

        # Q 레벨 루프만 유지 (Security 루프는 완전 제거됨)
        for Q in df[q_basis].unique():
            df_Q = df[df[q_basis] == Q].copy()

            if Q in available_Qs:
                # 해당 Q의 애널리스트별 평균 에러 추출 (Series 형태)
                q_errors = grouped_errors.xs(Q, level=0)

                # 현재 데이터에 존재하는 애널리스트만 필터링
                unique_securities = df_Q['Security'].unique()
                q_errors = q_errors[q_errors.index.isin(unique_securities)]

                if not q_errors.empty:
                    # 과거 에러 데이터가 존재하는 Row들만 필터링
                    valid_df_Q = df_Q[df_Q['Security'].isin(q_errors.index)].copy()

                    if len(valid_df_Q) >= min_count:
                        # 초성능 킬러 포인트: .apply() 대신 .map() 사용
                        valid_df_Q['PrevError'] = valid_df_Q['Security'].map(q_errors)
                        Q_result.append(valid_df_Q)
                        continue

            # 데이터가 없거나 기준 충족 못할 시 Fallback 처리
            df_Q['PrevError'] = 1.0
            Q_result.append(df_Q)

        if not Q_result:
            return pd.DataFrame()

        # 데이터 병합 및 가중치 계산 (Vectorized 연산)
        df = pd.concat(Q_result)
        df['PrevError'] += 0.01

        df['I_PrevError'] = 1.0 / df['PrevError']
        df_mean = df['I_PrevError'].mean()
        df_std = df['I_PrevError'].std()

        # 아웃라이어 클리핑 및 가중 ROE 계산
        df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 3 * df_std, upper=df_mean + 3 * df_std)
        df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']

        agg_df = df.groupby(q_basis).agg({
            'W_E_ROE': 'sum',
            'I_PrevError': 'sum',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 딕셔너리 스타일 구조 매핑으로 pd.concat 오버헤드 방지
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['W_E_ROE'] / agg_df['I_PrevError']
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __BAM__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        #shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 과거 11개년 기록 단 한 번만 미리 도려내기
        # ----------------------------------------------------------------
        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 11)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        coeff_map = {}

        if not hist_train.empty:
            # 중복 제거
            hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()

            # ----------------------------------------------------------------
            # OPTIMIZATION 2: LinearRegression 라이브러리를 대체하는 수학적 벡터화 (OLS)
            # fit_intercept=False 일 때, Slope = sum(X*Y) / sum(X^2)
            # ----------------------------------------------------------------
            hist_train['XY'] = hist_train['E_ROE'] * hist_train['A_ROE']
            hist_train['X2'] = hist_train['E_ROE'] ** 2

            # 모든 Q에 대한 필요한 통계량을 단 한 번의 groupby로 집계
            agg_hist = hist_train.groupby(q_basis).agg(
                sum_xy=('XY', 'sum'),
                sum_x2=('X2', 'sum'),
                unique_years=('Year', 'nunique')
            )

            # 조건 검증 (데이터 개수 조건 및 분모가 0이 아닌지 체크)
            valid_mask = (agg_hist['unique_years'] >= min_count) & (agg_hist['sum_x2'] > 0)

            # 기본값은 Slope=1, Intercept=0으로 세팅 후 유효한 값만 연산
            agg_hist['Slope'] = 1.0
            agg_hist.loc[valid_mask, 'Slope'] = agg_hist.loc[valid_mask, 'sum_xy'] / agg_hist.loc[valid_mask, 'sum_x2']
            agg_hist['Intercept'] = 0.0

            # 빠른 조회를 위해 딕셔너리로 변환
            coeff_map = agg_hist[['Slope', 'Intercept']].to_dict('index')

            # ----------------------------------------------------------------
            # OPTIMIZATION 3: 후반부 수많은 Groupby를 단 1번으로 일괄 집계
            # ----------------------------------------------------------------
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # ----------------------------------------------------------------
        # OPTIMIZATION 4: apply(axis=1) 제거 및 고속 매핑 보정 연산
        # ----------------------------------------------------------------
        # 각 Q에 맞는 Slope와 Intercept를 C-Level 속도로 매핑
        slopes = agg_df.index.map(lambda q: coeff_map.get(q, {'Slope': 1.0})['Slope'])
        intercepts = agg_df.index.map(lambda q: coeff_map.get(q, {'Intercept': 0.0})['Intercept'])

        # 결과 데이터프레임 구축 (BAM 보정 수식을 인라인 벡터 연산으로 처리)
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE'] * slopes + intercepts  # apply_bam 함수가 y = ax + b 구조일 때 기준
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __IMC__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        #shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
        df['SecAnl'] = df['Security'] + df['CoreAnalyst']

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 과거 11개년 기록 단 한 번만 슬라이싱 및 필터링
        # ----------------------------------------------------------------
        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 11)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        # 기본 매핑 딕셔너리 초기화
        s_slope_map = {}
        q_slope_map = {}

        if not hist_train.empty:
            hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
            hist_train['CoreAnalyst'] = hist_train['Analyst'].str.split(',').str[0]
            hist_train['SecAnl'] = hist_train['Security'] + hist_train['CoreAnalyst']

            # 현재 분석 타겟에 존재하는 애널리스트 데이터만 남겨 서치 오버헤드 최소화
            hist_train = hist_train[hist_train['SecAnl'].isin(df['SecAnl'].unique())]

            # ----------------------------------------------------------------
            # OPTIMIZATION 2: [1단계] Analyst 레벨 기울기 일괄 계산 (이중 루프 제거)
            # fit_intercept=False 일 때, Slope = sum(X*Y) / sum(X^2)
            # ----------------------------------------------------------------
            hist_train['XY'] = hist_train['E_ROE'] * hist_train['A_ROE']
            hist_train['X2'] = hist_train['E_ROE'] ** 2

            # Q와 SecAnl 조합으로 단 한 번에 그룹 집계
            s_agg = hist_train.groupby([q_basis, 'SecAnl']).agg(
                sum_xy=('XY', 'sum'),
                sum_x2=('X2', 'sum'),
                row_cnt=('XY', 'count'),
                nyear=('Year', 'nunique')
            )

            s_agg['Slope'] = 1.0
            valid_s = (s_agg['row_cnt'] >= 10) & (s_agg['nyear'] >= min_count) & (s_agg['sum_x2'] > 0)
            s_agg.loc[valid_s, 'Slope'] = s_agg.loc[valid_s, 'sum_xy'] / s_agg.loc[valid_s, 'sum_x2']
            s_slope_map = s_agg['Slope'].to_dict()

            # ----------------------------------------------------------------
            # OPTIMIZATION 3: [2단계] Company(Q) 레벨 기울기 일괄 계산
            # 1단계에서 계산된 Analyst Slope를 적용한 뒤 다시 OLS 수행
            # ----------------------------------------------------------------
            # MultiIndex 매핑을 통해 가중치 적용 (.apply 제거)
            hist_train['S_Slope'] = hist_train.set_index([q_basis, 'SecAnl']).index.map(s_slope_map).fillna(1.0)
            hist_train['E_ROE_adj'] = hist_train['E_ROE'] * hist_train['S_Slope']

            hist_train['XY_adj'] = hist_train['E_ROE_adj'] * hist_train['A_ROE']
            hist_train['X2_adj'] = hist_train['E_ROE_adj'] ** 2

            q_agg = hist_train.groupby(q_basis).agg(
                sum_xy_adj=('XY_adj', 'sum'),
                sum_x2_adj=('X2_adj', 'sum'),
                nyear_q=('Year', 'nunique')
            )

            q_agg['Slope'] = 1.0
            valid_q = (q_agg['nyear_q'] >= min_count) & (q_agg['sum_x2_adj'] > 0)
            q_agg.loc[valid_q, 'Slope'] = q_agg.loc[valid_q, 'sum_xy_adj'] / q_agg.loc[valid_q, 'sum_x2_adj']
            q_slope_map = q_agg['Slope'].to_dict()

            # ----------------------------------------------------------------
            # OPTIMIZATION 4: 현재 예측 대상 데이터(df)에 보정치 초고속 적용
            # ----------------------------------------------------------------
            # 1단계: Analyst 레벨 보정 (.apply 대신 .map 사용)
        df['S_Slope'] = df.set_index([q_basis, 'SecAnl']).index.map(s_slope_map).fillna(1.0)
        df['E_ROE'] = df['E_ROE'] * df['S_Slope']

        # 후반부 모든 무거운 Groupby 연산을 단 1번으로 일괄 처리
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 2단계: Company 레벨 보정 (최종 데이터프레임 구축 과정에서 인라인 연산)
        q_slopes = agg_df.index.map(lambda q: q_slope_map.get(q, 1.0))

        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE'] * q_slopes  # 별도의 apply_bam 함수 호출 없이 벡터 연산 처리
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __EW_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]
        q_basis = x[4]

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]

        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 부울 마스킹을 통한 Before/After 분기 연산 최적화
        # ----------------------------------------------------------------
        mask_bf = df['Date'] <= df['CutDate']
        spread_series = pd.Series(0.0, index=df.index)

        # Before CutDate 그룹 연산
        if mask_bf.any():
            # 만약 term_spread가 벡터화를 지원한다면: term_spread(df[mask_bf], *popt_bf) 가 베스트
            spread_series.loc[mask_bf] = df['DBtw'].loc[mask_bf].apply(lambda r: term_spread(r, *popt_bf))

        # After CutDate 그룹 연산 (~ 연산자로 반대 타겟 지정)
        if (~mask_bf).any():
            spread_series.loc[~mask_bf] = df['DBtw'].loc[~mask_bf].apply(lambda r: term_spread(r, *popt_af))

        # 보정치 차감 처리
        df['E_ROE'] = df['E_ROE'] - spread_series.fillna(0)

        # ----------------------------------------------------------------
        # OPTIMIZATION 2: 후반부 모든 무거운 Groupby를 단 1번으로 일괄 집계
        # ----------------------------------------------------------------
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 가로 병합(concat) 오버헤드 방지를 위한 다이렉트 딕셔너리 빌드
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE']
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅 및 결과 반환
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __PBest_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]
        q_basis = x[4]

        star_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]

        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 부울 마스킹을 통한 Before/After 분기 연산 최적화
        # ----------------------------------------------------------------
        mask_bf = df['Date'] <= df['CutDate']
        spread_series = pd.Series(0.0, index=df.index)

        # Before CutDate 그룹 연산
        if mask_bf.any():
            # 만약 term_spread가 벡터화를 지원한다면: term_spread(df[mask_bf], *popt_bf) 가 베스트
            spread_series.loc[mask_bf] = df['DBtw'].loc[mask_bf].apply(lambda r: term_spread(r, *popt_bf))

        # After CutDate 그룹 연산 (~ 연산자로 반대 타겟 지정)
        if (~mask_bf).any():
            spread_series.loc[~mask_bf] = df['DBtw'].loc[~mask_bf].apply(lambda r: term_spread(r, *popt_af))

        # 보정치 차감 처리
        df['E_ROE'] = df['E_ROE'] - spread_series.fillna(0)

        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 4)
        year_max = str(target_year - 1)

        # 전체 train 스캔 최소화
        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        # 미리 중복 제거 및 절댓값 에러 컬럼 생성 (벡터화 준비)
        hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
        hist_train['abs_Error'] = hist_train['Error'].abs()

        # 모든 Q와 Security별 평균 에러율을 한 번에 계산
        grouped_errors = hist_train.groupby([q_basis, 'Security'])['abs_Error'].mean()
        available_Qs = set(grouped_errors.index.get_level_values(0))

        unique_securities = df['Security'].unique()
        Q_result = []

        # Q에 대해서만 루프 수행 (Security 루프는 완전히 증발함)
        for Q in df[q_basis].unique():
            df_Q = df[df[q_basis] == Q]

            if Q in available_Qs:
                # 해당 Q의 애널리스트 에러 정보 가져오기
                q_errors = grouped_errors.xs(Q, level=0)
                # 현재 분석 중인 애널리스트(unique_securities)만 필터링
                q_errors = q_errors[q_errors.index.isin(unique_securities)]

                if not q_errors.empty:
                    # 상위 5명(star_count)의 애널리스트 추출 (nsmallest 활용)
                    top_secs = q_errors.nsmallest(star_count).index
                    check_star_count = df_Q[df_Q['Security'].isin(top_secs)]

                    if len(check_star_count) >= 2:
                        Q_result.append(check_star_count)
                        continue

            # 데이터가 없거나 조건 미달 시 원본 유지
            Q_result.append(df_Q)

        estEW = pd.DataFrame(df.groupby(q_basis)['E_ROE_o'].mean())
        if Q_result:
            df = pd.concat(Q_result)

        # 단 1번의 groupby로 나머지 연산 일괄 처리
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 가로 병합(concat) 대신 딕셔너리 스타일 매핑으로 오버헤드 방지
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE']
        data['EW'] = estEW  # 인덱스(q_basis) 기준으로 자동 정렬 및 매핑됨
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 결과 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __IMSE_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]

        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 부울 마스킹을 통한 Before/After 분기 연산 최적화
        # ----------------------------------------------------------------
        mask_bf = df['Date'] <= df['CutDate']
        spread_series = pd.Series(0.0, index=df.index)

        # Before CutDate 그룹 연산
        if mask_bf.any():
            # 만약 term_spread가 벡터화를 지원한다면: term_spread(df[mask_bf], *popt_bf) 가 베스트
            spread_series.loc[mask_bf] = df['DBtw'].loc[mask_bf].apply(lambda r: term_spread(r, *popt_bf))

        # After CutDate 그룹 연산 (~ 연산자로 반대 타겟 지정)
        if (~mask_bf).any():
            spread_series.loc[~mask_bf] = df['DBtw'].loc[~mask_bf].apply(lambda r: term_spread(r, *popt_af))

        # 보정치 차감 처리
        df['E_ROE'] = df['E_ROE'] - spread_series.fillna(0)

        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 3)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
        hist_train['abs_Error'] = hist_train['Error'].abs()

        grouped_errors = hist_train.groupby([q_basis, 'Security'])['abs_Error'].mean()
        available_Qs = set(grouped_errors.index.get_level_values(0))

        Q_result = []

        # Q 레벨 루프만 유지 (Security 루프는 완전 제거됨)
        for Q in df[q_basis].unique():
            df_Q = df[df[q_basis] == Q].copy()

            if Q in available_Qs:
                # 해당 Q의 애널리스트별 평균 에러 추출 (Series 형태)
                q_errors = grouped_errors.xs(Q, level=0)

                # 현재 데이터에 존재하는 애널리스트만 필터링
                unique_securities = df_Q['Security'].unique()
                q_errors = q_errors[q_errors.index.isin(unique_securities)]

                if not q_errors.empty:
                    # 과거 에러 데이터가 존재하는 Row들만 필터링
                    valid_df_Q = df_Q[df_Q['Security'].isin(q_errors.index)].copy()

                    if len(valid_df_Q) >= min_count:
                        # 초성능 킬러 포인트: .apply() 대신 .map() 사용
                        valid_df_Q['PrevError'] = valid_df_Q['Security'].map(q_errors)
                        Q_result.append(valid_df_Q)
                        continue

            # 데이터가 없거나 기준 충족 못할 시 Fallback 처리
            df_Q['PrevError'] = 1.0
            Q_result.append(df_Q)

        if not Q_result:
            return pd.DataFrame()

        # 데이터 병합 및 가중치 계산 (Vectorized 연산)
        df = pd.concat(Q_result)
        df['PrevError'] += 0.01

        df['I_PrevError'] = 1.0 / df['PrevError']
        df_mean = df['I_PrevError'].mean()
        df_std = df['I_PrevError'].std()

        # 아웃라이어 클리핑 및 가중 ROE 계산
        df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 3 * df_std, upper=df_mean + 3 * df_std)
        df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']

        agg_df = df.groupby(q_basis).agg({
            'W_E_ROE': 'sum',
            'I_PrevError': 'sum',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 딕셔너리 스타일 구조 매핑으로 pd.concat 오버헤드 방지
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['W_E_ROE'] / agg_df['I_PrevError']
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __BAM_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]

        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 부울 마스킹을 통한 Before/After 분기 연산 최적화
        # ----------------------------------------------------------------
        mask_bf = df['Date'] <= df['CutDate']
        spread_series = pd.Series(0.0, index=df.index)

        # Before CutDate 그룹 연산
        if mask_bf.any():
            # 만약 term_spread가 벡터화를 지원한다면: term_spread(df[mask_bf], *popt_bf) 가 베스트
            spread_series.loc[mask_bf] = df['DBtw'].loc[mask_bf].apply(lambda r: term_spread(r, *popt_bf))

        # After CutDate 그룹 연산 (~ 연산자로 반대 타겟 지정)
        if (~mask_bf).any():
            spread_series.loc[~mask_bf] = df['DBtw'].loc[~mask_bf].apply(lambda r: term_spread(r, *popt_af))

        # 보정치 차감 처리
        df['E_ROE'] = df['E_ROE'] - spread_series.fillna(0)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 과거 11개년 기록 단 한 번만 미리 도려내기
        # ----------------------------------------------------------------
        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 11)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        coeff_map = {}

        if not hist_train.empty:
            # 중복 제거
            hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()

            # ----------------------------------------------------------------
            # OPTIMIZATION 2: LinearRegression 라이브러리를 대체하는 수학적 벡터화 (OLS)
            # fit_intercept=False 일 때, Slope = sum(X*Y) / sum(X^2)
            # ----------------------------------------------------------------
            hist_train['XY'] = hist_train['E_ROE'] * hist_train['A_ROE']
            hist_train['X2'] = hist_train['E_ROE'] ** 2

            # 모든 Q에 대한 필요한 통계량을 단 한 번의 groupby로 집계
            agg_hist = hist_train.groupby(q_basis).agg(
                sum_xy=('XY', 'sum'),
                sum_x2=('X2', 'sum'),
                unique_years=('Year', 'nunique')
            )

            # 조건 검증 (데이터 개수 조건 및 분모가 0이 아닌지 체크)
            valid_mask = (agg_hist['unique_years'] >= min_count) & (agg_hist['sum_x2'] > 0)

            # 기본값은 Slope=1, Intercept=0으로 세팅 후 유효한 값만 연산
            agg_hist['Slope'] = 1.0
            agg_hist.loc[valid_mask, 'Slope'] = agg_hist.loc[valid_mask, 'sum_xy'] / agg_hist.loc[valid_mask, 'sum_x2']
            agg_hist['Intercept'] = 0.0

            # 빠른 조회를 위해 딕셔너리로 변환
            coeff_map = agg_hist[['Slope', 'Intercept']].to_dict('index')

            # ----------------------------------------------------------------
            # OPTIMIZATION 3: 후반부 수많은 Groupby를 단 1번으로 일괄 집계
            # ----------------------------------------------------------------
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # ----------------------------------------------------------------
        # OPTIMIZATION 4: apply(axis=1) 제거 및 고속 매핑 보정 연산
        # ----------------------------------------------------------------
        # 각 Q에 맞는 Slope와 Intercept를 C-Level 속도로 매핑
        slopes = agg_df.index.map(lambda q: coeff_map.get(q, {'Slope': 1.0})['Slope'])
        intercepts = agg_df.index.map(lambda q: coeff_map.get(q, {'Intercept': 0.0})['Intercept'])

        # 결과 데이터프레임 구축 (BAM 보정 수식을 인라인 벡터 연산으로 처리)
        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE'] * slopes + intercepts  # apply_bam 함수가 y = ax + b 구조일 때 기준
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data

    @staticmethod
    def __IMC_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        #model = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]
        q_basis = x[4]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if df.empty:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', q_basis], keep='last')
        df['E_ROE_o'] = df['E_ROE'].copy()

        res = Enhanced_EPS.__fill_missing_est__(df, q_basis)
        df = pd.concat([df, res])

        df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
        df['SecAnl'] = df['Security'] + df['CoreAnalyst']

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 부울 마스킹을 통한 Before/After 분기 연산 최적화
        # ----------------------------------------------------------------
        mask_bf = df['Date'] <= df['CutDate']
        spread_series = pd.Series(0.0, index=df.index)

        # Before CutDate 그룹 연산
        if mask_bf.any():
            # 만약 term_spread가 벡터화를 지원한다면: term_spread(df[mask_bf], *popt_bf) 가 베스트
            spread_series.loc[mask_bf] = df['DBtw'].loc[mask_bf].apply(lambda r: term_spread(r, *popt_bf))

        # After CutDate 그룹 연산 (~ 연산자로 반대 타겟 지정)
        if (~mask_bf).any():
            spread_series.loc[~mask_bf] = df['DBtw'].loc[~mask_bf].apply(lambda r: term_spread(r, *popt_af))

        # 보정치 차감 처리
        df['E_ROE'] = df['E_ROE'] - spread_series.fillna(0)

        # ----------------------------------------------------------------
        # OPTIMIZATION 1: 과거 11개년 기록 단 한 번만 슬라이싱 및 필터링
        # ----------------------------------------------------------------
        target_code = df['Code'].iloc[0]
        target_year = int(df['Year'].iloc[0])
        year_min = str(target_year - 11)
        year_max = str(target_year - 1)

        hist_train = train[(train['Code'] == target_code) &
                           (train['Year'] >= year_min) &
                           (train['Year'] <= year_max)]

        # 기본 매핑 딕셔너리 초기화
        s_slope_map = {}
        q_slope_map = {}

        if not hist_train.empty:
            hist_train = hist_train.drop_duplicates(subset=['E_ROE', 'Security', 'Year', q_basis]).copy()
            hist_train['CoreAnalyst'] = hist_train['Analyst'].str.split(',').str[0]
            hist_train['SecAnl'] = hist_train['Security'] + hist_train['CoreAnalyst']

            # 현재 분석 타겟에 존재하는 애널리스트 데이터만 남겨 서치 오버헤드 최소화
            hist_train = hist_train[hist_train['SecAnl'].isin(df['SecAnl'].unique())]

            # ----------------------------------------------------------------
            # OPTIMIZATION 2: [1단계] Analyst 레벨 기울기 일괄 계산 (이중 루프 제거)
            # fit_intercept=False 일 때, Slope = sum(X*Y) / sum(X^2)
            # ----------------------------------------------------------------
            hist_train['XY'] = hist_train['E_ROE'] * hist_train['A_ROE']
            hist_train['X2'] = hist_train['E_ROE'] ** 2

            # Q와 SecAnl 조합으로 단 한 번에 그룹 집계
            s_agg = hist_train.groupby([q_basis, 'SecAnl']).agg(
                sum_xy=('XY', 'sum'),
                sum_x2=('X2', 'sum'),
                row_cnt=('XY', 'count'),
                nyear=('Year', 'nunique')
            )

            s_agg['Slope'] = 1.0
            valid_s = (s_agg['row_cnt'] >= 10) & (s_agg['nyear'] >= min_count) & (s_agg['sum_x2'] > 0)
            s_agg.loc[valid_s, 'Slope'] = s_agg.loc[valid_s, 'sum_xy'] / s_agg.loc[valid_s, 'sum_x2']
            s_slope_map = s_agg['Slope'].to_dict()

            # ----------------------------------------------------------------
            # OPTIMIZATION 3: [2단계] Company(Q) 레벨 기울기 일괄 계산
            # 1단계에서 계산된 Analyst Slope를 적용한 뒤 다시 OLS 수행
            # ----------------------------------------------------------------
            # MultiIndex 매핑을 통해 가중치 적용 (.apply 제거)
            hist_train['S_Slope'] = hist_train.set_index([q_basis, 'SecAnl']).index.map(s_slope_map).fillna(1.0)
            hist_train['E_ROE_adj'] = hist_train['E_ROE'] * hist_train['S_Slope']

            hist_train['XY_adj'] = hist_train['E_ROE_adj'] * hist_train['A_ROE']
            hist_train['X2_adj'] = hist_train['E_ROE_adj'] ** 2

            q_agg = hist_train.groupby(q_basis).agg(
                sum_xy_adj=('XY_adj', 'sum'),
                sum_x2_adj=('X2_adj', 'sum'),
                nyear_q=('Year', 'nunique')
            )

            q_agg['Slope'] = 1.0
            valid_q = (q_agg['nyear_q'] >= min_count) & (q_agg['sum_x2_adj'] > 0)
            q_agg.loc[valid_q, 'Slope'] = q_agg.loc[valid_q, 'sum_xy_adj'] / q_agg.loc[valid_q, 'sum_x2_adj']
            q_slope_map = q_agg['Slope'].to_dict()

            # ----------------------------------------------------------------
            # OPTIMIZATION 4: 현재 예측 대상 데이터(df)에 보정치 초고속 적용
            # ----------------------------------------------------------------
            # 1단계: Analyst 레벨 보정 (.apply 대신 .map 사용)
        df['S_Slope'] = df.set_index([q_basis, 'SecAnl']).index.map(s_slope_map).fillna(1.0)
        df['E_ROE'] = df['E_ROE'] * df['S_Slope']

        # 후반부 모든 무거운 Groupby 연산을 단 1번으로 일괄 처리
        agg_df = df.groupby(q_basis).agg({
            'E_ROE': 'mean',
            'E_ROE_o': 'mean',
            'A_EPS_1': 'last',
            'BPS': 'last',
            'EQBtw': 'mean'
        })

        if agg_df.empty:
            return pd.DataFrame()

        # 2단계: Company 레벨 보정 (최종 데이터프레임 구축 과정에서 인라인 연산)
        q_slopes = agg_df.index.map(lambda q: q_slope_map.get(q, 1.0))

        data = pd.DataFrame(index=agg_df.index)
        data['Est'] = agg_df['E_ROE'] * q_slopes  # 별도의 apply_bam 함수 호출 없이 벡터 연산 처리
        data['EW'] = agg_df['E_ROE_o']
        data['EW_prev'] = agg_df['A_EPS_1'] / agg_df['BPS']
        data['EQBtw'] = np.round(agg_df['EQBtw'])

        # 최종 포맷팅
        data = Enhanced_EPS.__resform__(data, code, df)
        data['FY'] = symbol[-6:]

        return data
