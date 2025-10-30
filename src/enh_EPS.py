import pickle
from multiprocessing.shared_memory import SharedMemory
from typing import List

from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import *
import warnings
warnings.filterwarnings('ignore')


class Enhanced_EPS(object):
    def __init__(self):
        self.ucurve = dict()
        self.min_count = 2
        self.year_range = 10

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

    @staticmethod
    def __term_spread_adj__(sector, fy, train):

        train = train.dropna(subset=['Error'])
        if 'Q' in fy:
            currYear = int('20'+fy[2:4])
        else:
            currYear = int(fy)
        sector = sector

        # if sector is na
        if pd.isna(sector):
            prev_data_bf = train[(train.Year <= str(currYear - 2)) & (train.Year >= str(currYear - 11))]
            prev_data_af = train[(train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 10))]
        else:
            prev_data_bf = train[
                (train.SectorClass == sector) & (train.Year <= str(currYear - 2)) & (train.Year >= str(currYear - 11))]
            prev_data_af = train[
                (train.SectorClass == sector) & (train.Year <= str(currYear - 1)) & (train.Year >= str(currYear - 10))]

        num_of_obs = 5

        # case for data which announced before previous year's actual data
        if len(prev_data_bf) < 20:
            popt_bf = np.array([np.nan] * num_of_obs)
        else:
            try:
                prev_data_bf = Enhanced_EPS.__filter_outlier__(prev_data_bf, 'Code', 'Error')
                popt_bf, pcov_bf, _, _, _ = curve_fit(
                    term_spread
                    , prev_data_bf
                    , prev_data_bf.Error
                    , method='trf'
                    , p0=[0, 0.01, 0.01, 0.01, 1]  # b0, c, b1, b2, lam
                    , bounds=((-1, -np.inf, -np.inf, -np.inf, -np.inf), (1, np.inf, np.inf, np.inf, np.inf))
                    , full_output=True
                )
                if pcov_bf[0, 0] == np.inf:
                    raise Exception
            except:
                popt_bf = np.array([np.nan] * num_of_obs)

        # case for data which announced after previous year's actual data
        if len(prev_data_af) < 20:
            popt_af = np.array([np.nan] * num_of_obs)
        else:
            try:
                popt_af, pcov_af, _, _, _ = curve_fit(
                    term_spread
                    , xdata=prev_data_bf
                    , ydata=prev_data_bf.Error
                    , method='trf'
                    , p0=[0, 0.01, 0.01, 0.01, 1]  # b0, c, b1, b2, lam
                    , bounds=((-1, -np.inf, -np.inf, -np.inf, -np.inf), (1, np.inf, np.inf, np.inf, np.inf))
                    , full_output=True
                )
                if pcov_af[0, 0] == np.inf:
                    raise Exception
            except:
                popt_af = np.array([np.nan] * num_of_obs)

        return {'popt_bf': popt_bf, 'popt_af': popt_af}

    def calc_ucurve(self, country, prdFY, train, reuse:bool=False):
        if reuse:
            self.ucurve = pd.read_json(f'result/{country}/ucurve.json').T.to_dict(orient='index')
            return True

        # add np.nan to train.SectorClass.unique
        unique_sectors = set(train.SectorClass.unique().tolist() + [np.nan])

        listoftable = list(product(unique_sectors, prdFY))
        for table in tqdm(listoftable):
            self.ucurve[str(table[0])+str(table[1])] = term_spread_adj(*table, train=train)

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
            multiproclist = list(product(ncal_presym, model_list, [self.shm_train], [self.shm_ucurve]))
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
            symwithbm.append([sym, bestmodel.get(sym, 'EW'), self.shm_train, self.shm_ucurve])
        result = process_map(Enhanced_EPS.__calc__, symwithbm, max_workers=os.cpu_count()-1)

        return pd.concat(result)

    def calc_model(self, UniqueSymbol:List, model_name:str):
        if len(UniqueSymbol)==0:
            return pd.DataFrame()

        # set shared memory
        _ = self.__save_memory__()

        symwithbm = []
        for sym in UniqueSymbol:
            symwithbm.append([sym, model_name, self.shm_train, self.shm_ucurve])
        result = process_map(Enhanced_EPS.__calc__, symwithbm, max_workers=os.cpu_count()-1)

        return pd.concat(result)

    @staticmethod
    def __calc__(x):
        UniqueSymbol = x[0]
        model_name = x[1]
        shm_train = x[2]
        shm_ucurve = x[3]

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
        shm_train = x[-2]
        shm_ucurve = x[-3]

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol==symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()


        est = df.groupby('QBtw')['E_ROE'].mean()

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, est, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __PBest__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-3]

        star_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol==symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 4))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            # list to append previous year's error rate by analyst
            tempset = []

            unique_sec = df.Security.unique()
            for sec in unique_sec:
                df_sec = tempdata[tempdata.Security == sec]
                if len(df_sec) > 0:
                    df_sec_error = df_sec['Error'].abs().mean()
                    tempset.append([sec, df_sec_error])

            # if previous year's data exist, calculate smart consensus
            if len(tempset) > 0:
                prev_error = pd.DataFrame(tempset, columns=['Security', 'Error']).set_index('Security')
                # if prev_year's anaylst data is not enough(less than 5 data point), append all
                check_star_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.nsmallest(star_count, 'Error').index))]
                if len(check_star_count) < 2:
                    Q_result.append(df[df.QBtw == Q])
                else:
                    Q_result.append(check_star_count)
            else:
                Q_result.append(df[df.QBtw == Q])

        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())
        if len(Q_result) > 0:
            df = pd.concat(Q_result)

        est = df.groupby('QBtw')['E_ROE'].mean()

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __IMSE__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-3]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            # list to append previous year's error rate by analyst
            tempset = []

            unique_sec = df.Security.unique()
            for sec in unique_sec:
                df_sec = tempdata[tempdata.Security == sec]
                if len(df_sec) > 0:
                    df_sec_error = df_sec['Error'].abs().mean()
                    tempset.append([sec, df_sec_error])

            # if previous year's data exist, calculate smart consensus
            if len(tempset) > 0:
                prev_error = pd.DataFrame(tempset, columns=['Security', 'Error']).set_index('Security')
                # if prev_year's anaylst data is not enough(less than 5 data point), append all
                check_prev_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.index))]
                if len(check_prev_count) < min_count:
                    check_prev_count = df[df.QBtw == Q]
                    check_prev_count['PrevError'] = 1
                    Q_result.append(check_prev_count)
                else:
                    check_prev_count['PrevError'] = check_prev_count.apply(
                        lambda x: prev_error.loc[x.Security].values[0], axis=1)
                    Q_result.append(check_prev_count)
            else:
                check_prev_count = df[df.QBtw == Q]
                check_prev_count['PrevError'] = 1
                Q_result.append(check_prev_count)

        if len(Q_result) > 0:
            df = pd.concat(Q_result)
            df.PrevError += 0.01

        df['I_PrevError'] = df['PrevError'].pow(-1)
        df_mean = df['I_PrevError'].mean()
        df_std = df['I_PrevError'].std()
        df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 3 * df_std, upper=df_mean + 3 * df_std)
        df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']

        est = df.groupby('QBtw')['W_E_ROE'].sum() / df.groupby('QBtw')['I_PrevError'].sum()
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __BAM__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-3]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            # list to append previous year's error rate by analyst
            lenYear = len(tempdata.Year.unique())
            if lenYear >= min_count:
                # Linear Regression between E_ROE and A_ROE
                try:
                    lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']), tempdata['A_ROE'])
                    slope = lr_result.coef_[0]
                    intercept = lr_result.intercept_
                except:
                    slope = 1
                    intercept = 0
            else:
                slope = 1
                intercept = 0

            Q_result.append([Q, slope, intercept])

        coeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
        # with slope and intercept, calculate BAM
        est = df.groupby('QBtw')[['E_ROE']].mean()
        est = est.apply(lambda x: apply_bam(x, coeffset), axis=1)
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __IMC__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-3]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()
        df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
        df['SecAnl'] = df['Security'] + df['CoreAnalyst']

        year = symbol[-6:-2]
        if 'Q' in year:
            quarter = year[:2]
        else:
            quarter = 'NA'

        Q_result = []
        S_result = []

        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            if len(tempdata) > 0:
                tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
                tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

                # polyfit E_ROE with A_ROE per analyst
                tempset = []
                for S in df.SecAnl.unique():
                    temp = tempdata[tempdata.SecAnl == S]
                    if len(temp) >= 10 and len(temp.Year.unique()) >= min_count:
                        try:
                            lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),
                                                                                  tempdata['A_ROE'])
                            slope = lr_result.coef_[0]
                            intercept = lr_result.intercept_
                        except:
                            slope = 1
                            intercept = 0
                    else:
                        slope = 1
                        intercept = 0
                    tempset.append([S, Q, slope, intercept])

                # get S by S data
                SQ_df = pd.DataFrame(tempset, columns=['S', 'Q', 'Slope', 'Intercept']).set_index(['S', 'Q'])
                S_result.append(SQ_df)

                # polyfit E_ROE with A_ROE per company
                tempdata.E_ROE = tempdata.apply(lambda x: apply_imc(x, SQ_df), axis=1)

            # list to append previous year's error rate by analyst
            lenYear = len(tempdata.Year.unique())
            if lenYear >= min_count:
                # Linear Regression between E_ROE and A_ROE
                try:
                    lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),
                                                                          tempdata['A_ROE'])
                    slope = lr_result.coef_[0]
                    intercept = lr_result.intercept_
                except:
                    slope = 1
                    intercept = 0
            else:
                slope = 1
                intercept = 0

            Q_result.append([Q, slope, intercept])

        if len(S_result) > 0:
            Scoeffset = pd.concat(S_result)
            df['E_ROE'] = df.apply(lambda x: apply_imc(x, Scoeffset), axis=1)

        Qcoeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
        # with slope and intercept, calculate BAM
        est = df.groupby('QBtw')[['E_ROE']].mean()
        est = est.apply(lambda x: apply_bam(x, Qcoeffset), axis=1)
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __EW_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-1]

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        df['E_ROE'] = (df['E_ROE']
                       - df.apply(lambda x:
                                  term_spread(x, *popt_bf)
                                  if x.Date <= x.CutDate
                                  else term_spread(x, *popt_af)
                                  , axis=1).fillna(0))

        est = df.groupby('QBtw')['E_ROE'].mean()
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        estEW_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, estEW_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __PBest_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-1]

        star_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        df['E_ROE'] = (df['E_ROE']
                       - df.apply(lambda x:
                                  term_spread(x, *popt_bf)
                                  if x.Date <= x.CutDate
                                  else term_spread(x, *popt_af)
                                  , axis=1).fillna(0))

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 4))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            if Q < 4:
                if popt_af[0] == 0:
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
            else:
                if popt_bf[0] == 0:
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
            tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

            # list to append previous year's error rate by analyst
            tempset = []

            unique_sec = df.Security.unique()
            for sec in unique_sec:
                df_sec = tempdata[tempdata.Security == sec]
                if len(df_sec) > 0:
                    df_sec_error = df_sec['Error'].abs().mean()
                    tempset.append([sec, df_sec_error])

            # if previous year's data exist, calculate smart consensus
            if len(tempset) > 0:
                prev_error = pd.DataFrame(tempset, columns=['Security', 'Error']).set_index('Security')
                # if prev_year's anaylst data is not enough(less than 5 data point), append all
                check_star_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.nsmallest(star_count, 'Error').index))]
                if len(check_star_count) < 2:
                    Q_result.append(df[df.QBtw == Q])
                else:
                    Q_result.append(check_star_count)
            else:
                Q_result.append(df[df.QBtw == Q])

        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())
        if len(Q_result) > 0:
            df = pd.concat(Q_result)

        est = df.groupby('QBtw')['E_ROE'].mean()

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __IMSE_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-1]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        df['E_ROE'] = (df['E_ROE']
                       - df.apply(lambda x:
                                  term_spread(x, *popt_bf)
                                  if x.Date <= x.CutDate
                                  else term_spread(x, *popt_af)
                                  , axis=1).fillna(0))

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 3))]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            if Q < 4:
                if popt_af[0] == 0:
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
            else:
                if popt_bf[0] == 0:
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
            tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

            # list to append previous year's error rate by analyst
            tempset = []

            unique_sec = df.Security.unique()
            for sec in unique_sec:
                df_sec = tempdata[tempdata.Security == sec]
                if len(df_sec) > 0:
                    df_sec_error = df_sec['Error'].abs().mean()
                    tempset.append([sec, df_sec_error])

            # if previous year's data exist, calculate smart consensus
            if len(tempset) > 0:
                prev_error = pd.DataFrame(tempset, columns=['Security', 'Error']).set_index('Security')
                # if prev_year's anaylst data is not enough(less than 5 data point), append all
                check_prev_count = df[(df.QBtw == Q) & (df.Security.isin(prev_error.index))]
                if len(check_prev_count) < min_count:
                    check_prev_count = df[df.QBtw == Q]
                    check_prev_count['PrevError'] = 1
                    Q_result.append(check_prev_count)
                else:
                    check_prev_count['PrevError'] = check_prev_count.apply(
                        lambda x: prev_error.loc[x.Security].values[0], axis=1)
                    Q_result.append(check_prev_count)
            else:
                check_prev_count = df[df.QBtw == Q]
                check_prev_count['PrevError'] = 1
                Q_result.append(check_prev_count)

        if len(Q_result) > 0:
            df = pd.concat(Q_result)
            df.PrevError += 0.01

        df['I_PrevError'] = df['PrevError'].pow(-1)
        # limit upper and lower bound of I_PrevError as +- 2 stdev
        df_mean = df['I_PrevError'].mean()
        df_std = df['I_PrevError'].std()
        df['I_PrevError'] = df['I_PrevError'].clip(lower=df_mean - 5 * df_std, upper=df_mean + 5 * df_std)
        df['W_E_ROE'] = df['E_ROE'] * df['I_PrevError']

        est = df.groupby('QBtw')['W_E_ROE'].sum() / df.groupby('QBtw')['I_PrevError'].sum()
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __BAM_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-1]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        df['E_ROE'] = (df['E_ROE']
                       - df.apply(lambda x:
                                  term_spread(x, *popt_bf)
                                  if x.Date <= x.CutDate
                                  else term_spread(x, *popt_af)
                                  , axis=1).fillna(0))

        Q_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                                 & (train.QBtw == Q)]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            if Q < 4:
                if pd.isna(popt_af[0]):
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
            else:
                if pd.isna(popt_bf[0]):
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (
                            tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
            tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

            # list to append previous year's error rate by analyst
            lenYear = len(tempdata.Year.unique())
            if lenYear >= min_count:
                # Linear Regression between E_ROE and A_ROE
                try:
                    lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']), tempdata['A_ROE'])
                    slope = lr_result.coef_[0]
                    intercept = lr_result.intercept_
                except:
                    slope = 1
                    intercept = 0
            else:
                slope = 1
                intercept = 0

            Q_result.append([Q, slope, intercept])

        coeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
        # with slope and intercept, calculate BAM
        est = df.groupby('QBtw')[['E_ROE']].mean()
        est = est.apply(lambda x: apply_bam(x, coeffset), axis=1)
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data

    @staticmethod
    def __IMC_adp__(x):
        symbol = x[0]
        code = symbol[:-6]
        shm_train = x[-2]
        shm_ucurve = x[-1]

        min_count = 5

        train = Enhanced_EPS.__load_memory__(shm_train)
        ucurve = Enhanced_EPS.__load_memory__(shm_ucurve)
        df = train[train.UniqueSymbol == symbol]
        if len(df) == 0:
            return pd.DataFrame()

        df = df.drop_duplicates(subset=['E_ROE', 'Security', 'QBtw'])
        df['E_ROE_o'] = df['E_ROE'].copy()
        df['CoreAnalyst'] = df.Analyst.str.split(',', expand=True)[0]
        df['SecAnl'] = df['Security'] + df['CoreAnalyst']

        year = symbol[-6:]
        sector = df.SectorClass.iloc[-1]
        popt = ucurve[sector+year]
        popt_bf = np.asarray(popt['popt_bf'], dtype=np.float32)
        popt_af = np.asarray(popt['popt_af'], dtype=np.float32)

        df['E_ROE'] = (df['E_ROE']
                       - df.apply(lambda x:
                                  term_spread(x, *popt_bf)
                                  if x.Date <= x.CutDate
                                  else term_spread(x, *popt_af)
                                  , axis=1).fillna(0))

        Q_result = []
        S_result = []
        for Q in df.QBtw.unique():
            if Q < 4:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 1))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 10))
                                ]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])
            else:
                tempdata = train[(train.Code == df.Code.iloc[0])
                                 & (train.Year <= str(int(df.Year.iloc[0]) - 2))
                                 & (train.Year >= str(int(df.Year.iloc[0]) - 11))
                                ]
                tempdata = tempdata.drop_duplicates(subset=['E_ROE', 'Security', 'Year', 'QBtw'])

            if Q < 4:
                if pd.isna(popt_af[0]):
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_af), axis=1).fillna(0))
            else:
                if pd.isna(popt_bf[0]):
                    tempdata = tempdata[(tempdata.QBtw == Q)]
                tempdata['E_ROE'] = (tempdata['E_ROE'] - tempdata.apply(lambda x: term_spread(x, *popt_bf), axis=1).fillna(0))
            tempdata['Error'] = tempdata['E_ROE'] - tempdata['A_ROE']

            if len(tempdata) > 0:
                tempdata['CoreAnalyst'] = tempdata.Analyst.str.split(',', expand=True)[0]
                tempdata['SecAnl'] = tempdata['Security'] + tempdata['CoreAnalyst']

                # polyfit E_ROE with A_ROE per analyst
                tempset = []
                for S in df.SecAnl.unique():
                    temp = tempdata[tempdata.SecAnl == S]
                    if len(temp) >= 10 and len(temp.Year.unique()) >= min_count:
                        try:
                            lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']),tempdata['A_ROE'])
                            slope = lr_result.coef_[0]
                            intercept = lr_result.intercept_
                        except:
                            slope = 1
                            intercept = 0
                    else:
                        slope = 1
                        intercept = 0
                    tempset.append([S, Q, slope, intercept])

                # get S by S data
                SQ_df = pd.DataFrame(tempset, columns=['S', 'Q', 'Slope', 'Intercept']).set_index(['S', 'Q'])
                S_result.append(SQ_df)

                # polyfit E_ROE with A_ROE per company
                tempdata.E_ROE = tempdata.apply(lambda x: apply_imc(x, SQ_df), axis=1)

            # list to append previous year's error rate by analyst
            lenYear = len(tempdata.Year.unique())
            if lenYear >= min_count:
                # Linear Regression between E_ROE and A_ROE
                try:
                    lr_result = LinearRegression(fit_intercept=False).fit(pd.DataFrame(tempdata['E_ROE']), tempdata['A_ROE'])
                    slope = lr_result.coef_[0]
                    intercept = lr_result.intercept_
                except:
                    slope = 1
                    intercept = 0
            elif lenYear == 0:
                slope = 1
                intercept = 0
            else:
                slope = 1
                intercept = 0

            Q_result.append([Q, slope, intercept])

        if len(S_result) > 0:
            Scoeffset = pd.concat(S_result)
            df['E_ROE'] = df.apply(lambda x: apply_imc(x, Scoeffset), axis=1)

        Qcoeffset = pd.DataFrame(Q_result, columns=['Q', 'Slope', 'Intercept']).set_index('Q')
        # with slope and intercept, calculate BAM
        est = df.groupby('QBtw')[['E_ROE']].mean()
        est = est.apply(lambda x: apply_bam(x, Qcoeffset), axis=1)
        estEW = pd.DataFrame(df.groupby('QBtw')['E_ROE_o'].mean())

        est_prev = pd.DataFrame(df.groupby('QBtw')['A_EPS_1'].last() / df.groupby('QBtw')['BPS'].last())

        eqbtw = np.round(pd.DataFrame(df.groupby('QBtw')['EQBtw'].mean()))

        data = pd.concat([est, estEW, est_prev, eqbtw], axis=1)
        data.columns = ['Est', 'EW', 'EW_prev', 'EQBtw']

        if len(data) == 0 or len(df) == 0:
            return pd.DataFrame()
        else:
            data = Enhanced_EPS.__resform__(data, code, df)
            data['FY'] = symbol[-6:]
            return data
