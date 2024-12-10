import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.funs import *
import warnings
warnings.filterwarnings('ignore')

import pymysql
from sqlalchemy import create_engine
import configparser

class Enhanced_EPS(object):
    def __init__(self):
        self.ucurve = dict()
        self.min_count = 5
        self.year_range = 10

    #set base data for calculation
    def set_data(self, train):
        self.train = train

    def calc_ucurve(self, country, prdYear, train):
        listoftable = list(product(train.SectorClass.unique(), prdYear))
        for table in tqdm(listoftable):
            self.ucurve[table] = term_spread_adj(*table, train=train)

        pd.DataFrame(self.ucurve).to_json(f'result/{country}/ucurve.json')

    def calc(self, UniqueSymbol, model_name):
        if model_name == 'IMC_adp':
            dataset = self.calc_IMC_adp(UniqueSymbol)
        elif model_name == 'EW':
            dataset = self.calc_EW(UniqueSymbol)
        elif model_name == 'EW_adp':
            dataset = self.calc_EW_adp(UniqueSymbol)
        elif model_name == 'IMSE_adp':
            dataset = self.calc_IMSE_adp(UniqueSymbol)
        else:
            raise('Invalid model name')

        dataset_pd = pd.concat(dataset)
        dataset_pd.index.name = 'QBtw'
        dataset_pd = dataset_pd.reset_index()
        dataset_pd = dataset_pd.set_index('Code')

        return dataset_pd

    def calc_quantile(self, dataset_pd, model_name):
        counter = 8
        while True:
            try:
                dataset_pd['Q_GEst'] = dataset_pd.groupby(['QBtw']).GEst.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1

        if model_name == 'EW':
            return dataset_pd

        # get quantile of shock
        dataset_pd['MinMax'] = dataset_pd.groupby(['Code']).Shock.max() - dataset_pd.groupby(
            ['Code']).Shock.min().values

        counter = 8
        while True:
            try:
                dataset_pd['Q_Shock'] = dataset_pd.groupby(['QBtw']).Shock.transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1
        counter = 8
        while True:
            try:
                dataset_pd['Q_Vol'] = dataset_pd.groupby(['QBtw', 'Q_Shock'])['MinMax'].transform(lambda x: pd.qcut(x, counter, labels=[str(x) for x in range(counter)]))
                break
            except:
                counter -= 1

        return dataset_pd

    def calc_IMC_adp(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        return process_map(IMC_adp_cache, multi_arg, max_workers=os.cpu_count() - 1)


    def calc_IMSE_adp(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        return process_map(IMSE_adp_cache, multi_arg, max_workers=os.cpu_count() - 1)


    def calc_EW_adp(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        return process_map(EW_adp_cache, multi_arg, max_workers=os.cpu_count() - 1)


    def calc_EW(self, UniqueSymbol):

        multi_arg = list(product(UniqueSymbol, [self.min_count], [self.year_range], [self.ucurve]))

        return process_map(EW_cache, multi_arg, max_workers=os.cpu_count() - 1)
