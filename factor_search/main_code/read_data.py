import os
import sys
import pandas as pd
import numpy as np

package2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(package2_path)
from path import Path


class Read_data():
    """
    Summary: this class is used to read feature calculation results: 
    
    Attributes:
        df_data (DataFrame): dataframe stored the A value V.S. B value
    
    Methods:
        read_data()
    """
    def __init__(self):
        self.pro_path = Path


    def _cache_method1_result(self):
        if not hasattr(self, "_origin_list") or not hasattr(self, "_new_list"):
            origin_list, new_list = self.convert_columns()
            self._origin_list = origin_list
            self._new_list = new_list
        return self._origin_list, self._new_list


    def convert_columns(self):
        import json
        with open(self.pro_path+'/factor_search/config/columns.json', 'r') as config_file:
            config_data = json.load(config_file)
        return config_data['origin_columns'], config_data['new_columns']


    def read_future_data(self, data_path):

        origin_list, new_list = self._cache_method1_result()
        df_oneday = pd.read_parquet(data_path,columns = origin_list)
        df_oneday.columns = new_list
        df_oneday['TradingDate'] = pd.to_datetime(df_oneday['TradingDate'],format='%Y%m%d')
        df_oneday['TradingTime'] = pd.to_datetime(df_oneday['TradingTime'])

        for c in df_oneday.columns:
            try:
                df_oneday[c] = df_oneday[c].astype(np.float32)
            except:
                continue
        
        return df_oneday
    