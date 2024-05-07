import pandas as pd
import numpy as np
import sys
import os

package2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(package2_path)
from path import Path
file_path = Path

sys.path.append("./main_code")
from data_process import Data
from feature_cal import A_feature, B_feature, C_feature, D_feature, Special_future_feature
from feature_derive import Feature_derive

class Analysis(A_feature,B_feature,C_feature,D_feature,Special_future_feature,Data,Feature_derive):
    """
    Summary: this class is used to calculate: 
        1) factor value of different type; 
        2) return series of stock/future;
        3) IC value between above two variables;
        4) IC results analysis.
    
    Attributes:
        data (DataFrame): inherit from Data class, if None, data will be read from remote Database.
        data_underlying (str): data type, one of stock and future.
        data_level (int): data detail of price/volume info, stock is 10 and future is 5.
        factot_back_time (float): aggregation window time when calculate factors, (unit: minute).
    Methods:
        cal_xfeatures(): calculate the value of factors, x in [a,b,c,d]
        cal_return(ticks, mins, return_type): calculate return series using Investment target price
        cal_ic(ticks, mins, return_type): calculate ic series using factors and return
    """

    
    def __init__(self,symbol_list:list=None,begin_date=None,end_date=None,begin_time=None,end_time=None,
                 database='SEL2_TAQ_h',data=None,data_underlying=None,data_level=None,\
                    tick_filled=False,factor_back_time=None, factor_derive=None):
        
        super().__init__(symbol_list,begin_date,end_date,begin_time,end_time,
                 database,data,data_underlying,data_level,tick_filled,factor_back_time)
        self.data_underlying = data_underlying
        self.factor_derive = factor_derive
    
    def loop_cal_factors(self, methods, methods_params):    
        # loop function to cal factors
        results = pd.DataFrame()
        for method in methods:
            method_func = getattr(self, method) 
            
            if any(method == item[0] for item in methods_params):
                for param in [param for param in methods_params if param[0] == method]:
                    try:
                        result = method_func(param[1])
                    except:
                        result = method_func(param[1], param[2])
                    results = pd.concat([results, result], axis=1)

                    ###cal derive factor and add result to df
                    if self.factor_derive:
                        for method_name in dir(Feature_derive):
                            derive_method = getattr(self, method_name)
                            if callable(derive_method) and "_derive" in method_name:
                                derive_result = derive_method(result)
                                results = pd.concat([results, derive_result], axis=1)

                    results = results.loc[:, ~results.columns.duplicated()].round(6)
                    
            else:
                result = method_func() 
                results = pd.concat([results, result], axis=1)

                ### cal derive factor and add result to df
                if self.factor_derive:
                    for method_name in dir(Feature_derive):
                        derive_method = getattr(self, method_name)
                        if callable(derive_method) and "_derive" in method_name:
                            derive_result = derive_method(result)
                            results = pd.concat([results, derive_result], axis=1)

                results = results.loc[:, ~results.columns.duplicated()].round(6)
        
        print(results.shape)
        return results
        

    def cal_feature(self, feature_type):
        'Calculate all Class features'

        import json
        with open(file_path+'/factor_search/config/feature_method.json', 'r') as config_file:
            config_data = json.load(config_file)

            if feature_type == "All":
                methods=[]
                methods_params=[]
                for type_str in ['A','B','C']:
                    methods += config_data['%s_methods'%type_str]
                    methods_params += config_data['%s_methods_params'%type_str]
            else:        
                methods = config_data['%s_methods'%feature_type]
                methods_params = config_data['%s_methods_params'%feature_type]
        
        return self.loop_cal_factors(methods, methods_params)  



    def cal_future_special(self):
        
        methods = [ 'cal_trade_volume',
                    'cal_trade_amount',
                    'cal_total_positon',
                    'cal_prepositon_change',
                    'cal_change',
                    'cal_change_ratio',
                    'cal_position_change',
                    'cal_average_change',
                    'cal_order_rate',
                    'cal_order_diff',
                    'cal_amplitude',
                    'cal_volrate']
        
        methods_params = []                      
            
        # loop function to cal factors
        return self.loop_cal_factors(methods, methods_params)  
    
    
    def cal_return(self, ticks=1, mins=None, return_type='absolute_spread'): 
        """
        summary: calculate return at oneday level

        Args:
            :df (pd.DataFrame): data of investment target, including TradingDate,Symbol,LastPrice
            :ticks (int): decay tick number, under filled tick condition, 
                        Future: 1、3、5min -> ticks num: 120, 360, 600; 
                        Stock:  1、3、5min -> ticks num: 20, 60, 100.
            :mins (float): use minute cal, for future: 15s=0.25min=30tick, 30s=0.5min=60tick
        
        Returns:
            type: dataframe, including TradingTime, Symbol, return
        """
        
        df = self.df[['Symbol','TradingDate','TradingTime','LastPrice']]
        # group according symbol and date
        grouped_df = df.groupby(['TradingDate', 'Symbol'])

        result_df = pd.DataFrame()

        for (date, symbol), df1 in grouped_df:  # df1 is data in one day

            if mins is None: # at tick level
                df1['LastPrice_post'] = df1['LastPrice'].shift(-ticks)
                
            else: # at minute level
                time_unfilled = df1['TradingTime']
                df1 = self.future_tick_filled(data=df1)
                
                if self.data_underlying == 'future':
                    df1['LastPrice_post'] = df1['LastPrice'].shift(-int(mins*120))
                elif self.data_underlying == 'stock':
                    df1['LastPrice_post'] = df1['LastPrice'].shift(-int(mins*20))

                df1 = df1.loc[df1['TradingTime'].isin(time_unfilled)]

            if return_type == 'absolute_spread':
                df1['return'] = df1['LastPrice_post'] - df1['LastPrice']  
            else:
                df1['return'] = df1['LastPrice_post'] / df1['LastPrice'] - 1 # 'simple return'

            result_df = pd.concat([result_df, df1])[['Symbol','TradingTime','return']].dropna()

        return result_df.reset_index(drop=True)
    

    def cal_ic(self,feature_df,return_df,ic_type='rank_ic'):
        """
        summary: calculate IC at oneday level, delete nan factor value with calc IC of each factor 

        Args:
            :feature_df (pd.DataFrame): 
            :return_df (pd.DataFrame): 

        Returns:
            type: dataframe, including TradingTime, Symbol, IC value for each factor
        """

        # feature_df = feature_df.dropna()
        feature_df = feature_df.loc[feature_df['TradingTime'].isin(return_df['TradingTime'])]
        return_df = return_df.loc[return_df['TradingTime'].isin(feature_df['TradingTime'])]# for factor used dropna()
        column_list = feature_df.columns.drop(['Symbol','TradingTime'])
        result_df = pd.DataFrame()
        
        x = return_df['return'].reset_index(drop=True)
        y = feature_df.reset_index(drop=True)
        

        result_df['Symbol'] = [y['Symbol'].iloc[0]]
        result_df['TradingDate'] = [y['TradingTime'].dt.date.iloc[0]]

        if ic_type == 'rank_ic':
            x = x.rank(method='dense')
            y = y.drop(['Symbol','TradingTime'],axis=1).rank(method='dense')

        for column in column_list:
                
                # delete nan factor number
                return_data = x[y[column].notna()]
                factor_one = y[column].dropna()

                result_df[column] = [np.corrcoef(factor_one, return_data)[0, 1]]

        return_df.reset_index(drop=True, inplace=True)
        
        return result_df
    
    
    def cal_icmean(self,ic_df,window): 
    
        column_list = ic_df.columns.drop(['Symbol','TradingDate'])
        df = ic_df.set_index('TradingDate', inplace=False)

        result_df = pd.DataFrame()
        begin_time = df.index[0]
        for row in df.itertuples():
            index = row.Index
            next_time = index - pd.Timedelta(days=window)  #Calculate absolute time intervals, ignoring insufficient time intervals

            if next_time<begin_time:
                continue

            df_rolling = df.loc[next_time:index]

            result_each = pd.DataFrame()
            result_each['Symbol'] = [df_rolling['Symbol'].iloc[-1]]
            result_each['TradingDate'] = [index]

            for column in column_list:
                    mean = np.mean(df_rolling[column].values)
                    result_each[column] = [mean]

            result_df = pd.concat([result_df,result_each],ignore_index=True)
        return result_df 
    
    
    def cal_icstd(self,ic_df,window): 
    
        column_list = ic_df.columns.drop(['Symbol','TradingDate'])
        df = ic_df.set_index('TradingDate', inplace=False)

        result_df = pd.DataFrame()
        begin_time = df.index[0]
        for row in df.itertuples():
            index = row.Index
            next_time = index - pd.Timedelta(days=window)  #Calculate absolute time intervals, ignoring insufficient time intervals

            if next_time<begin_time:
                continue

            df_rolling = df.loc[next_time:index]

            result_each = pd.DataFrame()
            result_each['Symbol'] = [df_rolling['Symbol'].iloc[-1]]
            result_each['TradingDate'] = [index]

            for column in column_list:
                    std = np.std(df_rolling[column].values)
                    result_each[column] = [std]

            result_df = pd.concat([result_df,result_each],ignore_index=True)
        return result_df  
    
    
    def cal_icir(self,ic_df,window): 
    
        column_list = ic_df.columns.drop(['Symbol','TradingDate'])
        df = ic_df.set_index('TradingDate', inplace=False)

        result_ir = pd.DataFrame()
        result_mean = pd.DataFrame()
        
        begin_time = df.index[0]
        for row in df.itertuples():
            index = row.Index
            next_time = index - pd.Timedelta(days=window)  #Calculate absolute time intervals, ignoring insufficient time intervals

            if next_time<begin_time:
                continue 

            df_rolling = df.loc[next_time:index]

            result_each = pd.DataFrame()
            result_each['Symbol'] = [df_rolling['Symbol'].iloc[-1]]
            result_each['TradingDate'] = [index]

            for column in column_list:
                    mean = np.mean(df_rolling[column].values)
                    std = np.std(df_rolling[column])
                    result_each[column] = [mean/std]

            result_df = pd.concat([result_df,result_each],ignore_index=True)
            
        return result_df