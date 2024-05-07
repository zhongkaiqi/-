import pandas as pd
import numpy as np


class Feature_derive:
    """ Summary: derive cross section factor
    
    Attributes:
        df: factor series

    Methods:
        derive method: Aggregation cross section factor (sum(), mean(), sum(abs())...)

    Return:
        df: derived factors df

    """
    def sum_derive(self, factor_df):
        """Aggregation method is sum"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_info = factor_df[factor_name[0]]
        sum_data = np.sum([factor_info.shift(i) for i in range(factor_delta_num)],axis=0)
        sum_data = pd.DataFrame(sum_data, columns=[factor_name[0]+'_aggre_sum'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], sum_data], axis=1)

        return result_df
    
    # def mean_derive(self, factor_df):
    #     """Aggregation method is mean"""
    #     factor_delta_num  = int(120*self.factor_back_time)
    #     factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
    #     factor_info = factor_df[factor_name[0]]
    #     mean_data = np.mean([factor_info.shift(i) for i in range(factor_delta_num)],axis=0)
    #     mean_data = pd.DataFrame(mean_data, columns=[factor_name[0]+'_aggre_mean'],index=factor_df.index)
    #     result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], mean_data], axis=1)

    #     return result_df
    
    def abs_sum_derive(self, factor_df):
        """Aggregation method is abs+sum"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_info = factor_df[factor_name[0]].abs()
        abs_sum_data = np.sum([factor_info.shift(i) for i in range(factor_delta_num)],axis=0)
        abs_sum_data = pd.DataFrame(abs_sum_data, columns=[factor_name[0]+'_aggre_abs_sum'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], abs_sum_data], axis=1)

        return result_df
    

    def volatility_derive(self, factor_df):
        """Aggregation method is volatility"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_info = factor_df[factor_name[0]] 

        volatility_data = np.std([factor_info.shift(i) for i in range(factor_delta_num)],axis=0)
        volatility_data = pd.DataFrame(volatility_data, columns=[factor_name[0]+'_aggre_volatility'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], volatility_data], axis=1)

        return result_df
    
    def skew_derive(self, factor_df):
        """Aggregation method is skew"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_list = [factor_df[factor_name[0]].shift(i) for i in range(factor_delta_num)]
        factor_info = pd.concat(factor_list, axis=1)
        #calculate skew
        skew_data = factor_info.skew(axis=1)
        # skew_data = factor_info.apply(lambda row: row.skew(), axis=1)
        skew_data = pd.DataFrame(skew_data, columns=[factor_name[0]+'_aggre_skew'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], skew_data], axis=1)

        return result_df

    
    def kurtosis_derive(self, factor_df):
        """Aggregation method is kurtosis"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_list = [factor_df[factor_name[0]].shift(i) for i in range(factor_delta_num)]
        factor_info = pd.concat(factor_list, axis=1)
        # calculate kurtosis
        kurtosis_data = factor_info.kurt(axis=1)
        # kurtosis_data = factor_info.apply(lambda row: row.kurt(), axis=1)
        kurtosis_data = pd.DataFrame(kurtosis_data, columns=[factor_name[0]+'_aggre_kurtosis'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], kurtosis_data], axis=1)

        return result_df
    
    
    def bipower_var_derive(self, factor_df):
        """Aggregation method is bipower_variation"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_base = factor_df[factor_name[0]]*factor_df[factor_name[0]].shift(1)*(np.pi*factor_delta_num/(2*(factor_delta_num-2)))

        factor_list = [factor_base.shift(i) for i in range(factor_delta_num)]
        factor_info = pd.concat(factor_list, axis=1)
        bipower_var_data = factor_info.sum(axis=1)
        bipower_var_data = pd.DataFrame(bipower_var_data, columns=[factor_name[0]+'_aggre_bipower_var'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], bipower_var_data], axis=1)

        return result_df
    
    def Downside_volatility_derive(self, factor_df):
        """Aggregation method is Downside_volatilityiation"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_base = factor_df[factor_name[0]]
        
        data = [factor_base.shift(i) for i in range(factor_delta_num)]
        column_means = np.mean(data, axis=0)
        """consider mean value"""
        above_mean_values = np.where(data < column_means, data - column_means, 0)
        sum_square_less = np.sum(above_mean_values**2, axis=0)
        sum_square = factor_delta_num * np.std([factor_base.shift(i) for i in range(factor_delta_num)], axis=0)
        
        Downside_volatility_data = sum_square_less/sum_square
        Downside_volatility_data = pd.DataFrame(Downside_volatility_data, columns=[factor_name[0]+'_aggre_Downside_volatility'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], Downside_volatility_data], axis=1)

        return result_df
    
    
    def trend_strength_derive(self, factor_df):
        """Aggregation method is trend_strength"""
        factor_delta_num  = int(120*self.factor_back_time)
        factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
        factor_base = factor_df[factor_name[0]]
        data = [factor_base.shift(i) for i in range(factor_delta_num)]
        column_means = np.mean(data, axis=0)
        """consider mean value"""
        data = data - column_means
        
        factor_abs_sum = np.sum(np.abs(data), axis=0)
        factor_sum = np.sum(data, axis=0)
        trend_strength_data = factor_sum/factor_abs_sum
        trend_strength_data = pd.DataFrame(trend_strength_data, columns=[factor_name[0]+'_aggre_trend_strength'],index=factor_df.index)
        result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], trend_strength_data], axis=1)

        return result_df






###########################back up #########################  
    
    # def slope_derive(self, factor_df):
    #     """Aggregation method is slope"""

    #     factor_delta_num  = int(120*self.factor_back_time)
    #     factor_name = factor_df.columns.drop(['Symbol','TradingTime'])
    #     factor_info = factor_df[factor_name[0]].diff()        
    #     slope_data = np.sum([factor_info.shift(i) for i in range(factor_delta_num)],axis=0)
    #     slope_data = pd.DataFrame(slope_data, columns=[factor_name[0]+'_slope'],index=factor_df.index)
    #     result_df = pd.concat([factor_df[['Symbol', 'TradingTime']], slope_data], axis=1)

    #     return result_df

