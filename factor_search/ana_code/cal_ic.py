import pandas as pd
import numpy as np
import os
import glob

class Cal_ic():
    """
    Summary: this class is used to analysis the calculate results of factor: 
    
    Attributes:
        df_data (DataFrame): dataframe stored the A value V.S. B value
    
    Methods:
        cal_allday_mean()
        cal_IC_oneday()
        cal_IC_allday()
    """
    def __init__(self, file_path, factor_type,level,select_date=202001, time_intervel=None):
        self.file_path = file_path
        self.feature_path = file_path+"data/ic_results_data/%s_factor_data"%factor_type
        self.factor_type = factor_type
        self.level = level
        self.select_date = select_date
        self.time_intervel = time_intervel

    def data_downsample(self, data_path):
        """down sampling the factor data"""

        df_oneday = pd.read_hdf(data_path)
        if self.time_intervel is not None:
            df_oneday = df_oneday.set_index('TradingTime')
            sam_result_df = df_oneday.resample(rule='1min').last()
            sam_result_df = sam_result_df.reset_index()
        else:
            sam_result_df = df_oneday

        return sam_result_df

    def cal_allday_mean(self):

        matching_files = glob.glob(os.path.join(self.feature_path, '*%s*'%str(self.select_date)))
        first_df = pd.read_hdf(matching_files[0])
        columns = first_df.columns.drop(['Symbol','TradingTime'])
        mean_dict = {}
        
        for col in columns:
            mean_dict[col] = []

        for factor_file in matching_files:
            
            df_oneday = self.data_downsample(factor_file)
            for col in columns:
                mean_dict[col].append(df_oneday[col].mean()) 
        mean_series = pd.DataFrame(mean_dict).mean()

        return mean_series


    def cal_IC_oneday(self, mean_value_method="normal_mean"):
        """ cal IC value at one day level"""

        mean_series = self.cal_allday_mean()  
        allday_return_mean = mean_series['return']  
        
        matching_files = glob.glob(os.path.join(self.feature_path, '*%s*'%str(self.select_date)))
        matching_files.sort()
        first_df = pd.read_hdf(matching_files[0])
        columns_list = first_df.columns.drop(['Symbol','TradingTime','return'])
        result_df = pd.DataFrame()

        for factor_file in matching_files:
            
            df_oneday = self.data_downsample(factor_file)
            date = df_oneday['TradingDate'] = df_oneday['TradingTime'].dt.date.iloc[0]
            result = pd.DataFrame(index=[date])
            result['Symbol'] = df_oneday['Symbol'].iloc[0]
            result['TradingDate'] = df_oneday['TradingTime'].dt.date.iloc[0]
            for column in range(len(columns_list)):
                x = df_oneday[columns_list[column]]
                y = df_oneday['return']
                
                """delete nan factor number"""
                return_data = y[x.notna()]
                factor_one = x.dropna()
                if mean_value_method == "normal_mean":
                    ic_value = [np.corrcoef(factor_one, return_data)[0, 1]]
                elif mean_value_method == "allday_mean":
                    allday_factor_mean = mean_series[column]
                    ic_value = np.mean((factor_one-allday_factor_mean)*(return_data-allday_return_mean))/\
                        (np.std(factor_one)*np.std(return_data))
                
                result[columns_list[column]] = ic_value

            result_df = pd.concat([result_df, result]).reset_index(drop=True).round(4)
        result_df.to_csv(self.file_path+"data/ic_recal_data/%s_type_%s_%s_IC_eachday.csv"%(self.factor_type,mean_value_method,self.level))
        return result_df
    

    def cal_IC_allday(self):
        """ cal IC value at one day level"""

        mean_series = self.cal_allday_mean()  
        print(mean_series[0],"just a test")
        allday_return_mean = mean_series['return']  
        
        matching_files = glob.glob(os.path.join(self.feature_path, '*%s*'%str(self.select_date)))
        matching_files.sort()
        first_df = pd.read_hdf(matching_files[0])
        columns_list = list(first_df.columns.drop(['Symbol','TradingTime','return']))
        result_df = pd.DataFrame()

        expect_dict = {}
        f_std_dict = {}
        notNAN_dict = {}
        return_dict = {}
        corr_dict = {}
        for col in columns_list:
            expect_dict[col] = 0
            f_std_dict[col] = 0
            notNAN_dict[col] = 0
            corr_dict[col] = 0
            return_dict[col] = 0

        for factor_file in matching_files:
            
            df_oneday = self.data_downsample(factor_file)       
            for column in columns_list:
                x = df_oneday[column]
                y = df_oneday['return']
                
                """delete nan factor number"""
                not_nan_count = x.count()
                return_data = y[x.notna()]
                factor_one = x.dropna()

                oneday_cov = np.sum((factor_one - mean_series[column])*(return_data - allday_return_mean))
                oneday_factor_std = np.sum((factor_one - mean_series[column])**2)
                return_std = np.sum((return_data - mean_series['return'])**2)
                
                expect_dict[column] += oneday_cov
                f_std_dict[column] += oneday_factor_std
                notNAN_dict[column] += not_nan_count
                return_dict[column] += return_std

        """cal corr value"""
        print(expect_dict)
        print(notNAN_dict)
        print(return_dict)
        for col in columns_list:
            cov = expect_dict[col]/notNAN_dict[col]
            factor_std = np.sqrt(f_std_dict[col]/(notNAN_dict[col] -1))
            ret_std = np.sqrt(return_dict[col]/(notNAN_dict[col] -1))
            corr_dict[col] = cov/(factor_std*ret_std)

        print(corr_dict)
        keys = list(corr_dict.keys())
        values = list(corr_dict.values())

        result_df = pd.DataFrame({'features': keys, 'All_day_IC': values})
        result_df.to_csv(self.file_path+"data/ic_recal_data/%s_type_%s_IC_allday.csv"%(self.factor_type,self.level))
        return result_df
