import os
import pandas as pd
import time
from main_code.analysis import Analysis
from main_code.read_data import Read_data
import warnings
warnings.filterwarnings("ignore")

main_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(main_script_path)
os.chdir(script_dir)
file_path='/dybfs2/nEXO/fuys/others/sw_work'


def cal_all_ic(args):#'rank_ic'or normal_ic
    """calculate feature value, return and stored for further analysis"""

    start_time = time.time()

    os.chdir(file_path+'/data/Tick_data_20230606')
    fname_list = [x for x in os.listdir() if args.process_date in x ]
    raw_data_obj = Read_data()

    for fname in fname_list:

        df_oneday = raw_data_obj.read_future_data(fname)
        ana = Analysis(data=df_oneday,data_underlying='future',data_level=5,\
            factor_derive=args.factor_derive, factor_back_time=args.factor_back_time)
        return_df = ana.cal_return(mins=args.return_time)

        factor_df = ana.cal_feature(args.factor_type)
        print(factor_df)

        factor_df = factor_df.loc[factor_df['TradingTime'].isin(return_df['TradingTime'])].reset_index(drop=True)
        return_df = return_df.loc[return_df['TradingTime'].isin(factor_df['TradingTime'])].reset_index(drop=True)
        factor_return_df = pd.concat([factor_df, return_df['return']],axis=1)#, axis=1, ignore_index=True)


        time_info =fname.split('.')[0]
        h5_file_name = 'type_%s_factor_and_return_%s_min_back_%s_min_%s_%s'\
            %(args.factor_type, args.return_time, args.factor_back_time, args.ic_types,time_info)
        factor_return_file_path = file_path+'/data/ic_results_data/%s_factor_data/'%args.factor_type
        if not os.path.exists(factor_return_file_path):
            os.makedirs(factor_return_file_path)
        factor_return_df.to_hdf(factor_return_file_path+h5_file_name+".h5", index=False,key="results")

    print("total time is %0.2f s"%(time.time()-start_time))



import argparse

parser = argparse.ArgumentParser(description='set factor search key paras')
parser.add_argument('--process_date', type=str, help='date need to processed')
parser.add_argument('--return_time', type=int, help='calculate return time [min]',default=1)
parser.add_argument('--factor_back_time', type=int, help='factor back time [min]',default=1)
parser.add_argument('--ic_types', type=str, help='methods for calc ic',default='normal_ic')
parser.add_argument('--factor_type', type=str,choices=['All','A','B','C','D'], help='types of factor to be calculated',default='All')
parser.add_argument('--factor_derive', action="store_true", help='whether to perform factor', default=False)

args = parser.parse_args()


cal_all_ic(args)
