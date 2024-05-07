import pandas as pd
from datetime import datetime

class Data:
    
    """
    Summary: 获取指定时间的tick级特征
    
    Attributes:
    
        df: 根据范围筛选的数据
        
        level: 行情档位，10档行情或者5档行情
        
        underlying：股票市场或十年国债期货
    """
    
    def __init__(self,symbol_list:list=None,begin_date=None,end_date=None,begin_time=None,end_time=None,
                 database='SEL2_TAQ_h',data=None,data_underlying=None,data_level=None,tick_filled=False,factor_back_time=None):
        
        """
        summary：根据给定范围获取数据
        
        :param database: 开发环境库名,默认为'SEL2_TAQ_h'上交所L2十档行情；‘SZL2_TAQ_H’深交所L2十档行情；
                        'TFL2_TAQ_H'中金所国债期货分笔
                        
        :param symbol_list: 标的代码的字符串列表，如['600518','600519']
        
        :param begin_date: 数据开始日期的字符串，如'2023.06.26'
        
        :param end_date: 数据结束日期的字符串，如'2023.06.28'
        
        :param begin_time: 数据开始时间的字符串，默认为None，此时股票默认为'09:30:00'，国债期货默认为'09:15:00'(2020.07.20前)，
                        或'09:30:00'(2020.07.20后)
                        
        :param end_time: 数据结束时间的字符串，默认为None，此时股票默认为'15:30:01'，国债期货默认为'15:15:00.500'
        
        :param data: 传入数据（有格式要求，列名、起始时间等），默认为None，即默认使用数据库计算
        
        :param data_underlying: 传入数据的标的，underling='stock' or 'future'，默认为None
        
        :param data_level:传入数据的行情档位，level=5 or 10，默认为None
        
        :param tick_filled:是否补全tick，默认为False。注意如果是传入数据，则要求数据起始时间正确
        
        return:None
        
        """
        
        # TODO:区分主力合约
        if data is None:
            trade = s.loadTable(tableName= 'data', dbPath= "dfs://CSMAR_{}".format(database))
            if 'SEL' in database or 'SZL' in database: # 股票
                
                if begin_time == None:
                    begin_time = '09:30:00'
                    
                if end_time == None:
                    end_time = '15:00:01'
            
#                 if symbol_list is None:
#                     self.df = trade.select("*")\
#                         .where("tradingdate between {} and {}".format(begin_date,end_date))\
#                         .where("second(tradingtime) between {}s and {}s".format(begin_time,end_time)).toDF()
#                 else:    
                self.df = trade.select("*").where("symbol in set({})".format(symbol_list))\
                        .where("tradingdate between {} and {}".format(begin_date,end_date))\
                        .where("second(tradingtime) between {}s and {}s".format(begin_time,end_time)).toDF()
    
                self.underlying = 'stock'
                self.level = 5
                
                if tick_filled == True:
                    self.df = self.stock_tick_filled()
                
            if 'SEL2' in database or 'SZL2' in database: # 判断是否有十档行情
                self.level = 10
                
            else: #期货
                #2020.07.20交易时间改变
                    
                date = datetime.strptime('2020.07.20', '%Y.%m.%d')
                if (datetime.strptime(begin_date, '%Y.%m.%d')<date)&(datetime.strptime(end_date,'%Y.%m.%d')>=date): 
                    df_before = trade.select("*").where("symbol in set({})".format(symbol_list))\
                        .where("tradingdate between {} and 2020.07.17".format(begin_date))\
                        .where("second(tradingtime) between 09:15:00s and 15:15:00s").toDF()

                    df_after = trade.select("*").where("symbol in set({})".format(symbol_list))\
                        .where("tradingdate between 2020.07.20 and {}".format(end_date))\
                        .where("second(tradingtime) between 09:30:00s and 15:15:00s").toDF()
                    self.df = pd.concat([df_before,df_after],ignore_index = True)
                    self.underlying = 'future'
                    self.level = 5
                    
                else:
                    if (begin_time == None)&(datetime.strptime(begin_date, '%Y.%m.%d')<date):
                        begin_time = '09:15:00'
                        
                    elif (begin_time == None)&(datetime.strptime(begin_date, '%Y.%m.%d')>=date):
                        begin_time = '09:30:00'
                    
                    if end_time == None:
                        end_time = '15:15:00'
                        
                    self.df = trade.select("*").where("symbol in set({})".format(symbol_list))\
                        .where("tradingdate between {} and {}".format(begin_date,end_date))\
                        .where("second(tradingtime) between {}s and {}s".format(begin_time,end_time)).toDF()
                    
                self.underlying = 'future'
                self.level = 5
                
                if tick_filled == True:
                    self.df = self.future_tick_filled()
                
        else:
            self.underlying = data_underlying
            self.level = data_level
            self.df = data
            
            if tick_filled == True:
                if self.underlying == 'stock':
                    self.df = self.stock_tick_filled()
                if self.underlying == 'future':
                    self.df = self.future_tick_filled()
                
        self.df = self.time_selected()
        self.factor_back_time = factor_back_time
        
    def future_tick_filled(self,data=None): # tick数据填充，可以填充多日的Tick数据，tick数据500ms更新
        # TODO: 国债期货到期前最后交易日11:30收盘 不能和其他日期同时回测
        
        if data is not None:
            df = data
        else:
            df = self.df
            
        date_list = df['TradingDate'].drop_duplicates()
        result_df = pd.DataFrame()   

        for date in date_list:
            df1 = df.loc[(df['TradingDate']==date)]

            date_time = pd.Timestamp(df1['TradingTime'].values[0])
            td = pd.Timedelta(milliseconds=500)
            start_time = date_time
            end_time = pd.Timestamp(df1['TradingTime'].values[-1])
            time_range = pd.date_range(start=start_time, end=end_time, freq='500ms')
            time_range = time_range[(time_range.time < (pd.Timestamp('11:30:00') + td).time())|(time_range.time >= pd.Timestamp('13:00:00').time())]

            # 2020年7月20日交易时间有调整
            if date.date() >= pd.Timestamp('2020-07-20').date(): 
                time_range = time_range[time_range.time >= pd.Timestamp('09:30:00').time()]
#                 if time_range[-1]<pd.Timestamp(df1['TradingTime'].values[-1]): 
#                     time_range = time_range.append(pd.DatetimeIndex([df1['TradingTime'].values[-1]]))

            time_df = pd.DataFrame(index=time_range)
            filled_df = pd.merge(time_df, df1, how='left', left_index=True, right_on='TradingTime')    
            filled_df = filled_df.fillna(method='ffill')

            result_df = pd.concat([result_df,filled_df],ignore_index=True)

        return result_df 
    
    
    def stock_tick_filled(self): # tick数据填充，可以填充多日、多个股票的Tick数据，tick数据3s更新
        
        df = self.df
        date_list = df['TradingDate'].drop_duplicates()
        symbol_list = df['Symbol'].drop_duplicates()
        result_df = pd.DataFrame()

        for date in date_list:
            for symbol in symbol_list:
                df1 = df.loc[(df['TradingDate']==date)&(df['Symbol']==symbol)]

                date_time1 = pd.Timestamp(df1['TradingTime'].values[0])
                date_time2 = pd.Timestamp(df1['TradingTime'].values[1])

                if (date_time2-date_time1).total_seconds()%3 == 0:
                    date_time = date_time1
                else:
                    date_time = date_time2

                td = pd.Timedelta(seconds=3)

                start_time = date_time
                end_time = pd.Timestamp.combine(date, pd.Timestamp('15:00:00').time())
                time_range = pd.date_range(start=start_time, end=end_time, freq='3s')
                time_range = time_range[(time_range.time <= (pd.Timestamp('11:30:00') + td).time()) | (time_range.time >= pd.Timestamp('13:00:00').time())]

                if time_range[-1]<pd.Timestamp(df1['TradingTime'].values[-1]): # 最后的交易时间不是3s的整数倍的情况
                        time_range = time_range.append(pd.DatetimeIndex([df1['TradingTime'].values[-1]]))

                time_df = pd.DataFrame(index=time_range)
                filled_df = pd.merge(time_df, df1, how='left', left_index=True, right_on='TradingTime')    
                filled_df = filled_df.fillna(method='ffill')
                result_df = pd.concat([result_df,filled_df],ignore_index=True)

        return result_df
    
            
    def time_selected(self):# 筛选时间
        
        df = self.df
        start_time = pd.to_datetime('09:15:00').time()
        start_time1 = pd.to_datetime('09:30:00').time()
        end_time = pd.to_datetime('15:15:01').time()
        date = pd.to_datetime('2020-07-20')
        
        if df['TradingDate'].iloc[-1]<date:
            result_df = df.loc[(df['TradingTime'].dt.time >= start_time) & (df['TradingTime'].dt.time < end_time)]
        if df['TradingDate'].iloc[0]>=date:
            result_df = df.loc[(df['TradingTime'].dt.time >= start_time1) & (df['TradingTime'].dt.time < end_time)]
        else:
            result_df1 = df.loc[(df['TradingDate']<date)&(df['TradingTime'].dt.time>=start_time)&(df['TradingTime'].dt.time<end_time)]
            result_df2 = df.loc[(df['TradingDate']>=date)&(df['TradingTime'].dt.time>=start_time1)&(df['TradingTime'].dt.time<end_time)]
            result_df = pd.concat([result_df1,result_df2],ignore_index=True)
        
        return result_df