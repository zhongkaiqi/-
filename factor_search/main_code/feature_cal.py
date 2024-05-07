import pandas as pd
import numpy as np


class A_feature:
    
    # A1-A10
    def cal_bid_ask_spreads(self, i):
        'i=1,2,...10'
        df = self.future_tick_filled(data=self.df)
        ask = df['SellPrice' + str(i).zfill(2)]
        bid = df['BuyPrice' + str(i).zfill(2)]
        spread = ask - bid
        spread = pd.DataFrame(spread, columns=['bid_ask_spreads_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], spread], axis=1)
        return result_df
    
    
    # A11-A20
    def cal_mid_prices(self, i):
        'i=1,2,...10'
        
        df = self.future_tick_filled(data=self.df)
        ask = df['SellPrice' + str(i).zfill(2)]
        bid = df['BuyPrice' + str(i).zfill(2)]
        mid_prices = (ask + bid)/2
        mid_prices = pd.DataFrame(mid_prices, columns=['mid_prices_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], mid_prices], axis=1)

        return result_df
    
    
    # A21-A29
    def cal_ask_prices_differences(self, i):
        'i=2,...10'
        df = self.future_tick_filled(data=self.df)
        ask = df['SellPrice' + str(i).zfill(2)]
        ask_1 = df['SellPrice01']
        diff = ask - ask_1
        diff = pd.DataFrame(diff, columns=['ask_prices_differences_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], diff], axis=1)

        return result_df
    
    
    # A30-A37
    def cal_ask_prices_differences_abs(self, i):
        'i=3,...10'

        df = self.future_tick_filled(data=self.df)
        abs_diff = abs(df['SellPrice' + str(i).zfill(2)] - df['SellPrice' + str(i-1).zfill(2)])
        abs_diff = pd.DataFrame(abs_diff, columns=['ask_prices_differences_abs_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], abs_diff], axis=1)
        
        return result_df
    
    
    # A38-A46
    def cal_bid_prices_differences(self, i):
        'i=2,...10'
        df = self.future_tick_filled(data=self.df)
        bid = df['BuyPrice' + str(i).zfill(2)]
        bid_1 = df['BuyPrice01']
        diff = bid - bid_1
        diff = pd.DataFrame(diff, columns=['bid_prices_differences_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], diff], axis=1)

        return result_df
    
    
    # A47-A54
    def cal_bid_prices_differences_abs(self, i):
        'i=3,...10'
        df = self.future_tick_filled(data=self.df)
        abs_diff = abs(df['BuyPrice' + str(i).zfill(2)] - df['BuyPrice' + str(i-1).zfill(2)])
        abs_diff = pd.DataFrame(abs_diff, columns=['bid_prices_differences_abs_'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], abs_diff], axis=1)

        return result_df

    
    # A55
    def cal_ask_mean_prices(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        ask_acc_prices = np.zeros(len(df))
        for index in range(1,level+1):
            ask_acc_prices = ask_acc_prices + df['SellPrice' + str(index).zfill(2)]
        ask_mean_prices = ask_acc_prices/level
        ask_mean_prices = pd.DataFrame(ask_mean_prices, columns=['ask_mean_prices'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], ask_mean_prices], axis=1)

        return result_df
    
     
    # A56
    def cal_bid_mean_prices(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        bid_acc_prices = np.zeros(len(df))
        for index in range(1,level+1):
            bid_acc_prices = bid_acc_prices + df['BuyPrice' + str(index).zfill(2)]
        bid_mean_prices = bid_acc_prices/level
        bid_mean_prices = pd.DataFrame(bid_mean_prices, columns=['bid_mean_prices'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], bid_mean_prices], axis=1)

        return result_df
    

    # A57
    def cal_ask_mean_volume(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        ask_acc_prices = np.zeros(len(df))
        for index in range(1,level+1):
            ask_acc_prices = ask_acc_prices + df['SellVolume' + str(index).zfill(2)]
        ask_mean_volume = ask_acc_prices/level
        ask_mean_volume = pd.DataFrame(ask_mean_volume, columns=['ask_mean_volume'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], ask_mean_volume], axis=1)

        return result_df
    
    
    # A58
    def cal_bid_mean_volume(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        bid_acc_prices = np.zeros(len(df))
        for index in range(1,level+1):
            bid_acc_prices = bid_acc_prices + df['BuyVolume' + str(index).zfill(2)]
        bid_mean_volume = bid_acc_prices/level
        bid_mean_volume = pd.DataFrame(bid_mean_volume, columns=['bid_mean_volume'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], bid_mean_volume], axis=1)

        return result_df

    
    # A59
    def cal_accumulated_price_spreads(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        accumulated_price_spreads = np.zeros(len(df))
        for index in range(1,level+1):
            temp_bid =  df['SellPrice' + str(index).zfill(2)] - df['BuyPrice' + str(index).zfill(2)]
            accumulated_price_spreads = accumulated_price_spreads + temp_bid
        accumulated_price_spreads = pd.DataFrame(accumulated_price_spreads, columns=['accumulated_price_spreads'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], accumulated_price_spreads], axis=1)

        return result_df

    
    # A60
    def cal_accumulated_volume_spreads(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        accumulated_volume_spreads = np.zeros(len(df))
        for index in range(1,level+1):
            temp_bid =  df['SellVolume' + str(index).zfill(2)] - df['BuyVolume' + str(index).zfill(2)]
            accumulated_volume_spreads = accumulated_volume_spreads + temp_bid
        accumulated_volume_spreads = pd.DataFrame(accumulated_volume_spreads, columns=['accumulated_volume_spreads'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], accumulated_volume_spreads], axis=1)
        
        return result_df


    # A61
    def cal_log_quote_slope(self):
        df = self.future_tick_filled(data=self.df)
        log_price_slope =  df['SellPrice01'].apply(np.log) - df['BuyPrice01'].apply(np.log)
        log_volume_slope =  df['SellVolume01'].apply(np.log) + df['BuyVolume01'].apply(np.log)
        log_quote_slope = log_price_slope/log_volume_slope
        log_quote_slope = pd.DataFrame(log_quote_slope, columns=['log_quote_slope'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], log_quote_slope], axis=1)
        return result_df

    
    # A62
    def cal_spdr(self):
        df = self.future_tick_filled(data=self.df)
        spdr_price =  df['SellPrice01'] - df['BuyPrice01']
        spdr_volume =  (df['SellPrice01'] + df['BuyPrice01'])/2
        spdr = spdr_price/spdr_volume
        spdr = pd.DataFrame(spdr, columns=['spdr'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], spdr], axis=1)
    

        return result_df
    
    
    # A63-A71
    def cal_price_diff_imbalance_minus(self,i):
        'i=2,3...10'
        df = self.future_tick_filled(data=self.df)
        tmp_bid = (df['BuyPrice' + str(i).zfill(2)] - df['BuyPrice01']).abs()
        tmp_ask = (df['SellPrice' + str(i).zfill(2)]- df['SellPrice01']).abs()
        price_diff_imbalance = tmp_bid - tmp_ask
        price_diff_imbalance = pd.DataFrame(price_diff_imbalance, columns=['price_diff_imbalance_minus'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], price_diff_imbalance], axis=1)

        return result_df

    
    # A72-A80
    def cal_price_diff_imbalance_divide(self,i):
        'i=2,3,...,10'
        df = self.future_tick_filled(data=self.df)
        tmp_bid = (df['BuyPrice' + str(i).zfill(2)] - df['BuyPrice01']).abs()
        tmp_ask = (df['SellPrice' + str(i).zfill(2)]- df['SellPrice01']).abs()
        price_diff_imbalance = tmp_bid/tmp_ask
        price_diff_imbalance = pd.DataFrame(price_diff_imbalance, columns=['price_diff_imbalance_divide'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], price_diff_imbalance], axis=1)
        return result_df

    
    # A81-A90
    def cal_volume_imbalance(self,i):
        'i=1,2,...,10'
        df = self.future_tick_filled(data=self.df)
        volume_minus = (df['BuyVolume' + str(i).zfill(2)] - df['SellVolume' + str(i).zfill(2)])
        volume_add = (df['BuyVolume' + str(i).zfill(2)] + df['SellVolume' + str(i).zfill(2)])
        volume_imbalance = volume_minus/volume_add
        volume_imbalance = pd.DataFrame(volume_imbalance, columns=['volume_imbalance'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], volume_imbalance], axis=1)
        return result_df

    
    # A91-A99
    def cal_acc_volume_imbalance(self, i):
        'i=2,3,...,10'
        df = self.future_tick_filled(data=self.df)
        accumulated_bid_volume = np.zeros(len(df))
        accumulated_ask_volume = np.zeros(len(df))
        for index in range(1,i+1):
            accumulated_bid_volume = accumulated_bid_volume + df['BuyVolume' + str(index).zfill(2)] 
            accumulated_ask_volume = accumulated_ask_volume + df['SellVolume' + str(index).zfill(2)] 
        acc_volume_imbalance = (accumulated_bid_volume - accumulated_ask_volume)/(accumulated_bid_volume + accumulated_ask_volume)
        acc_volume_imbalance = pd.DataFrame(acc_volume_imbalance, columns=['acc_volume_imbalance'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], acc_volume_imbalance], axis=1)

        return result_df


    # A100-A101
    def cal_acc_num_imbalance(self, i):
        'i=1,10'
        df = self.future_tick_filled(data=self.df)
        accumulated_bid_volume = np.zeros(len(df))
        accumulated_ask_volume = np.zeros(len(df))
        for index in range(1,i+1):
            accumulated_bid_volume = accumulated_bid_volume + df['TotalBuyOrderNo' + str(index).zfill(2)] 
            accumulated_ask_volume = accumulated_ask_volume + df['TotalSellOrderNo' + str(index).zfill(2)] 
        acc_num_imbalance = (accumulated_bid_volume - accumulated_ask_volume)/(accumulated_bid_volume + accumulated_ask_volume)
        acc_num_imbalance = pd.DataFrame(acc_num_imbalance, columns=['acc_num_imbalance'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], acc_num_imbalance], axis=1)

        return result_df

    
    # A102-A103
    def cal_average_volume_imbalance(self, i):
        'i=1,10'
        df = self.future_tick_filled(data=self.df)
        accumulated_bid_volume = np.zeros(len(df))
        accumulated_ask_volume = np.zeros(len(df))
        for index in range(1,i+1):
            accumulated_bid_volume = accumulated_bid_volume + \
                df['BuyVolume' + str(index).zfill(2)]/df['TotalBuyOrderNo' + str(index).zfill(2)]
            accumulated_ask_volume = accumulated_ask_volume + \
                df['SellVolume' + str(index).zfill(2)]/df['TotalSellOrderNo' + str(index).zfill(2)]         
        ave_bid_volume = accumulated_bid_volume/i
        ave_ask_volume = accumulated_ask_volume/i
        average_volume_imbalance = (ave_bid_volume - ave_ask_volume)/(ave_bid_volume + ave_ask_volume)
        average_volume_imbalance = pd.DataFrame(average_volume_imbalance, columns=['average_volume_imbalance'+ str(i)])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], average_volume_imbalance], axis=1)
        return result_df
    
    
    # A104
    def cal_weighted_volume_imbalance(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        bid_omega_volume = np.zeros(len(df))
        ask_omega_volume = np.zeros(len(df))
        for index in range(1,level+1):
            weighted_bid = (level + 1 -index)/sum(range(level+1))\
                        *df['BuyVolume' + str(index).zfill(2)]
            bid_omega_volume = bid_omega_volume + weighted_bid
            weighted_ask = (level + 1 -index)/sum(range(level+1))\
                        *df['SellVolume' + str(index).zfill(2)]
            bid_omega_volume = bid_omega_volume + weighted_bid
            ask_omega_volume = ask_omega_volume + weighted_ask
        VIR = (bid_omega_volume - ask_omega_volume)/(bid_omega_volume + ask_omega_volume)
        VIR = pd.DataFrame(VIR, columns=['weighted_volume_imbalance'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], VIR], axis=1)

        return result_df


    # A105
    def cal_weighted_amount_imbalance(self):
        '特征含义：订单金额加权总量的不平衡性，数值为正体现出委买订单量大，买方交易意愿强'
        
        df = self.future_tick_filled(data=self.df)
        level = self.level
        bid = np.zeros(len(df))
        ask = np.zeros(len(df))
        for j in range(level):
            weighted_bid = (level - j) / sum(range(level + 1)) \
                           * df['BuyPrice' + str(j + 1).zfill(2)] * df['BuyVolume' + str(j + 1).zfill(2)]
            weighted_ask = (level - j) / sum(range(level + 1)) \
                           * df['SellPrice' + str(j + 1).zfill(2)] * df['SellVolume' + str(j + 1).zfill(2)]
            bid = bid + weighted_bid
            ask = ask + weighted_ask
        spread = (bid - ask) / (bid + ask)
        spread = pd.DataFrame(spread, columns=['weighted_amount_imbalance'])
        dataframe = pd.concat([df[['Symbol', 'TradingTime']], spread], axis=1)

        return dataframe


    # A106-A108
    def cal_liquidity_imbalance(self,Q):
        """
        Feature meaning: 
            The comparison of the distance between the buying and selling market price and the 
            middle price under a specific trading volume, with a positive value indicating that 
            the bid price is relatively low

        Parameter Q: 
            Trade volume (Q=500, 1000, 2000)
            for Treasury bond futures, according 5 level volume data (range(1,1000)),
            Q could set as (50, 200, 500)
        """
        df = self.future_tick_filled(data=self.df)
        level = self.level
        V_Bid = np.array(df[['BuyVolume' + str(j + 1).zfill(2) for j in range(level)]])
        V_Ask = np.array(df[['SellVolume' + str(j + 1).zfill(2) for j in range(level)]])
        P_Bid = np.array(df[['BuyPrice' + str(j + 1).zfill(2) for j in range(level)]])
        P_Ask = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]])

        def calV(V_input, Q):
            cumsum1 = V_input.cumsum(axis=1)
            cumsum2 = cumsum1 - V_input
            V = np.where((cumsum1 >= Q) & (cumsum2 < Q), Q - cumsum2, V_input)
            V = np.where(cumsum2 >= Q, 0, V)
            return V

        def LR(Q):
            Mid = (P_Bid[:, 0] + P_Ask[:, 0]) / 2
            LB = Mid - (calV(V_Bid, Q) * P_Bid).sum(axis=1) / Q
            LA = (calV(V_Ask, Q) * P_Ask).sum(axis=1) / Q - Mid
            LR = (LB - LA) / (LB + LA + 1e-5)
            return LR

        LR = pd.DataFrame(LR(Q), columns=['liquidity_imbalance_' + str(Q)],index=df.index)
        dataframe = pd.concat([df[['Symbol', 'TradingTime']], LR], axis=1)
        return dataframe


    # A109-A117
    def cal_volume_proportion_imbalance(self, i):
        """
        特征含义：基于价格百分比排序的买卖订单量不平衡性，数值为正体现出买方前k档订单价格接近且订单量大
        参数i：第i个委托买价或卖家，这里i=1，2，3，...,9,与档位无关
        """
        
        df = self.future_tick_filled(data=self.df)
        level = self.level
        # 设置上下限
        buy_l = df['BuyPrice01'] - (df['BuyPrice01'] - df['BuyPrice' + str(level).zfill(2)]) * 0.1 * i
        sell_u = df['SellPrice01'] + (df['SellPrice' + str(level).zfill(2)] - df['SellPrice01']) * 0.1 * i

        # 复制上下限用作对比
        buy_l = np.array(buy_l).reshape([len(buy_l), 1])
        sell_u = np.array(sell_u).reshape([len(sell_u), 1])
        ones = np.ones([1, level])
        buy_l = np.dot(buy_l, ones)
        sell_u = np.dot(sell_u, ones)

        # 提取df用于对比
        buy_p = np.array(df[['BuyPrice' + str(j + 1).zfill(2) for j in range(level)]])
        buy_v = np.array(df[['BuyVolume' + str(j + 1).zfill(2) for j in range(level)]])
        sell_p = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]])
        sell_v = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]])

        volume_buy = (buy_v * (buy_p > buy_l)).sum(axis=1)
        volume_sell = (sell_v * (sell_p < sell_u)).sum(axis=1)

        imbalance = (volume_buy - volume_sell) / (volume_buy + volume_sell)
        imbalance = pd.DataFrame(imbalance, columns=['volume_proportion_imbalance_' + str(i)],index=df.index)
        dataframe = pd.concat([df[['Symbol', 'TradingTime']], imbalance], axis=1)

        return dataframe


    # A118-A120
    def cal_weighted_price_imbalance(self, k):
        """
        特征含义：加权价差不平衡性，数值为正体现出买方交易意愿较弱
        参数k：对于股票十档：k=2，5，10；对于五档：k=2,3,5
        """
        
        df = self.future_tick_filled(data=self.df)
        level = self.level
        # 为便于计算，提取 price 和 volume 信息
        buy_p = np.array(df[['BuyPrice' + str(j + 1).zfill(2) for j in range(level)]])
        buy_v = np.array(df[['BuyVolume' + str(j + 1).zfill(2) for j in range(level)]])
        sell_p = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]])
        sell_v = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]])

        # 累加
        sum_buy_v = buy_v.cumsum(axis=1)
        sum_sell_v = sell_v.cumsum(axis=1)

        sum_buy_pv = (buy_p * buy_v).cumsum(axis=1)
        sum_sell_pv = (sell_p * sell_v).cumsum(axis=1)

        buy_p_bar = sum_buy_pv[:, k - 1] / sum_buy_v[:, k - 1]
        sell_p_bar = sum_sell_pv[:, k - 1] / sum_sell_v[:, k - 1]

        # 直接得出对应的 imbalance
        imbalance = abs(buy_p_bar - buy_p[:, 0]) - abs(sell_p_bar - sell_p[:, 0])
        imbalance = pd.DataFrame(imbalance, columns=['weighted_price_imbalance_' + str(k)],index=df.index)
        dataframe = pd.concat([df[['Symbol', 'TradingTime']], imbalance], axis=1)

        return dataframe


    # A121
    def cal_orderbook_dispersion(self):
        '特征含义：订单簿分散性，数值为正体现出买方档位价差较大'
        
        df = self.future_tick_filled(data=self.df)
        level = self.level
        k = level
        # 处理分母的交易量加和
        buy_volume = np.array(df[['BuyVolume' + str(j + 1).zfill(2) for j in range(level)]])
        sell_volume = np.array(df[['SellVolume' + str(j + 1).zfill(2) for j in range(level)]])

        sum_buy_v = buy_volume.cumsum(axis=1)
        sum_sell_v = sell_volume.cumsum(axis=1)

        # 处理价格差
        buy_price_diff = np.array(df[['BuyPrice' + str(j + 1).zfill(2) for j in range(level)]] \
                                  .diff(axis=1).abs())
        sell_price_diff = np.array(df[['SellPrice' + str(j + 1).zfill(2) for j in range(level)]] \
                                   .diff(axis=1).abs())
        mp_diff = np.array((df['SellPrice01'] - df['BuyPrice01']) / 2)
        buy_price_diff[:, 0] = mp_diff
        sell_price_diff[:, 0] = mp_diff

        # 得到价差交易量乘积的加和矩阵
        buy_numerator = (buy_volume * buy_price_diff).cumsum(axis=1)
        sell_numerator = (sell_volume * sell_price_diff).cumsum(axis=1)

        dispimb = (buy_numerator[:, k - 1] / sum_buy_v[:, k - 1]) - (sell_numerator[:, k - 1] / sum_sell_v[:, k - 1])
        dispimb = pd.DataFrame(dispimb, columns=['orderbook_dispersion_' + str(k)],index=df.index)
        dataframe = pd.concat([df[['Symbol', 'TradingTime']], dispimb], axis=1)

        return dataframe
    
    
    # A122-A123
    def cal_shape_imbalance(self,n):
        '参数n：代表两种不同算法n=1,2'
        data = self.future_tick_filled(data=self.df)
        level = self.level
        # i取1和2，分别计算shape_imbalance_1 和 shape_imbalance_2
        def PricesAndVolumes(data):
            List=np.array([x+str(y).zfill(2) for x in ['BuyPrice','BuyVolume','SellPrice','SellVolume'] for y in range(1,level+1)]).reshape((4,-1))
            P_Bid=data[List[0]].values
            V_Bid=data[List[1]].values
            P_Ask=data[List[2]].values
            V_Ask=data[List[3]].values
            return P_Bid,V_Bid,P_Ask,V_Ask

        P_Bid,V_Bid,P_Ask,V_Ask = PricesAndVolumes(data)
        def MeanSigmaSkew(P,V):   #计算分布的均值，标准差和偏度
            prob=V/((V.sum(axis=1)).reshape((-1,1)))
            mu=(P*prob).sum(axis=1)
            sigma2=np.abs((P**2*prob).sum(axis=1)-mu**2)
            skew=((P**3*prob).sum(axis=1)-3*mu*sigma2-mu**3)/(sigma2**1.5+1e-9)
            return mu,sigma2**0.5,skew

        mean_Bid,sigma_Bid,skew_Bid=MeanSigmaSkew(P_Bid,V_Bid)
        mean_Ask,sigma_Ask,skew_Ask=MeanSigmaSkew(P_Ask,V_Ask)

    #     mean=(P_Bid[:,0]-mean_Bid)-(mean_Ask-P_Ask[:,0])  #与一档距离之差的imbalance
        sigma=pd.DataFrame((sigma_Bid-sigma_Ask),columns=['shape_imbalance_1'],index=data.index)   #标准差的imbalance
        skew=pd.DataFrame((skew_Bid+skew_Ask),columns=['shape_imbalance_2'],index=data.index)    #偏度的imbalance（一方取负）

        if n == 1:
            return pd.concat([data[['Symbol', 'TradingTime']],sigma],axis=1)
        elif n == 2: 
            return pd.concat([data[['Symbol', 'TradingTime']],skew],axis=1)
        else:
            return '参数错误'
        
        
    # A124    
    def cal_average_local_slope_imbalance(self):
        data = self.future_tick_filled(data=self.df)
        def PricesAndVolumes(data):
            List=np.array([x+str(y).zfill(2) for x in ['BuyPrice','BuyVolume','SellPrice','SellVolume'] for y in range(1,self.level+1)]).reshape((4,-1))
            P_Bid=data[List[0]].values
            V_Bid=data[List[1]].values
            P_Ask=data[List[2]].values
            V_Ask=data[List[3]].values
            return P_Bid,V_Bid,P_Ask,V_Ask

        P_Bid,V_Bid,P_Ask,V_Ask = PricesAndVolumes(data)
        def get_slopes(P,V,midprice):
            log_cumV=np.log(V.cumsum(axis=1)+1e-9)
            slopes=(log_cumV[:,1:]/(log_cumV[:,:-1]+1e-9)-1)/(P[:,1:]/(P[:,:-1]+1e-9)-1)
            slope1=log_cumV[:,0]/(P[:,0]/midprice-1+1e-9)
            slopes=np.hstack((slope1.reshape(-1,1),slopes))
            return slopes

        midprice=(P_Bid[:,0]+P_Ask[:,0])/2
        Bid_slopes=get_slopes(P_Bid,V_Bid,midprice).mean(axis=1)
        Ask_slopes=get_slopes(P_Ask,V_Ask,midprice).mean(axis=1)
        slopeImbalance=(Bid_slopes-Ask_slopes)/(Ask_slopes+Bid_slopes+1e-9)
        slopeImbalance=pd.DataFrame(slopeImbalance.reshape((-1,1)),columns=['average_local_slope_imbalance'],index=data.index)

        result  = pd.concat([data[['Symbol','TradingTime']],slopeImbalance],axis=1)

        return result


    # A125
    def cal_weighted_volume_imbalance_1(self):
        data = self.future_tick_filled(data=self.df)
        if self.underlying == 'future':
            pass
        else:
            new_order = (['Symbol','TradingTime']+['BuyPrice{:02d}'.format(i) for i in range(1, 11)] + 
            ['SellPrice{:02d}'.format(i) for i in range(1, 11)]+
            ['BuyVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['SellVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalBuyOrderNo{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalSellOrderNo{:02d}'.format(i) for i in range(1, 11)])

            data = data.reindex(columns = new_order)

        data['PriceMid'] = (data['BuyPrice01'] + data['SellPrice01'])/2
        def cal_weighted_volume(name):
            volume_columns = [col for col in data.columns if col.startswith(f'{name}Volume')]
            price_columns = [col for col in data.columns if col.startswith(f'{name}Price')]
            volume_columns = volume_columns [1:]
            price_columns = price_columns [1:]
            A = data[price_columns] 
            B = data[f'{name}Price01']
            count = len(price_columns)
            B = pd.concat([B] * count, axis=1)
            C = data[volume_columns]
            def reset_columns(dataframe):
                dataframe.columns = range(len(dataframe.columns))
            for dataframe in [A,B,C]:
                reset_columns(dataframe)
            data[f'{name}WeightedVolume'] = np.sum(C/np.abs(A - B), axis = 1)
        cal_weighted_volume('Buy') 
        cal_weighted_volume('Sell')   
        data['weighted_volume_imbalance_1'] = (data['BuyWeightedVolume'] - data['SellWeightedVolume'])/ (data['BuyWeightedVolume'] + data['SellWeightedVolume'])
        weighted_volume_imbalance_01  = data[['Symbol','TradingTime','weighted_volume_imbalance_1']]
        
        return weighted_volume_imbalance_01 


    # A126
    def cal_weighted_volume_imbalance_2(self):
        data = self.future_tick_filled(data=self.df)
        if self.underlying == 'future':
            pass
        else:
            new_order = (['Symbol','TradingTime']+['BuyPrice{:02d}'.format(i) for i in range(1, 11)] + 
            ['SellPrice{:02d}'.format(i) for i in range(1, 11)]+
            ['BuyVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['SellVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalBuyOrderNo{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalSellOrderNo{:02d}'.format(i) for i in range(1, 11)])

            data = data.reindex(columns = new_order)
        
        data['PriceMid'] = (data['BuyPrice01'] + data['SellPrice01'])/2


        def cal_weighted_volume(name):
            volume_columns = [col for col in data.columns if col.startswith(f'{name}Volume')]
            price_columns = [col for col in data.columns if col.startswith(f'{name}Price')]
            order_columns = [col for col in data.columns if col.startswith(f'Total{name}OrderNo')]
            volume_columns = volume_columns [1:]
            price_columns = price_columns [1:]
            order_columns = order_columns[1:]
            A = data[price_columns] 
            B = data[f'{name}Price01']
            count = len(price_columns)
            B = pd.concat([B] * count, axis=1)
            C = data[volume_columns]
            D = data[order_columns]
            def reset_columns(dataframe):
                dataframe.columns = range(len(dataframe.columns))
            for dataframe in [A,B,C,D]:
                reset_columns(dataframe)
            data[f'{name}WeightedVolume'] = np.sum((C + D*3000)/np.abs(A - B), axis = 1)
        cal_weighted_volume('Buy') 
        cal_weighted_volume('Sell')   
        data['weighted_volume_imbalance_02'] = (data['BuyWeightedVolume'] - data['SellWeightedVolume'])/ (data['BuyWeightedVolume'] + data['SellWeightedVolume'])
        weighted_volume_imbalance_02  = data[['Symbol','TradingTime','weighted_volume_imbalance_02']]
        
        return weighted_volume_imbalance_02 


    # A127
    def cal_press_imbalance(self):
        data = self.future_tick_filled(data=self.df)
        if self.underlying == 'future':
            pass
        else:
            new_order = (['Symbol','TradingTime']+['BuyPrice{:02d}'.format(i) for i in range(1, 11)] + 
            ['SellPrice{:02d}'.format(i) for i in range(1, 11)]+
            ['BuyVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['SellVolume{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalBuyOrderNo{:02d}'.format(i) for i in range(1, 11)]+
            ['TotalSellOrderNo{:02d}'.format(i) for i in range(1, 11)])

            data = data.reindex(columns = new_order)

        data['PriceMid'] = (data['BuyPrice01'] + data['SellPrice01'])/2
        
        def cal_press(name):
            volume_columns = [col for col in data.columns if col.startswith(f'{name}Volume')]
            price_columns = [col for col in data.columns if col.startswith(f'{name}Price')]
            A = data['PriceMid']
            count = len(price_columns)
            A = pd.concat([A] * count, axis=1)
            B = data[price_columns]
            C = data[volume_columns]
            def reset_columns(dataframe):
                dataframe.columns = range(len(dataframe.columns))
            for dataframe in [A,B,C]:
                reset_columns(dataframe)
            weight_sum = np.sum(A/np.abs(A - B) , axis=1)
            weight_sum = pd.concat([weight_sum] * count, axis=1)
            for dataframe in [A,B,C,weight_sum]:
                reset_columns(dataframe)
            data[f'{name}Press'] = np.sum(C*(A/np.abs(A-B))/weight_sum, axis = 1)
        cal_press('Buy')
        cal_press('Sell')
        data['press_imbalance'] = np.log(data['BuyPress']) - np.log(data['SellPress'])
        press_imbalance = data[['Symbol','TradingTime','press_imbalance']]
        return press_imbalance
    
    
    # A128
    def cal_slope_imbalance(self):
        df = self.future_tick_filled(data=self.df)
        tbv = df['BuyVolume01'] + df['BuyVolume02'] + df['BuyVolume03'] + df['BuyVolume04'] + df['BuyVolume05']
        tav = df['SellVolume01'] + df['SellVolume02'] + df['SellVolume03'] + df['SellVolume04'] + df['SellVolume05']
        mp = (df['BuyPrice01'] + df['SellPrice01']) / 2
        sbid = np.abs(mp - df['BuyPrice05']) / (tbv + 1e-9) # 避免无穷大的结果
        sask = np.abs(mp - df['SellPrice05']) / (tav + 1e-9)
        signs = df[['Symbol','TradingTime']]
        signs['slope_imbalance'] = sbid - sask
        return signs
    
    
    # A129
    def cal_slope_imbalance_fine(self):
        
        df = self.future_tick_filled(data=self.df)

        bp = df[['BuyPrice01', 'BuyPrice02', 'BuyPrice03', 'BuyPrice04', 'BuyPrice05']].values
        bv = df[['BuyVolume01', 'BuyVolume02', 'BuyVolume03', 'BuyVolume04', 'BuyVolume05']].values
        ap = df[['SellPrice01', 'SellPrice02', 'SellPrice03', 'SellPrice04', 'SellPrice05']].values
        av = df[['SellVolume01', 'SellVolume02', 'SellVolume03', 'SellVolume04', 'SellVolume05']].values

        mp = (bp[:, 0] + ap[:, 0]) / 2
        mp = mp.reshape(-1, 1)

        cumulative_vask = np.cumsum(av, axis=1)
        cumulative_vbid = np.cumsum(bv, axis=1)

        cumulative_vask = np.insert(cumulative_vask, 0, 0, axis=1)
        cumulative_vbid = np.insert(cumulative_vbid, 0, 0, axis=1)

        beta_bid_numo = 2/3 * np.sum(abs(bp-mp) * (cumulative_vbid[:, 1:]**2 - cumulative_vbid[:, :-1]**2), axis=1)
        beta_bid_deno = np.sum(cumulative_vbid[:, 1:]**3 - cumulative_vbid[:, :-1]**3, axis=1)
        beta_bid = beta_bid_numo / (beta_bid_deno + 1e-9)

        beta_ask_nume = 2/3 * np.sum(abs(ap-mp) * (cumulative_vask[:, 1:]**2 - cumulative_vask[:, :-1]**2), axis=1)
        beta_ask_deno = 2/3 * np.sum(cumulative_vask[:, 1:]**3 - cumulative_vask[:, :-1]**3, axis=1)
        beta_ask = beta_ask_nume / (beta_ask_deno+ 1e-9)

        signs = df[['Symbol', 'TradingTime']]
        signs["slope_imbalance_fine"] = beta_bid - beta_ask
        return signs
    
    
    # A130
    def cal_next_up_probability(self):
    
        df = self.future_tick_filled(data=self.df)
        def fun1(x,bid_num,ask_num):
            return (2-np.cos(x)-np.sqrt((2-np.cos(x))**2-1))**ask_num*(np.sin(bid_num*x)*np.cos(x/2)/np.sin(x/2))

        Nums=np.vstack((df['TotalBuyOrderNo01'],df['TotalSellOrderNo01'])).T
        Nums=Nums[~np.isnan(Nums).all(axis=1)].T  #去掉有nan的行
        set1=list(set(list(zip(Nums[0],Nums[1]))))

        # 蒙特卡洛模拟算积分
        sample=np.random.rand(10000)*np.pi 
        value=[fun1(sample,x,y).mean() for x,y in set1]
        dict1=dict(zip(set1,value))
        dict1[(np.nan,np.nan)]=0.5
        prob=pd.Series(zip(df['TotalBuyOrderNo01'],df['TotalSellOrderNo01'])).map(dict1).values
        prob=np.array(prob).reshape((-1,1))

        result_df = df[['Symbol', 'TradingTime']]
        result_df['next_up_probability'] = prob
        return result_df
    
    
    # A131
    def cal_microprice(self):
        df = self.future_tick_filled(data=self.df)
        bp1, bp2, bp3, bp4, bp5 = df['BuyPrice01'], df['BuyPrice02'], df['BuyPrice03'], df['BuyPrice04'], df['BuyPrice05']
        ap1, ap2, ap3, ap4, ap5 = df['SellPrice01'], df['SellPrice02'], df['SellPrice03'], df['SellPrice04'], df['SellPrice05']
        bv1, bv2, bv3, bv4, bv5 = df['BuyVolume01'], df['BuyVolume02'], df['BuyVolume03'], df['BuyVolume04'], df['BuyVolume05']
        av1, av2, av3, av4, av5 = df['SellVolume01'], df['SellVolume02'], df['SellVolume03'], df['SellVolume04'], df['SellVolume05']

        midprice = (bp1 + ap1) / 2
        totalv = bv1 + bv2 + bv3 + bv4 + bv5 + av1 + av2 + av3 + av4 + av5

        fn1 = (bp1 * av1 + bp2 * av2 + bp3 * av3 + bp4 * av4 + bp5 * av5 + ap1 * bv1 + ap2 * bv2 + ap3 * bv3 + ap4 * bv4 + ap5 * bv5) / totalv
        fn2 = (bp1 * bv1 + bp2 * bv2 + bp3 * bv3 + bp4 * bv4 + bp5 * bv5 + ap1 * av1 + ap2 * av2 + ap3 * av3 + ap4 * av4 + ap5 * av5) / totalv

        signs = df[['Symbol', 'TradingTime']]
        signs['microprice'] = fn1 + fn2 - midprice * 2
        return signs

class B_feature:
    
    # TODO:1、总体设定delta_t参数，考虑用时间计算
    # TODO:2、多个股票、多个日期；方案一：在函数里循环；方案二：读取数据时循环
    
    # B1-B2
    def cal_snapshot_direct_imbalance_WithdrawNo(self,n,delta_t=1):
        
        df = self.df
        d_buy = df['WithdrawBuyNo'] - df['WithdrawBuyNo'].shift(delta_t)
        d_sell = df['WithdrawSellNo'] - df['WithdrawSellNo'].shift(delta_t)
        imbalance1 = d_buy - d_sell

        if n==1:
            imbalance1 = pd.DataFrame(imbalance1,columns=['snapshot_direct_imbalance1_WithdrawNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance1],axis=1).dropna()
        else:
            imbalance2 = imbalance1/(np.abs(d_buy)+np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2,columns=['snapshot_direct_imbalance2_WithdrawNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance2],axis=1).dropna()

        return result


    # B3-B4
    def cal_snapshot_direct_imbalance_WithdrawVolume(self,n,delta_t=1):
        
        df = self.df
        d_buy = df['WithdrawBuyVolume'] - df['WithdrawBuyVolume'].shift(delta_t)
        d_sell = df['WithdrawSellVolume'] - df['WithdrawSellVolume'].shift(delta_t)
        imbalance1 = d_buy - d_sell

        if n==1:
            imbalance1 = pd.DataFrame(imbalance1,columns=['snapshot_direct_imbalance1_WithdrawVolume'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance1],axis=1).dropna()
        else:
            imbalance2 = imbalance1/(np.abs(d_buy)+np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2,columns=['snapshot_direct_imbalance2_WithdrawVolume'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance2],axis=1).dropna()

        return result


    # B5-B6
    def cal_snapshot_direct_imbalance_TotalNo(self,n,delta_t=1):
        
        df = self.df
        d_buy = df['TotalBuyNo'] - df['TotalBuyNo'].shift(delta_t)
        d_sell = df['TotalSellNo'] - df['TotalSellNo'].shift(delta_t)
        imbalance1 = d_buy - d_sell

        if n==1:
            imbalance1 = pd.DataFrame(imbalance1,columns=['snapshot_direct_imbalance1_TotalNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance1],axis=1).dropna()
        else:
            imbalance2 = imbalance1/(np.abs(d_buy)+np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2,columns=['snapshot_direct_imbalance2_TotalNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance2],axis=1).dropna()

        return result


    # B7-B8
    def cal_snapshot_direct_imbalance_TotalOrderVolumne(self,n,delta_t=1):
        
        df = self.df
        d_buy = df['TotalBuyOrderVolume'] - df['TotalBuyOrderVolume'].shift(delta_t)
        d_sell = df['TotalSellOrderVolume'] - df['TotalSellOrderVolume'].shift(delta_t)
        imbalance1 = d_buy - d_sell

        if n==1:
            imbalance1 = pd.DataFrame(imbalance1,columns=['snapshot_direct_imbalance1_TotalOrderVolumne'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance1],axis=1).dropna()
        else:
            imbalance2 = imbalance1/(np.abs(d_buy)+np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2,columns=['ssnapshot_direct_imbalance2_TotalOrderVolumne'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance2],axis=1).dropna()

        return result


    # B9-B10
    def cal_snapshot_direct_imbalance_OrderNo(self,n,delta_t=1):
        
        df = self.df
        d_buy = df['BuyOrderNo'] - df['BuyOrderNo'].shift(delta_t)
        d_sell = df['SellOrderNo'] - df['SellOrderNo'].shift(delta_t)
        imbalance1 = d_buy - d_sell

        if n==1:
            imbalance1 = pd.DataFrame(imbalance1,columns=['snapshot_direct_imbalance1_OrderNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance1],axis=1).dropna()
        else:
            imbalance2 = imbalance1/(np.abs(d_buy)+np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2,columns=['ssnapshot_direct_imbalance2_OrderNo'])
            result = pd.concat([df[['Symbol','TradingTime']],imbalance2],axis=1).dropna()

        return result
    

    # B11-B14
    def average_volumne_change_imbalance(self,k,delta_t=1):
        
        df = self.df
        def cal_pct(trade_type):
            ratio = pd.DataFrame()
            for i in range(k):
                ratio[i+1] = df['{}Volume'.format(trade_type) + str(i + 1).zfill(2)]\
                                        /df['Total{}OrderNo'.format(trade_type) + str(i + 1).zfill(2)]
            avevol = ratio.mean(axis=1)/k
            avevol.loc[avevol==np.inf] = 0
            pct = (avevol - avevol.shift(1))/(avevol.shift(1))
            pct.loc[(pct==np.inf)] = 0
            pct = pct.fillna(0)
            return pct

        result = pd.DataFrame((cal_pct('Buy') - cal_pct('Sell')),columns=['average_volumne_change_imbalance_{}'.format(k)])
        result = pd.concat([df[['Symbol','TradingTime']],result],axis=1)

        return result

        # B15-16
    def cal_snapshot_direct_imbalance_VOL(self, n):
        """
        标的：国债期货
        变量：BuyVOL、SellVOL：内外盘
        含义：衡量内外盘增量的imbalance，B16进行了去量纲
        """
        df = self.future_tick_filled(data=self.df)
    
        d_buy = df['BuyVol'] - df['BuyVol'].shift(1)
        d_sell = df['SellVol'] - df['SellVol'].shift(1)
        imbalance1 = d_buy - d_sell

        if n == 1:
            imbalance1 = pd.DataFrame(imbalance1, columns=['snapshot_direct_imbalance_vol1'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imbalance1], axis=1).dropna()
        else:
            imbalance2 = imbalance1 / (np.abs(d_buy) + np.abs(d_sell))
            imbalance2 = pd.DataFrame(imbalance2, columns=['snapshot_direct_imbalance_vol2'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imbalance2], axis=1).dropna()

        return result

    # B17-18
    def cal_increment_imbalance_volume(self, n):

        df = self.future_tick_filled(data=self.df)
        level = self.level

        def cal_delta_volume(variable):
            acc_v = np.zeros(len(df))
            for index in range(1, level + 1):
                acc_v = acc_v + df['{}Volume'.format(variable) + str(index).zfill(2)]
            acc_v = acc_v.cumsum()
            return acc_v - acc_v.shift(1)

        d_buy = cal_delta_volume('Buy')
        d_sell = cal_delta_volume('Sell')
        imb1 = d_buy - d_sell

        if n == 1:
            imb1 = pd.DataFrame(imb1, columns=['increment_imbalance_volume1'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb1], axis=1).dropna()
        else:
            imb2 = imb1 / (np.abs(d_buy) + np.abs(d_sell))
            imb2 = pd.DataFrame(imb2, columns=['increment_imbalance_volume2'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb2], axis=1).dropna()

        return result

    # B19-28 
    def cal_increment_imbalance_price(self, n, i):
        """
        参数i：档位，股票：1...10；期货：1...5
        """
        df = self.future_tick_filled(data=self.df)

        def cal_delta_price(variable):
            p = df['{}Price'.format(variable) + str(i).zfill(2)]
            return p - p.shift(1)

        d_buy = cal_delta_price(variable='Buy')
        d_sell = cal_delta_price(variable='Sell')
        imb1 = d_buy - d_sell

        if n == 1:
            imb1 = pd.DataFrame(imb1, columns=['increment_imbalance_price1_%s'%i])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb1], axis=1).dropna()
        elif n == 2:
            imb2 = imb1 / (np.abs(d_buy) + np.abs(d_sell))
            imb2 = pd.DataFrame(imb2, columns=['increment_imbalance_price2_%s'%i])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb2], axis=1).dropna()

        return result

    # B29-30
    def cal_increment_imbalance_amount(self, n):

        df = self.future_tick_filled(data=self.df)
        level = self.level

        def cal_delta_amount(variable):
            amo = np.zeros(len(df))
            for j in range(level):
                weighted_amo = (level - j) / sum(range(level + 1)) \
                               * df['{}Price'.format(variable) + str(j + 1).zfill(2)] \
                               * df['{}Volume'.format(variable) + str(j + 1).zfill(2)]
                amo = amo + weighted_amo
            amo = amo.cumsum()
            return amo - amo.shift(1)

        d_buy = cal_delta_amount(variable='Buy')
        d_sell = cal_delta_amount(variable='Sell')
        imb1 = d_buy - d_sell

        if n == 1:
            imb1 = pd.DataFrame(imb1, columns=['increment_imbalance_amount1'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb1], axis=1).dropna()
        elif n == 2:
            imb2 = imb1 / (np.abs(d_buy) + np.abs(d_sell))
            imb2 = pd.DataFrame(imb2, columns=['increment_imbalance_amount2'])
            result = pd.concat([df[['Symbol', 'TradingTime']], imb2], axis=1).dropna()
        return result

    # B31
    def cal_relative_volume_change(self):

        df = self.future_tick_filled(data=self.df)

        # 计算开盘到现在的时间
        t = df['TradingTime']
        t = t - t.iloc[0]
        mins = t.dt.total_seconds() / 60  # 转换为分钟
        mins = mins - (90 * (mins > 210))  # 午盘时间调整

        vol = df['TotalVolume'] / mins
        change = vol / vol.shift(1) - 1

        change = pd.DataFrame(change, columns=['relative_volume_change'])
        change = pd.concat([df[['Symbol', 'TradingTime']], change], axis=1).dropna()

        return change

    # B32
    def cal_relative_amount_change(self):

        df = self.future_tick_filled(data=self.df)

        # 计算开盘到现在的时间
        t = df['TradingTime']
        t = t - t.iloc[0]
        mins = t.dt.total_seconds() / 60  # 转换为分钟
        mins = mins - (90 * (mins > 210))  # 午盘时间调整

        amo = df['TotalAmount'] / mins
        change = amo / amo.shift(1) - 1

        change = pd.DataFrame(change, columns=['relative_amount_change'])
        change = pd.concat([df[['Symbol', 'TradingTime']], change], axis=1).dropna()

        return change

class C_feature:
    #%% C1 
    def cal_C1(self):

        df = self.future_tick_filled(data=self.df) #用了一个自定义的函数
        level = self.level                         #level应该是继承来的
        factor_delta_num  = int(120*self.factor_back_time) #时间间隔的设置调整
        prev_rows = df.shift(1)

        def calculate_new_column(row,level, prev_rows,types):
            new_value = 0  
            cals_var = ['%sPrice'%types,'%sVolume'%types]
            List=np.array([x+str(y).zfill(2) for x in cals_var \
                           for y in range(1,level+1)]).reshape((2,-1))
            price_list = List[0]
            volume_list = List[1]
            packed_lists = zip(price_list, volume_list) #zip函数的作用再复习一下

            result = list(packed_lists)
            for col_price, col_vol in result:
                if row[col_price] >=  prev_rows.at[row.name, '%s01'%cals_var[0]]:  #一个判断语句，判断当前时间戳委买价大于等于前一个时间戳买一价的所有委买单的委托金额之和
                    new_value += row[col_price]*row[col_vol]           #01是指一价
            new_value = new_value - prev_rows.at[row.name, '%s01'%cals_var[0]]*prev_rows.at[row.name, '%s01'%cals_var[1]]
            
            return new_value
        
        def calculate_new_column1(row,level, prev_rows,types):
            new_value = 0  
            cals_var = ['%sPrice'%types,'%sVolume'%types]
            List=np.array([x+str(y).zfill(2) for x in cals_var \
                           for y in range(1,level+1)]).reshape((2,-1))
            price_list = List[0]
            volume_list = List[1]
            packed_lists = zip(price_list, volume_list)

            result = list(packed_lists)
            for col_price, col_vol in result:
                if row[col_price] <=  prev_rows.at[row.name, '%s01'%cals_var[0]]:  #一个判断语句，判断当前时间戳委买价大于等于前一个时间戳买一价的所有委买单的委托金额之和
                    new_value += row[col_price]*row[col_vol]
            new_value = new_value - prev_rows.at[row.name, '%s01'%cals_var[0]]*prev_rows.at[row.name, '%s01'%cals_var[1]]

            return new_value
    #注意这里委买增额和委卖增额的算法是有细微差别的！

        Increase_purchases = df.apply(lambda row: calculate_new_column(row,level,prev_rows,"Buy"), axis=1)
        Increase_sales = df.apply(lambda row: calculate_new_column1(row,level,prev_rows,"Sell"), axis=1)
        Net_increase = Increase_purchases - Increase_sales    
        Net_increase_all = Net_increase
        # Net_increase_all = sum([Net_increase.shift(i+1) for i in range(factor_delta_num)])
        Net_increase_all = pd.DataFrame(Net_increase_all, columns=['C1'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], Net_increase_all], axis=1).dropna()
        return result_df
    
    #%% C2-C3   这其中N指代的到底是什么
    def cal_VOI(self, n):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)
        prev_df = df.shift(1)

        def cal_voi(row, prev_df, types="Buy", level_num=1):
            cals_var = ['%sPrice%s'%(types,str(level_num).zfill(2)),'%sVolume%s'%(types,str(level_num).zfill(2))]  #这里level_number是指定了买卖档数
            price_change = row[cals_var[0]] - prev_df.at[row.name, cals_var[0]]                                    #zfill(2) 方法确保它至少有两位数字（如果不足两位，前面会补零）。这个值将被放置在第二个 %s 的位置。
            if price_change < 0:
                return 0
            elif price_change > 0:
                return row[cals_var[1]]
            else:
                return row[cals_var[1]] - prev_df.at[row.name, cals_var[1]]
    
        if n == 2:
            Bid_VOI_1 = df.apply(lambda row: cal_voi(row, prev_df, "Buy",level_num=1), axis=1)
            Ask_VOI_1 = df.apply(lambda row: cal_voi(row, prev_df, "Sell",level_num=1), axis=1)
            VOI = Bid_VOI_1 - Ask_VOI_1
        elif n == 3:
            Bid_VOI_Weighted = np.zeros(len(df))
            Ask_VOI_Weighted = np.zeros(len(df))
            
            for index in range(1,level+1):
                omgea_value = (level+1-index)/(sum(range(level+1)))   #这里应该要再除以2，买卖的N都应该被计算进去
                Bid_VOI_Weighted +=  df.apply(lambda row: cal_voi(row, prev_df, "Buy",level_num=index), axis=1)*omgea_value
                Ask_VOI_Weighted +=  df.apply(lambda row: cal_voi(row, prev_df, "Sell",level_num=index), axis=1)*omgea_value

            VOI = Bid_VOI_Weighted - Ask_VOI_Weighted
        else:
            print("Please choice correct imbalance !")

        VOI_all = VOI
        # VOI_all = sum([VOI.shift(i) for i in range(factor_delta_num)])
        VOI_all = pd.DataFrame(VOI_all, columns=['C_%s'%n])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], VOI_all], axis=1).dropna()
        
        return result_df
    
    #%% C4-C6  #权重设置还是有点问题
    def cal_MOFI(self, n):
        """
        Feature meaning: 

        Parameter Q: 
            n:  C type (4, 5, 6)
        """

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)
        prev_df = df.shift(1)

        def cal_voi(row, prev_df, types="Buy", level_num=1):
            cals_var = ['%sPrice%s'%(types,str(level_num).zfill(2)),'%sVolume%s'%(types,str(level_num).zfill(2))]
            price_change = row[cals_var[0]] - prev_df.at[row.name, cals_var[0]]

            if price_change < 0:
                return prev_df.at[row.name, cals_var[1]]*(-1)
            elif price_change > 0:
                return row[cals_var[1]]
            else:
                return row[cals_var[1]] - prev_df.at[row.name, cals_var[1]]

        MOFI = np.zeros(len(df))
        for index in range(1,level+1):
            Bid_VOI_i = df.apply(lambda row: cal_voi(row, prev_df, "Buy",level_num=index), axis=1)
            Ask_VOI_i = df.apply(lambda row: cal_voi(row, prev_df,"Sell",level_num=index), axis=1)
            VOI = Bid_VOI_i - Ask_VOI_i
            if n == 4:
                MOFI += VOI/level  #统一权重,有几个level就填几层
            elif n == 5:
                omgea_value = (level+1-index)/sum(range(level+1))
                MOFI += VOI*omgea_value
            elif n == 6:
                omgea_value = (index)/sum(range(level+1))
                MOFI += VOI*omgea_value
            else:
                print("Please choice correct imbalance type !")

        VOI_all = MOFI
        # VOI_all = sum([MOFI.shift(i) for i in range(factor_delta_num)])
        VOI_all = pd.DataFrame(VOI_all, columns=['C_%s'%n])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], VOI_all], axis=1).dropna()

        return result_df
    
    #%% C7
    def cal_C7(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        def cal_voi(df, types="Buy"):
            cals_var = ['%sPrice'%types,'%sVolume'%types]
            List=np.array([x+str(y).zfill(2) for x in cals_var \
                           for y in range(1,level+1)]).reshape((2,-1))
            price_list = List[0]
            volume_list = List[1]
            packed_lists = zip(price_list, volume_list)
            result = list(packed_lists)

            if types=="Buy":
                min_price = price_list[-1]
                max_price = price_list[0]
            elif types=="Sell":
                min_price = price_list[0]
                max_price = price_list[-1]

            df['Shifted_%s'%max_price] = df[max_price].shift(1)
            price_up_min = df[[max_price, 'Shifted_%s'%max_price]].min(axis=1,skipna=False).values
            df['Shifted_%s'%min_price] = df[min_price].shift(1)
            price_down_max = df[[min_price, 'Shifted_%s'%min_price]].max(axis=1).values
            result_min = df[price_list] >= price_up_min[:, np.newaxis]
            result_max = df[price_list] <= price_down_max[:, np.newaxis]
            
            result = result_max & result_min
            sum_data = (df[volume_list] * result.values).sum(axis=1)
            delta_vol_imb = sum_data - sum_data.shift(1)
            return delta_vol_imb

        buy_sell_vol_imb = cal_voi(df, types="Buy") - cal_voi(df, types="Sell")
        vol_imb_sum = buy_sell_vol_imb
        # vol_imb_sum = sum([buy_sell_vol_imb.shift(i) for i in range(factor_delta_num)])
        vol_imb_sum = pd.DataFrame(vol_imb_sum, columns=['C_7'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], vol_imb_sum], axis=1).dropna()

        return result_df


    #%% C8
    def cal_C8(self):
        
        df= self.df
        df= self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['TP'] = df['TotalAmount'].diff()  / df['TotalVolume'].diff() 
        df = df.fillna(method='ffill')

        df['MidPrice'] = (df['BuyPrice01'] + df['SellPrice01'])/2
        df['MidPrice_delta'] = (df['MidPrice'] + df['MidPrice'].shift(1))/2
        
        """divided 1e4 for Government Bond Futures feture: 1e2 euqals 1e6"""
        df['MPB'] = df['TP']/1e4 - df['MidPrice_delta']  #这里不是很懂
        mean_value = df['MPB']
        # mean_value = df['MPB'].rolling(factor_delta_num).mean().rename(None)
        mean_value = pd.DataFrame(mean_value, columns=['C_8'])
        result_df = pd.concat([df[['Symbol','TradingTime']], mean_value], axis=1).dropna()
        
        return result_df

    #%% C9
    def cal_C9(self):
        """Buying sentiment, positive indicates strong market buying and possible price increase"""
        df = self.future_tick_filled(data=self.df)
        factor_delta_num  = int(120*self.factor_back_time)

        delta_vol = df['TotalVolume'].diff()
        df['TP'] = df['TotalAmount'].diff()  / delta_vol
        df['TP'] = df['TP'].fillna(method='ffill')/1e4
        delta_vol_ask_sel = df['TP'] > df['SellPrice01'].shift(1)
        delta_vol_bid_sel = df['TP'] < df['BuyPrice01'].shift(1)
        delta_vol_ask = delta_vol*delta_vol_ask_sel
        delta_vol_bid = delta_vol*delta_vol_bid_sel
        buy_sell_vol_imb = delta_vol_ask - delta_vol_bid

        vol_imb_sum = buy_sell_vol_imb
        # vol_imb_sum = buy_sell_vol_imb.rolling(factor_delta_num).sum()
        vol_imb_sum = pd.DataFrame(vol_imb_sum, columns=['C_9'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], vol_imb_sum], axis=1).dropna()

        return result_df
    
    def PricesAndVolumes(self):
        df = self.future_tick_filled(data=self.df)
        level = self.level
        List = np.array(
            [x + str(y).zfill(2) for x in
             ['BuyPrice', 'BuyVolume', 'SellPrice', 'SellVolume']
             for y in range(1, level + 1)]).reshape((4, -1))  # reshape(6,-1) 意为自动转为6行
        P_Bid = df[List[0]].values
        V_Bid = df[List[1]].values
        P_Ask = df[List[2]].values
        V_Ask = df[List[3]].values
        return P_Bid, V_Bid, P_Ask, V_Ask
    
    
    def cal_initiative_trade(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        P_Bid, V_Bid, P_Ask, V_Ask = self.PricesAndVolumes()

        # 特征C10，净主买额，正数表示为买方力量强
        def get_sum2(P, V, p_flag2):
            before_V = np.where(P[:-1] == p_flag2.reshape(-1, 1), V[:-1], 0).sum(axis=1)
            after_V = np.where(P[1:] == p_flag2.reshape(-1, 1), V[1:], 0).sum(axis=1)
            sum2 = np.where(before_V > after_V, before_V - after_V, 0)
            return sum2

        p_flag1 = P_Bid[
            range(1, len(P_Bid)), (P_Bid[1:] <= P_Bid[:-1, 0].reshape(-1, 1)).argmax(axis=1)]
        # 小于等于上一时刻一档价格的最大值
        p_flag2 = P_Bid[
            range(0, len(P_Bid) - 1), (P_Bid[:-1] <= p_flag1.reshape(-1, 1)).argmax(axis=1)]
        # 上一时刻小于等于p_flag1的价格的最大值

        sum1 = np.where(P_Bid[:-1] > p_flag2.reshape(-1, 1), P_Bid[:-1] * V_Bid[:-1], 0).sum(axis=1)
        sum_totalAsk = sum1 + get_sum2(P_Bid, V_Bid, p_flag2) * p_flag2

        p_flag1 = P_Ask[range(1, len(P_Ask)), (P_Ask[1:] >= P_Ask[:-1, 0].reshape(-1, 1)).argmax(axis=1)]
        p_flag2 = P_Ask[range(0, len(P_Ask) - 1), (P_Ask[:-1] >= p_flag1.reshape(-1, 1)).argmax(axis=1)]

        sum1 = np.where(P_Ask[:-1] < p_flag2.reshape(-1, 1), P_Ask[:-1] * V_Ask[:-1], 0).sum(axis=1)
        sum_totalBid = sum1 + get_sum2(P_Ask, V_Ask, p_flag2) * p_flag2
        initiative_trade = np.hstack([np.nan, sum_totalBid - sum_totalAsk])

        return initiative_trade/1e4

    def cal_C10(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)
        
        initiative_trade = self.cal_initiative_trade()
        initiative_trade = pd.Series(initiative_trade)
        initiative_trade_sum = initiative_trade
        # initiative_trade_sum = initiative_trade.rolling(factor_delta_num).sum()
        initiative_trade_sum = pd.DataFrame(initiative_trade_sum, columns=['C_10'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], initiative_trade_sum], axis=1).dropna()
    
        return result_df
    
    def cal_NetInnerAmount(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        P_Bid, V_Bid, P_Ask, V_Ask = self.PricesAndVolumes()

        # 特征C11，净内部增额
        def write_easy(P, V):
            return P[:-1], P[1:], V[:-1], V[1:]

        # 撤单信息分解
        # 符合条件的撤单求和
        def sum_cancel(boundary, P_Before, P_After, V_Before, V_After, level=10):
            sumV = np.zeros(P_Before.shape[0])
            for i in range(level):
                NowP = np.where((P_Before[:, i] <= boundary[:, 0]) & (P_Before[:, i] >= boundary[:, 1]),
                                P_Before[:, i], 0)
                sum_now = np.where(P_Before == NowP.reshape(-1, 1), V_Before, 0).sum(axis=1) - np.where(
                    P_After == NowP.reshape(-1, 1), V_After, 0).sum(axis=1)
                sumV += np.where(sum_now > 0, sum_now, 0) * NowP
            return sumV

        import math
        P_Before, P_After, V_Before, V_After = write_easy(P_Bid, V_Bid)
        boundary1 = np.vstack((P_After[:, math.ceil(level / 2)], P_After[:, -1])).T
        Cancel_Bid5plus = sum_cancel(boundary1, P_Before, P_After, V_Before, V_After, level)  # 5档以外的撤单
        boundary2 = np.vstack((P_After[:, 0], P_After[:, math.ceil(level / 2) - 1])).T
        Cancel_Bid5minus = sum_cancel(boundary2, P_Before, P_After, V_Before, V_After, level)  # 5档以内的撤单

        P_Before, P_After, V_Before, V_After = write_easy(P_Ask, V_Ask)
        boundary1 = np.vstack((P_After[:, -1], P_After[:, math.ceil(level / 2)])).T
        Cancel_Ask5plus = sum_cancel(boundary1, P_Before, P_After, V_Before, V_After, level)
        boundary2 = np.vstack((P_After[:, math.ceil(level / 2)], P_After[:, 0])).T
        Cancel_Ask5minus = sum_cancel(boundary2, P_Before, P_After, V_Before, V_After, level)

        # 挂单信息分解
        # 符合条件的挂单求和
        def sum_add(boundary, P_Before, P_After, V_Before, V_After, level=10):
            sumV = np.zeros(P_Before.shape[0])
            for i in range(level):
                NowP = np.where((P_After[:, i] <= boundary[:, 0]) & (P_After[:, i] >= boundary[:, 1]), P_After[:, i], 0)
                sum_now = np.where(P_After == NowP.reshape(-1, 1), V_After, 0).sum(axis=1) - np.where(
                    P_Before == NowP.reshape(-1, 1), V_Before, 0).sum(axis=1)
                sumV += np.where(sum_now > 0, sum_now, 0) * NowP
            return sumV

        P_Before, P_After, V_Before, V_After = write_easy(P_Bid, V_Bid)  # TODO
        boundary1 = np.vstack((P_Before[:, 0], P_Before[:, math.ceil(level / 2) - 1])).T
        Add_Bid5minus = sum_add(boundary1, P_Before, P_After, V_Before, V_After, level)  # 5档以内的挂单
        boundary2 = np.vstack((P_Before[:, math.ceil(level / 2)], P_Before[:, -1])).T
        Add_Bid5plus = sum_add(boundary2, P_Before, P_After, V_Before, V_After, level)  # 5档以外的挂单

        P_Before, P_After, V_Before, V_After = write_easy(P_Ask, V_Ask)
        boundary1 = np.vstack((P_Before[:, math.ceil(level / 2) - 1], P_Before[:, 0])).T
        Add_Ask5minus = sum_add(boundary1, P_Before, P_After, V_Before, V_After, level)
        boundary2 = np.vstack((P_Before[:, -1], P_Before[:, math.ceil(level / 2)])).T
        Add_Ask5plus = sum_add(boundary2, P_Before, P_After, V_Before, V_After, level)
        NetInnerAmount = (Add_Bid5minus - Cancel_Bid5minus) - (Add_Ask5minus - Cancel_Ask5minus)
        NetInnerAmount = np.hstack([np.nan, NetInnerAmount])

        return NetInnerAmount/1e4

    def cal_C11(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        NetInnerAmount =  self.cal_NetInnerAmount()
        NetInnerAmount = pd.Series(NetInnerAmount)
        NetInnerAmount_sum = NetInnerAmount
        # NetInnerAmount_sum = NetInnerAmount.rolling(factor_delta_num).sum()
        NetInnerAmount_sum = pd.DataFrame(NetInnerAmount_sum, columns=['C_11'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], NetInnerAmount_sum], axis=1).dropna()
    
        return result_df
    

    def cal_C12(self):

        df = self.future_tick_filled(data=self.df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        P_Bid, V_Bid, P_Ask, V_Ask = self.PricesAndVolumes()

        # 特征C12，净买入意愿=净内部增额+净委买增额+净主买额
        # 净委买增额
        def SameP_DeltaV(P, V):
            V_after = np.where(P[1:] == P[:-1, 0].reshape((-1, 1)), V[1:], 0).sum(axis=1)
            sum2 = np.where(V_after > V[:-1, 0], (V_after - V[:-1, 0]) * P[:-1, 0], 0)
            return sum2

        sum1 = np.where(P_Bid[1:] > P_Bid[:-1, 0].reshape((-1, 1)), P_Bid[1:] * V_Bid[1:], 0).sum(axis=1)
        BidIncrease = sum1 + SameP_DeltaV(P_Bid, V_Bid)
        sum1 = np.where(P_Ask[1:] < P_Ask[:-1, 0].reshape((-1, 1)), P_Ask[1:] * V_Ask[1:], 0).sum(axis=1)
        AskIncrease = sum1 + SameP_DeltaV(P_Ask, V_Ask)
        NetBidIncrease = np.hstack([np.nan, BidIncrease - AskIncrease])/1e4
        
        initiative_trade = self.cal_initiative_trade()
        NetInnerAmount =  self.cal_NetInnerAmount()
        NetWill = NetInnerAmount + NetBidIncrease + initiative_trade

        NetWill = pd.Series(NetWill)
        NetWill_sum = NetWill
        # NetWill_sum = NetWill.rolling(factor_delta_num).sum()
        NetWill_sum = pd.DataFrame(NetWill_sum, columns=['C_12'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], NetWill_sum], axis=1).dropna()

        return result_df

class D_feature:
 
    #D1
    def cal_realized_absolute_variation(self):
        
        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = abs(df['LastPrice'].shift(-1) - df['LastPrice'])
        sum_return = sum([df['return'].shift(i) for i in range(factor_delta_num)])
        RAbsVar = np.sqrt(np.pi/(2*factor_delta_num))*sum_return
        RAbsVar = RAbsVar.rename(None)
        RAbsVar = pd.DataFrame(RAbsVar, columns=['realized_absolute_variation'])

        result_df = pd.concat([df[['Symbol', 'TradingTime']], RAbsVar], axis=1).dropna()
        return result_df
    

    #%% D2
    def cal_realized_volatility(self):
        
        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return_square'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**2
        RVol = sum([df['return_square'].shift(i) for i in range(factor_delta_num)])
        RVol = RVol.rename(None)
        RVol = pd.DataFrame(RVol, columns=['realized_volatility'])

        result_df = pd.concat([df[['Symbol', 'TradingTime']], RVol], axis=1).dropna()
        return result_df
    
    #%% D3
    def cal_realized_skew(self):
        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return_square'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**2
        df['return_cube'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**3

        RVol = sum([df['return_square'].shift(i) for i in range(factor_delta_num)])
        RCube = sum([df['return_cube'].shift(i) for i in range(factor_delta_num)])

        RSkew = np.sqrt(factor_delta_num)*RCube/pow(RVol, 3/2)
        RSkew = RSkew.rename(None)
        RSkew = pd.DataFrame(RSkew, columns=['realized_skew'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], RSkew], axis=1).dropna()
        
        return result_df
    
    #%% D4
    def cal_realized_kurtosis(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return_square'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**2
        df['return_four_power'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**4

        RVol = sum([df['return_square'].shift(i) for i in range(factor_delta_num)])
        Rfour_power = sum([df['return_four_power'].shift(i) for i in range(factor_delta_num)])

        RKur = factor_delta_num * Rfour_power/pow(RVol, 2)
        RKur = RKur.rename(None)
        RKur = pd.DataFrame(RKur, columns=['realized_kurtosis'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], RKur], axis=1).dropna()
        
        return result_df
    
    #%% D5
    def cal_integrated_quarticity(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = df['LastPrice'].shift(-1) - df['LastPrice']
        
        Rquarticity = sum([(100*df['return'].shift(i))**4 for i in range(factor_delta_num)])
        Intq = factor_delta_num / 3 * Rquarticity
        Intq = Intq.rename(None)
        Intq = pd.DataFrame(Intq, columns=['integrated_quarticity'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], Intq], axis=1).dropna()
        
        return result_df
    
    #%% D6
    def cal_realized_bipower_variation(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])
        df['return_quarticity'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**4

        RR_delta = abs(df['return']*df['return'].shift(1))
        Sum_RR_delta = sum([RR_delta.shift(i) for i in range(factor_delta_num-1)])

        RBip = (np.pi * factor_delta_num) / (2*(factor_delta_num-2)) * Sum_RR_delta
        RBip = RBip.rename(None)
        RBip = pd.DataFrame(RBip, columns=['realized_bipower_variation'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], RBip], axis=1).dropna()
        
        return result_df
    
    #%% D7
    def cal_jump_process(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])
        df['return_quarticity'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**4
        df['return_square'] = (df['LastPrice'].shift(-1) - df['LastPrice'])**2

        RVol = sum([df['return_square'].shift(i) for i in range(factor_delta_num)])
        RR_delta = abs(df['return']*df['return'].shift(1))
        Sum_RR_delta = sum([RR_delta.shift(i+1) for i in range(factor_delta_num-1)])

        RBip = (np.pi * factor_delta_num) / (2*(factor_delta_num-2)) * Sum_RR_delta
        zero = np.zeros(len(df))
        Jump = np.max([RVol-RBip, zero],axis=0)
        Jump = pd.DataFrame(Jump, columns=['jump_process'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], Jump], axis=1).dropna()

        return result_df
    
    #%% D8
    def cal_downside_volatility_proportion(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])
        df['return_square'] = df['return']**2
        df['return_less_zeros'] = df['return']
        df['return_less_zeros'][df['return']>0] = 0

        RVol_less_zero = sum([(df['return_less_zeros'].shift(i))**2 for i in range(factor_delta_num)])
        RVol = sum([df['return_square'].shift(i) for i in range(factor_delta_num)])
        DVol = RVol_less_zero / RVol
        DVol = DVol.rename(None)
        DVol = pd.DataFrame(DVol, columns=['downside_volatility_proportion'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], DVol], axis=1).dropna()
        
        return result_df
    
    #%% D9
    def cal_trend_strength(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])

        RAbsVar = sum([abs(df['return'].shift(i)) for i in range(factor_delta_num)])
        RVar = sum([df['return'].shift(i) for i in range(factor_delta_num)])
        trendstrength = RVar / RAbsVar
        trendstrength = trendstrength.rename(None)
        trendstrength = pd.DataFrame(trendstrength, columns=['trend_strength'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], trendstrength], axis=1).dropna()
        
        return result_df
#%% D10
    def cal_net_inflow_ratio(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])

        df['deltavolume'] = df['TotalVolume'] - df['TotalVolume'].shift(1)
        df['deltaamount'] = df['TotalAmount'] - df['TotalAmount'].shift(1)

        df['deltavolume_large_zero'] = np.where(df['return']>0, df['deltavolume'], 0)
        df['deltaamount_large_zero'] = np.where(df['return']>0, df['deltaamount'], 0) 
        df['deltavolume_small_zero'] = np.where(df['return']<0, df['deltavolume'], 0)  
        df['deltaamount_small_zero'] = np.where(df['return']<0, df['deltaamount'], 0)
        df.at[0,'deltavolume_small_zero'] = 0
        df.at[0,'deltaamount_small_zero'] = 0

        deltavolume_large_zero = sum([df['deltavolume_large_zero'].shift(i) for i in range(factor_delta_num)])
        deltaamount_large_zero = sum([df['deltaamount_large_zero'].shift(i) for i in range(factor_delta_num)])
        deltavolume_small_zero = sum([df['deltavolume_small_zero'].shift(i) for i in range(factor_delta_num)])
        deltaamount_small_zero = sum([df['deltaamount_small_zero'].shift(i) for i in range(factor_delta_num)])

        AmtPerTrdInFlow = deltaamount_large_zero / deltavolume_large_zero
        AmtPerTrdOutFlow = deltaamount_small_zero / deltavolume_small_zero
        NetInflowRatio = AmtPerTrdInFlow / AmtPerTrdOutFlow
        NetInflowRatio = NetInflowRatio.rename(None)
        NetInflowRatio = pd.DataFrame(NetInflowRatio, columns=['net_inflow_ratio'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], NetInflowRatio], axis=1).dropna()

        return result_df
    
    #%% D11
    def cal_amount_flow_imbalance(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])
        df['deltaamount'] = df['TotalAmount'] - df['TotalAmount'].shift(1)
        df['deltaamount_large_zero'] = np.where(df['return']>0, df['deltaamount'], 0) 
        df['deltaamount_small_zero'] = np.where(df['return']<0, df['deltaamount'], 0)
        df.at[0,'deltaamount_small_zero'] = 0

        deltaamount_large_zero = sum([df['deltaamount_large_zero'].shift(i) for i in range(factor_delta_num)])
        deltaamount_small_zero = sum([df['deltaamount_small_zero'].shift(i) for i in range(factor_delta_num)])

        AmtFI = deltaamount_large_zero - deltaamount_small_zero
        AmtFI = AmtFI.rename(None)
        AmtFI = pd.DataFrame(AmtFI, columns=['amount_flow_imbalance'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], AmtFI], axis=1).dropna()

        return result_df

    #%% D12
    def cal_num_flow_imbalance(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        factor_delta_num  = int(120*self.factor_back_time)
        df['return'] = (df['LastPrice'].shift(-1) - df['LastPrice'])
        df['deltavolume'] = df['TotalVolume'] - df['TotalVolume'].shift(1)
        df['deltavolume_large_zero'] = np.where(df['return']>0, df['deltavolume'], 0)
        df['deltavolume_small_zero'] = np.where(df['return']<0, df['deltavolume'], 0)  
        df.at[0,'deltavolume_small_zero'] = 0

        deltavolume_large_zero = sum([df['deltavolume_large_zero'].shift(i) for i in range(factor_delta_num)])
        deltavolume_small_zero = sum([df['deltavolume_small_zero'].shift(i) for i in range(factor_delta_num)])

        NumFI = deltavolume_large_zero -  deltavolume_small_zero
        NumFI = NumFI.rename(None)
        NumFI = pd.DataFrame(NumFI, columns=['num_flow_imbalance'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], NumFI], axis=1).dropna()

        return result_df
    
    #%% D13 only use the information between  and T-deltaT
    def cal_modeling_shape_feature(self):

        df = self.df
        df = self.future_tick_filled(data=df)
        level = self.level
        factor_delta_num  = int(120*self.factor_back_time)

        def PricesAndVolumes(data):
            List=np.array([x+str(y).zfill(2) for x in ['BuyPrice','BuyVolume','SellPrice','SellVolume'] \
                           for y in range(1,level+1)]).reshape((4,-1))
            P_Bid=data[List[0]].values
            V_Bid=data[List[1]].values
            P_Ask=data[List[2]].values
            V_Ask=data[List[3]].values

            return P_Bid,V_Bid,P_Ask,V_Ask

        P_Bid,V_Bid,P_Ask,V_Ask = PricesAndVolumes(df)

        def MeanSigmaSkew(P,V):   #calcuate mean and sigma of price according volume distribution

            prob=V/((V.sum(axis=1)).reshape((-1,1)))
            mu=(P*prob).sum(axis=1)
            sigma2=np.abs((P**2*prob).sum(axis=1)-mu**2)
            return mu,sigma2

        mean_Bid,sigma2_Bid = MeanSigmaSkew(P_Bid,V_Bid)
        mean_Ask,sigma2_Ask = MeanSigmaSkew(P_Ask,V_Ask)
        df['mean_Bid'] = mean_Bid
        df['sigma2_Bid'] = sigma2_Bid
        df['mean_Ask'] = mean_Ask
        df['sigma2_Ask'] = sigma2_Ask
        alpha_t_bid = df['mean_Bid'] - df['mean_Bid'].shift(factor_delta_num)
        alpha_t_ask = df['mean_Ask'] - df['mean_Ask'].shift(factor_delta_num)
        beta_t_bid = sum([df['sigma2_Bid'].shift(i+1)/2 for i in range(5)])/5
        beta_t_ask = sum([df['sigma2_Ask'].shift(i+1)/2 for i in range(5)])/5
        alpha_e = (alpha_t_bid*beta_t_ask + alpha_t_ask*beta_t_bid)/(beta_t_bid+beta_t_ask)

        df['MidPrice'] = (df['BuyPrice01'] + df['SellPrice01'])/2
        pricing_error = np.log(df['MidPrice']).shift(factor_delta_num) - alpha_e - np.log(df['MidPrice'])
        pricing_error = pd.DataFrame(pricing_error, columns=['modeling_shape_feature'])
        result_df = pd.concat([df[['Symbol', 'TradingTime']], pricing_error], axis=1).dropna()

        return result_df

class Special_future_feature:
                    
    #TradeVolume
    def cal_trade_volume(self):
        
        df = self.df
        result_df = df[['Symbol', 'TradingTime', 'TradeVolume']]

        return result_df 
    
    #TradeAmount
    def cal_trade_amount(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','TradeAmount']]

        return result_df

     #TotalPosition
    def cal_total_positon(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','TotalPosition']]

        return result_df

    #PrePositionChange
    def cal_prepositon_change(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','PrePositionChange']]

        return result_df   

     #Change
    def cal_change(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','Change']]

        return result_df

    #ChangeRatio
    def cal_change_ratio(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','ChangeRatio']]

        return result_df   

     #PositionChange
    def cal_position_change(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','PositionChange']]

        return result_df

    #AveragePrice
    def cal_average_change(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','AveragePrice']]

        return result_df   
    
     #OrderRate
    def cal_order_rate(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','OrderRate']]

        return result_df

    #OrderDiff
    def cal_order_diff(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','OrderDiff']]

        return result_df  

    #Amplitude
    def cal_amplitude(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','Amplitude']]

        return result_df 

    #VolRate
    def cal_volrate(self):

        df = self.df
        result_df = df[['Symbol', 'TradingTime','VolRate']]

        return result_df  
