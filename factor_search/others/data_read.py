import pandas as pd
import numpy as np

#数据读取
import os
os.chdir('D:/OneDrive/test-LL/swhy/sw_work/data/Tick_data_20230606')
drs = [pd.read_parquet(fname,columns=
            ['symbol','tradingdate','tradingtime','lastprice',
          'buyprice01','buyprice02','buyprice03','buyprice04','buyprice05',
          'sellprice01','sellprice02','sellprice03','sellprice04','sellprice05',
          'buyvolume01','buyvolume02','buyvolume03','buyvolume04','buyvolume05',
          'sellvolume01','sellvolume02','sellvolume03','sellvolume04','sellvolume05']
                      )for fname in os.listdir() if '2020' in fname or '2021' in fname]
       
df = pd.concat(drs,ignore_index=True)

df.columns = ['Symbol','TradingDate','TradingTime','LastPrice',
          'BuyPrice01','BuyPrice02','BuyPrice03','BuyPrice04','BuyPrice05',
          'SellPrice01','SellPrice02','SellPrice03','SellPrice04','SellPrice05',
          'BuyVolume01','BuyVolume02','BuyVolume03','BuyVolume04','BuyVolume05',
          'SellVolume01','SellVolume02','SellVolume03','SellVolume04','SellVolume05']

df['TradingDate'] = pd.to_datetime(df['TradingDate'],format='%Y%m%d')
df['TradingTime'] = pd.to_datetime(df['TradingTime'])
         
for c in df.columns:
    try:
        df[c] = df[c].astype(np.float64)
    except:
        continue
print(df)