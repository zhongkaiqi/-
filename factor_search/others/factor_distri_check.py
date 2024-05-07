import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def factor_distri_check(factor_result_df,check_day_num,columns=["ask_prices_differences_4"]):

    # columns = ["ask_prices_differences_4"]
    with PdfPages("./temp.pdf") as pdf:
        for column_one in columns:
            df = factor_result_df
            df['TradingDate'] = df['TradingTime'].dt.date
            grouped_df = df.groupby(['TradingDate'])
            index=0

            def rolling_skew(rolling_series):
                return rolling_series.skew()

            for date, df_oneday in grouped_df: 
                index+=1
                if index > check_day_num:
                    break
                
                sampled_df = df_oneday[0:400]
                fig, ax = plt.subplots(figsize=(15,7))

                column_data = sampled_df[column_one].dropna()
                rolling_skewness = column_data.rolling(window=120).apply(rolling_skew)

                ax2 = ax.twinx()
                ax.plot(range(len(column_data)), column_data,'r.',label=column_one)
                ax2.plot(range(len(column_data)),  rolling_skewness,'b.',label="ts_skew")
                ax.legend(frameon=True,loc=[0.5,1.02])
                ax2.legend(frameon=True,loc=[0.0,1.02])
                ax.set_ylabel("base A value")
                ax2.set_ylabel("Skew")
                ax2.set_xlabel("tick")
                plt.tight_layout()
                pdf.savefig(fig)

                print(rolling_skewness)