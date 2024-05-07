import pandas as pd
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("AGG")
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("/afs/ihep.ac.cn/users/f/fuys/.config/matplotlib/Paper.mplstyle")

data_path = "/dybfs2/nEXO/fuys/others/sw_work"

class Base_ana():
    """
    Summary: this class is used to analysis the calculate results of factor: 
    
    Attributes:
        df_data (DataFrame): dataframe stored the A value V.S. B value
    
    Methods:
        corr_heatmap()
    """
    def __init__(self, df_data, factor_type, level=None):
        self.df = df_data
        self.factor_type = factor_type
        self.level = level

    def corr_heatmap(self):
        """ calculate corr between IC/(factor value) of factors."""

        df = self.df
        try:
            df = df.drop(['Symbol','TradingDate'])

        except:
            df = df
            print("Don't find columns Unnamed: 0")
        correlation_matrix = df.corr()
        import seaborn as sns
        plt.figure(figsize=(20, 20))
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 20})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60,fontsize=30)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=30)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(data_path+"/plots/results_ana/%s_%s_level_corr_heatmap.pdf"%(self.factor_type,self.level))



class IC_results_ana(Base_ana):
    """
    Summary: this class is used to analysis the calculate results of factor: 
    
    Attributes:
        df_data (DataFrame): dataframe stored the IC value V.S. time/IC value 
    
    Methods:
        ic_vs_ic_delta()
        ic_vs_time()
        cal_icstat()
        cal_t_teststat()
    """
    def __init__(self, df_data, factor_type, level=None):
        self.df = df_data
        self.factor_type = factor_type
        self.level = level
        super().__init__(df_data, factor_type, level)

    def ic_vs_ic_delta(self):
        """ visualize the corr between IC of factors"""
        df = self.df
        try:
            columns_list = df.columns.drop(['Symbol','TradingDate'])
        except:
            columns_list = df.columns
        with PdfPages(data_path+'/plots/results_ana/%s_type_IC_vs_IC_corr_multipage_plots.pdf'%self.factor_type) as pdf:
            for column in range(len(columns_list)):
                for column_in in range(column+1,len(columns_list)):
                    fig, ax = plt.subplots(figsize=(9, 9))
                    correlation = df[columns_list[column]].corr(df[columns_list[column_in]])
                    ax.plot(df[columns_list[column]].to_numpy(),df[columns_list[column_in]].to_numpy(),'.',\
                            label="corr:%0.3f" %( correlation))

                    ax.set_title("IC V.S IC")
                    ax.set_xlabel(columns_list[column])
                    ax.set_ylabel(columns_list[column_in])
                    ax.legend(frameon=True, edgecolor='red',fontsize=20)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)  

    def ic_vs_time(self):
        """Study IC stability varies with time"""
        df = self.df
        from scipy import optimize as op
        try:
            columns_list = df.columns.drop(['Symbol','TradingDate','Unnamed: 0'])
        except:
            columns_list = df.columns
        with PdfPages(data_path+'/plots/results_ana/%s_type_IC_vs_time_multipage_plots.pdf'%self.factor_type) as pdf:
            for column in columns_list:
                fig = plt.figure(figsize=(15,7))
                ax1 = fig.add_subplot(1,1,1)  
                df[column] = df[column]
                ax1.plot(df[column],'.',\
                        label="IC mean:%0.3f \n IC std:%0.3f"%(np.mean(df[column]),np.std(df[column])))
                
                import statsmodels.api as sm

                result = sm.tsa.adfuller(df[column])
                """Extract some key information from ADF test results"""
                adf_statistic = result[0]
                critical_values = result[4]
                p_value = result[1]
                print(critical_values)

                def f_1(x, A, B):
                    return A * x + B
                A, B = op.curve_fit(f_1, range(len(df)),df[column])[0]

                correlation_matrix = np.corrcoef(range(len(df)), df[column])
                correlation = correlation_matrix[0, 1]
                
                """use random time series to compare IC values ranges with time"""
                # random_numbers = np.random.normal(np.mean(df[column]), np.std(df[column]), len(df[column]))
                # A_rand, B_rand = op.curve_fit(f_1, range(len(df)),random_numbers)[0]
                # ax1.plot(x,random_numbers,label="random ",color='red',marker='.',markersize=1.2, linestyle='None',markerfacecolor='none')
                # y_rand = A_rand*x + B_rand
                # ax1.plot(x,y_rand,label="fit A_rand:%2.4f \n fit B_rand:%2.4f"%(A_rand,B_rand),color='red',linestyle="--")

                x = np.arange(len(df))
                y = A*x + B
                ax1.plot(x,y,label="fit A:%2.4f \n fit B:%2.4f"%(A,B),color='blue')
                ax1.set_title(column+ " \n Time corr: %0.3f  ADF stat: %0.2f P(0.01): %0.2f"\
                              %(correlation,adf_statistic,critical_values['1%']))
                ax1.set_xlabel('Time (day)')
                ax1.set_ylabel('IC')
                ax1.legend(frameon=True, edgecolor='red',fontsize=16,ncol=5)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)  
        print("PDF saved successfully.")

    def cal_icstat(self,ic_range=[0.01, 0.06] ,ic_file_info = "default_name"):
        """analysis IC result of factor"""
        ic_df = self.df
        try:
            column_list = ic_df.columns.drop(['Symbol','TradingDate','Unnamed: 0'])
        except:
            column_list = ic_df.columns
        result_df = pd.DataFrame()
        for column in column_list:
            ic = ic_df[column]
            result = pd.DataFrame(index=[column])
            result['mean'] = np.mean(ic)
            result['std'] = np.std(ic)
            result['ir'] =result['mean']/result['std']
            result['p'] = (ic>0).sum()/len(ic)
            result['nan_p'] = ic.isna().sum()/len(ic)
            result['mean_abs'] = np.abs(np.mean(ic))
            result_df = pd.concat([result_df,result]).round(4)

        result_df.to_csv(data_path+"/data/ic_results_data/ana_IC_type_{}_{}.csv".format(self.factor_type, ic_file_info))
        
        selected_ic_df_index = list(result_df[(result_df['mean_abs'] >= ic_range[0]) & (result_df['mean_abs'] <= ic_range[1])].index)
        
        """delete feature has nan oneday IC value"""
        problem_columns = result_df[result_df['nan_p'] > 0].index
        print(len(selected_ic_df_index),"before nan IC")
        for column in problem_columns:
            if column in selected_ic_df_index:
                selected_ic_df_index.remove(column)
        print(len(selected_ic_df_index),"after nan IC")
        selected_ic_info = result_df.loc[selected_ic_df_index,['mean','mean_abs']].round(3)
        print(selected_ic_info)

        return result_df, selected_ic_df_index
    
    

    def cal_t_teststat(self, t_test_file_name = "default_name"):
        """analysis t test result of factor"""
        t_test_df = self.df
        try:
            column_list = t_test_df.columns.drop(['Symbol','TradingDate'])
        except:
            column_list = t_test_df.columns
        result_df = pd.DataFrame()
        for column in column_list:
            t_test = t_test_df[column]
            result = pd.DataFrame(index=[column])
            result['mean'] = np.mean(t_test)
            result['std'] = np.std(t_test)
            result['abs_mean'] = np.mean(np.abs(t_test))
            result['abs_mean-mean_abs'] = np.mean(np.abs(t_test)) - np.abs(np.mean(t_test))
            result['T_large_2'] = (np.abs(t_test)>2).sum()/len(t_test)
            result_df = pd.concat([result_df,result]).round(4)
            result_df.to_csv(data_path+"/data/ic_results_data/ana_IC_type_%s_%s.csv"%(self.factor_type,t_test_file_name))


class Factor_results_ana(Base_ana):
    """
    Summary: this class is used to analysis the calculate results of factor: 
    
    Attributes:
        df_data (DataFrame): dataframe stored the factor value V.S. return 

    Methods:
        corr_between_factors_allday()
        corr_between_factors_oneday()

    """
    def __init__(self, df_data, factor_type, level):
        self.df = df_data
        self.factor_type = factor_type
        self.level = level
        super().__init__(df_data, factor_type, level)

    def corr_between_factors_allday(self,sample_size=10000,columns_list=None):
        """visualize the corr between values of factors use all days data"""
        df = self.df
        if columns_list is None:
            try:
                columns_list = df.columns.drop(['TradingDate','Symbol','TradingTime'])
            except:
                columns_list = columns_list
        with PdfPages(data_path+'/plots/results_ana/%s_factor_VS_factor_corr_multipage_plots.pdf'%self.factor_type) as pdf:
            for column in range(len(columns_list)):
                for column_in in range(column+1,len(columns_list)):
                    correlation = df[columns_list[column]].corr(df[columns_list[column_in]])
                    """draw figures downsample data point"""
                    sampled_df = df.sample(n=sample_size)
                    fig, ax = plt.subplots(figsize=(9, 9))
                    ax.plot(sampled_df[columns_list[column]].to_numpy(),sampled_df[columns_list[column_in]].to_numpy(),'.',markersize=2,\
                            label="corr:%0.3f" %( correlation))
                    ax.set_xlabel(columns_list[column])
                    ax.set_ylabel(columns_list[column_in])
                    plt.legend(edgecolor='red',frameon=True)
                    plt.title("feature V.S feature")
                    plt.tight_layout()
                    pdf.savefig(fig)
    
    def corr_between_factors_oneday(self,columns_list = ['realized_skew','return']):
        """ visualize the corr between values of factors at one day level"""
        df = self.df
        df['TradingDate'] = df['TradingTime'].dt.date
        grouped_df = df.groupby(['TradingDate'])
        factor_std = []
        with PdfPages(data_path+'/plots/results_ana/one_day_type_%s_%s_%s_factor_VS_factor_corr_multipage_plots.pdf'%\
                      (self.factor_type,columns_list[0],columns_list[-1])) as pdf:
            for date, df_oneday in grouped_df: 
                for column in range(len(columns_list)):
                    for column_in in range(column+1,len(columns_list)):
                        correlation = df_oneday[columns_list[column]].corr(df_oneday[columns_list[column_in]])

                        #draw figures
                        fig, ax = plt.subplots(figsize=(9, 9))
                        ax.plot(df_oneday[columns_list[column]],df_oneday[columns_list[column_in]],'.',markersize=2,\
                                label="corr:%0.3f" %( correlation))
                        ax.set_xlabel(columns_list[column])
                        ax.set_ylabel(columns_list[column_in])
                        plt.legend(edgecolor='red',frameon=True)
                        plt.title(columns_list[column]+"  "+columns_list[column_in])
                        plt.tight_layout()
                        pdf.savefig(fig)
            
                max_value = np.max(df_oneday[columns_list[0]])
                min_value = np.min(df_oneday[columns_list[0]])
                factor_std.append([max_value, min_value])

        """convert list to numpy array and save"""
        my_array = np.array(factor_std)
        np.save(data_path+'/data/middle_test_data/type_%s_%s_array.npy'%(self.factor_type,columns_list[0]), my_array)


    def select_factor(self,selected_rows,ic_ana_result,corr_threshold=0.5):
        
        factor_result_df = self.df
        row_names = [column for column in selected_rows if "aggre_mean" not in column]
        print("through the IC range select factor num: ",len(row_names))
        corr_df = factor_result_df[row_names].corr()
        corr_df.to_csv(data_path+"/data/corr_data/corr_result_%s_all.csv"%self.factor_type)

        import networkx as nx
        """read data"""
        corr_matrix = corr_df
        col_list = corr_matrix.columns

        """Create an undirected graph"""
        G = nx.Graph()
        """add edge to graph, set IC value as feature of Nodes"""
        for col in col_list:
            G.add_node(col, ic=ic_ana_result['mean_abs'][col] )
        """add ic info to graph (corr value > threshold)"""
        for i in range(len(col_list)):
            for j in range(i + 1,len(col_list)):
                if np.abs(corr_matrix.loc[col_list[i], col_list[j]]) < corr_threshold:
                    G.add_edge(col_list[i], col_list[j])
        """fina max subset"""
        max_cliques = list(nx.find_cliques(G))
        max_clique = max(max_cliques, key=len)
        """get nodes need to delete"""
        nodes_to_delete = [col_list[i] for i in range(len(col_list)) if col_list[i] not in max_clique]
        """print results"""
        print("features need to del: %s"%len(nodes_to_delete))
        print("features selected: %s"%len(max_clique))
        print(max_clique)

        corr_df = factor_result_df[max_clique].corr()
        corr_df.to_csv(data_path+"/data/corr_data/corr_result_%s_select.csv"%self.factor_type)
        print(ic_ana_result.loc[max_clique,['mean','mean_abs']])
        return max_clique
