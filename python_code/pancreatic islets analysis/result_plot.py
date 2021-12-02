from collections import OrderedDict

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import ttest_ind,levene


def boxplot(reference='Baron'):
    #"MuSiC","SCDC","BisqueRNA"
    for i,method in enumerate(["DecOT","NNLS","MuSiC","SCDC","BisqueRNA"]):
        ax = plt.subplot(1,5,i+1)
        lambda_path = '../../results/HbA1c/'+reference+'_results/lambda_'+method+'.txt'

        if method == 'DecOT':
            lambda_m = pd.read_csv(lambda_path,sep=' ',header=None)
            lambda_m.index = ['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma']
        else:
            lambda_m = pd.read_csv(lambda_path, sep='\t', header=0,index_col=0)

        lambda_m = pd.DataFrame(lambda_m.values.T, index=lambda_m.columns, columns=lambda_m.index)
        sns.boxplot(x='Cell type',y='Proportions',data=lambda_m.melt(var_name='Cell type',value_name='Proportions'), whis=[0, 100], width=.6, palette="vlag")
        sns.stripplot(x='Cell type',y='Proportions',data=lambda_m.melt(var_name='Cell type',value_name='Proportions'), size=2, color=".3", linewidth=0)
        ax.set(xlabel="")
        plt.ylim(0,1)
        if i>0:
            ax.set(ylabel="")
            plt.yticks([])
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        plt.title(method)
    plt.show()


def boxplot_decot():
    lambda_list = []
    for i,reference in enumerate(["EMTAB","Baron","EMTAB+Baron"]):

        lambda_path = '../../results/HbA1c/'+reference+'_results/lambda_DecOT.txt'

        lambda_m = pd.read_csv(lambda_path,sep=' ',header=None)
        lambda_m.index = ['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma']
        lambda_m = pd.DataFrame(lambda_m.values.T, index=lambda_m.columns, columns=lambda_m.index).melt(var_name='Cell type',value_name='Proportions')
        lambda_m['Bulk'] = np.repeat(reference,lambda_m.shape[0])
        lambda_list.append(lambda_m)
    lambda_m = pd.concat(lambda_list,axis=0,ignore_index=True)

    f,ax = plt.subplots()
    ax.yaxis.grid(True)
    sns.boxplot(x='Cell type',y='Proportions',hue='Bulk',data=lambda_m, whis=[0, 100], width=.6, palette="vlag")
    # sns.stripplot(x='Cell type',y='Proportions',hue='Bulk',data=lambda_m, size=2, color=".3", linewidth=0,dodge=True)
    ax.set(xlabel="")
    plt.ylim(0,1)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    plt.show()


def violin_decot():
    bulk_phenoData = pd.read_csv('../../results/HbA1c/GSE50244_phenoData.txt', sep='\t', header=0, index_col=0)
    fig = plt.figure(figsize=(15, 6))
    for i,reference in enumerate(["Segerstolpe","Baron",'Segerstolpe+Baron']):
        lambda_path = '../../results/HbA1c/'+reference+'_results/lambda_DecOT.txt'

        lambda_m = pd.read_csv(lambda_path,sep=' ',header=None)
        lambda_m.index = ['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma']
        lambda_m = pd.DataFrame(lambda_m.values.T, index=bulk_phenoData.index, columns=lambda_m.index)
        # lambda_m['Bulk'] = np.repeat(reference,lambda_m.shape[0])
        lambda_m['hba1c'] = bulk_phenoData['hba1c']
        lambda_m = lambda_m.dropna(axis=0)
        lambda_m['Individual type'] = (lambda_m['hba1c'] > 6).replace({True:'T2D',False:'Normal'})
        lambda_m = lambda_m.drop(['hba1c'],axis=1)
        lambda_m = lambda_m.melt(id_vars='Individual type',var_name='Cell type',value_name='Proportions')
        ax = plt.subplot(1, 3, i + 1)
        sns.violinplot(x='Cell type',y='Proportions',hue='Individual type',data=lambda_m,split=True, inner="quart", linewidth=1,palette="vlag")
        # sns.stripplot(x='Cell type',y='Proportions',hue='Bulk',data=lambda_m, size=2, color=".3", linewidth=0,dodge=True)
        ax.yaxis.grid(True)
        ax.set(xlabel="")
        plt.ylim(0,1)
        plt.title(reference,fontsize=14)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if i > 0:
            ax.set(ylabel="")
            # ax.get_yaxis().set_visible(False)
            plt.yticks(alpha=0)
            plt.tick_params(axis='y',width=0)
            ax.get_legend().remove()
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.91, 0.5), loc="center left", frameon=True)
        else:
            ax.get_legend().remove()
            # ax.set(ylabel="Beta cell prop")
            ax.set_ylabel("Proportions", fontsize=14)
    plt.subplots_adjust(wspace=0.05)
    plt.show()


def regression(reference='Segerstolpe'):
    #"MuSiC","SCDC","BisqueRNA"
    bulk_phenoData = pd.read_csv('../../results/HbA1c/GSE50244_phenoData.txt', sep='\t', header=0,index_col=0)
    # plt.figure(figsize=(15, 6))
    fig, ax0 = plt.subplots(nrows=1, ncols=6, figsize=(15, 6))
    plt.suptitle(reference, fontsize=18)
    for i, method in enumerate(["DecOT","NNLS","MuSiC","SCDC","BisqueRNA"]):
        ax = plt.subplot(1,5,i+1)
        lambda_path = '../../results/HbA1c/'+reference+'_results/lambda_'+method+'.txt'

        if method == 'DecOT':
            lambda_m = pd.read_csv(lambda_path,sep=' ',header=None)
            lambda_m.index = ['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma']
        else:
            lambda_m = pd.read_csv(lambda_path, sep='\t', header=0,index_col=0)

        reg_m = bulk_phenoData[['age', 'bmi', 'hba1c', 'gender']].copy()
        if method != "SCDC_ENSEMBLE":
            lambda_m = pd.DataFrame(lambda_m.values.T, index=bulk_phenoData.index, columns=lambda_m.index)
            reg_m['beta'] = lambda_m['beta']
            reg_m = reg_m.dropna(axis=0)
        else:
            reg_m = reg_m.dropna(axis=0)
            reg_m['beta'] = lambda_m['beta']


        e1 = LabelEncoder()
        reg_m['gender'] = e1.fit_transform(reg_m['gender'])

        x=sm.add_constant(reg_m[['age', 'bmi', 'hba1c', 'gender']])
        y=reg_m['beta']
        regr = sm.OLS(y,x)
        res = regr.fit()
        # st, data, ss2 = summary_table(res,alpha=0.5)

        # lr = LinearRegression()
        # lr.fit(reg_m[['age', 'bmi', 'hba1c', 'gender']],reg_m['beta'])
        # reg_m['beta_pred'] = lr.predict(reg_m[['age', 'bmi', 'hba1c', 'gender']])

        # Plot outputs
        reg_m['Individual type'] = (reg_m['hba1c'] > 6).replace({True: 'T2D', False: 'Normal'})
        reg_m['beta_adj'] = reg_m['beta'] - res.params['age'] * reg_m['age'] - res.params['bmi'] * reg_m['bmi'] - res.params['gender'] * reg_m['gender']
        plt.grid(True, axis='both',linestyle='-.')
        sns.regplot(x='hba1c',y='beta_adj',data=reg_m,scatter=False)
        # sns.scatterplot(x='hba1c', y='beta', data=reg_m[reg_m['hba1c'] > 6.5])
        # sns.scatterplot(x='hba1c', y='beta', data=reg_m[reg_m['hba1c'] <= 6.5])
        sns.scatterplot(x='hba1c', y='beta', data=reg_m, hue='Individual type')

        ax.get_legend().remove()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.91, 0.5), loc="center left", frameon=True)

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.set(xlabel="")
        plt.ylim(0, 1)
        if i > 0:
            ax.set(ylabel="")
            # ax.get_yaxis().set_visible(False)
            plt.yticks(alpha=0)
            plt.tick_params(axis='y',width=0)
        else:
            # ax.set(ylabel="Beta cell prop")
            ax.set_ylabel("Beta cell prop",fontsize=14)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        plt.title(method,fontsize=14)
        plt.text(5,0.9,'p-value=%.4f'%res.pvalues['hba1c'],fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    fig.text(0.5, 0.01, 'HbA1c', ha='center',fontsize=14)
    plt.show()
    # plt.savefig()
    pass


def t_test(reference='Baron'):
    ttest = pd.DataFrame(columns=['statistic', 'pvalue'])
    bulk_phenoData = pd.read_csv('../../results/HbA1c/GSE50244_phenoData.txt', sep='\t', header=0, index_col=0)
    for i,method in enumerate(["DecOT","NNLS","MuSiC","SCDC_ENSEMBLE","BisqueRNA"]):
        lambda_path = '../../results/HbA1c/'+reference+'_results/lambda_'+method+'.txt'
        if method == 'DecOT':
            lambda_m = pd.read_csv(lambda_path,sep=' ',header=None)
            lambda_m.index = ['acinar', 'alpha', 'beta', 'delta', 'ductal', 'gamma']
            lambda_m = pd.DataFrame(lambda_m.values.T, index=bulk_phenoData.index, columns=lambda_m.index)
        else:
            lambda_m = pd.read_csv(lambda_path, sep='\t', header=0,index_col=0)
            if method != "SCDC_ENSEMBLE":
                lambda_m = pd.DataFrame(lambda_m.values.T, index=lambda_m.columns, columns=lambda_m.index)

        lambda_m['hba1c'] = bulk_phenoData['hba1c']
        lambda_m = lambda_m.dropna(axis=0)
        lambda_m['Individual type'] = (lambda_m['hba1c'] > 6).replace({True: 'T2D', False: 'Normal'})
        print(levene(lambda_m[lambda_m['Individual type']=="T2D"]['beta'],lambda_m[lambda_m['Individual type']=="Normal"]['beta']))
        t = ttest_ind(lambda_m[lambda_m['Individual type']=="T2D"]['beta'],lambda_m[lambda_m['Individual type']=="Normal"]['beta'])
        ttest = ttest.append(pd.Series({'statistic':t[0],'pvalue':t[1]},name=method))
    # ttest.to_csv('../../results/HbA1c/t_test/t-test-{}.csv'.format(reference),header=True,index=True,sep=',')
    print(ttest)


if __name__ == '__main__':
    # boxplot('Baron')
    # boxplot_decot()
    regression('Baron')
    # violin_decot()
    # t_test('Segerstolpe+Baron')
    pass