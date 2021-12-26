from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
from utils import load_data, make_dir
import seaborn as sns
import matplotlib.pyplot as plt
import os


def result_analysis(lambda_true, lambda_pred, method_name, verbose=1):
    # print(method_name, '计算出的占比：')
    # print(lambda_pred)
    if verbose:
        print('-------------------------------------------------')
        print(method_name, ' RMSE：')
    # print(sqrt(mean_squared_error(lambda_true, lambda_pred)))

    rmse = []
    for i in range(lambda_true.shape[1]):
        rmse.append(sqrt(mean_squared_error(lambda_true[:, i], lambda_pred[:, i])))
    rmse = np.array(rmse)
    if verbose:
        print('rmse.mean: ', str(np.mean(rmse)), 'rmse.std: ', str(np.std(rmse)))

        # print(method_name, ' r2：')
        # print(r2_score(lambda_true, lambda_pred))

        print(method_name, ' cor：')
    # lambda_true_ser = pd.Series(lambda_true.flatten())
    # lambda_pred_ser = pd.Series(lambda_pred.flatten())
    # print(lambda_true_ser.corr(lambda_pred_ser))
    lambda_pred_df = pd.DataFrame(lambda_pred)
    lambda_true_df = pd.DataFrame(lambda_true)
    cor = []
    for i in range(lambda_true.shape[1]):
        cor.append(lambda_true_df[i].corr(lambda_pred_df[i]))
    cor = np.array(cor)
    if verbose:
        print('cor.mean: ', str(np.mean(cor)), 'cor.std: ', str(np.std(cor)))
        print('-------------------------------------------------\r\n')
    return rmse, cor


def load_results_from_R(methods_list, dataset, root_path='/media/user/新加卷/lg/deconv_benchmark-master'):
    print(dataset, ' results:')
    Y, C_init, P, pDataC, D = load_data(dataset)
    lambda_true = P.loc[D.index].astype('float')

    results_df = pd.DataFrame()
    dataset_p = root_path + '/' + dataset + '/results'
    for method in methods_list:
        lambda_m = pd.read_csv(dataset_p + '/lambda_' + method + '.txt', index_col=0, header=0, sep='\t')
        rmse, cor = result_analysis(lambda_true.values, lambda_m.loc[lambda_true.index].values, method)
        results_df[method + '_rmse'] = rmse
        results_df[method + '_cor'] = cor

    return results_df


def results_boxplot(results_df, output_path, dataset, order):
    make_dir(output_path)
    methods = set(['_'.join(col.split('_')[0:-1]) for col in results_df.columns])
    resluts_df_seaborn = pd.DataFrame(columns=['RMSE', 'Cor', 'method'])
    for i in results_df.index:
        for m in methods:
            resluts_df_seaborn = resluts_df_seaborn.append(
                {'RMSE': results_df.loc[i, m + '_rmse'], 'Cor': results_df.loc[i, m + '_cor'],
                 'method': m}, ignore_index=True)
    # order = ['wasserstein_dissTOM', 'wasserstein_euclidean', 'wasserstein_cosine', 'wasserstein_correlation',
    #          'WS_dissTOM_music_basis', 'WS_euclidean_music_basis', 'WS_cosine_music_basis',
    #          'WS_correlation_music_basis', 'NNLS', 'MuSiC', 'SCDC', 'BisqueRNA']

    for d in ['RMSE', 'Cor']:
        save_path = output_path + dataset + '_' + d + '_boxplot.png'
        boxplot(d, "method", resluts_df_seaborn, order, dataset + ' ' + d, save_path)


def boxplot(x, y, df, order, title, save_path, hue=None, hue_order=None, strip_plot=True):
    f, ax = plt.subplots(figsize=(20, 10))
    sns.set_theme(style="ticks")
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x=x, y=y, data=df, width=.6, palette="vlag", order=order, hue=hue, hue_order=hue_order)

    # Add in points to show each observation
    if strip_plot:
        sns.stripplot(x=x, y=y, data=df, size=2, color=".3", linewidth=0, order=order, hue=hue, hue_order=hue_order)
        ax.xaxis.grid(True)
        # ax.set(ylabel="")
        # sns.despine(trim=True, left=True)
    plt.title(title)
    plt.savefig(save_path, dpi=400, pad_inchs=5)
    # plt.show()
    plt.clf()


# def results_output_df(results_df, dataset,
#                       output_path):
#     methods = ['Wasserstein_dissTOM', 'Wasserstein_euclidean', 'Wasserstein_cosine', 'Wasserstein_correlation', 'NNLS',
#                'MuSiC', 'SCDC', 'BisqueRNA']
#     output = pd.DataFrame(index=["RMSE", 'Cor'], columns=methods)
#
#     for m in methods:
#         try:
#             m_ = m.split('_')[1]
#         except IndexError:
#             m_ = m.split('_')[0]
#         for d in output.index:
#             output.loc[d, m] = str(np.mean(results_df['_'.join([m_, d.lower()])]).round(5)) + '(' + str(
#                 np.std(results_df['_'.join([m_, d.lower()])]).round(5)) + ')'
#
#     output.to_csv(output_path + dataset + '_results.csv', index=True, header=True)


def results_output_df(results_df, output_path, dataset, methods):
    make_dir(output_path)
    # methods = ['wasserstein_dissTOM', 'wasserstein_euclidean', 'wasserstein_cosine', 'wasserstein_correlation',
    #          'WS_dissTOM_music_basis', 'WS_euclidean_music_basis', 'WS_cosine_music_basis',
    #          'WS_correlation_music_basis', 'NNLS', 'MuSiC', 'SCDC', 'BisqueRNA']
    output = pd.DataFrame(index=["RMSE", 'Cor'], columns=methods)

    for m in methods:
        for d in output.index:
            output.loc[d, m] = str(np.mean(results_df['_'.join([m, d.lower()])]).round(5)) + '(' + str(
                np.std(results_df['_'.join([m, d.lower()])]).round(5)) + ')'

    output.to_csv(output_path + dataset + '_results.csv', index=True, header=True)


def main():
    # 'baron','GSE81547','Kidney_HCL','EMTAB5061'
    for dataset in ['baron', 'GSE81547', 'Kidney_HCL', 'EMTAB5061']:
        results_df_r = load_results_from_R(methods_list=["MuSiC", "SCDC", "BisqueRNA"], dataset=dataset)
        results_df_p = pd.read_csv('/media/user/新加卷/lg/cell_type_deconvolution/results/' + dataset + '/result_df.csv',
                                   index_col=0, header=0, sep='\t')
        results_df = results_df_r.join(results_df_p)
        results_boxplot(results_df.iloc[0:500, ], '/media/user/新加卷/lg/cell_type_deconvolution/results/boxplot/',
                        dataset)
        results_output_df(results_df.iloc[0:500, ],
                          '/media/user/新加卷/lg/cell_type_deconvolution/results/output_all_methods/', dataset)
    pass


def experiment_param_analysis(results_init_df, dataset):  # results_init_df,dataset
    # results_init_df = pd.read_csv('../experiment_param_results/Kidney_HCL/results_init_df.csv', index_col=0, header=0, sep='\t')
    # dataset = 'Kidney_HCL'

    results_sns_df = pd.DataFrame(columns=['RMSE', 'Cor', 'method', 'gamma', 'rho'])
    params = set(['_'.join(col.split('_')[0:3]) for col in results_init_df.columns])
    for i in results_init_df.index:
        for j in params:
            results_sns_df = results_sns_df.append(
                {'RMSE': results_init_df.loc[i, j + '_rmse'], 'Cor': results_init_df.loc[i, j + '_cor'],
                 'method': j.split('_')[0], 'gamma': float(j.split('_')[1]), 'rho': float(j.split('_')[2])},
                ignore_index=True)
    order = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    for d in ['RMSE', 'Cor']:
        for rho in [0.001, 0.005, 0.01, 0.05, 0.1]:
            save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/boxplot/gamma_' + d + '_(rho=' + str(
                rho) + ')_boxplot.png'
            title = "Kidney_HCL param analysis gamma-" + d + ' (rho=' + str(rho) + ')'
            boxplot('gamma', d, results_sns_df[results_sns_df['rho'] == rho], None, title, save_path, hue="method",
                    hue_order=order, strip_plot=False)

        for gamma in [0.001, 0.005, 0.01, 0.05, 0.1]:
            save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/boxplot/rho_' + d + '_(gamma=' + str(
                gamma) + ')_boxplot.png'
            title = "Kidney_HCL param analysis rho-" + d + ' (gamma=' + str(gamma) + ')'
            boxplot('rho', d, results_sns_df[results_sns_df['gamma'] == gamma], None, title, save_path, hue="method",
                    hue_order=order, strip_plot=False)

        save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/boxplot/gamma_rho_' + d + '_boxplot.png'
        title = "Kidney_HCL param analysis gamma_rho-" + d + ' (gamma=rho)'
        boxplot('gamma', d, results_sns_df[results_sns_df['gamma'] == results_sns_df['rho']], None, title, save_path,
                hue="method",
                hue_order=order, strip_plot=False)
    pass


def experiment_time_analysis():  # results_init_df,dataset
    results_sns_df = pd.read_csv('../experiment_param_results/Kidney_HCL/result_running_time_df.csv', index_col=0,
                                 header=0, sep='\t')
    dataset = 'Kidney_HCL'
    make_dir('/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/time_plot')

    order = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    for rho in [0.001, 0.005, 0.01, 0.05, 0.1]:
        save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/time_plot/gamma_time_(rho=' + str(
            rho) + ')_boxplot.png'
        title = 'Kidney_HCL param analysis gamma-time (rho=' + str(rho) + ')'
        barplot('gamma', 'time', results_sns_df[results_sns_df['rho'] == rho], None, title, save_path, hue="method",
                hue_order=order, strip_plot=False)

    for gamma in [0.001, 0.005, 0.01, 0.05, 0.1]:
        save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/time_plot/rho_time_(gamma=' + str(
            gamma) + ')_boxplot.png'
        title = 'Kidney_HCL param analysis rho-time (gamma=' + str(gamma) + ')'
        barplot('rho', 'time', results_sns_df[results_sns_df['gamma'] == gamma], None, title, save_path, hue="method",
                hue_order=order, strip_plot=False)

    save_path = '/media/user/新加卷/lg/cell_type_deconvolution/experiment_param_results/' + dataset + '/time_plot/gamma_rho_time_boxplot.png'
    title = 'Kidney_HCL param analysis gamma_rho-time (gamma=rho)'
    barplot('gamma', 'time', results_sns_df[results_sns_df['gamma'] == results_sns_df['rho']], None, title, save_path,
            hue="method",
            hue_order=order, strip_plot=False)
    pass


def barplot(x, y, df, order, title, save_path, hue=None, hue_order=None, strip_plot=True):
    f, ax = plt.subplots(figsize=(18, 10))
    sns.set_theme(style="ticks")
    # Plot the orbital period with horizontal boxes
    sns.barplot(x=x, y=y, data=df, palette="vlag", order=order, hue=hue, hue_order=hue_order)

    plt.title(title)
    plt.savefig(save_path, dpi=400)
    # plt.show()
    plt.clf()


def experiment_main_analysis():
    order = ['wasserstein_dissTOM', 'wasserstein_euclidean', 'wasserstein_cosine', 'wasserstein_correlation',
             'wasserstein_dissTOM_weighted', 'NNLS', 'MuSiC', 'SCDC', 'BisqueRNA']
    r_result_path = '/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/Kidney_HCL/Donor34/results_/'
    p_result_path = '/media/user/新加卷/lg/cell_type_deconvolution/results_e5/Kidney_HCL/Donor34/lambda/'

    r_results = [r_result_path + i for i in os.listdir(r_result_path)]
    p_results = [p_result_path + i for i in os.listdir(p_result_path)]
    p_results.remove(p_result_path + "lambda_true.txt")

    results_df = pd.DataFrame()
    Y, C_init, P, pDataC, D = load_data('/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/Kidney_HCL/Donor34/',
                                        'Kidney_HCL')
    lambda_true = P.loc[D.columns].astype('float')
    for r in r_results:
        method = r.split('/')[-1].split('.')[0].split('_')[1]
        lambda_m = pd.read_csv(r, index_col=0, header=0, sep='\t')
        rmse, cor = result_analysis(lambda_true.loc[lambda_m.index].values, lambda_m.values, method)
        results_df[method + '_rmse'] = rmse
        results_df[method + '_cor'] = cor

    for p in p_results:
        method = p.split('/')[-1].split('.')[0][7:]
        lambda_m = np.loadtxt(p)
        rmse, cor = result_analysis(lambda_true.values, lambda_m, method)
        results_df[method + '_rmse'] = rmse
        results_df[method + '_cor'] = cor
    make_dir('/media/user/新加卷/lg/cell_type_deconvolution/results_e5/Kidney_HCL/Donor34/boxplot/')
    make_dir('/media/user/新加卷/lg/cell_type_deconvolution/results_e5/Kidney_HCL/Donor34/results_table/')
    results_boxplot(results_df, '/media/user/新加卷/lg/cell_type_deconvolution/results_e5/Kidney_HCL/Donor34/boxplot/',
                    'baron_pca_human2', order)
    results_output_df(results_df,
                      '/media/user/新加卷/lg/cell_type_deconvolution/results_e5/Kidney_HCL/Donor34/results_table/', 'baron_pca_human2',
                      order)


if __name__ == '__main__':
    # main()
    # experiment_time_analysis()

    experiment_main_analysis()

    # p_result_path = '/media/user/新加卷/lg/cell_type_deconvolution/results/baron_marker_hvg/lambda/'
    # p_results = [p_result_path + i for i in os.listdir(p_result_path)]
    # p_results.remove(p_result_path + "lambda_true.txt")
    # lambda_true = np.loadtxt(p_result_path + "lambda_true.txt")[:,0:500]
    # results_df = pd.DataFrame()
    # for p in p_results:
    #     method = p.split('/')[-1].split('.')[0][7:]
    #     lambda_m = np.loadtxt(p)
    #     rmse, cor = result_analysis(lambda_true, lambda_m, method)
    #     results_df[method + '_rmse'] = rmse
    #     results_df[method + '_cor'] = cor
    # results_output_df(results_df, '/media/user/新加卷/lg/cell_type_deconvolution/results/baron_marker_hvg/result_table/', 'baron_marker_hvg')
