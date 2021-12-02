from collections import OrderedDict

from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np
import pandas as pd
from utils import load_data, make_dir
from wasserstein_NMF_coefficient_step import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def params_plot():
    params_results = pd.read_csv('../results/params_analysis/params_analysis_df.csv',header=0,index_col=0,sep='\t')

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(1, figsize=(11, 9))
    sns.heatmap(params_results.pivot('gamma', 'rho', 'time'), cmap=cmap,annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.figure(2, figsize=(11, 9))
    sns.heatmap(params_results.pivot('gamma', 'rho', 'rmse'), cmap=cmap, annot=True, fmt='.5f',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.figure(3, figsize=(11, 9))
    sns.heatmap(params_results.pivot('gamma', 'rho', 'cor'), cmap=cmap, annot=True, fmt='.5f',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    pass

from methods import default_logger,NNLS
from result_analysis import *
from utils import basis_matrix
from sklearn.manifold import TSNE
# def weight_plot(root_path,dataset,subject):
#     result_path = root_path + dataset + "/" + subject + "/"
#     make_dir(root_path + "weight_analysis")
#     logger = default_logger(name='-'.join([dataset, subject]), save_file_path=root_path + "weight_analysis/base-log.log")
#
#     # result_df = pd.DataFrame()
#     logger.info("Reading data...")
#     Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
#
#     C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)
#     C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
#     C['cellID'] = C.index
#     C = C.merge(pDataC, on='cellID')
#     D = C.groupby('cellType').mean()
#     D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
#
#     logger.info("Preprocessing data...")
#     all_celltype = D_all.columns.intersection(P_df.index)
#     lambda_true = P_df.loc[all_celltype].values.astype('float')
#     np.savetxt(root_path + '/weight_analysis/lambda_true.txt', lambda_true)
#     Y = Y_df.values.astype('float')
#     logger.info('   genes: ' + str(Y.shape[0]))
#     logger.info('   cells: ' + str(C_init_df.shape[1]))
#     C = C_init_df.values.astype('float')
#
#     pDataC_group = pDataC.set_index(['sampleID'])
#
#     # data info
#     sampleid = list(set(pDataC['sampleID']))
#
#     # Wasserstein dissTOM weighted
#     D_subjects = dict()
#     lambda_true_subjects = dict()
#     m='dissTOM'
#
#     sampleid.remove('54_male')
#     lambda_subject_df = pd.DataFrame(columns=sampleid)
#     y_subject_df = pd.DataFrame(columns=sampleid)
#     for sam in sampleid:
#         C_sam = C_init_df[pDataC_group.loc[sam]['cellID']]
#         pDataC_sam = pDataC[pDataC["sampleID"] == sam]
#
#         C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
#         C_sam['cellID'] = C_sam.index
#         C_sam = C_sam.merge(pDataC_sam, on='cellID')
#
#         if sam == '54_male':
#             C_sam = C_sam[(C_sam['cellType'] != 'acinar') & (C_sam['cellType'] != 'ductal')]
#
#         D = C_sam.groupby('cellType').mean()
#         D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
#         print(D)
#         print(D.columns.tolist())
#         lambda_sam = np.loadtxt('../results/weight_analysis/lambda_individual/lambda_{}.txt'.format(sam))
#
#         if len(D.columns.tolist()) < len(all_celltype):
#             missing_list = []
#             for i, j in enumerate(D_all.columns):
#                 if j not in D.columns:
#                     missing_list.append(i)
#                     logger.info(sam + ' ' + j + " is missing!")
#             lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
#             D_subjects[sam] = D.values.astype('float')
#
#             lambda_0 = lambda_sam.copy()
#             for j, jj in enumerate(missing_list):
#                 if jj < lambda_0.shape[0]:
#                     lambda_0 = np.insert(lambda_0, jj, values=np.zeros(lambda_0.shape[1]), axis=0)
#                 elif jj == lambda_0.shape[0]:
#                     lambda_0 = np.vstack((lambda_0, np.zeros(lambda_0.shape[1])))
#                 print(lambda_0.shape)
#             lambda_subject_df[sam] = lambda_0.reshape(-1)
#         else:
#             lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
#             D_subjects[sam] = D.values.astype('float')
#             lambda_subject_df[sam] = lambda_sam.reshape(-1)
#
#         result_analysis(lambda_true_subjects[sam], lambda_sam, 'wasserstein' + '_' + m + ' ' + sam)
#         y_subject_df[sam] = np.dot(D_subjects[sam], lambda_sam).reshape(-1)
#
#     weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
#     print(list(weight.T))
#     lambda_weighted = lambda_subject_df * weight.T
#     lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
#     result_analysis(lambda_true, lambda_weighted, 'wasserstein' + '_' + m + '_weighted')


# def weight_plot(root_path,dataset,subject,imputation=True):
#     result_path = root_path + dataset + "/" + subject + "/"
#     make_dir(root_path + "weight_analysis")
#     logger = default_logger(name='-'.join([dataset, subject]), save_file_path=root_path + "weight_analysis/base-log.log")
#
#     # result_df = pd.DataFrame()
#     logger.info("Reading data...")
#     Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
#
#     C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)
#     C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
#     C['cellID'] = C.index
#     C = C.merge(pDataC, on='cellID')
#     D = C.groupby('cellType').mean()
#     D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
#
#     logger.info("Preprocessing data...")
#     all_celltype = D_all.columns.intersection(P_df.index)
#     lambda_true = P_df.loc[all_celltype].values.astype('float')
#     np.savetxt(root_path + '/weight_analysis/lambda_true.txt', lambda_true)
#     Y = Y_df.values.astype('float')
#     logger.info('   genes: ' + str(Y.shape[0]))
#     logger.info('   cells: ' + str(C_init_df.shape[1]))
#     C = C_init_df.values.astype('float')
#
#     pDataC_group = pDataC.set_index(['sampleID'])
#
#     # data info
#     sampleid = list(set(pDataC['sampleID']))
#
#     # Wasserstein dissTOM weighted
#     D_subjects = dict()
#     lambda_true_subjects = dict()
#     m='dissTOM'
#
#     rmse_all_df = pd.DataFrame()
#
#     # sampleid.remove('54_male')
#     lambda_subject_df = pd.DataFrame(columns=sampleid)
#     y_subject_df = pd.DataFrame(columns=sampleid)
#     for sam_n,sam in enumerate(sampleid):
#         C_sam = C_init_df[pDataC_group.loc[sam]['cellID']]
#         pDataC_sam = pDataC[pDataC["sampleID"] == sam]
#
#         # model = TSNE()
#         # C_sam_tsne = model.fit_transform(C_sam.values.astype('float').T)
#         # plt.figure(sam_n)
#         # labels = pDataC_sam['cellType'].map(dict(zip(D_all.columns,range(5))))
#         # plt.scatter(C_sam_tsne[:, 0], C_sam_tsne[:, 1], 5, labels)
#         # plt.title(sam)
#         # plt.legend(labels)
#         # plt.show()
#
#         C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
#         C_sam['cellID'] = C_sam.index
#         C_sam = C_sam.merge(pDataC_sam, on='cellID')
#
#         if sam == '54_male':
#             C_sam = C_sam[(C_sam['cellType'] != 'acinar') & (C_sam['cellType'] != 'ductal')]
#
#         D = C_sam.groupby('cellType').mean()
#         D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
#         print(D)
#         print(D.columns.tolist())
#         lambda_sam = np.loadtxt('../results/weight_analysis/lambda_individual/lambda_{}.txt'.format(sam))
#
#         if len(D.columns.tolist()) < len(all_celltype):
#             missing_list = []
#             for i, j in enumerate(D_all.columns):
#                 if j not in D.columns:
#                     missing_list.append(i)
#                     logger.info(sam + ' ' + j + " is missing!")
#             lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
#             D_subjects[sam] = D.values.astype('float')
#
#             lambda_0 = lambda_sam.copy()
#             for j, jj in enumerate(missing_list):
#                 if jj < lambda_0.shape[0]:
#                     if not imputation:
#                         lambda_0 = np.insert(lambda_0, jj, values=np.zeros(lambda_0.shape[1]), axis=0)
#                     D_subjects[sam] = np.insert(D_subjects[sam], jj, values=D_all[D_all.columns[jj]].values, axis=1)
#                 elif jj == lambda_0.shape[0]:
#                     if not imputation:
#                         lambda_0 = np.vstack((lambda_0, np.zeros(lambda_0.shape[1])))
#                     D_subjects[sam] = np.hstack((D_subjects[sam], D_all[D_all.columns[jj]].values))
#                 print(lambda_0.shape)
#             lambda_subject_df[sam] = lambda_0.reshape(-1)
#         else:
#             lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
#             D_subjects[sam] = D.values.astype('float')
#             lambda_subject_df[sam] = lambda_sam.reshape(-1)
#
#         if imputation:
#             result_analysis(lambda_true, lambda_sam, 'wasserstein' + '_' + m + ' ' + sam)
#         else:
#             result_analysis(lambda_subject_df[sam], lambda_sam, 'wasserstein' + '_' + m + ' ' + sam)
#         y_subject_df[sam] = np.dot(D_subjects[sam], lambda_sam).reshape(-1)
#
#
#
#
#     plt.show()
#     weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
#     print(sampleid)
#     print(list(weight.T))
#     lambda_weighted = lambda_subject_df * weight.T
#     lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
#     rmse, cor = result_analysis(lambda_true, lambda_weighted, 'wasserstein' + '_' + m + '_weighted')
#     print(np.median(rmse))


def weight_plot(root_path, dataset, subject, imputation=True):
    result_path = root_path + dataset + "/" + subject + "/"
    make_dir(root_path + "weight_analysis")
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=root_path + "weight_analysis/base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)

    C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)
    C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    all_celltype = D_all.columns.intersection(P_df.index)
    lambda_true = P_df.loc[all_celltype].values.astype('float')
    np.savetxt(root_path + '/weight_analysis/lambda_true.txt', lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cells: ' + str(C_init_df.shape[1]))
    C = C_init_df.values.astype('float')

    pDataC_group = pDataC.set_index(['sampleID'])

    # data info
    sampleid = list(set(pDataC['sampleID']))

    # Wasserstein dissTOM weighted
    D_subjects = dict()
    lambda_true_subjects = dict()
    m = 'dissTOM'

    rmse_all_df = pd.DataFrame(columns=['RMSE','celltype','individual','bulk'])

    # sampleid.remove(subject)
    sampleid.sort()
    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    ae_individual_df = pd.DataFrame(columns=['AE', 'celltype', 'individual', 'bulk'])
    target_ind_celltype = []
    for sam_n, sam in enumerate(sampleid):
        C_sam = C_init_df[pDataC_group.loc[sam]['cellID']]
        pDataC_sam = pDataC[pDataC["sampleID"] == sam]

        # model = TSNE()
        # C_sam_tsne = model.fit_transform(C_sam.values.astype('float').T)
        # plt.figure(sam_n)
        # labels = pDataC_sam['cellType'].map(dict(zip(D_all.columns,range(5))))
        # plt.scatter(C_sam_tsne[:, 0], C_sam_tsne[:, 1], 5, labels)
        # plt.title(sam)
        # plt.legend(labels)
        # plt.show()

        C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
        C_sam['cellID'] = C_sam.index
        C_sam = C_sam.merge(pDataC_sam, on='cellID')

        # if sam == subject:
        #     C_sam = C_sam[(C_sam['cellType'] != 'acinar') & (C_sam['cellType'] != 'ductal')]

        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        # print(D)
        print(D.columns.tolist())
        if sam == subject:
            target_ind_celltype = D.columns.tolist()
        lambda_sam = np.loadtxt('../results/weight_analysis/lambda_individual/lambda_{}.txt'.format(sam))

        if len(D.columns.tolist()) < len(all_celltype):
            missing_list = []
            for i, j in enumerate(D_all.columns):
                if j not in D.columns:
                    missing_list.append(i)
                    logger.info(sam + ' ' + j + " is missing!")
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')

            lambda_0 = lambda_sam.copy()
            for j, jj in enumerate(missing_list):
                if jj < lambda_0.shape[0]:
                    if not imputation:
                        lambda_0 = np.insert(lambda_0, jj, values=np.zeros(lambda_0.shape[1]), axis=0)
                    else:
                        D_subjects[sam] = np.insert(D_subjects[sam], jj, values=D_all[D_all.columns[jj]].values, axis=1)
                elif jj == lambda_0.shape[0]:
                    if not imputation:
                        lambda_0 = np.vstack((lambda_0, np.zeros(lambda_0.shape[1])))
                    else:
                        D_subjects[sam] = np.hstack((D_subjects[sam], D_all[D_all.columns[jj]].values))
                print(lambda_0.shape)
            lambda_subject_df[sam] = lambda_0.reshape(-1)
        else:
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')
            lambda_subject_df[sam] = lambda_sam.reshape(-1)

        if imputation:
            rmse, cor = result_analysis(lambda_true, lambda_sam, 'wasserstein' + '_' + m + ' ' + sam)
            ae = np.abs(lambda_true - lambda_sam)
        else:
            rmse, cor = result_analysis(lambda_true_subjects[sam], lambda_sam, 'wasserstein' + '_' + m + ' ' + sam)
            ae = np.abs(lambda_true_subjects[sam] - lambda_sam)

        y_subject_df[sam] = np.dot(D_subjects[sam], lambda_sam).reshape(-1)
        if sam == subject:
            rmse_target = rmse
            for i in range(ae.shape[0]):
                for j in range(ae.shape[1]):
                    ae_individual_df = ae_individual_df.append({'AE': ae[i, j], 'celltype': all_celltype[i], 'individual': 'DecOT-pair', 'bulk': j}, ignore_index=True)
        for i in range(rmse.shape[0]):
            rmse_all_df = rmse_all_df.append({'RMSE':rmse[i],'individual':sam,'bulk':i},ignore_index=True)

    ae_weighted_df = pd.DataFrame(columns=['AE','celltype','individual','bulk'])
    weight_df = pd.DataFrame(columns=sampleid)

    weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
    print(sampleid)
    print(list(weight.T))
    weight_df = weight_df.append(pd.Series(dict(zip(sampleid,list(weight))), name='{}-all'.format(subject)))
    weight_df.loc['{}-all'.format(subject), 'RMSE-individual'] = np.mean(rmse_target)
    lambda_weighted = lambda_subject_df * weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    ae = np.abs(lambda_true-lambda_weighted)
    rmse, cor = result_analysis(lambda_true, lambda_weighted, 'wasserstein_dissTOM_weighted')
    for i in range(rmse.shape[0]):
        rmse_all_df = rmse_all_df.append({'RMSE': rmse[i], 'individual': 'DecOT-pair', 'bulk': i}, ignore_index=True)
    weight_df.loc['{}-all'.format(subject),'RMSE-weighted'] = np.mean(rmse)
    for i in range(ae.shape[0]):
        for j in range(ae.shape[1]):
            ae_weighted_df = ae_weighted_df.append({'AE': ae[i,j], 'celltype': all_celltype[i],'individual': 'DecOT-pair', 'bulk': j}, ignore_index=True)

    rmse_target_df = pd.DataFrame(columns=['RMSE', 'missing celltype', 'bulk'])
    for i in range(rmse_target.shape[0]):
        rmse_target_df = rmse_target_df.append({'RMSE': rmse_target[i], 'missing celltype': 0, 'bulk': i}, ignore_index=True)
    files = os.listdir('../results/weight_analysis/lambda_individual/')
    files.sort()
    sam = subject
    for file in files:
        if file.startswith('lambda_{}_'.format(sam)):
            deletion = file[0:-4].split('_')[3:]

            lambda_i = np.loadtxt('../results/weight_analysis/lambda_individual/{}'.format(file))
            D_subject_temp = D_subjects[sam].copy()
            lambda_true_subject_temp = lambda_true_subjects[sam].copy()
            if imputation:
                for i in deletion:
                    D_subject_temp[:,np.where(all_celltype==i)[0][0]]=D_all[i].values
            else:
                D_subject_temp = np.delete(D_subject_temp,np.where(all_celltype.isin(deletion)),1)
                lambda_true_subject_temp = np.delete(lambda_true_subject_temp,np.where(all_celltype.isin(deletion)),0)

            if imputation:
                rmse, cor = result_analysis(lambda_true, lambda_i, 'wasserstein' + '_' + m + ' ' + file)
                ae = np.abs(lambda_true - lambda_i)
            else:
                rmse, cor = result_analysis(lambda_true_subject_temp, lambda_i.reshape(lambda_true_subject_temp.shape), 'wasserstein' + '_' + m + ' ' + file)
                ae = np.abs(lambda_true_subject_temp - lambda_i.reshape(lambda_true_subject_temp.shape))
            for i in range(rmse.shape[0]):
                rmse_target_df = rmse_target_df.append({'RMSE': rmse[i], 'missing celltype': len(deletion), 'bulk': i},ignore_index=True)
            for i in range(ae.shape[0]):
                for j in range(ae.shape[1]):
                    ae_individual_df = ae_individual_df.append({'AE': ae[i,j], 'celltype': all_celltype[i], 'individual': 'Missing: '+'-'.join(deletion), 'bulk': j}, ignore_index=True)
            y_subject_df[sam] = np.dot(D_subject_temp, lambda_i.reshape(lambda_true_subject_temp.shape)).reshape(-1)
            weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
            weight_df = weight_df.append(pd.Series(dict(zip(sampleid, list(weight))), name=sam+'-'+'-'.join(deletion)))
            weight_df.loc[sam + '-' + '-'.join(deletion), 'RMSE-individual'] = np.mean(rmse)
            print(sampleid)
            print(list(weight.T))

            lambda_0 = lambda_i.reshape(lambda_true_subject_temp.shape).copy()
            for j, jj in enumerate(np.where(all_celltype.isin(deletion))[0]):
                if jj < lambda_0.shape[0]:
                    if not imputation:
                        lambda_0 = np.insert(lambda_0, jj, values=np.zeros(lambda_0.shape[1]), axis=0)
                elif jj == lambda_0.shape[0]:
                    if not imputation:
                        lambda_0 = np.vstack((lambda_0, np.zeros(lambda_0.shape[1])))
                print(lambda_0.shape)
            lambda_subject_df[sam] = lambda_0.reshape(-1)

            lambda_weighted = lambda_subject_df * weight.T
            lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
            ae = np.abs(lambda_true - lambda_weighted)
            rmse, cor = result_analysis(lambda_true, lambda_weighted, 'wasserstein_dissTOM_weighted')
            weight_df.loc[sam+'-'+'-'.join(deletion), 'RMSE-weighted'] = np.mean(rmse)
            for i in range(ae.shape[0]):
                for j in range(ae.shape[1]):
                    ae_weighted_df = ae_weighted_df.append({'AE': ae[i,j], 'celltype': all_celltype[i], 'individual': 'Missing: '+'-'.join(deletion), 'bulk': j}, ignore_index=True)

    sampleid_deletion = sampleid.copy()
    sampleid_deletion.remove(sam)
    y_subject_df = y_subject_df.drop(sam, axis=1)
    lambda_subject_df = lambda_subject_df.drop(sam, axis=1)
    weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
    weight_df = weight_df.append(pd.Series(dict(zip(sampleid_deletion, list(weight))), name=sam + '-' + 'unpair'))
    print(sampleid_deletion)
    print(list(weight.T))
    lambda_weighted = lambda_subject_df * weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    ae = np.abs(lambda_true - lambda_weighted)
    rmse, cor = result_analysis(lambda_true, lambda_weighted, 'wasserstein_dissTOM_weighted')
    weight_df.loc[sam + '-' + 'unpair', 'RMSE-weighted'] = np.mean(rmse)
    for i in range(rmse.shape[0]):
        rmse_all_df = rmse_all_df.append({'RMSE': rmse[i], 'individual': 'DecOT-unpair', 'bulk': i}, ignore_index=True)
    for i in range(ae.shape[0]):
        for j in range(ae.shape[1]):
            ae_weighted_df = ae_weighted_df.append({'AE': ae[i, j], 'celltype': all_celltype[i], 'individual': 'DecOT-unpair', 'bulk': j}, ignore_index=True)

    weight_df.astype('float').to_csv('../results/weight_analysis/weight.csv',index=True,header=True)
    sns.set_theme(style="whitegrid")
    plt.figure(1,figsize=(10,6))
    g = sns.boxenplot(y="RMSE", x="individual", data=rmse_all_df,palette="vlag",)
    for label in g.get_xticklabels():
        label.set_rotation(30)
        # label.set_fontsize('small')
    plt.savefig('/home/liugan/文档/celltype_deconvolution/results/weight_analysis/Fig-unimpute/fig1.png', dpi=600, bbox_inches='tight')

    plt.figure(2,figsize=(10,6))
    g = sns.boxplot(y="AE", x="celltype", hue='individual', data=ae_weighted_df, palette="vlag")
    for label in g.get_xticklabels():
        label.set_rotation(30)
        # label.set_fontsize('small')
    g.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.savefig('/home/liugan/文档/celltype_deconvolution/results/weight_analysis/Fig-unimpute/fig2.png', dpi=600, bbox_inches='tight')

    plt.figure(3,figsize=(6,4))
    rmse_target_df['missing celltype'] = rmse_target_df['missing celltype'].astype('int')
    g = sns.boxplot(y="RMSE", x="missing celltype", data=rmse_target_df,palette="vlag",)
    plt.savefig('/home/liugan/文档/celltype_deconvolution/results/weight_analysis/Fig-unimpute/fig3.png', dpi=600, bbox_inches='tight')
    # for label in g.get_xticklabels():
        # label.set_rotation(30)
        # label.set_fontsize('small')
    plt.figure(4,figsize=(10,6))
    g = sns.boxplot(y="AE", x="celltype", hue='individual', data=ae_individual_df, palette="vlag")
    for label in g.get_xticklabels():
        label.set_rotation(30)
        # label.set_fontsize('small')
    g.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.savefig('/home/liugan/文档/celltype_deconvolution/results/weight_analysis/Fig-unimpute/fig4.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # params_plot()
    weight_plot(root_path='/home/liugan/文档/celltype_deconvolution/results/', dataset='GSE81547', subject='54_male',imputation=False)