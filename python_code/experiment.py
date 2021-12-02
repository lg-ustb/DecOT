from methods import *


def experiment_4(dataset, data_root_path, result_root_path, subject, ground_cost_path):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = result_root_path + dataset + "/" + subject + "/"
    ground_cost_path = ground_cost_path + dataset + "/" + subject + "/"

    make_dir(result_path)
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=result_path + "base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all= load_data(data_root_path, dataset.split("_")[0])


    logger.info("Preprocessing data...")
    lambda_true = P_df.loc[D_all.columns].values.astype('float')
    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    C = C_init_df.values.astype('float')
    pDataC_group = pDataC.set_index(['sampleID'])
    # mixture_num = 200
    # Y = Y[:, 0:mixture_num]
    # lambda_true = lambda_true[:, 0:mixture_num]
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))
    all_celltype = D_all.columns

    ## PCA before compute ground cost
    # for c in set(pDataC['cellType']):
    #     pca = PCA(n_components=0.95)
    #     pca.fit(C_init[pDataC['cellID'][pDataC['cellType'] == c]].values.T)
    #     components = pca.components_.T
    #     try:
    #         C_pca = np.hstack((C_pca, components))
    #     except NameError:
    #         C_pca = components
    # print('PCA components:', C_pca.shape[1])
    #
    # for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
    #     logger.info("Compute ground cost: " + m)
    #     ground_cost(C, ground_cost_path, metric=m)
    lambda1 = preprocess(NNLS(Y.T, D_all.T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
    result_analysis(lambda_true, lambda1, 'NNLS')

    D_subjects = dict()
    lambda_true_subjects = dict()
    missing_celltype = dict()
    for sam in sampleid:
        logger.info("------------------------------------------------")
        logger.info("Prepare data for "+sam+":")
        C_sam = C_init_df[pDataC_group.loc[sam]['cellID']]
        pDataC_sam = pDataC[pDataC["sampleID"]==sam]

        C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
        C_sam['cellID'] = C_sam.index
        C_sam = C_sam.merge(pDataC_sam, on='cellID')
        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        logger.info(sam+' celltype: ' + str(D.columns.tolist()))
        logger.info(sam + ' celltype counts: ' + str(len(D.columns.tolist())))
        if len(D.columns.tolist()) < len(all_celltype):
            # temp = pd.DataFrame(index=D_all.index,columns=D_all.columns)
            # temp[D.columns] = D
            # D_subjects[sam] = temp.fillna(0).values.astype('float')
            # lambda_true_sam = pd.DataFrame(index=D_all.columns,columns=P_df.columns)
            # lambda_true_sam.loc[D.columns] = P_df.loc[D.columns]
            # lambda_true_subjects[sam] = lambda_true_sam.fillna(0).values.astype('float')
            missing_list = []
            for i,j in enumerate(D_all.columns):
                if j not in D.columns:
                    missing_list.append(i)
                    logger.info(sam+' '+j+" is missing!")
            missing_celltype[sam] = missing_list
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')
        else:
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')

        # logger.info("Start NNLS deconvolution...")
        # lambda1 = preprocess(NNLS(Y.T, D.T).T)
        # # np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
        # result_analysis(lambda_true_sam, lambda1,'NNLS_'+sam)
        # logger.info("NNLS finished...")
        # logger.info("------------------------------------------------")
    print(missing_celltype)
    logger.info("Start wasserstein deconvolution for all individuals...")
    # methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)
    m = 'dissTOM'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   load_ground_cost(m, ground_cost_path,
                                                                                    select_genes=None),
                                                                   m+' '+sam, 0.001, 0.001)) for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        # np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_analysis(lambda_true_subjects[sam], lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j,jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, j+jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Compute weighted wasserstein_"+m+" results...")
    weight = preprocess(NNLS(Y.reshape(-1,1).T, y_subject_df.values.T).T)
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df*weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    result_analysis(lambda_true, lambda_weighted,'wasserstein_dissTOM_weighted')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('wasserstein_dissTOM_weighted'), lambda_weighted)
    logger.info("Weighted wasserstein_" + m + " completed!")


def experiment_5(dataset, data_root_path, result_root_path, subject, ground_cost_path ,if_pca=True):
    '''
    deconvolution for every individual(not include test subject), then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    r_dataset = dataset
    if if_pca:
        dataset = dataset+"_pca"
    result_path = result_root_path + dataset + "/" + subject + "/"
    ground_cost_path = ground_cost_path + dataset + "/" + subject + "/"

    make_dir(result_path)
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=result_path + "base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all= load_data(data_root_path, "Kidney_HCL")

    pDataC_group = pDataC.set_index(['sampleID'])
    C_init = C_init_df[C_init_df.columns[~C_init_df.columns.isin(pDataC_group.loc[subject]['cellID'])]]
    C = pd.DataFrame(C_init.values.T, index=C_init.columns, columns=C_init.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    lambda_true = P_df.loc[D_all.columns].values.astype('float')
    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))

    # mixture_num = 200
    # Y = Y[:, 0:mixture_num]
    # lambda_true = lambda_true[:, 0:mixture_num]
    sampleid = list(set(pDataC['sampleID']))
    sampleid.remove(subject)
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))
    all_celltype = D_all.columns

    ## PCA before compute ground cost
    # for c in set(pDataC['cellType']):
    #     pca = PCA(n_components=0.95)
    #     pca.fit(C_init[pDataC['cellID'][pDataC['cellType'] == c]].values.T)
    #     components = pca.components_.T
    #     try:
    #         C_pca = np.hstack((C_pca, components))
    #     except NameError:
    #         C_pca = components
    # print('PCA components:', C_pca.shape[1])
    #
    # for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
    #     logger.info("Compute ground cost: " + m)
    #     ground_cost(C, ground_cost_path, metric=m)

    logger.info("Start wasserstein deconvolution...")
    methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)

    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_all.values.astype('float'), standardized=0, normalized=0),
                                                                   load_ground_cost(m, ground_cost_path, select_genes=None),
                                                                   m, 0.001, 0.001)) for m in methods]

    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        m = methods[i]
        logger.info(' '.join([' ', m, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m)

    lambda1 = preprocess(NNLS(Y.T, D_all.values.astype('float').T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
    result_analysis(lambda_true, lambda1, 'NNLS')

    D_subjects = dict()
    lambda_true_subjects = dict()
    missing_celltype = dict()
    for sam in sampleid:
        logger.info("------------------------------------------------")
        logger.info("Prepare data for "+sam+":")
        C_sam = C_init_df[pDataC_group.loc[sam]['cellID']]
        pDataC_sam = pDataC[pDataC["sampleID"]==sam]

        C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
        C_sam['cellID'] = C_sam.index
        C_sam = C_sam.merge(pDataC_sam, on='cellID')
        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        logger.info(sam+' celltype: ' + str(D.columns.tolist()))
        logger.info(sam + ' celltype counts: ' + str(len(D.columns.tolist())))
        if len(D.columns.tolist()) < len(all_celltype):
            # temp = pd.DataFrame(index=D_all.index,columns=D_all.columns)
            # temp[D.columns] = D
            # D_subjects[sam] = temp.fillna(0).values.astype('float')
            # lambda_true_sam = pd.DataFrame(index=D_all.columns,columns=P_df.columns)
            # lambda_true_sam.loc[D.columns] = P_df.loc[D.columns]
            # lambda_true_subjects[sam] = lambda_true_sam.fillna(0).values.astype('float')
            missing_list = []
            for i,j in enumerate(D_all.columns):
                if j not in D.columns:
                    missing_list.append(i)
                    logger.info(sam+' '+j+" is missing!")
            missing_celltype[sam] = missing_list
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')
        else:
            lambda_true_subjects[sam] = P_df.loc[D.columns].values.astype('float')
            D_subjects[sam] = D.values.astype('float')

        # logger.info("Start NNLS deconvolution...")
        # lambda1 = preprocess(NNLS(Y.T, D.T).T)
        # # np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
        # result_analysis(lambda_true_sam, lambda1,'NNLS_'+sam)
        # logger.info("NNLS finished...")
        # logger.info("------------------------------------------------")
    print(missing_celltype)
    logger.info("Start wasserstein deconvolution for all individuals...")
    # methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)
    m = 'dissTOM'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   load_ground_cost(m, ground_cost_path,
                                                                                    select_genes=None),
                                                                   m+' '+sam, 0.001, 0.001)) for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        # np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_analysis(lambda_true_subjects[sam], lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j,jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, j+jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Compute weighted wasserstein_"+m+" results...")
    weight = preprocess(NNLS(Y.reshape(-1,1).T, y_subject_df.values.T).T)
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df*weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    result_analysis(lambda_true, lambda_weighted,'wasserstein_dissTOM_weighted')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('wasserstein_dissTOM_weighted'), lambda_weighted)
    logger.info("Weighted wasserstein_" + m + " completed!")


if __name__ == '__main__':
    # experiment_4(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/baron/human2/',
    #                 dataset='baron_pca', result_root_path='../results_e4/', subject='human2', ground_cost_path='../results_e3/')

    # experiment_4(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/Kidney_HCL/Donor34/',
    #              dataset='Kidney_HCL', result_root_path='../results_e4/', subject='Donor34', ground_cost_path='../results_e3/')

    experiment_5(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/Kidney_HCL/Donor34/',
                 dataset='Kidney_HCL', result_root_path='../results_e5/', subject='Donor34', ground_cost_path='../results_e3/',if_pca=False)