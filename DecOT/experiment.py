from methods import *


def experiment_decot(root_path, dataset, subject):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = root_path + dataset + "/" + subject + "/"

    make_dir(result_path)
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=result_path + "base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
    # logger.info("Select hvg and marker genes...")
    # Y_df, C_init_df, P_df, pDataC, D_all= load_data(result_path, dataset)
    C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)
    C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    all_celltype = D_all.columns.intersection(P_df.index)
    lambda_true = P_df.loc[all_celltype].values.astype('float')

    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cells: ' + str(C_init_df.shape[1]))
    C = C_init_df.values.astype('float')

    pDataC_group = pDataC.set_index(['sampleID'])
    # mixture_num = 200
    # Y = Y[:, 0:mixture_num]
    # lambda_true = lambda_true[:, 0:mixture_num]

    # data info
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))

    # NNLS
    lambda1 = preprocess(NNLS(Y.T, D_all.values.astype('float').T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
    result_analysis(lambda_true, lambda1, 'NNLS')

    # # PCA before compute ground cost
    # logger.info("Start PCA...")
    # for c in set(pDataC['cellType']):
    #     pca = PCA(n_components=0.95)
    #     pca.fit(C_init_df[pDataC['cellID'][pDataC['cellType'] == c]].values.T)
    #     components = pca.components_.T
    #     try:
    #         C_pca = np.hstack((C_pca, components))
    #     except NameError:
    #         C_pca = components
    # print('PCA components:', C_pca.shape[1])

    for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
        logger.info("Compute ground cost: " + m)
        ground_cost(C_init_df.values, result_path, metric=m)

    # Wasserstein
    logger.info("Start wasserstein deconvolution...")
    methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_all.values.astype('float'), standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m, 0.001, 0.001)) for m in methods]
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        m = methods[i]
        logger.info(' '.join([' ', m, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m)

    # Wasserstein dissTOM weighted
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

    print(missing_celltype)
    logger.info("Start wasserstein deconvolution for all individuals...")
    # methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(10)
    m = 'dissTOM'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m+' '+sam, 0.001, 0.001)) for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    make_dir(result_path + '/lambda_individual')
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda_individual/lambda_{}.txt'.format(sam), lambda0)
        result_analysis(lambda_true_subjects[sam], lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j,jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Compute weighted wasserstein_"+m+" results...")
    weight = preprocess(NNLS(Y.reshape(-1,1).T, y_subject_df.values.T).T)
    print(list(weight.T))
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df*weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    result_analysis(lambda_true, lambda_weighted,'wasserstein' + '_' + m + '_weighted')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('wasserstein' + '_' + m + '_weighted'), lambda_weighted)
    logger.info("Weighted wasserstein_" + m + " completed!")

    experiment_main_analysis(root_path=result_path)


def experiment_decot_encemble(root_path, dataset, subject):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = root_path + dataset + "/" + subject + "/"

    make_dir(result_path)
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=result_path + "base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
    # logger.info("Select hvg and marker genes...")
    # Y_df, C_init_df, P_df, pDataC, D_all= load_data(result_path, dataset)
    # Weight_gene = pd.read_csv(result_path+'{}_Weight_gene.csv'.format(dataset),index_col=0,header=0,sep='\t')

    pDataC_group = pDataC.set_index(['sampleID'])
    # remove test subject
    C_init_df = C_init_df[C_init_df.columns[~C_init_df.columns.isin(pDataC_group.loc[subject.split("#")[0]]['cellID'])]]
    pDataC = pDataC[pDataC['sampleID'] != subject.split("#")[0]]
    # convert cell matrix into histogram
    C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)

    C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    all_celltype = D_all.columns.intersection(P_df.index)
    lambda_true = P_df.loc[all_celltype].values.astype('float')

    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cells: ' + str(C_init_df.shape[1]))
    C = C_init_df.values.astype('float')

    pDataC_group = pDataC.set_index(['sampleID'])
    # mixture_num = 200
    # Y = Y[:, 0:mixture_num]
    # lambda_true = lambda_true[:, 0:mixture_num]

    # data info
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))

    # NNLS
    lambda1 = preprocess(NNLS(Y.T, D_all.values.astype('float').T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
    result_analysis(lambda_true, lambda1, 'NNLS')

    # PCA before compute ground cost
    # logger.info("Start PCA...")
    # for c in set(pDataC['cellType']):
    #     pca = PCA(n_components=0.95)
    #     pca.fit(C_init_df[pDataC['cellID'][pDataC['cellType'] == c]].values.T)
    #     components = pca.components_.T
    #     try:
    #         C_pca = np.hstack((C_pca, components))
    #     except NameError:
    #         C_pca = components
    # print('PCA components:', C_pca.shape[1])

    for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
        logger.info("Compute ground cost: " + m)
        ground_cost(C_init_df.values, result_path, metric=m)

    # Wasserstein
    logger.info("Start wasserstein deconvolution...")
    methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_all.values.astype('float'), standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m, 0.001, 0.001)) for m in methods]
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        m = methods[i]
        logger.info(' '.join([' ', m, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m)

    # Wasserstein dissTOM weighted
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
    pool = mp.Pool(10)
    m = 'dissTOM'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m+' '+sam, 0.001, 0.001)) for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    make_dir(result_path + '/lambda_individual')
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda_individual/lambda_{}.txt'.format(sam), lambda0)
        result_analysis(lambda_true_subjects[sam], lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j,jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Compute weighted wasserstein_"+m+" results...")
    weight = preprocess(NNLS(Y.reshape(-1,1).T, y_subject_df.values.T).T)
    print(list(weight.T))
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df*weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    result_analysis(lambda_true, lambda_weighted,'wasserstein' + '_' + m + '_weighted')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('wasserstein' + '_' + m + '_weighted'), lambda_weighted)
    logger.info("Weighted wasserstein_" + m + " completed!")

    experiment_main_analysis(root_path=result_path)


def params_experiment(root_path, dataset, subject):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = root_path + dataset + "/" + subject + "/"
    make_dir(result_path)
    make_dir(root_path + 'params_analysis/')
    make_dir(root_path + 'params_analysis/lambda')
    logger = default_logger(name='-'.join([dataset, subject]),
                            save_file_path=root_path + "params_analysis/base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Params(gamma and rho) analysis start...")
    logger.info("Data path: " + result_path)
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
    # logger.info("Select hvg and marker genes...")
    # Y_df, C_init_df, P_df, pDataC, D_all= load_data(result_path, dataset)
    # Weight_gene = pd.read_csv(result_path+'{}_Weight_gene.csv'.format(dataset),index_col=0,header=0,sep='\t')

    pDataC_group = pDataC.set_index(['sampleID'])
    # remove test subject
    C_init_df = C_init_df[C_init_df.columns[~C_init_df.columns.isin(pDataC_group.loc[subject.split("#")[0]]['cellID'])]]
    pDataC = pDataC[pDataC['sampleID'] != subject.split("#")[0]]
    # convert cell matrix into histogram
    C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)

    C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    all_celltype = D_all.columns.intersection(P_df.index)
    lambda_true = P_df.loc[all_celltype].values.astype('float')

    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cells: ' + str(C_init_df.shape[1]))
    C = C_init_df.values.astype('float')
    pDataC_group = pDataC.set_index(['sampleID'])

    # data info
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))

    # for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
    #     logger.info("Compute ground cost: " + m)
    #     ground_cost(C_init_df.values, result_path, metric=m)

    # Wasserstein
    logger.info("Start wasserstein deconvolution...")
    # methods = ['dissTOM']
    m = 'dissTOM'
    params_list = [0.1, 0.05, 0.01, 0.005, 0.001]

    # num_cores = int(mp.cpu_count())
    pool = mp.Pool(7)
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_all.values.astype('float'),
                                                                              standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m, gamma, rho)) for gamma in params_list for rho in
               params_list]

    params_analysis = pd.DataFrame(columns=['method', 'gamma', 'rho', 'time', 'rmse', 'cor'])
    params_error = pd.DataFrame(columns=['method', 'gamma', 'rho'])

    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        params = [str(i) for i in params]
        try:
            rmse, cor = result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m)
            params_analysis = params_analysis.append(
                {'method': params[0], 'gamma': params[1], 'rho': params[2], 'time': running_time, 'rmse': np.mean(rmse),
                 'cor': np.mean(cor)}, ignore_index=True)
        except ValueError:
            params_error = params_error.append({'method': params[0], 'gamma': params[1], 'rho': params[2]},
                                               ignore_index=True)
        logger.info(' '.join([' ', params[0], params[1], params[2], 'cost:', str(running_time), '(s)']))
        np.savetxt(root_path + 'params_analysis/lambda/{}_{}_{}.txt'.format(params[0], params[1], params[2]), lambda0)
    params_analysis.to_csv(root_path + 'params_analysis/params_analysis_df.csv', sep='\t', index=True, header=True)
    params_error.to_csv(root_path + 'params_analysis/params_error.csv', sep='\t', index=True, header=True)


def hba1c_analysis():
    result_path = '/media/user/新加卷/lg/cell_type_deconvolution/results/HbA1c/'
    data_set = 'hba1c'
    logger = default_logger(name='HbA1c', save_file_path=result_path + "base-log.log")
    logger.info("Reading data...")
    Y_df = pd.read_csv(result_path + '{}_T.csv'.format(data_set), index_col=0, header=0, sep='\t')
    # C_init_df = pd.read_csv(result_path + '{}_C.csv'.format(data_set), index_col=0, header=0, sep='\t')
    # pDataC = pd.read_csv(result_path + '{}_pDataC.csv'.format(data_set), index_col=0, header=0, sep='\t').astype('str')

    C_init_df = pd.read_csv(result_path + '{}_C.csv'.format('baron'), index_col=0, header=0, sep='\t')
    pDataC = pd.read_csv(result_path + '{}_pDataC.csv'.format('baron'), index_col=0, header=0, sep='\t').astype('str')
    C_init_emtab_df = pd.read_csv(result_path + '{}_C.csv'.format(data_set), index_col=0, header=0, sep='\t')
    pDataC_emtab = pd.read_csv(result_path + '{}_pDataC.csv'.format(data_set), index_col=0, header=0, sep='\t').astype(
        'str')
    C_init_df = pd.concat([C_init_df,C_init_emtab_df],axis=1,join='inner')
    pDataC = pd.concat([pDataC,pDataC_emtab],axis=0)


    select_celltype = ['alpha', 'beta', 'delta', 'gamma', 'acinar', 'ductal']
    pDataC = pDataC[pDataC['cellType'].isin(select_celltype)]
    C_init_df = C_init_df[C_init_df.columns[C_init_df.columns.isin(pDataC['cellID'])]]

    Y_df = Y_df.drop(C_init_df.index[np.where(C_init_df.sum(1) == 0)[0]])
    C_init_df = C_init_df.drop(C_init_df.index[np.where(C_init_df.sum(1) == 0)[0]])

    common_genes = C_init_df.index.intersection(Y_df.index)
    Y_df = Y_df.loc[common_genes]
    C_init_df = C_init_df.loc[common_genes]

    sampleid = list(set(pDataC['sampleID']))
    all_celltype = list(set(pDataC['cellType']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))
    pDataC_group = pDataC.set_index(['sampleID'])
    Y = Y_df.values.astype('float')

    # Wasserstein dissTOM weighted
    D_subjects = dict()
    missing_celltype = dict()
    for sam in sampleid:
        logger.info("------------------------------------------------")
        logger.info("Prepare data for " + str(sam) + ":")
        C_sam_init = C_init_df[pDataC_group.loc[sam]['cellID']]
        pDataC_sam = pDataC[pDataC["sampleID"] == sam]

        C_sam = pd.DataFrame(C_sam_init.values.T, index=C_sam_init.columns, columns=C_sam_init.index)
        C_sam['cellID'] = C_sam.index
        C_sam = C_sam.merge(pDataC_sam, on='cellID')
        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        logger.info(str(sam) + ' celltype: ' + str(D.columns.tolist()))
        logger.info(str(sam) + ' celltype counts: ' + str(len(D.columns.tolist())))
        if len(D.columns.tolist()) < len(all_celltype):
            missing_list = []
            for i, j in enumerate(all_celltype):
                if j not in D.columns:
                    missing_list.append(i)
                    logger.info(str(sam) + ' ' + j + " is missing!")
            missing_celltype[sam] = missing_list
            D_subjects[sam] = D.values.astype('float')
        else:
            D_subjects[sam] = D.values.astype('float')

    # ground_cost(C_init_df.values, result_path, metric='dissTOM')
    print(missing_celltype)
    logger.info("Start wasserstein deconvolution for all individuals...")
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(4)
    m = 'dissTOM'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0,
                                                                              normalized=0),
                                                                   m, result_path,
                                                                   m + ' ' + str(sam), 0.001, 0.001)) for sam in
               sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    make_dir(result_path + '/lambda_individual')
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda_individual/lambda_{}.txt'.format(sam), lambda0)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j, jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Compute weighted wasserstein_" + m + " results...")
    weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
    print(list(weight.T))
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df * weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape((len(all_celltype), len(Y_df.columns)))
    np.savetxt(result_path + '/lambda_{}.txt'.format('wasserstein' + '_' + m + '_weighted'), lambda_weighted)
    logger.info("Weighted wasserstein_" + m + " completed!")


def weight_analysis(root_path, dataset, subject):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = root_path + dataset + "/" + subject + "/"
    make_dir(root_path + "weight_analysis")
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=root_path + "weight_analysis/base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
    # logger.info("Select hvg and marker genes...")
    # Y_df, C_init_df, P_df, pDataC, D_all= load_data(result_path, dataset)
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
    # mixture_num = 200
    # Y = Y[:, 0:mixture_num]
    # lambda_true = lambda_true[:, 0:mixture_num]

    # data info
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))

    # for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
    #     logger.info("Compute ground cost: " + m)
    #     ground_cost(C_init_df.values, result_path, metric=m)

    # Wasserstein dissTOM weighted
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

        if sam == '22_male':
            C_sam = C_sam[(C_sam['cellType'] != 'ductal')&(C_sam['cellType'] != 'acinar')&(C_sam['cellType'] != 'delta')]
        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        logger.info(sam+' celltype: ' + str(D.columns.tolist()))
        logger.info(sam + ' celltype counts: ' + str(len(D.columns.tolist())))
        if len(D.columns.tolist()) < len(all_celltype):
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

    print(missing_celltype)
    for i in missing_celltype.keys():
        for j, jj in enumerate(missing_celltype[i]):
            if jj < len(D_all.columns):
                D_subjects[i] = np.insert(D_subjects[i], jj, values=D_all[D_all.columns[jj]].values, axis=1)
            elif jj == len(D_all.columns):
                D_subjects[i] = np.hstack((D_subjects[i], D_all[D_all.columns[jj]].values))
            print(D_subjects[i].shape)
    missing_celltype = dict()
    logger.info("Start wasserstein deconvolution for all individuals...")
    # methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(10)
    m = 'dissTOM'

    sam='22_male'
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   m, result_path,
                                                                   m+' '+sam, 0.001, 0.001))]# for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    make_dir(root_path + '/weight_analysis/lambda_individual')
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        # sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        np.savetxt(root_path + '/weight_analysis/lambda_individual/lambda_{}.txt'.format(sam), lambda0)
        # result_analysis(lambda_true_subjects[sam], lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m + ' ' + sam)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j,jj in enumerate(missing_celltype[sam]):
                if jj < lambda0.shape[0]:
                    lambda0 = np.insert(lambda0, jj, values=np.zeros(lambda0.shape[1]), axis=0)
                elif jj == lambda0.shape[0]:
                    lambda0 = np.vstack((lambda0,np.zeros(lambda0.shape[1])))
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    # logger.info("Compute weighted wasserstein_"+m+" results...")
    # weight = preprocess(NNLS(Y.reshape(-1,1).T, y_subject_df.values.T).T)
    # print(list(weight.T))
    # logger.info("Weight:" + str(list(weight.T)))
    # lambda_weighted = lambda_subject_df*weight.T
    # lambda_weighted = lambda_weighted.sum(1).values.reshape(lambda_true.shape)
    # result_analysis(lambda_true, lambda_weighted,'wasserstein' + '_' + m + '_weighted')
    # np.savetxt(root_path + '/weight_analysis/lambda_{}.txt'.format('wasserstein' + '_' + m + '_weighted'), lambda_weighted)
    # logger.info("Weighted wasserstein_" + m + " completed!")


def genes_num_experiment(root_path, dataset, subject):
    '''
    deconvolution for every individual, then weighted sum their proportion.
    :param dataset:
    :param data_root_path:
    :param result_root_path:
    :param subject:
    :param ground_cost_path:
    :return:
    '''
    result_path = root_path + dataset + "/" + subject + "/"
    make_dir(result_path)
    make_dir(root_path + 'params_analysis/')
    make_dir(root_path + 'params_analysis/lambda')
    logger = default_logger(name='-'.join([dataset, subject]),
                            save_file_path=root_path + "params_analysis/base-log.log")

    # result_df = pd.DataFrame()
    logger.info("Params(gamma and rho) analysis start...")
    logger.info("Data path: " + result_path)
    logger.info("Reading data...")
    Y_df, C_init_df, P_df, pDataC, D_all, D_music, Sigma, S = basis_matrix(result_path, dataset)
    # logger.info("Select hvg and marker genes...")
    # Y_df, C_init_df, P_df, pDataC, D_all= load_data(result_path, dataset)
    # Weight_gene = pd.read_csv(result_path+'{}_Weight_gene.csv'.format(dataset),index_col=0,header=0,sep='\t')

    pDataC_group = pDataC.set_index(['sampleID'])
    # remove test subject
    C_init_df = C_init_df[C_init_df.columns[~C_init_df.columns.isin(pDataC_group.loc[subject.split("#")[0]]['cellID'])]]
    pDataC = pDataC[pDataC['sampleID'] != subject.split("#")[0]]
    # convert cell matrix into histogram
    C_init_df = pd.DataFrame(data=preprocess(C_init_df.values.astype('float')), columns=C_init_df.columns, index=C_init_df.index)

    C = pd.DataFrame(C_init_df.values.T, index=C_init_df.columns, columns=C_init_df.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D_all = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    logger.info("Preprocessing data...")
    all_celltype = D_all.columns.intersection(P_df.index)
    lambda_true = P_df.loc[all_celltype].values.astype('float')

    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y_df.values.astype('float')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cells: ' + str(C_init_df.shape[1]))
    C = C_init_df.values.astype('float')
    pDataC_group = pDataC.set_index(['sampleID'])

    # data info
    sampleid = list(set(pDataC['sampleID']))
    logger.info('All celltype: ' + str(set(pDataC['cellType'])))
    logger.info('All celltype counts: ' + str(len(set(pDataC['cellType']))))

    # for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
    #     logger.info("Compute ground cost: " + m)
    #     ground_cost(C_init_df.values, result_path, metric=m)

    # Wasserstein
    logger.info("Start wasserstein deconvolution...")
    # methods = ['dissTOM']
    m = 'dissTOM'
    genes_list = [2000,4000,6000,8000,10000]

    running_time = pd.DataFrame(columns=['genes','time'])

    for i in genes_list:
        print('sample_genes:',i)
        D_sample = D_all.sample(n=i,replace=False)
        Y_sample = Y_df.loc[D_sample.index]
        ground_cost(C_init_df.loc[D_sample.index].values, '/media/user/新加卷/lg/cell_type_deconvolution/results/params_analysis/', metric=m)
        lambda0, t, params = wasserstein_NMF_coefficient(preprocess(Y_sample.values.astype('float'), standardized=0, normalized=0),
                                    preprocess(D_sample.values.astype('float'), standardized=0, normalized=0),
                                    m, '/media/user/新加卷/lg/cell_type_deconvolution/results/params_analysis/', m, 0.001, 0.001)

        running_time = running_time.append({'genes':i,'time':t},ignore_index=True)
        print('time:',t)
    running_time.to_csv(root_path + 'params_analysis/running_time_df.csv', sep='\t', index=True, header=True)


if __name__ == '__main__':
    # experiment_decot(root_path='../results/', dataset='Kidney_HCL', subject='Donor37')
    # experiment_decot_encemble(root_path='/media/user/新加卷/lg/cell_type_deconvolution/results/', dataset='Kidney_HCL', subject='Donor37#')
    # params_experiment(root_path='/media/user/新加卷/lg/cell_type_deconvolution/results/', dataset='GSE81547', subject='54_male')

    # weight_analysis(root_path='/media/user/新加卷/lg/cell_type_deconvolution/results/', dataset='GSE81547',subject='22_male')
    # hba1c_analysis()

    genes_num_experiment(root_path='/media/user/新加卷/lg/cell_type_deconvolution/results/', dataset='GSE81547',
                      subject='54_male')
