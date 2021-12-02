from wasserstein_NMF_coefficient_step import *
from result_analysis import *
from utils import *
from music_iter import music_basic

from sklearn.decomposition import PCA
import multiprocessing as mp
import time
import warnings

warnings.filterwarnings("ignore")

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import matlab
import matlab.engine

import logging
import logging.handlers


def NNLS(M, U):
    """
    NNLS performs non-negative constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).

    References:
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    import scipy.optimize as opt

    N, p1 = M.shape
    q, p2 = U.shape

    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        # opt.nnls() return a tuple, the first element is the result
        X[n1] = opt.nnls(MtM, np.dot(U, M[n1]))[0]
    return X


def wasserstein_music(Y, D, M, S, Sigma):
    Est_prop_allgene = np.zeros((D.shape[1], Y.shape[1]))
    Est_prop_weighted = np.zeros((D.shape[1], Y.shape[1]))
    weight_gene = np.zeros(Y.shape)

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores-2)

    start = time.time()

    gamma = 0.01
    K = np.exp(-M / gamma)
    K[np.where(K < 1e-200)] = 1e-200
    options = {'stop': 1e-3, 'Kmultiplication': 'symmetric', 'verbose': 1,
               'dual_descent_stop': 5e-4, 'alpha': 0.5}

    results = [pool.apply_async(music_basic, args=(Y[:, i], D, S, Sigma, K, options)) for i in range(Y.shape[1])]
    for i,p in enumerate(results):
        lm_D1_weighted = p.get()
        Est_prop_allgene[:, i] = lm_D1_weighted['p_nnls']
        Est_prop_weighted[:, i] = lm_D1_weighted['p_weight']
        weight_gene[:, i] = lm_D1_weighted['weight_gene']

    end = time.time()
    time_m = end - start
    print("wasserstein_music cost time: " + str(time_m))
    return Est_prop_allgene, Est_prop_weighted


def run_wasserstein(Y, D, M, m, gamma=0.001, rho=0.001):
    start = time.time()
    print(m, ' start...')
    K = np.exp(-M / gamma)
    K[np.where(K < 1e-200)] = 1e-200
    options = {'stop': 1e-3, 'Kmultiplication': 'symmetric', 'verbose': 1,
               'dual_descent_stop': 5e-4, 'alpha': 0.5}
    H = np.zeros(Y.shape)
    lambda0, H, obj, gradient = wasserstein_NMF_coefficient_step(Y, K, D, gamma, rho, H, options)
    end = time.time()
    print(m, ' cost:', str(end - start), '(s)')
    return lambda0, H, obj, gradient


def wasserstein_NMF(data, ground_cost):
    # a = preprocess(np.random.rand(10, 10), standardized=0, normalized=0)
    # M = ground_cost(a, metric='euclidean')
    engine = matlab.engine.start_matlab()
    start = time.time()
    print(' start...')
    D, lambda0, objectives = engine.run_wasserstein_DL(matlab.double(data.tolist()),
                                                       matlab.double(ground_cost.tolist()), nargout=3)
    end = time.time()
    print(' cost:', str(end - start), '(s)')
    return np.array(D), np.array(lambda0)


def wasserstein_NMF_coefficient(data, D, ground_cost, m, gamma=0.001, rho=0.001):
    # a = preprocess(np.random.rand(10, 10), standardized=0, normalized=0)
    # M = ground_cost(a, metric='euclidean')
    # print(data.shape)
    # print(D.shape)
    # print(ground_cost.shape)
    engine = matlab.engine.start_matlab()
    start = time.time()
    print(' '.join([' ', m, str(gamma), str(rho), 'start...']))
    lambda0, H, obj, gradient = engine.run_wasserstein_NMF_coefficient(matlab.double(data.tolist()),
                                                                       matlab.double(ground_cost.tolist()),
                                                                       matlab.double(D.tolist()), gamma, rho, nargout=4)
    end = time.time()
    running_time = end - start
    # logger.info(' '.join([' ', m, str(gamma), str(rho), 'cost:', str(running_time), '(s)']))
    print(m, ' cost:', str(running_time), '(s)')
    return np.array(lambda0), running_time, [m, gamma, rho]


def dissTOM(D, result_path):
    robjects.r.source('dissTOM.R')
    nr, nc = D.shape
    temp_D = robjects.r.matrix(D, nrow=nr, ncol=nc)
    robjects.r.assign("D", temp_D)
    dissTOM_mat = robjects.r.dissTOM(robjects.r["D"], result_path, min_max=False, z_score=True)
    return np.array(list(dissTOM_mat)).reshape(nr, nr)


def ground_cost(C, result_path, metric='euclidean'):
    if metric == 'dissTOM':
        M = dissTOM(C, result_path)
    else:
        # D = preprocess(D, standardized=True, normalized=False, histogram=False)
        M = ot.dist(C, metric=metric)
    make_dir(result_path + 'ground_cost/')
    file_name = result_path + 'ground_cost/' + metric
    np.save(file_name, M)
    print(M.shape)
    return M


def load_ground_cost(method, result_path, select_genes=None):
    try:
        M = np.load(result_path + 'ground_cost/{}.npy'.format(method))
        if select_genes is None:
            # print(M)
            return M
        else:
            return M[select_genes, :][:, select_genes]
    except FileNotFoundError:
        print('{} ground cost matrix is not computed.'.format(method))


def deconvolution_external_methods(T, C, pDataC, method):
    # run methods from R
    robjects.r.source('deconvolution.R')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        T_df = robjects.conversion.py2rpy(T)
        print('T convert complete.')
        C_df = robjects.conversion.py2rpy(C)
        print('C convert complete.')
        pDataC_df = robjects.conversion.py2rpy(pDataC)
        print('pDataC convert complete.')

    robjects.r.assign("T_df", T_df)
    robjects.r.assign("C_df", C_df)
    robjects.r.assign("pDataC_df", pDataC_df)
    result = robjects.r.Deconvolution(T=robjects.r["T_df"], C=robjects.r["C_df"], method=method,
                                      phenoDataC=robjects.r["pDataC_df"])
    return np.array(list(result))


def default_logger(name, save_file_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler1 = logging.FileHandler(save_file_path)
    handler1.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    return logger


def experiment_main(dataset, data_root_path, result_root_path, subject):
    result_path = result_root_path + dataset + "/" + subject + "/"
    make_dir(result_path)
    logger = default_logger(name='-'.join([dataset, subject]), save_file_path=result_path + "base-log.log")

    result_df = pd.DataFrame()
    # dataset = "baron_marker_hvg"
    # Y, D, lambda_true = mix_data()
    # Y, D, M, lambda_true = norm_data_mix()
    # Y, D, lambda_true, C = load_data("Kidney_HCL")
    logger.info("Reading data...")
    Y, C_init, P, pDataC, D, D_music, Sigma, S = basis_matrix(data_root_path, "baron")
    # for method in ['MuSiC']:
    #     result = deconvolution_external_methods(Y, C_init, pDataC, method)

    # dataset = "Kidney_HCL_music_basis"
    logger.info("Preprocessing data...")
    lambda_true = P.loc[D.columns].values.astype('float')
    make_dir(result_path + '/lambda')
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('true'), lambda_true)
    Y = Y.values.astype('float')
    D = D.values.astype('float')
    D_music = D_music.values.astype('float')
    C = C_init.values.astype('float')

    # Y_full, C_init_full, P_full, pDataC_full, D_full, D_music_full = basis_matrix(data_root_path, dataset)
    #
    # C_init_full['n'] = [i for i, j in enumerate(C_init_full.index)]
    # select_genes = C_init_full.loc[C_init.index]['n'].to_list()

    mixture_num = 200
    Y = Y[:, 0:mixture_num]
    lambda_true = lambda_true[:, 0:mixture_num]
    logger.info("Preprocess data finished...")
    ## dataset info
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   celltype: ' + str(lambda_true.shape[0]))

    # PCA before compute ground cost
    logger.info("Start PCA...")
    # for c in set(pDataC['cellType']):
    #     pca = PCA(n_components=0.95)
    #     pca.fit(C_init[pDataC['cellID'][pDataC['cellType'] == c]].values.T)
    #     components = pca.components_.T
    #     try:
    #         C_pca = np.hstack((C_pca, components))
    #     except NameError:
    #         C_pca = components
    # print('PCA components:', C_pca.shape[1])

    pca = PCA(n_components=0.95)
    pca.fit(C_init.values.T)
    C_pca = pca.components_.T
    print('PCA components:', C_pca.shape[1])

    for m in ['dissTOM', 'euclidean', 'cosine', 'correlation']:
        logger.info("Compute ground cost: "+m)
        ground_cost(C, result_path, metric=m)

    # Est_prop_allgene, Est_prop_weighted = wasserstein_music(preprocess(Y, standardized=0, normalized=0), preprocess(D_music, standardized=0, normalized=0), load_ground_cost('dissTOM', result_path, select_genes=None), S, Sigma)
    # np.savetxt(result_path + '/lambda/lambda_wasserstein_music_{}.txt'.format('dissTOM'), Est_prop_weighted)

    logger.info("Start wasserstein deconvolution...")
    methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)

    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D, standardized=0, normalized=0),
                                                                   load_ground_cost(m, result_path, select_genes=None), 
                                                                   m, 0.001, 0.001)) for m in methods]

    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        m = methods[i]
        logger.info(' '.join([' ', m, 'cost:', str(running_time), '(s)']))
        np.savetxt(result_path + '/lambda/lambda_wasserstein_{}.txt'.format(m), lambda0)
        result_df[m + '_rmse'], result_df[m + '_cor'] = result_analysis(lambda_true, lambda0, 'wasserstein' + '_' + m)

    # 'dissTOM', 'euclidean', 'cosine', 'correlation'
    logger.info("Start NNLS deconvolution...")
    lambda1 = preprocess(NNLS(Y.T, D.T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS'), lambda1)
    result_df['NNLS_rmse'], result_df['NNLS_cor'] = result_analysis(lambda_true, lambda1, 'NNLS')
    logger.info("NNLS finished...")
    lambda2 = preprocess(NNLS(Y.T, D_music.T).T)
    np.savetxt(result_path + '/lambda/lambda_{}.txt'.format('NNLS_music_basis'), lambda2)
    result_df['NNLS_music_basis_rmse'], result_df['NNLS_music_basis_cor'] = result_analysis(lambda_true, lambda2,
                                                                                            'NNLS_music_basis')
    # result_df.to_csv(result_path + '/result_df.csv', sep='\t', index=True, header=True)


def experiment_param(dataset="Kidney_HCL"):
    make_dir('../experiment_param_results/' + dataset)
    make_dir('../experiment_param_results/' + dataset + '/boxplot')
    result_df = pd.DataFrame()
    result_running_time_df = pd.DataFrame(columns=['time', 'method', 'gamma', 'rho'])
    Y, C_init, P, pDataC, D = load_data(dataset)
    lambda_true = P.loc[D.index].values.astype('float')
    Y = Y.values.astype('float')
    D = D.values.T.astype('float')
    C = C_init.values.astype('float')

    mixture_num = 50
    Y = Y[:, 0:mixture_num]
    lambda_true = lambda_true[:, 0:mixture_num]
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(10)
    methods = ['dissTOM', 'euclidean', 'cosine', 'correlation']
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D, standardized=0, normalized=0),
                                                                   load_ground_cost(m, dataset), m, gamma, rho)) for m
               in methods
               for gamma in [0.001, 0.005, 0.01, 0.05, 0.1] for rho in [0.001, 0.005, 0.01, 0.05, 0.1]]
    # [0.001, 0.005, 0.01, 0.05, 0.1]

    params_error = pd.DataFrame(columns=['method', 'gamma', 'rho'])
    for p in results:
        lambda0, running_time, params = p.get()
        params = [str(i) for i in params]
        m = '_'.join(params)
        result_running_time_df = result_running_time_df.append(
            {'time': running_time, 'method': params[0], 'gamma': params[1], 'rho': params[2]}, ignore_index=True)
        try:
            result_df[m + '_rmse'], result_df[m + '_cor'] = result_analysis(lambda_true, lambda0,
                                                                            'wasserstein' + '_' + m)
        except ValueError:
            params_error = params_error.append({'method': params[0], 'gamma': params[1], 'rho': params[2]},
                                               ignore_index=True)
    result_running_time_df.to_csv('../experiment_param_results/' + dataset + '/result_running_time_df.csv', sep='\t',
                                  index=True, header=True)
    result_df.to_csv('../experiment_param_results/' + dataset + '/results_init_df.csv', sep='\t', index=True,
                     header=True)
    params_error.to_csv('../experiment_param_results/' + dataset + '/params_error.csv', sep='\t', index=True,
                        header=True)

    experiment_param_analysis(result_df, dataset)


def dataset_info():
    for dataset in ['baron', 'GSE81547', 'Kidney_HCL', 'EMTAB5061']:
        Y, C_init, P, pDataC, D = load_data(dataset)
        print(dataset, ':')
        print('genes: ', Y.shape[0])

        lambda_true = P.loc[D.index].values.astype('float')
        print('celltype: ', lambda_true.shape[0])
        print('------------------------------------------------')


if __name__ == '__main__':
    # 'baron', 'GSE81547', 'Kidney_HCL', 'EMTAB5061'
    # for dataset in ['baron','GSE81547','EMTAB5061']:
    #     experiment_param(dataset)

    # dataset_info()
    # basis_matrix('Kidney_HCL')
    # experiment_main(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/train_data/',dataset='baron',result_path='../results/baron/')

    experiment_main(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/baron/human2/',
                    dataset='baron_pca', result_root_path='../results_e3/', subject='human2')
    # experiment_main(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/data_for_nextflow/Kidney_HCL/Donor34/',
    #                 dataset='Kidney_HCL', result_root_path='../results_e3/', subject='Donor34')
