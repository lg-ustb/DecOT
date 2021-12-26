from wasserstein_NMF_coefficient_step import wasserstein_NMF_coefficient_step
from result_analysis import result_analysis
from utils import *

import multiprocessing as mp
import time
import warnings
import ot
from ot.datasets import make_1D_gauss as gauss

warnings.filterwarnings("ignore")

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
import matlab
import matlab.engine

import logging
import logging.handlers
import argparse


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
    return lambda0


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


def wasserstein_NMF_coefficient(data, D, m, ground_cost_path, name, gamma=0.001, rho=0.001):
    # a = preprocess(np.random.rand(10, 10), standardized=0, normalized=0)
    # M = ground_cost(a, metric='euclidean')
    # print(data.shape)
    # print(D.shape)
    # print(ground_cost.shape)
    ground_cost = load_ground_cost(m, ground_cost_path, select_genes=None)
    engine = matlab.engine.start_matlab()
    start = time.time()
    print(' '.join([' ', name, str(gamma), str(rho), 'start...']))
    lambda0, H, obj, gradient = engine.run_wasserstein_NMF_coefficient(matlab.double(data.tolist()),
                                                                       matlab.double(ground_cost.tolist()),
                                                                       matlab.double(D.tolist()), gamma, rho, nargout=4)
    end = time.time()
    running_time = end - start
    # logger.info(' '.join([' ', m, str(gamma), str(rho), 'cost:', str(running_time), '(s)']))
    print(name, ' cost:', str(running_time), '(s)')
    return np.array(lambda0), running_time, [name, gamma, rho]


def dissTOM(D, result_path):
    robjects.r.source('dissTOM.R')
    nr, nc = D.shape
    temp_D = robjects.r.matrix(D, nrow=nr, ncol=nc)
    robjects.r.assign("D", temp_D)
    dissTOM_mat = robjects.r.dissTOM(robjects.r["D"], result_path, min_max=False, z_score=True)
    return np.array(list(dissTOM_mat)).reshape(nr, nr)


def ground_cost(C, result_path, metric='dissTOM'):
    if metric == 'dissTOM':
        M = dissTOM(C, result_path)
    else:
        M = ot.dist(C, metric=metric)
    make_dir(result_path + 'ground_cost/')
    file_name = result_path + 'ground_cost/' + metric
    np.save(file_name, M)
    print(M.shape)
    return M


def load_ground_cost(metric, result_path, select_genes=None):
    try:
        M = np.load(result_path + 'ground_cost/{}.npy'.format(metric))
        if select_genes is None:
            # print(M)
            return M
        else:
            return M[select_genes, :][:, select_genes]
    except FileNotFoundError:
        print('{} ground cost matrix is not found.'.format(metric))


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


def random_gauss_test(bins=100, mixtures=5, samples=10):
    random_seed = np.random.RandomState(0)

    mix_num = random_seed.randint(0, 100, (mixtures, samples)).astype('float')
    random_seed = np.random.RandomState(0)
    mean = random_seed.randint(0, 100, mixtures).astype('float')
    random_seed = np.random.RandomState(0)
    std = random_seed.randint(1, 10, mixtures).astype('float')

    D = np.zeros((bins, mixtures))
    for i in range(mixtures):
        D[:, i] = gauss(bins, m=mean[i], s=std[i])
        plt.plot(np.arange(bins, dtype=np.float64), D[:, i])
    plt.savefig('Gauss distributions.jpg')
    # plt.show()
    mix_ratio = preprocess(mix_num, standardized=0, normalized=0)
    Y_mix = np.dot(D, mix_ratio)

    M = ot.dist(np.arange(bins, dtype=np.float64).reshape((bins, 1)), np.arange(bins, dtype=np.float64).reshape((bins, 1)), metric='euclidean')
    return Y_mix, D, M, mix_ratio


def decot(Y, C, pDataC, ground_cost_path, metric, save_path, gamma=0.001, rho=0.001):
    '''
    Parameters:
        Y: `pandas dataframe`
            bulk mixtures (genes x mixtures).

        C: `pandas dataframe`
            reference cells (genes x cells).

        pDataC: `pandas dataframe`
            cell phenotype data (cells x ['cellID', 'cellType', 'sampleID'])

        ground_cost_path: str
            ground cost matrix file path

        metric: str
            metric used to compute the ground cost (['dissTOM', 'euclidean', 'cosine', 'correlation'])

        save_path: str
            path to save deconvolution results

        gamma: regularization parameter
            default 0.001

        rho: regularization parameter
            default 0.001

    Returns: `numpy array`
        cell type proportion (celltypes x mixtures).
    '''
    common_genes = C.index.intersection(Y.index)
    Y = Y.loc[common_genes]
    C = C.loc[common_genes]

    logger = default_logger(name='Decot', save_file_path=save_path + "Decot-log.log")
    logger.info('Data set information:')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cell types: ' + str(len(set(pDataC['cellType']))))

    logger.info("Computing ground cost matrix...")
    M = ground_cost(C.values.astype('float'), ground_cost_path, metric)

    logger.info("Computing cell type-specific expression matrix...")
    C_temp = pd.DataFrame(C.values.T, index=C.columns, columns=C.index)
    C_temp['cellID'] = C_temp.index
    C_temp = C_temp.merge(pDataC, on='cellID')
    D = C_temp.groupby('cellType').mean()
    D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    lambda0, running_time, params = wasserstein_NMF_coefficient(preprocess(Y.values.astype('float'), standardized=0, normalized=0),
                                                                preprocess(D.values.astype('float'), standardized=0, normalized=0),
                                                                metric, ground_cost_path, 'Decot-' + metric, gamma, rho)
    logger.info(' '.join(['cost time:', str(running_time), '(s)']))
    np.savetxt(save_path + '/lambda/lambda_decot_{}.txt'.format(metric), lambda0)

    return lambda0


def decot_ensemble(Y, C, pDataC, ground_cost_path, metric, save_path, cores=4, gamma=0.001, rho=0.001):
    '''
    Parameters:
        Y: `pandas dataframe`
            bulk mixtures (genes x mixtures).

        C: `pandas dataframe`
            reference cells (genes x cells).

        pDataC: `pandas dataframe`
            cell phenotype data (cells x ['cellID', 'cellType', 'sampleID'])

        ground_cost_path: str
            ground cost matrix file path

        metric: str
            metric used to compute the ground cost (['dissTOM', 'euclidean', 'cosine', 'correlation'])

        save_path: str
            path to save deconvolution results

        cores: int
            number of cores used for parallel computing

        gamma: regularization parameter
            default 0.001

        rho: regularization parameter
            default 0.001

    Returns: `numpy array`
        cell type proportion (celltypes x mixtures).
    '''
    common_genes = C.index.intersection(Y.index)
    Y = Y.loc[common_genes]
    C = C.loc[common_genes]
    sampleid = list(set(pDataC['sampleID']))
    sampleid.sort()

    logger = default_logger(name='Decot', save_file_path=save_path + "Decot-log.log")
    logger.info('Data set information:')
    logger.info('   genes: ' + str(Y.shape[0]))
    logger.info('   cell types: ' + str(len(set(pDataC['cellType']))))
    logger.info('   individuals: ' + str(len(sampleid)))

    C_temp = pd.DataFrame(C.values.T, index=C.columns, columns=C.index)
    C_temp['cellID'] = C_temp.index
    C_temp = C_temp.merge(pDataC, on='cellID')
    D_all = C_temp.groupby('cellType').mean()
    D_all = pd.DataFrame(D_all.values.T, index=D_all.columns, columns=D_all.index)

    logger.info("Computing ground cost matrix...")
    M = ground_cost(C.values.astype('float'), ground_cost_path, metric)

    logger.info('Computing cell type-specific expression matrix for each individual...')
    pDataC_group = pDataC.set_index(['sampleID'])
    all_celltype = list(set(pDataC['cellType']))
    all_celltype.sort()
    D_subjects = dict()
    missing_celltype = dict()
    for sam in sampleid:
        logger.info("------------------------------------------------")
        logger.info("Prepare data for " + sam + ":")
        C_sam = C[pDataC_group.loc[sam]['cellID']]
        pDataC_sam = pDataC[pDataC["sampleID"] == sam]

        C_sam = pd.DataFrame(C_sam.values.T, index=C_sam.columns, columns=C_sam.index)
        C_sam['cellID'] = C_sam.index
        C_sam = C_sam.merge(pDataC_sam, on='cellID')
        D = C_sam.groupby('cellType').mean()
        D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)
        logger.info(sam + ' celltype: ' + str(D.columns.tolist()))
        logger.info(sam + ' celltype counts: ' + str(len(D.columns.tolist())))
        if len(D.columns.tolist()) < len(all_celltype):
            missing_list = []
            for i, j in enumerate(all_celltype):
                if j not in D.columns:
                    missing_list.append(i)
                    logger.info(sam + ' ' + j + " is missing!")
            missing_celltype[sam] = missing_list
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
    pool = mp.Pool(cores)
    results = [pool.apply_async(wasserstein_NMF_coefficient, args=(preprocess(Y, standardized=0, normalized=0),
                                                                   preprocess(D_subjects[sam], standardized=0, normalized=0),
                                                                   metric, save_path, metric + '-' + sam, gamma, rho)) for sam in sampleid]

    lambda_subject_df = pd.DataFrame(columns=sampleid)
    y_subject_df = pd.DataFrame(columns=sampleid)
    make_dir(save_path + '/lambda_individual')
    for i, p in enumerate(results):
        lambda0, running_time, params = p.get()
        sam = sampleid[i]
        logger.info(' '.join([' ', sam, 'cost:', str(running_time), '(s)']))
        np.savetxt(save_path + '/lambda_individual/lambda_{}.txt'.format(sam), lambda0)
        y_subject_df[sam] = np.dot(D_subjects[sam], lambda0).reshape(-1)
        if sam in missing_celltype.keys():
            for j, jj in enumerate(missing_celltype[sam]):
                lambda0 = np.insert(lambda0, jj, values=np.zeros(lambda0.shape[1]), axis=0)
                print(lambda0.shape)
            lambda_subject_df[sam] = lambda0.reshape(-1)
        else:
            lambda_subject_df[sam] = lambda0.reshape(-1)
    logger.info("Computing ensemble results...")
    weight = preprocess(NNLS(Y.reshape(-1, 1).T, y_subject_df.values.T).T)
    print(list(weight.T))
    logger.info("Weight:" + str(list(weight.T)))
    lambda_weighted = lambda_subject_df * weight.T
    lambda_weighted = lambda_weighted.sum(1).values.reshape(len(all_celltype), Y.shape[1])
    np.savetxt(save_path + '/lambda/lambda_{}.txt'.format('DecOT_ensemble_' + metric), lambda_weighted)
    logger.info("DecOT ensemble-" + metric + " completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose a method and pass the corresponding parameters')
    parser.add_argument('--random_gauss_test', required=False, help='Random gauss test (toy model).', action="store_true")
    parser.add_argument('--genes', required=False, help='Gauss distribution bins (genes).')
    parser.add_argument('--types', required=False, help='Gauss distribution nums (cell types).')
    parser.add_argument('--mixtures', required=False, help='Bulk mixtures.')

    parser.add_argument('--ensemble', required=False, help='DecOT with or without ensemble.', action="store_true")
    parser.add_argument('--y', required=False, help='Bulk mixtures (genes x mixtures).')
    parser.add_argument('--c', required=False, help='Reference cells (genes x cells).')
    parser.add_argument('--pDataC', required=False, help="Cell phenotype data (cells x ['cellID', 'cellType', 'sampleID']).")
    parser.add_argument('--ground_cost_path', required=False, help='Ground cost matrix file path.')
    parser.add_argument('--metric', required=False, help="Metric used to compute the ground cost (['dissTOM', 'euclidean', 'cosine', 'correlation'])")
    parser.add_argument('--save_path', required=False, help='Path to save deconvolution results.')
    parser.add_argument('-cores', required=False, help='Number of cores used for parallel computing.')
    parser.add_argument('-gamma', required=False, help='Regularization parameter.')
    parser.add_argument('-rho', required=False, help='Regularization parameter.')

    args = parser.parse_args()
    if args.random_gauss_test:
        # random gauss distribution mixtures
        Y_mix, D, M, mix_ratio = random_gauss_test(bins=int(args.genes), mixtures=int(args.types), samples=int(args.mixtures))
        pred_ratio = run_wasserstein(Y_mix, D, M, '')
        result_analysis(mix_ratio, pred_ratio, '')
    else:
        if args.ensemble:
            Y = pd.read_csv(args.y, index_col=0, header=0, sep='\t')
            C = pd.read_csv(args.c, index_col=0, header=0, sep='\t')
            pDataC = pd.read_csv(args.pDataC, index_col=0, header=0, sep='\t')
            decot_ensemble(Y, C, pDataC, args.ground_cost_path, args.metric, args.save_path, cores=int(args.cores), gamma=float(args.gamma), rho=float(args.rho))
        else:
            Y = pd.read_csv(args.y, index_col=0, header=0, sep='\t')
            C = pd.read_csv(args.c, index_col=0, header=0, sep='\t')
            pDataC = pd.read_csv(args.pDataC, index_col=0, header=0, sep='\t')
            decot(Y, C, pDataC, args.ground_cost_path, args.metric, args.save_path, gamma=float(args.gamma), rho=float(args.rho))
