import pandas as pd
import numpy as np
from tqdm import tqdm
from wasserstein_NMF_coefficient_step import *
import heapq
import multiprocessing as mp
import time
import matlab
import matlab.engine
import warnings
warnings.filterwarnings('ignore')


def music_basic(Y, X, S, Sigma, K, options, iter_max=1000, nu=1e-4, eps=0.01):
    # k = X.shape[0]
    lm_D = wassertein_regression(Y, X, K, options)
    r = lm_D['reside']
    x = lm_D['x']
    weight_gene = 1 / (nu + r ** 2 + ((x * S.T) ** 2 * Sigma).sum(1))
    weight_gene = preprocess(weight_gene, standardized=0)


    Y_weight = Y * np.sqrt(weight_gene)
    D_weight = (X.T * np.sqrt(weight_gene)).T

    # Y_weight = preprocess(Y_weight, standardized=0)
    # D_weight = preprocess(D_weight, standardized=0)
    lm_D_weight = wassertein_regression(Y_weight, D_weight, K, options)
    p_weight = lm_D_weight['x'] / sum(lm_D_weight['x'])
    # p_weight_iter = p_weight
    r = lm_D_weight['reside']
    for iter in range(1, iter_max + 1):
        weight_gene = 1 / (nu + r ** 2 + ((lm_D_weight['x'] * S.T) ** 2 * Sigma).sum(1))
        weight_gene = preprocess(weight_gene, standardized=0)
        Y_weight = Y * np.sqrt(weight_gene)
        D_weight = (X.T * np.sqrt(weight_gene)).T
        # Y_weight = preprocess(Y_weight, standardized=0)
        # D_weight = preprocess(D_weight, standardized=0)
        lm_D_weight = wassertein_regression(Y_weight, D_weight, K, options)

        p_weight_new = lm_D_weight['x'] / sum(lm_D_weight['x'])
        r_new = lm_D_weight['reside']
        if np.sum(np.abs(p_weight_new - p_weight)) < eps:
            p_weight = p_weight_new
            r = r_new
            R_squared = 1 - sample_variance(Y - np.dot(X, lm_D_weight['x'])) / sample_variance(Y)
            fitted = np.dot(X, lm_D_weight['x'])
            var_p = np.diag(np.linalg.inv(np.dot(D_weight.T, D_weight))) * np.mean(r ** 2) / lm_D_weight['x'].sum() ** 2
            return dict(p_nnls=lm_D['x'] / sum(lm_D['x']), q_nnls=lm_D['x'], fit_nnls=lm_D['fitted'],
                        resid_nnls=lm_D['reside'], p_weight=p_weight, q_weight=lm_D_weight['x'],
                        fit_weight=fitted, resid_weight=Y - np.dot(X, lm_D_weight['x']), weight_gene=weight_gene,
                        converge="Converge at " + str(iter), rsd=r, R_squared=R_squared, var_p=var_p)
        p_weight = p_weight_new
        r = r_new
    R_squared = 1 - sample_variance(Y - np.dot(X, lm_D_weight['x'])) / sample_variance(Y)
    fitted = np.dot(X, lm_D_weight['x'])
    var_p = np.diag(np.linalg.inv(np.dot(D_weight.T), D_weight)) * np.mean(r ** 2) / lm_D_weight['x'].sum() ** 2
    return dict(p_nnls=lm_D['x'] / sum(lm_D['x']), q_nnls=lm_D['x'], fit_nnls=lm_D['fitted'],
                resid_nnls=lm_D['reside'], p_weight=p_weight, q_weight=lm_D_weight['x'],
                fit_weight=fitted, resid_weight=Y - np.dot(X, lm_D_weight['x']), weight_gene=weight_gene,
                converge="Reach Max iter", rsd=r, R_squared=R_squared, var_p=var_p)


def sample_variance(X):
    return np.var(X) / (len(X) - 1) * len(X)


def wassertein_regression(Y, D, K, options, gamma=0.01, rho=0.01, ):
    H = np.zeros(Y.shape)
    lambda0, H, obj, gradient = wasserstein_NMF_coefficient_step(Y, K, D, gamma, rho, H, options)

    # engine = matlab.engine.start_matlab()
    # lambda0, H, obj, gradient = engine.run_wasserstein_NMF_coefficient(matlab.double(Y.tolist()),
    #                                                                    matlab.double(K.tolist()),
    #                                                                    matlab.double(D.tolist()), gamma, rho, nargout=4)
    return {'x': lambda0, 'reside': Y-np.dot(D, lambda0), 'fitted': np.dot(D, lambda0)}


def music_iter(Y, X, S, Sigma, iter_max=1000, nu=1e-04, eps=0.01, centered=False, normalize=False):
    if centered:
        X = X - np.mean(X, 0)
        Y = Y - np.mean(Y)
    if normalize:
        X = X / np.std(X, 0)
        S = S * np.std(X, 0)
        Y = Y / np.std(Y)
    else:
        Y = Y * 100
    lm_D = music_basic(Y, X, S, Sigma, iter_max=iter_max, nu=nu, eps=eps)
    return lm_D


def max_weight_genes(weight_gene, n=100):
    max_index = set()
    for i in range(weight_gene.shape[1]):
        max_index = max_index.union(heapq.nlargest(n, range(len(weight_gene[:, i])), weight_gene[:, i].take))
    return list(max_index)


def main():
    Y, D, weight_gene, S, Sigma = read_data()
    max_index = max_weight_genes(weight_gene, 20)
    print('seleted genes:', len(max_index))
    Y = Y[max_index]
    D = D[max_index]
    Sigma = Sigma[max_index]

    # Y = Y[:50, :]
    # D = D[:50, :]
    # Sigma = Sigma[:50, :]

    Y = preprocess(Y, standardized=0)
    D = preprocess(D, standardized=0)

    # M = ground_cost(D)
    M = ot.dist(D, D)*len(max_index)
    print('cost 矩阵创建完成')
    gamma = 0.01
    rho = 0.1

    K = np.exp(-M / gamma)
    K[np.where(K < 1e-200)] = 1e-200
    options = {'stop': 1e-3, 'Kmultiplication': 'symmetric', 'verbose': 2, 'D_step_stop': 5e-5,
               'lambda_step_stop': 5e-4, 'alpha': 0.5}

    Est_prop_allgene = np.zeros((D.shape[1], Y.shape[1]))
    Est_prop_weighted = np.zeros((D.shape[1], Y.shape[1]))
    weight_gene = np.zeros(Y.shape)

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores-2)

    start = time.time()
    # for i in tqdm(range(Y.shape[1])):
    #     lm_D1_weighted = music_basic(Y[:, i], D, S, Sigma, K, options)

    #     Est_prop_allgene[:, i] = lm_D1_weighted['p_nnls']
    #     Est_prop_weighted[:, i] = lm_D1_weighted['p_weight']
    #     weight_gene[:, i] = lm_D1_weighted['weight_gene']

    results = [pool.apply_async(music_basic, args=(Y[:, i], D, S, Sigma, K, options)) for i in range(Y.shape[1])]
    for i,p in enumerate(results):
        lm_D1_weighted = p.get()
        Est_prop_allgene[:, i] = lm_D1_weighted['p_nnls']
        Est_prop_weighted[:, i] = lm_D1_weighted['p_weight']
        weight_gene[:, i] = lm_D1_weighted['weight_gene']

    end = time.time()
    time_m = end - start
    print("time: " + str(time_m))
    np.savetxt('../results/Est_prop_allgene.txt', Est_prop_allgene)
    np.savetxt('../results/Est_prop_weighted.txt', Est_prop_weighted)
    pass


if __name__ == '__main__':
    main()
