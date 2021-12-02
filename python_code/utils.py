import os
from wasserstein_NMF_coefficient_step import *
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import seaborn as sns


def load_data(data_root_path='/media/user/新加卷/lg/deconv_benchmark-master/train_data/', data_set='baron'):
    Y = pd.read_csv(data_root_path+'{}_T.csv'.format(data_set),index_col=0,header=0,sep='\t')
    C_init = pd.read_csv(data_root_path+'{}_C.csv'.format(data_set),index_col=0,header=0,sep='\t')
    P = pd.read_csv(data_root_path+'{}_P.csv'.format(data_set),index_col=0,header=0,sep='\t')
    pDataC = pd.read_csv(data_root_path+'{}_pDataC.csv'.format(data_set),index_col=0,header=0,sep='\t')

    C = pd.DataFrame(C_init.values.T, index=C_init.columns, columns=C_init.index)
    C['cellID'] = C.index
    C = C.merge(pDataC, on='cellID')
    D = C.groupby('cellType').mean()
    D = pd.DataFrame(D.values.T, index=D.columns, columns=D.index)

    return Y, C_init, P, pDataC, D


def basis_matrix(data_root_path, dataset):
    Y, C_init, P, pDataC, D = load_data(data_root_path=data_root_path, data_set=dataset)

    pDataC_group = pDataC.set_index(['cellType','sampleID'])
    celltype = set(pDataC['cellType'])
    sampleid = set(pDataC['sampleID'])

    D_music = pd.DataFrame(columns=celltype)
    Sigma = pd.DataFrame(columns=celltype)
    S = pd.DataFrame(index=celltype, columns=sampleid)
    for c in celltype:
        theta = pd.DataFrame(index=C_init.index,columns=sampleid)
        for sam in sampleid:
            try:
                r = C_init[pDataC_group.loc[c, sam]['cellID']].sum(1)
                theta[sam] = r/r.sum()
                S.loc[c,sam] = r.sum()/pDataC_group.loc[c, sam]['cellID'].shape[0]
            except KeyError:
                S.loc[c, sam] = 0
        m_theta = theta.mean(1)
        Sigma[c] = theta.var(1)
        D_music[c] = m_theta*S.loc[c].mean()

    return Y, C_init, P, pDataC, D, D_music[D.columns], Sigma, S


def mix_data():
    Y, D, weight_gene, S, Sigma = read_data()
    max_index = max_weight_genes(weight_gene, 30)
    print('seleted genes:', len(max_index))

    D = D[max_index]
    pd.DataFrame(D).to_csv('../data/SC_seleted_100.csv', header=False, index=False)
    cor = 1-ground_cost(D, metric='correlation')
    plt.hist(np.power(np.abs(cor.flatten()), 3))
    plt.show()
    # D = D[0:1000, :]

    sample_num = 10
    random_seed = np.random.RandomState(0)
    mix_num = random_seed.randint(0, 10, (D.shape[1], sample_num)).astype('float')
    Y_mix = np.dot(D, mix_num)
    print(mix_num)

    mix_ratio = preprocess(mix_num, standardized=0, normalized=0)
    np.save('../data/mix_data/Y.npy', Y_mix)
    np.save('../data/mix_data/D.npy', D)
    np.save('../data/mix_data/mix_ratio.npy', mix_ratio)
    return Y_mix, D, mix_ratio


def norm_data_mix(bins=100, mixture=5, sample=10):
    random_seed = np.random.RandomState(0)

    mix_num = random_seed.randint(0, 100, (mixture, sample)).astype('float')
    random_seed = np.random.RandomState(0)
    mean = random_seed.randint(0, 100, mixture).astype('float')
    random_seed = np.random.RandomState(0)
    std = random_seed.randint(1, 10, mixture).astype('float')

    D = np.zeros((bins, mixture))
    for i in range(mixture):
        D[:, i] = gauss(bins, m=mean[i], s=std[i])
        plt.plot(np.arange(bins, dtype=np.float64), D[:, i])

    plt.show()
    mix_ratio = preprocess(mix_num, standardized=0, normalized=0)
    Y_mix = np.dot(D, mix_ratio)

    # Y_temp = np.dot(D, mix_num)
    # Y_temp = preprocess(Y_temp, standardized=0, normalized=0)

    # print(mean_squared_error(Y_mix, Y_temp))

    M = ot.dist(np.arange(bins, dtype=np.float64).reshape((bins, 1)), np.arange(bins, dtype=np.float64).reshape((bins, 1)), metric='euclidean')
    return Y_mix, D, M, mix_ratio


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


if __name__ == '__main__':
    # Y, D, lambda_true = mix_data()
    # Y, D, lambda_true = load_data()
    # print(ground_cost(D, 'dissTOM'))
    Y, C_init, P, pDataC, D = load_data("GSE81547")
    C = C_init.values.astype('float')
    M = ground_cost(C, metric="dissTOM")
