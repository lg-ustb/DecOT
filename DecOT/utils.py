import os
import pandas as pd
import numpy as np


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


def preprocess(data, standardized=0, normalized=0, histogram=1):
    try:
        for i in range(data.shape[1]):
            if standardized:
                data[:, i] = standardization(data[:, i])
            if normalized:
                data[:, i] = normalization(data[:, i])
            if histogram:
                data[:, i] = data[:, i] / sum(data[:, i])
    except IndexError:
        if standardized:
            data = standardization(data)
        if normalized:
            data = normalization(data)
        if histogram:
            data = data / sum(data)
    return data


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def dataset_info():
    for dataset in ['baron', 'GSE81547', 'Kidney_HCL', 'EMTAB5061']:
        Y, C_init, P, pDataC, D = load_data(dataset)
        print(dataset, ':')
        print('genes: ', Y.shape[0])

        lambda_true = P.loc[D.index].values.astype('float')
        print('celltype: ', lambda_true.shape[0])
        print('------------------------------------------------')


if __name__ == '__main__':
    pass
