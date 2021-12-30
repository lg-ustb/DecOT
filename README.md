# DecOT
DecOT is a bulk gene expression deconvolution method that uses the optimal transport distance as loss and apply an ensemble framework to integrate reference information from scRNA-seq data of multiple individuals. 

## Installation

DecOT is a python script developed in a Windows and Linux 64-bit architecture. To use DecOT, the following packages need to be installed：

python packages:

* Pandas
* Numpy
* Scipy
* POT
* rpy2
* multiprocess
* matlab

R packages:

* WGCNA

DecOT use wasserstein_NMF_coefficient_step.m developed by Rolet(2016) to calculate Wasserstein loss. We rewrite the .m files into .py format, which is suitable for small batches of data. For data with thousands of genes, we recommend using DecOT with matlab interface. 

* Rolet, A., Cuturi, M., & Peyré, G. (2016, May). Fast dictionary learning with a smoothed Wasserstein loss. In Artificial Intelligence and Statistics (pp. 630-638). PMLR.


## Input file
The input files are tab-delimited .txt or .csv files, which contain:

* Y.csv : Bulk mixtures (genes * mixtures)
* C.csv : Reference cells (genes * cells)
* pDataC.csv : Cell phenotype data with three columns ('cellID', 'cellType', 'sampleID')

## Parameters
      def decot_ensemble(Y, C, pDataC, ground_cost_path, metric, save_path, cores=4, gamma=0.001, rho=0.001):
        Y: `pandas dataframe`
            Bulk mixtures (genes x mixtures).

        C: `pandas dataframe`
            Reference cells (genes x cells).

        pDataC: `pandas dataframe`
            Cell phenotype data (cells x ['cellID', 'cellType', 'sampleID'])

        ground_cost_path: str
            Ground cost matrix file path.

        metric: str
            Metric used to compute the ground cost (['dissTOM', 'euclidean', 'cosine', 'correlation']).

        save_path: str
            Path to save deconvolution results.

        cores: int
            Number of cores used for parallel computing. (for ensemble framework)

        gamma: regularization parameter
            Default: 0.001

        rho: regularization parameter
            Default: 0.001


## Example 

An example dataset (simulation/Batch1) is provided to run DecOT with default parameters (gamma: 0.001, rho: 0.001). To run DecOT, please follow commands below from the DecOT directory.

* python DecOT.py --working_dir ../results/simulation/Batch1/ --y sim0_Y.csv --c sim0_C.csv --pDataC sim0_pDataC.csv --metric dissTOM --gamma 0.001 --rho 0.001
* python DecOT.py --ensemble --working_dir ../results/simulation/Batch1/ --y sim0_Y.csv --c sim0_C.csv --pDataC sim0_pDataC.csv --metric dissTOM --cores 6 --gamma 0.001 --rho 0.001

We also provide a toy dataset which are mixtures of gauss distributions:

* python DecOT.py --random_gauss_test --genes 100 --types 5 --mixtures 10

You can download our source code and use DecOT in command line or install the DecOT package and directly call the functions in it:

        python setup.py install
