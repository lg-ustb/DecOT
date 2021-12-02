function [lambda, H, obj, gradient]=run_wasserstein_NMF_coefficient(Y,M,D,gamma,rho)
%% Set the parameters of wasserstein_DL

options.stop=1e-3;
options.verbose=2;
options.dual_descent_stop=5e-4;
options.alpha=0.5;
options.Kmultiplication='symmetric';
options.GPU=0;

K=exp(-M/gamma);
K(K<1e-200)=1e-200;

%% Perform Wasserstein NMF
[lambda, H, obj, gradient]=wasserstein_NMF_coefficient_step(Y,K,D,gamma,rho,zeros(size(Y)),options,1);