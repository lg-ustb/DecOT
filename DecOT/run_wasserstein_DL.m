function [D, lambda, objectives]=run_wasserstein_DL(data,M)
%% Set the parameters of wasserstein_DL

options.stop=1e-2;
options.verbose=2;
options.D_step_stop=5e-4;
options.lambda_step_stop=5e-4;
options.alpha=0.5;
options.Kmultiplication='symmetric';
options.GPU=0;
k=3;

gamma=0.01;

wassersteinOrder=1;

%% Perform Wasserstein NMF
rho1=.01;
rho2=.01;
[D, lambda, objectives]=wasserstein_DL(data,k,M.^wassersteinOrder,gamma,rho1,rho2,options);

