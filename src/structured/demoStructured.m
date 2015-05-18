%%
% here we choose the number of tasks as the number of Lables 
% and threat the binary nodes separately
clear; clc; close all;
rng(1110, 'twister');


%% Get data for chunking problem
NTRAIN              = 2; % Number of sequences
DATA_PATH           = '~/Documents/research/projects/structGP/gpstruct_vi/data/chunking';
FOLD_ID             = 1;

[data_train, data_test, ll_train,  Y_test_vector] = ...
    getDataSmall(NTRAIN, DATA_PATH, FOLD_ID);


%% Parameters for memory allocation
m       = data_train; % model
clear data_train;
Q       = m.nLabels; % for unary nodes. we treat the binary nodes separately
N       = m.TT;
D       = size(m.X,2);
nUnary  = N*Q;
nBinary = Q^2;
nTotal  = m.max;  % nUnary + nBinary;
%
m.N     = N;
m.Q     = Q;

% 

%% Parameter initialization
% variational parameters for the total number of latent variables
m.pars.M                      = zeros(nTotal,1);     % the mean parameters
m.pars.L                      = -2*(1./1e10)*ones(nTotal,1);     % the linear parametrisation of the cov matrix

%% covariance hyperparameters
m.pars.hyp.covfunc  = @covLINone;                   % cov function
m.pars.hyp.cov      = cell(Q,1);                       % cov hyperparameters
nhyper              = eval(feval(m.pars.hyp.covfunc));
matHyper            = log(ones(Q,nhyper));
m.pars.hyp.cov      = mat2cell(matHyper,ones(Q,1),nhyper);

%% Optimization settings
conf.nsamples                 = 2000;               % number of samples for gradient estimator
conf.ntestsamples             = 10000; 
conf.covfunc                  = m.pars.hyp.covfunc; % covariance function
conf.maxiter                  = 100;                % how many optimization iterations to run?
conf.variter                  = 10;                 % maxiter for optim the variational hyperparameter (per iteration)
conf.hypiter                  = 5;                  % maxiter for optim the cov hyperparameter (per iteration)
conf.likiter                  = 5;                  % maxiter for optim the likelihood hyperparameter (per iteration)
conf.displayInterval          = 20;                 % intervals to display some progress 
conf.checkVarianceReduction   = false;              % show diagnostic for variance reduction?
conf.learnhyp                 = true;             
conf.latentnoise              = 1e-3;                  % minimum noise level of the latent function

%% Model setting
m.likfunc                     = ll_train;         % likelihood function
m.pars.hyp.likfunc            = ll_train;         % I AM HERE      
m.pred                        = @predStructured;  % prediction 
 m.pars.hyp.lik =             [];                 % likelihood parameters

 
m               = learnFullGaussianStructured(m,conf);
marginals       = feval(m.pred, m, conf, data_test);
[avgError, nlp] =  computeErrorStructured(marginals, Y_test_vector);

str = datestr(now);
save(['final-',str,'.mat']);

















