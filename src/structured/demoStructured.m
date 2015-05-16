%%
% here we choose the number of tasks as the number of Lables 
% and threat the binary nodes separately
clear; clc; close all;
rng(1110, 'twister');

%% Get data for chunking problem
NTRAIN              = 50; % Number of sequences
DATA_PATH           = '~/Documents/research/projects/structGP/gpstruct_vi/data/chunking';
FOLD_ID             = 1;

[ll_train, predictiveMarginalsN, Y_test_vector, nLabels, data_train, data_test] = ...
    getData(NTRAIN, DATA_PATH, FOLD_ID);


%% Parameters for memory allocation
Q       = nLabels; % for unary nodes. we treat the binary nodes separately
N       = data_train.TT;
D       = size(data_train.X,2);
nUnary  = N*nLabels;
nBinary = nLabels^2;
nTotal  = data_train.max;  % nUnary + nBinary;
%
m       = data_train; % model
m.N     = N;
m.Q     = Q;

% 

%% Parameter initialization
% variational parameters for the total number of latent variables
m.pars.M                      = zeros(nTotal,1);     % the mean parameters
m.pars.L                      = -2*(1./1e10)*ones(nTotal,1);     % the linear parametrisation of the cov matrix

%% Initialisation of posterior covariances of unary nodes
for j = 1 : Q
  % free initial values 
  m.pars.S{j} = eye(N);                  % the cov matrix
end

%% Initialise psoterior covariance of binary nodes (we assume it's diagonal so this wastes a bit of computation) 
 m.pars.S{Q+1} = eye(nBinary); 

%
%% covariance hyperparameters
m.pars.hyp.covfunc  = @covLINone;                   % cov function
m.pars.hyp.cov      = cell(Q,1);                       % cov hyperparameters
nhyper              = eval(feval(m.pars.hyp.covfunc));
matHyper            = log(ones(Q,nhyper));
m.pars.hyp.cov      = mat2cell(matHyper,ones(Q,1),nhyper);

%% Optimization settings
conf.nsamples                 = 2000;               % number of samples for gradient estimator
conf.covfunc                  = m.pars.hyp.covfunc; % covariance function
conf.maxiter                  = 100;                % how many optimization iterations to run?
conf.variter                  = 10;                 % maxiter for optim the variational hyperparameter (per iteration)
conf.hypiter                  = 5;                  % maxiter for optim the cov hyperparameter (per iteration)
conf.likiter                  = 5;                  % maxiter for optim the likelihood hyperparameter (per iteration)
conf.displayInterval          = 20;                 % intervals to display some progress 
conf.checkVarianceReduction   = false;              % show diagnostic for variance reduction?
conf.learnhyp                 = false;             
conf.latentnoise              = 0;                  % minimum noise level of the latent function

%% Model setting
m.likfunc                     = ll_train;         % likelihood function
m.pars.hyp.likfunc            = ll_train;         % I AM HERE      
m.pred                        = @predRegression;  % prediction 
 m.pars.hyp.lik =             [];                 % likelihood parameters

tic;
m = learnFullGaussianStructured(m,conf);
toc





















