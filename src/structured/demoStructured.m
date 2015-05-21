function demoStructured(fold_id)
% here we choose the number of tasks as the number of Lables 
% and treat the binary nodes separately
%clear; clc; close all;
rng(1110, 'twister');
DATA_PATH           = '~/Documents/research/projects/structGP/gpstruct_vi/data/chunking';


%% Get data for chunking problem
NTRAIN              = 50; % Number of sequences
[data_train, data_test, ll_train,  Y_test_vector] = ...
    getData(NTRAIN, DATA_PATH, fold_id);

%NTRAIN              = 3; % Number of sequences
%[data_train, data_test, ll_train,  Y_test_vector] = ...
%    getDataSmall(NTRAIN, DATA_PATH, fold_id);


%% Parameters for memory allocation
Q       = data_train.nLabels; % for unary nodes. we treat the binary nodes separately
Nx      = data_train.TT;      % total number of feature fectors
D       = size(data_train.X,2);
nUnary  = Nx*Q;
nBinary = Q^2;
nTotal  = data_train.max;  % Total number of nodes
%
m.Nx        = Nx;
m.Q         = Q;
m.D         = D;
m.nUnary    = nUnary;
m.nBinary   = nBinary;
m.nSeq      = data_train.N; % Number of sequences
m.nTotal    = nTotal;
m.idxUnary  = data_train.unary;
m.idxBinary = data_train.binary; 
% 

%% Parameter initialization
% variational parameters for the total number of latent variables
m.pars.M                      = zeros(nTotal,1);     % the mean parameters
m.pars.L                      = -2*(1./1e10)*ones(nTotal,1);     % the linear parametrisation of the cov matrix

% for Q+1 we have the actual variances
m.pars.L(nTotal-nBinary+1:nTotal) = ones(nBinary,1);

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
conf.variter                  = 50;                 % maxiter for optim the variational hyperparameter (per iteration)
conf.hypiter                  = 10;                  % maxiter for optim the cov hyperparameter (per iteration)
conf.likiter                  = 5;                  % maxiter for optim the likelihood hyperparameter (per iteration)
conf.displayInterval          = 20;                 % intervals to display some progress 
conf.checkVarianceReduction   = false;              % show diagnostic for variance reduction?
conf.learnhyp                 = true;             
conf.latentnoise              = 1e-4;                  % minimum noise level of the latent function

%% Model setting
m.likfunc                     = ll_train;         % likelihood function
m.pars.hyp.likfunc            = ll_train;         % I AM HERE      
m.pred                        = @predStructured;  % prediction 
 m.pars.hyp.lik =             [];                 % likelihood parameters

m               = learnFullGaussianStructured(m, conf, data_train);
marginals       = feval(m.pred, m, conf, data_train.X, data_test);
[avgError, nlp, maxMargPost] =  computeErrorStructured(marginals, Y_test_vector);

str = datestr(now);
save(['final-fold-',num2str(fold_id),'-',str,'.mat']);

















