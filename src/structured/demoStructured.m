function demoStructured(fold_id, previousRunFile, strScale)
% here we choose the number of tasks as the number of Lables 
% and treat the binary nodes separately
%clear; clc; close all;
rng(1110, 'twister');
if ( numel(previousRunFile) > 0 )
    runFromFile(previousRunFile);
else
    runSingle(fold_id, strScale);
end

end

%% runSingle
function runSingle(fold_id, strScale)
DATA_PATH           = '~/Documents/research/projects/structGP/gpstruct_vi/data/chunking';

switch strScale 
    case 'small'
        NTRAIN              = 2; % Number of sequences
        [data_train, data_test, ll_train,  Y_test_vector] = ...
            getDataSmall(NTRAIN, DATA_PATH, fold_id);
    otherwise
        %% Get data for chunking problem
        NTRAIN              = 50; % Number of sequences
        [data_train, data_test, ll_train,  Y_test_vector] = ...
        getData(NTRAIN, DATA_PATH, fold_id);
end


m = initModelStructured(data_train);

%% Optimization settings
conf.nsamples                 = 2000;               % number of samples for gradient estimator
conf.ntestsamples             = 10000; 
conf.covfunc                  = m.pars.hyp.covfunc; % covariance function
conf.maxiter                  = 50;                % how many optimization iterations to run?
conf.variter                  = 20;                 % maxiter for optim the variational hyperparameter (per iteration)
conf.hypiter                  = 10;                  % maxiter for optim the cov hyperparameter (per iteration)
conf.likiter                  = 5;                  % maxiter for optim the likelihood hyperparameter (per iteration)
conf.displayInterval          = 20;                 % intervals to display some progress 
conf.checkVarianceReduction   = false;              % show diagnostic for variance reduction?
conf.learnhyp                 = true;             
conf.latentnoise              = 1e-4;              % minimum noise level of the latent function
conf.fvalTol                  = 1e-3; 


%% Model setting
m.likfunc                     = ll_train;         % likelihood function
m.pars.hyp.likfunc            = ll_train;         % I AM HERE      
m.pred                        = @(m_) predStructured(m_,conf,data_train.X, data_test);  % prediction 
m.pars.hyp.lik                =  [];                 % likelihood parameters
m.fval                        = [];

m            = learnFullGaussianStructured(m, conf, data_train);
[marginals, avgError, nlp, maxMargPost]       = feval(m.pred, m);

str = datestr(now);
save(['final-fold-',num2str(fold_id),'-',str,'.mat']);

end


%% runFromFile
function runFromFile(fname)
load(fname);

m               = learnFullGaussianStructured(m, conf, data_train);
marginals       = feval(m.pred, m);
[avgError, nlp, maxMargPost] =  computeErrorStructured(marginals, Y_test_vector);

str = datestr(now);
save(['final-fold-',num2str(fold_id),'-',str,'.mat']);


end
















