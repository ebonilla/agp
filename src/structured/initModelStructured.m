function m = initModelStructured(data_train)

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
m.varPriorBinary = 1e-4; % prior variance of binary functions
% 

%% Variational Paramemter  initialization (for all latent variable)% variational parameters for the total number of latent variables
m.pars.M                      = ones(nTotal,1);     % the mean parameters
m.pars.L                      = -2*(1./1e2)*ones(nTotal,1);     % the linear parametrisation of the cov matrix
% for Q+1 we have the actual variances (in log scale)
varPosterior                      =  1 ; %   m.varPriorBinary;
m.pars.L(nTotal-nBinary+1:nTotal) = log(varPosterior)*ones(nBinary,1);
 
%% covariance hyperparameters
m.pars.hyp.covfunc  = @covLINone;                   % cov function
m.pars.hyp.cov      = cell(Q,1);                       % cov hyperparameters
nhyper              = eval(feval(m.pars.hyp.covfunc));
matHyper            = log(ones(Q,nhyper));
m.pars.hyp.cov      = mat2cell(matHyper,ones(Q,1),nhyper);

end

