function m = learnFullGaussianStructured(m, conf, data_train)
%LEARNFULLGAUSSIAN m = learnFullGaussian(m,conf)
%   
% Automated inference with (factorized) full Gaussian posterior.
% Fast implementation.
%
% This code uses LBFGS or CG to optimize the variational parameters instead
% of stochastic optimization.
%
% 30/09/14
% N here is total number of feature vectors
% 
% m.exitFlat: (0) Max number of global iterations reached 
%             (1) tolerance on parameter space reached
%             (2) Tolerance on objective function reached 

fHyper   =  @(hyp, m, sn2) elboCovhypStructured(hyp, m, sn2, data_train.X);
Q   = m.Q;
Nx  = m.Nx;
conf.cvsamples = 200; % for control variates
if ~isfield(conf,'learnhyp') % compatability with previous version
  conf.learnhyp = true;
end
if ~isfield(conf,'latentnoise')
  sn2 = 0;
else
  sn2 = conf.latentnoise;
end
K           = cell(Q+1,1); 
LKchol      = cell(Q+1,1);
nBinary     = m.nBinary;
K{Q+1}      = m.varPriorBinary*eye(nBinary); % This is inefficient but want to hard code less things
LKchol{Q+1} = sqrt(K{Q+1}); 
s_rows      = (0:(Q-1))'*Nx + 1;
e_rows      = (1:Q)'*Nx;
s_rows      = [s_rows; e_rows(end)+1];         % Adding entries for binary nodes
e_rows      = [e_rows; e_rows(end) + nBinary];
%
for j = 1 : Q
    K{j}      = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, data_train.X) + sn2*eye(Nx);
    LKchol{j} = jit_chol(K{j});
end

if ( ~isfield(conf, 'fvalTol') )
    conf.fvalTol = 1e-3;
end

 
%% Main loop
if ( isfield(m,'fval') &&  numel(m.fval) > 0 )
    fvalOld  = m.fval(end);
else
    fvalOld  = Inf;
end
iter     = 0;
while ( iter <= conf.maxiter )
  %% E-step : optimize variational parameters
  theta = [m.pars.M; m.pars.L];
  
  % Check derivatives
  % theta = rand(size(theta));
  %[diff_deriv, gfunc, gnum] = derivativeCheck(@elboVarStructured, theta, 1, 1, m, conf, K, LKchol, s_rows, e_rows, true);
  
  [theta,fX,~] = minimize(theta, @elboVarStructured, conf.variter, m, conf, K, LKchol, s_rows, e_rows, true);
  fvalNew     = fX(end);
  m.fval        = [m.fval; fX(end)];
  
  delta_m = mean(abs(m.pars.M(:)-theta(1:numel(m.pars.M))));
  delta_l = mean(abs(m.pars.L(:)-theta(numel(m.pars.M)+1:end)));
  fprintf('variational change m= %.4f\n', delta_m);
  fprintf('variational change s= %.4f\n', delta_l);
  
  m.pars.M = theta(1:numel(m.pars.M));
  m.pars.L = theta(numel(m.pars.M)+1:end);
  
  %% update S for binary node functions
  j            = Q + 1;
  sBin          = exp(m.pars.L(s_rows(j):e_rows(j)));
  m.pars.S{j}  =  diag(sBin);
  
  %% Update S for unary node functions
  for j = 1 : Q
    % S = (K^{-1} - 2*diag(lambda))^{-1}
    lambda = m.pars.L(s_rows(j):e_rows(j));
    m.pars.S{j} = K{j} - K{j}*((-diag(1./(2*lambda))+K{j})\K{j});
  end

  %% Gradient-based optimization for covariance hyperparameters
  if conf.learnhyp
   % [diff_deriv, gfunc, gnum] = derivativeCheck(fHyper, cell2mat(m.pars.hyp.cov), 1, 1, m, sn2);
  
    hyp0 = minimize(m.pars.hyp.cov, fHyper, conf.hypiter, m, sn2);
    m.pars.hyp.cov = hyp0;
    for j = 1 : Q   % updating covariances with new hyperparameters
      K{j} = feval(m.pars.hyp.covfunc, hyp0{j}, data_train.X) + sn2*eye(Nx);
      LKchol{j} = jit_chol(K{j});
    end
    fhyp0 = elboVarStructured(theta, m, conf, K, LKchol, s_rows, e_rows, false);
    m.fval = [m.fval; fhyp0];
  end
  
  %% Update likelihood parameters
  if numel(m.pars.hyp.lik) > 0
    fs = zeros(m.Q*m.Nx, conf.nsamples);
    for j = 1 : m.Q
      fs(s_rows(j):e_rows(j),:) = mvnrnd(m.pars.M(s_rows(j):e_rows(j),:)', ...
        diag(m.pars.S{j})', conf.nsamples)';
    end
    lik0 = minimize(m.pars.hyp.lik(end),@elbolik,conf.likiter,m,fs);
    m.pars.hyp.lik(end) = lik0;
    m.fval = [m.fval; elboVarStructured(theta,m,conf,K,LKchol,s_rows,e_rows,false)];
    disp('new lik hyp')
    disp(exp(2*m.pars.hyp.lik(end)))
  end
  if ( mod(iter,5)==0 )
      [marginals, avgError, nlp, maxMargPost]       = feval(m.pred, m);
      save(['model','.mat'], 'm'); 
      fprintf('**** Error rate = %.2f **** \n', avgError);
  end

%  if ( (delta_m + delta_l)/2 < 1e-3 ) 
%      m.exitFlag = 1; 
%    return;
%  end

  if (iter > 1 && ( (fvalOld - fvalNew) < conf.fvalTol) )
    m.exitFlag = 2;      
    return;
  end
  
  fvalOld = fvalNew;
  iter = iter + 1;
  fprintf('Iteration %d done \n', iter);
end

m.exitFlag = 0; % exit due to  maxiter
      

end






















