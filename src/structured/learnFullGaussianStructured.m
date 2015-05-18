function m = learnFullGaussianStructured(m,conf)
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
Q = m.Q;
N = m.N;
conf.cvsamples = 200; % for control variates
if ~isfield(conf,'learnhyp') % compatability with previous version
  conf.learnhyp = true;
end
if ~isfield(conf,'latentnoise')
  sn2 = 0;
else
  sn2 = conf.latentnoise;
end
fval = [];
K = cell(Q+1,1); 
LKchol = cell(Q+1,1);
nBinary = m.nLabels^2;
K{Q+1}      = eye(nBinary); % This is inefficient but want to hard code less things
LKchol{Q+1} = eye(nBinary); 
s_rows = (0:(Q-1))'*N + 1;
e_rows = (1:Q)'*N;
s_rows = [s_rows; e_rows(end)+1];         % Adding entries for binary nodes
e_rows = [e_rows; e_rows(end) + nBinary];

%% Main loop
for j = 1 : Q
    K{j}    = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.X) + sn2*eye(N);
    LKchol{j} = jit_chol(K{j});
end
iter = 0;
while true
  %% E-step : optimize variational parameters
  theta = [m.pars.M; m.pars.L];
  
  % Check derivatives
  % theta = rand(size(theta));
  %[diff_deriv, gfunc, gnum] = derivativeCheck(@elbo, theta, 1, 1, m, conf, K, LKchol, s_rows, e_rows, true);
  
  [theta,fX,~] = minimize(theta, @elbo, conf.variter, m, conf, K, LKchol, s_rows, e_rows, true);
  
  delta_m = mean(abs(m.pars.M(:)-theta(1:numel(m.pars.M))));
  delta_l = mean(abs(m.pars.L(:)-theta(numel(m.pars.M)+1:end)));
  fprintf('variational change m= %.4f\n', delta_m);
  fprintf('variational change s= %.4f\n', delta_l);
  m.pars.M = theta(1:numel(m.pars.M));
  m.pars.L = theta(numel(m.pars.M)+1:end);
  % update S using new lambda
  for j = 1 : Q + 1
    % S = (K^{-1} - 2*diag(lambda))^{-1}
    lambda = m.pars.L(s_rows(j):e_rows(j));
    m.pars.S{j} = K{j} - K{j}*((-diag(1./(2*lambda))+K{j})\K{j});
  end
  fval = [fval; fX(end)];

  %% Gradient-based optimization for covariance hyperparameters
  if conf.learnhyp
    hyp0 = minimize(m.pars.hyp.cov, @elboCovhyp, conf.hypiter, m, m.pars.S, sn2);
    m.pars.hyp.cov = hyp0;
    for j = 1 : Q
      K{j} = feval(m.pars.hyp.covfunc, hyp0{j}, m.X) + sn2*eye(N);
      LKchol{j} = jit_chol(K{j});
    end
    fhyp0 = elbo(theta,m,conf,K,LKchol,s_rows,e_rows,false);
    fval = [fval; fhyp0];
  end
  if (delta_m + delta_l)/2 < 1e-3 || (iter > 1 && fval(end-1) - fval(end) < 1e-5)
    break;
  end
  
  %% Update likelihood parameters
  if numel(m.pars.hyp.lik) > 0
    fs = zeros(m.Q*m.N, conf.nsamples);
    for j=1:m.Q
      fs(s_rows(j):e_rows(j),:) = mvnrnd(m.pars.M(s_rows(j):e_rows(j),:)', ...
        diag(m.pars.S{j})', conf.nsamples)';
    end
    lik0 = minimize(m.pars.hyp.lik(end),@elbolik,conf.likiter,m,fs);
    m.pars.hyp.lik(end) = lik0;
    fval = [fval; elbo(theta,m,conf,K,LKchol,s_rows,e_rows,false)];
    disp('new lik hyp')
    disp(exp(2*m.pars.hyp.lik(end)))
  end
  if ( mod(iter,10)==0 )
      str = datestr(now);
      save(['model-',str,'.mat'], 'm');
  end
  % Commented out by EVB
  %predictionFullHelper(iter,conf,m);
  if iter > conf.maxiter %|| delta < 1e-2
    break
  end
  
  iter = iter + 1;
  fprintf('Iteration %d done \n', iter);
end

m.fval = fval;
%figure; hold off;
%fval = -fval;
%plot(1:numel(fval),fval,'-');
%title('objective values');
%m.fval = fval;
end

%% the negative elbo and its gradient wrt variational parameters
function [fval,grad] = elbo(theta, m, conf, K, LKchol, s_rows, e_rows, updateS)
  rng(10101,'twister');
  Q         = m.Q; 
  vec_N     = [m.N*ones(Q,1); size(K{Q+1},1)]; % vector of matrix sizes
  m.pars.M  = theta(1:numel(m.pars.M));
  m.pars.L  = theta(numel(m.pars.M)+1:end);
  dM        = zeros(size(m.pars.M));
  dL        = zeros(size(m.pars.L));
  
  %% KL term [entropy + neg cross entropy part]
  for j = 1 : Q + 1
    % new value of L leads to new value for S
    if updateS
        lambda = m.pars.L(s_rows(j):e_rows(j));
        m.pars.S{j} = K{j} - K{j}*((-diag(1./(2*lambda))+K{j})\K{j});
    end  
    LSchol = jit_chol(m.pars.S{j})';
    Kinvm = solve_chol(LKchol{j},m.pars.M(s_rows(j):e_rows(j))); % K^{-1}m
    KinvLj = solve_chol(LKchol{j},LSchol);  % K^{-1} Lj
    fvalEnt = sum(log(diag(LSchol))); % entropy
    fvalNCE = 2*sum(log(diag(LKchol{j}))) + m.pars.M(s_rows(j):e_rows(j))'*Kinvm...
      + trAB(KinvLj,LSchol');
    %
    % gradient 
    if nargout > 1
      dM(s_rows(j):e_rows(j)) = -Kinvm;
      A = inv(eye(vec_N(j))-2*AdiagB(K{j},diag(m.pars.L(s_rows(j):e_rows(j)))));
      dL(s_rows(j):e_rows(j)) = diag(m.pars.S{j}) - diagProd(m.pars.S{j},A');
    end
  end

  %% sample from the marginal posteriors
  nTotal = numel(m.pars.M);
  Fs = zeros(nTotal,  conf.nsamples);
%  z  = randn(nTotal, conf.nsamples);
  for j =1 : Q + 1
     Fs(s_rows(j):e_rows(j),:) = mvnrnd(m.pars.M(s_rows(j):e_rows(j),:)', diag(m.pars.S{j})', conf.nsamples)';
%    ptr = s_rows(j):e_rows(j);    mu_fs  = m.pars.M(ptr,:);    dev_fs = sqrt(diag(m.pars.S{j}));   Fs(ptr,:) = bsxfun(@plus, mu_fs,  bsxfun(@times,z(ptr,:), dev_fs));
  end
  
  %% ELL and its gradients
  if nargout == 1
    ell = computeNoisyGradient(m, Fs, s_rows, e_rows, conf);
  else
    [ell, dell_dm, dell_dl] = computeNoisyGradient(m, Fs, s_rows, e_rows, conf);
    % grad_{lambda} E_q log p(y|f) = 2(S.*S) grad_{diag(S)} E_q logp(y|f)
    for j = 1 : Q + 1
      dell_dl(s_rows(j):e_rows(j)) = 2*(m.pars.S{j}.^2)*dell_dl(s_rows(j):e_rows(j));
    end
    dM = dM + dell_dm;
    dL = dL + dell_dl;
    grad = -[dM; dL];
  end
  fval = -(fvalEnt - 0.5*fvalNCE + ell);
end 

%% compute ELL and gradients using the given samples
function [fval, dM, dL] = computeNoisyGradient(m, fs, s_rows, e_rows, conf)
  % m : model
  % fs : the samples f ~ q(f | lambda_k)
  % conf.cvsamples : number of samples used for estimating the optimal control
  % variate factor
  Q        = m.Q;
  nSamples = size(fs,2);
  nTotal   = m.max; % Total number of latent variables
  nSeq     = size(m.Y,1); % Number of sequences
  
  %% pre-computation of the inverses
  sinv = zeros(nTotal,1);
  for j = 1 : Q + 1
    s_row = s_rows(j);
    e_row = e_rows(j);
    sinv(s_row:e_row) = 1./diag(m.pars.S{j});
  end
  
  % EVB: Replaced this with structured likelihood
  %logllh = fastLikelihood(m.likfunc,m.Y,fs,m.pars.hyp,N,Q);
  % 
  vec_ll   = zeros(nSeq,nSamples); % log likelihood for all data-points
  for s = 1 : nSamples
    vec_ll(:,s) = feval(m.likfunc, fs(:,s)); % the labels are included in the anonymous function
  end  
  fsum_n = sum(vec_ll,1); % sum over n
  fval = sum(fsum_n)/nSamples; %  sum of average likelihoods over empirical distribution (samples)
  
  %% gradients are required
  if nargout > 1
    f0 = fs(:)-repmat(m.pars.M,nSamples,1);
    dM = f0.*repmat(sinv,nSamples,1);
    dL = 0.5*(dM.^2 - repmat(sinv,nSamples,1));
    
    logllh = zeros(nTotal, nSamples);
    %% assign likelihood to respective unary nodes
    for i = 1 : nSeq
        idx = m.unary{i}(:);
        logllh(idx, :) = bsxfun(@plus, logllh(idx, :), vec_ll(i,:));
    end
    clear vec_ll;
    
    %% assing likelihood to binary nodes
    idx            = m.binary(:);
    logllh(idx,:)  =  bsxfun(@plus, logllh(idx, :), fsum_n);  % weight of grads for binary nodes is the sum of the ELL

    %% some useful constructs
    %logllh = reshape(logllh,N,nSamples);
    %logllh = repmat(logllh,2*Q,1); % size (2*N*Q)xS
    %dML = [reshape(dM,N*Q,nSamples); reshape(dL,N*Q,nSamples)]; % size (2*N*Q)xS
    %
    %
    logllh = repmat(logllh, 2, 1); % size (2*nTotal)xS
    dML = [reshape(dM, nTotal, nSamples); reshape(dL,nTotal,nSamples)];
            
    %% the noisy gradient using the control variates
    grads = logllh.*dML; 
    
    %% adding control variates
    pz = dML(:,1 : conf.cvsamples)';
    py = logllh(:,1:conf.cvsamples)'.*pz;
    above = sum((py - repmat(mean(py),conf.cvsamples,1)).*pz)/(conf.cvsamples-1);
    below = sum(pz.^2)/(conf.cvsamples-1); % var(z) with E(z) = 0
    cvopt = above ./ below;
    cvopt(isnan(cvopt)) = 0;
    ah    =  repmat(cvopt',1,nSamples).*dML;
    grads = grads - ah; 
    
    grad = mean(grads,2);
    
    % NOTE: this is not the true variance reduction as this
    % gradient is wrt diag(S), not Lambda
    if conf.checkVarianceReduction
      ugrads = logllh.*dML;
      ugrad = mean(ugrads,2);
      vargrad = var(grads,0,2);
      varugrad = var(ugrads,0,2);
      disp('diff in estimated grad')
      disp(mean(abs(grad-ugrad)))
      reduction = 100*(varugrad - vargrad)./varugrad; 
      disp('min, max, mean percentage of variance reduction:')
      disp([min(reduction), max(reduction), mean(reduction)])
      disp('min, max, mean controlled variance:')
      disp([min(vargrad), max(vargrad), mean(vargrad)])
    end

    dM = grad(1:nTotal);
    dL = grad(nTotal+1:end);
  end
end


% wrapper for negative cross entropy function and its derivatives wrt
% hyperparameters so that it can be used with minimize function
function [fval,dhyp] = elboCovhyp(hyp, m, S, sn2)
%TODO: pass Lj to this function
if iscell(hyp)
  m.pars.hyp.cov = hyp;
else
  m.pars.hyp.cov = rewrap(m.pars.hyp.cov,hyp);
end
N = m.N; Q = m.Q; fval = 0;
dhyp = cell(Q,1);
for j=1:Q
  s_row = (j-1)*N+1;
  e_row = j*N;
  K = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.X) + sn2*eye(N);
  LKchol = jit_chol(K);
  Lj = jit_chol(S{j})';
  Kinvm = solve_chol(LKchol, m.pars.M(s_row:e_row)); % K^{-1}m
  KinvLj = solve_chol(LKchol, Lj); % K^{-1} Lj
  fval = fval + 2*sum(log(diag(LKchol))) + m.pars.M(s_row:e_row)'*Kinvm ...
    + trAB(KinvLj,Lj');
  dhyp{j} = zeros(size(m.pars.hyp.cov{j}));
  Kjinv = invChol(LKchol);
  for i=1:numel(m.pars.hyp.cov{j})
    dK = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.X, [], i);
    dhyp{j}(i) = 0.5*trAB(Kinvm*Kinvm' - Kjinv + KinvLj*KinvLj', dK);
  end
  % use numerically more stable gradient for covSEard
  if strcmp(func2str(m.pars.hyp.covfunc),'covSEard') && sn2 == 0
    dhyp{j}(i) = m.pars.M(s_row:e_row)'*Kinvm - N + trAB(KinvLj,Lj');
  end
end
fval = 0.5*fval; % for minimization
if iscell(hyp)
  dhyp = rewrap(m.pars.hyp.cov,-unwrap(dhyp));
else
  dhyp = -unwrap(dhyp);
end
end

function [fval,dlikhyp] = elbolik(hyp,m,fs)
m.pars.hyp.lik(end) = hyp;
nSamples = size(fs,2);
[logllh,dlikhyp] = fastLikelihood(m.likfunc,m.Y,fs,m.pars.hyp,m.N,m.Q);
dlikhyp = -dlikhyp(end)/nSamples;
fval = -sum(logllh)/nSamples;
end


