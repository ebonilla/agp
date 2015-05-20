function [fval, dM, dL] = computeELLStructured(m, Fs, s_rows, e_rows, conf)
%% compute ELL and gradients using the given samples

  % m : model
  % Fs : the samples f ~ q(f | lambda_k)
  % conf.cvsamples : number of samples used for estimating the optimal control
  % variate factor
  Q        = m.Q;
  nSamples = size(Fs,2);
  nTotal   = m.nTotal; % Total number of latent variables
  nSeq     = m.nSeq; % Number of sequences
  
  %% pre-computation of the inverses
  sinv = zeros(nTotal,1);
  for j = 1 : Q + 1
    s_row = s_rows(j);
    e_row = e_rows(j);
    sinv(s_row:e_row) = 1./diag(m.pars.S{j});
  end
  
  % EVB: Replaced this with structured likelihood
  %logllh = fastLikelihood(m.likfunc,m.Y,Fs,m.pars.hyp,N,Q);
  % 
  vec_ll   = zeros(nSeq,nSamples); % log likelihood for all data-points
  for s = 1 : nSamples
    vec_ll(:,s) = feval(m.likfunc, Fs(:,s)); % the labels are included in the anonymous function
  end  
  fsum_n = sum(vec_ll,1); % sum over n
  fval = mean(fsum_n); %  sum of average likelihoods over empirical distribution (samples)
  
  %% gradients are required
  if nargout > 1
    f0 = Fs(:)-repmat(m.pars.M,nSamples,1);
    dM = f0.*repmat(sinv,nSamples,1);
    dL = 0.5*(dM.^2 - repmat(sinv,nSamples,1));
    
    logllh = zeros(nTotal, nSamples);
    %% assign likelihood to respective unary nodes
    for i = 1 : nSeq
        idx = m.idxUnary{i}(:);
        logllh(idx, :) = bsxfun(@plus, logllh(idx, :), vec_ll(i,:));
    end
    clear vec_ll;
    
    %% assing likelihood to binary nodes
    idx            = m.idxBinary(:);
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

