function [fval, dM, dL] = computeELLStructuredNodes(m, Fs, s_rows, e_rows, conf)
%% compute ELL and gradients using the given samples
% use marginals instead of the sequence likelihoods
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
  logllh   = zeros(nTotal,nSamples); % log likelihood for all data-points
  sll      = zeros(nSamples,1);
  for s = 1 : nSamples
    [sll(s), logllh(:,s)] = feval(m.likfunc, Fs(:,s)); % the labels are included in the anonymous function
  end  
  % fval = mean(sll); %  sum of average likelihoods over empirical distribution (samples)
  % USE ABOVE DELETE BELOW
  fval =mean(sum(logllh,1));
  
  %% gradients are required
  if nargout > 1
    %F0 = Fs - repmat(m.pars.M, 1, nSamples);
    %Sinv = repmat(sinv, 1, nSamples);
    %dM = F0.*Sinv;
    %dL = 0.5*(dM.^2 - Sinv);    
    F0 = bsxfun(@minus, Fs, m.pars.M);
    dM  = bsxfun(@times, F0, sinv);
    dL  = 0.5*bsxfun(@minus, dM.^2, sinv);
    
    % gradients wrt lambfda parameter --> did't help bring this here
%     for j = 1 : Q
%         ptr = s_rows(j) : e_rows(j);
%         dL(ptr,:) = 2*(m.pars.S{j}.^2)*dL(ptr,:);
%     end
     
    logllh = repmat(logllh, 2, 1); % size (2*nTotal)xS
    dML = [dM; dL];
            
    %% the  gradients are weighted by their lokelihood
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

