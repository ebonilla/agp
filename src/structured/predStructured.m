function P = predStructured(m, conf, data_test)

[fmu,fvar] = predRegStructured(m, conf, data_test.X);

%% Need to sample for q(f*), then compute prediction through averaging
nTestSamples = conf.ntestsamples;
Z = randn(data_test.max,nTestSamples);
%fSample      = fmu + sqrt(fvar).*Z;
fSample       = bsxfun(@plus, fmu, bsxfun(@times, sqrt(fvar), Z));
nChunks       =  sum(cellfun('length', data_test.Y));
P            = zeros(nChunks, data_test.nLabels, nTestSamples);
for j  = 1 : nTestSamples
     P(:,:,j) = predictiveMarginalsN(fSample,data_test);
end
%
% prob = squeeze(mean(P,3));

end


%% pred regression for structured case
function [fmu,fvar] = predRegStructured(m,conf,xstar)
%PREDREGRESSION  [fmu,fvar,yvar] = predRegression(m,conf,xstar)
%   Prediction by a regression model with single latent function.
Q = m.Q; N = m.N;
s_rows = (0:(Q-1))'*N + 1;
e_rows = (1:Q)'*N;
nBinary = Q^2;
s_rows = [s_rows; e_rows(end)+1];         % Adding entries for binary nodes
e_rows = [e_rows; e_rows(end)+nBinary];
if ~isfield(conf,'latentnoise')
  sn2 = 0;
else
  sn2 = conf.latentnoise;
end
ntest = size(xstar,1);
Fmu  = zeros(ntest, Q);
Fvar = zeros(ntest, Q);
for j = 1 : Q
  M = m.pars.M(s_rows(j):e_rows(j),:);
  if isfield(m.pars,'S')
    % the full gaussian with efficient parametrization case, S is given
    S = m.pars.S{j};
  else
    % the mixture case (L is diagonal) and blackbox case (L is block)
    S = m.pars.L(s_rows(j):e_rows(j),:)*m.pars.L(s_rows(j):e_rows(j),:)';
  end
  covhyp = m.pars.hyp.cov{j};
  likhyp = m.pars.hyp.lik;
  [Fmu(:,j),Fvar(:,j)] = predOne(M,S,m.X,xstar,conf.covfunc,covhyp,likhyp,sn2);
end
valMean = m.pars.M(s_rows(Q+1):e_rows(Q+1),:);
valVar   = diag(m.pars.S{Q+1});
fmuUnary = Fmu(:);
fvarUnary = Fvar(:);
fmuBinary  = valMean;
fvarBinary = valVar;

fmu  = [fmuUnary;fmuBinary];
fvar = [fvarUnary; fvarBinary]; 

end

%% prediction for one latent function
function [fmu,fvar,yvar] = predOne(M,S,x,xstar,covfunc,covhyp,likhyp,sn2)
Kff = feval(covfunc, covhyp, x) + sn2*eye(size(x,1));
Lff = jit_chol(Kff,3);
invKff = invChol(Lff);

Kss = feval(covfunc, covhyp, xstar, 'diag') + sn2;
Kfs = feval(covfunc, covhyp, x, xstar);
fmu =  Kfs'*(invKff*M);

% we can also compute full covariance at a higher cost
% diag(Ksm * kmminv * S * Kmmonv *Kms) 
var_1 =  sum(Kfs.*(invKff*S*invKff*Kfs),1)';
var_2 =  sum(Kfs.*(invKff*Kfs),1)';
fvar = var_1 + Kss - var_2;
fvar = max(fvar,1e-10); % remove numerical noise i.e. negative variance
if nargout == 3
  yvar = fvar + exp(2*likhyp(end));
end
end

