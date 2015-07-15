%% the negative elbo and its gradient wrt variational parameters
function [fval,grad] = elboVarStructuredCoupled(theta, m, conf, K, LKchol, s_rows, e_rows, updateS)
% samples are drawn now from coupled Gaussians

rng(10101,'twister');
Q         = m.Q; 
vec_N     = [m.Nx*ones(Q,1); size(K{Q+1},1)]; % vector of matrix sizes
m.pars.M  = theta(1:numel(m.pars.M));
m.pars.L  = theta(numel(m.pars.M)+1:end);
dM        = zeros(size(m.pars.M));
dL        = zeros(size(m.pars.L));
  
%% KL term [entropy + neg cross entropy part] for Q + 1
j = Q + 1;
mu           =  m.pars.M(s_rows(j):e_rows(j));
sBin            =  exp(m.pars.L(s_rows(j):e_rows(j)));
dimS         = length(sBin); % dimensionality of the Gaussian
sinv         = 1./sBin;
m.pars.S{j}  =  diag(sBin);
k            =  diag(K{j});
kinv         = 1./k;
fvalEnt      =  0.5*sum(log(sBin));       % (1/2) log det (S) : S is a diagonal matrix
fvalNCE      =  -0.5*sum(log(k)) ...   % -0.5 log det K:  K{j} is diagonal here
                -0.5*(mu.^2)'*kinv ... %  -0.5  m' K^{-1} m
                -0.5*kinv'*sBin  ...      % -0.5 trace (K^-1 S)
                +0.5*dimS;
ptr         =  s_rows(j):e_rows(j);
if (nargout > 1) % gradient
    dM(ptr)     = -kinv.*mu;
    dL(ptr)     = 0.5*(sinv - kinv);
end

%% KL term [entropy + neg cross entropy part] for all j <= Q 
for j = 1 : Q 
    % new value of L leads to new value for S
    if updateS
        lambda = m.pars.L(s_rows(j):e_rows(j));
        m.pars.S{j} = K{j} - K{j}*((-diag(1./(2*lambda))+K{j})\K{j});
    end  
    LSchol   = jit_chol(m.pars.S{j})';
    m.pars.LowerCholS{j} = LSchol;
    
    Kinvm = solve_chol(LKchol{j},m.pars.M(s_rows(j):e_rows(j))); % K^{-1}m
    KinvLj = solve_chol(LKchol{j},LSchol);  % K^{-1} Lj
    fvalEnt = fvalEnt +  sum(log(diag(LSchol))); % entropy
    fvalNCE = fvalNCE - sum(log(diag(LKchol{j}))) ... 
                      - 0.5*m.pars.M(s_rows(j):e_rows(j))'*Kinvm...
                      - 0.5*trAB(KinvLj,LSchol') ...
                      + 0.5*m.Nx;
    %
    % gradient 
    if nargout > 1
      dM(s_rows(j):e_rows(j)) = -Kinvm;
      A = inv(eye(vec_N(j))-2*AdiagB(K{j},diag(m.pars.L(s_rows(j):e_rows(j)))));
      dL(s_rows(j):e_rows(j)) = diag(m.pars.S{j}) - diagProd(m.pars.S{j},A');
    end
end

Fs = sampleFromPosterior(m, conf);

%   
%% ELL and its gradients
if nargout == 1
    ell = feval(m.ellfun, m, Fs, s_rows, e_rows, conf);
else
    [ell, dell_dm, dell_dl] = feval(m.ellfun, m, Fs, s_rows, e_rows, conf);
   for j = 1 : Q      % grad_{lambda} E_q log p(y|f) = 2(S.*S) grad_{diag(S)} E_q logp(y|f)
       dell_dl(s_rows(j):e_rows(j)) = 2*(m.pars.S{j}.^2)*dell_dl(s_rows(j):e_rows(j));
   end
    dM = dM + dell_dm;
    dL = dL + dell_dl;
     
    % For Q+1 variance we work on log scale
    j = Q + 1;
    ptr         =  s_rows(j):e_rows(j);
    dL(ptr)     =  dL(ptr).*sBin;
    grad = -[dM; dL];
end
%
fval = -(fvalEnt +  fvalNCE  + ell);


end 

%% Samples from current posterior
function Fs = sampleFromPosterior(m, conf)
%% sample from the marginal posteriors
% TODO: we actually do not need to sample from the 
% full covariances as there are not correlations acros functions
nTotal = m.nTotal;
Z      = randn(nTotal, conf.nsamples);
Fs     = zeros(nTotal,  conf.nsamples);
for j =1 : Q + 1
     ptr    = s_rows(j):e_rows(j);
     mu_fs  = m.pars.M(ptr,:);
     Fs(ptr,:) =   bsxfun(@plus, mu_fs, m.pars.LowerCholS{j}*Z);
end
% nTotal = m.nTotal;
% Fs = zeros(nTotal,  conf.nsamples);
% %  z  = randn(nTotal, conf.nsamples);
% for j =1 : Q + 1
%     Fs(s_rows(j):e_rows(j),:) = mvnrnd(m.pars.M(s_rows(j):e_rows(j),:)', diag(m.pars.S{j})', conf.nsamples)';
% %    ptr = s_rows(j):e_rows(j);    mu_fs  = m.pars.M(ptr,:);    dev_fs = sqrt(diag(m.pars.S{j}));   Fs(ptr,:) = bsxfun(@plus, mu_fs,  bsxfun(@times,z(ptr,:), dev_fs));
% end


end




