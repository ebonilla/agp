% wrapper for negative cross entropy function and its derivatives wrt
% hyperparameters so that it can be used with minimize function
function [fval, dhyp] = elboCovhypStructured(hyp, m, sn2, x)
%TODO: pass Lj to this function
if iscell(hyp)
  m.pars.hyp.cov = hyp;
else
  m.pars.hyp.cov = rewrap(m.pars.hyp.cov,hyp);
end
Nx = m.Nx; 
Q = m.Q; 
fval = 0;
dhyp = cell(Q,1);
for j = 1 : Q
  s_row = (j-1)*Nx+1;
  e_row = j*Nx;
  K = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, x) + sn2*eye(Nx);
  LKchol = jit_chol(K);
  Lj = jit_chol(m.pars.S{j})';
  Kinvm = solve_chol(LKchol, m.pars.M(s_row:e_row)); % K^{-1}m
  KinvLj = solve_chol(LKchol, Lj); % K^{-1} Lj
  fval = fval + 2*sum(log(diag(LKchol))) + m.pars.M(s_row:e_row)'*Kinvm ...
    + trAB(KinvLj,Lj');
  dhyp{j} = zeros(size(m.pars.hyp.cov{j}));
  Kjinv = invChol(LKchol);
%  
 for i = 1 : numel(m.pars.hyp.cov{j})
    dK = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, x, [], i);
    dhyp{j}(i) = 0.5*trAB(Kinvm*Kinvm' - Kjinv + KinvLj*KinvLj', dK);
 end
 % use numerically more stable gradient for covSEard
 if strcmp(func2str(m.pars.hyp.covfunc),'covSEard') && sn2 == 0
    dhyp{j}(i) = m.pars.M(s_row:e_row)'*Kinvm - Nx + trAB(KinvLj,Lj');
 end
end
%
%
fval = 0.5*fval; % for minimization
if iscell(hyp)
  dhyp = rewrap(m.pars.hyp.cov,-unwrap(dhyp));
else
  dhyp = -unwrap(dhyp);
end

end

