function [fval,dlikhyp] = elboLikStructured(hyp,m,fs)
% Untested
m.pars.hyp.lik(end) = hyp;
nSamples = size(fs,2);
[logllh,dlikhyp] = fastLikelihood(m.likfunc,m.Y,fs,m.pars.hyp,m.Nx,m.Q);
dlikhyp = -dlikhyp(end)/nSamples;
fval = -sum(logllh)/nSamples;

end
