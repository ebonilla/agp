function [avgError, nlp] =  computeErrorStructured(marginals, Y_test_vector)
TT_test = length(Y_test_vector);
[foo, maxMargPost] = max(sum(marginals, 3), [], 2);
avgError = sum(maxMargPost ~= Y_test_vector) / TT_test;

P = squeeze(mean(marginals,3));
[R, C] = size(P);
idx = sub2ind([R C], (1 : R)', Y_test_vector);
nlp = sum(log(P(idx)));

end


  