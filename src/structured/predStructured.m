function P = predStructured(m, conf, xtrain, data_test)

[fmu,fvar] = predRegStructured(m, conf, xtrain, data_test.X);

%% Need to sample for q(f*), then compute prediction through averaging
nTestSamples = conf.ntestsamples;
Z = randn(data_test.max,nTestSamples);
%fSample      = fmu + sqrt(fvar).*Z;
fSample       = bsxfun(@plus, fmu, bsxfun(@times, sqrt(fvar), Z));
nChunks       =  sum(cellfun('length', data_test.Y));
P            = zeros(nChunks, data_test.nLabels, nTestSamples);
for j  = 1 : nTestSamples
     P(:,:,j) = predictiveMarginalsN(fSample(:,j), data_test);
end
%
% prob = squeeze(mean(P,3));

end
   

 