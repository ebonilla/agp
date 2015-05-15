function nodesBel = predictiveMarginalsN(fStar, dataset_test)
nodesBel = zeros(dataset_test.TT, dataset_test.nLabels);
edgePotSingle = exp(fStar(dataset_test.binary));
for n_test = 1:dataset_test.N
    edgeStruct = dataset_test.edgeStruct{n_test};
    edgePot = repmat(edgePotSingle, [1,1, dataset_test.T(n_test)]);
    nodePot = exp(fStar(dataset_test.unary{n_test}));
    
    % Compute marginals
    [nodesBel(dataset_test.dataStarts(n_test):dataset_test.dataEnds(n_test), :),foo,foo] = ...
        UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
end
end

