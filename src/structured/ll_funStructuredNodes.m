function [sll, vec_ll] = ll_funStructuredNodes(f, dataset, useMex)
% return the log likelihood  of the data 
% and the marginals vecl_ll for all nodes 
N = dataset.N;
vec_ll = zeros(dataset.max,1);
edgePotSingle = exp(f(dataset.binary));
idxBinary = dataset.binary(:);
sll = 0;
for n = 1 : N
    edgeStruct  = dataset.edgeStruct{n};
    nodePot     = exp(f(dataset.unary{n}));    
    edgePot     = repmat(edgePotSingle, [1,1, dataset.T(n)]);

    % Compute logZ
    [nodeBel, edgeBel,logZ] = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
    
    vec_ll(dataset.unary{n}(:)) = nodeBel(:);
    sumEdgeBel                  = sum(edgeBel, 3);
    vec_ll(idxBinary)           =  vec_ll(idxBinary) + sumEdgeBel(:);
    
    % Update LL
    if useMex
        sll =  sll + UGM_LogConfigurationPotentialC(int32(dataset.Y{n}),nodePot,edgePot,edgeStruct.edgeEnds ) - logZ;
    else
        sll =  sll + UGM_LogConfigurationPotential(int32(dataset.Y{n}),nodePot,edgePot,edgeStruct.edgeEnds ) - logZ;
    end
end

end

