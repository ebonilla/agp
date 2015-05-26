function vec_ll = ll_funStructured(f, dataset, useMex)
% returns NLL 
% EVB for all N
N = dataset.N;
vec_ll = zeros(N,1);
edgePotSingle = exp(f(dataset.binary));
for n = 1 : N
    edgeStruct = dataset.edgeStruct{n};
    edgePot = repmat(edgePotSingle, [1,1, dataset.T(n)]);
    nodePot = exp(f(dataset.unary{n}));

    % Compute logZ
    [foo,foo,logZ] = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
    
    % Update LL
    if useMex
        vec_ll(n) = UGM_LogConfigurationPotentialC(int32(dataset.Y{n}),nodePot,edgePot,edgeStruct.edgeEnds ) - logZ;
    else
        vec_ll(n) =  UGM_LogConfigurationPotential(int32(dataset.Y{n}),nodePot,edgePot,edgeStruct.edgeEnds ) - logZ;
    end
end

end

