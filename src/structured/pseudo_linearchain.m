function L = pseudo_linearchain(nodePot,edgePot, x)

L       = 0;

for i=1:size(x,1) %1:model.nnodes
    % get conditional distribution
    logb = zeros(size(nodePot,2),1);
    for xi=1:size(nodePot,2)
        logb(xi) = nodePot(i,xi);
        
        % 2-neighbourhood model
        if i~=size(x,1)
            logb(xi) = logb(xi) + edgePot(x(i),xi,i);
        end
    end
    
    L = L + logb(x(i)) - log_sum_exp(logb); 
    
end