function [newHypers,newR] = AlignVBResults(oldHypers,oldR,base_align)
    %reorder the clusters
    
    [D,K] = size(oldHypers.Mu);
    
    newHypers = struct('invW',zeros(D,D,K),'alpha',zeros(1,K),'v',D*ones(1,K),...
                'beta',zeros(1,K),'Mu',zeros(D,K));  
    newR = zeros(size(oldR));        

    [idxmap,cost] = cluster_algin(oldHypers.Mu', base_align');      
    
    newHypers.alpha = oldHypers.alpha(idxmap);
    newHypers.beta = oldHypers.beta(idxmap);  
    newHypers.Mu = oldHypers.Mu(:,idxmap);
    newHypers.v = oldHypers.v(idxmap);         
    newHypers.invW = oldHypers.invW(:,:,idxmap);
    newR = oldR(:,idxmap);
end