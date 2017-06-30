function [ perm_idx,cost ] = cluster_algin( vec, base_vec )
%LABEL_ALGIN Summary of this function goes here
%   Detailed explanation goes here
%   vec, base_vec: K x dim        
    
    % Eulidean distance
    cost_mat = bsxfun(@plus, bsxfun(@plus,-2*vec*base_vec',sum(vec.^2,2)),sum(base_vec.^2,2)');    
    [assignment,cost] = munkres(cost_mat);
    [perm_idx,~]=find(assignment');    
end

