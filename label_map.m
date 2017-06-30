function [ new_label ] = label_map( label, gnd )
% label_map Summary of this function goes here
%   Detailed explanation goes here
    
    K = length(unique(gnd));
    cost_mat = zeros(K,K);
    for i=1:K
        idx = find(label==i);
        for j=1:K        
            cost_mat(i,j) = length(find(gnd(idx)~=j));
        end
    end
    [assignment,cost] = munkres(cost_mat);
    [assignedrows,dum]=find(assignment');
    new_label = label;
    for i=1:K
        idx = find(label==i);
        new_label(idx) = assignedrows(i);
    end
end

