function [ sel_fea, sel_gnd, NodeSample, GroundTruth ] = splitdata_func( Network, fea, gnd ,K)
%SPLITDATA_FUNC Summary of this function goes here
%   Detailed explanation goes here

    nodenum = Network.Conf.NodeNumber;
    NodeSample = cell(nodenum,1);    

    [nsample,dim] = size(fea);
    split = floor(nsample/nodenum);
    select_id = randperm(nsample, split*nodenum);
    sel_fea = fea(select_id,:);
    sel_gnd = gnd(select_id,:);
    nsample = split*nodenum;

    sortedidx = randperm(nsample);
    sel_fea = sel_fea(sortedidx,:);
    sel_gnd = sel_gnd(sortedidx,:);

    base_align = zeros(dim,K);
    for i=1:K
        idx = find(sel_gnd==i);
        base_align(:,i) = mean(sel_fea(idx,:),1)';
    end

    beginp = 1;
    for i=1:nodenum    
        endp = beginp + split-1;
        NodeSample{i}.data = sel_fea(beginp:endp,:);      
        GroundTruth.nodegnd{i} = sel_gnd(beginp:endp,:);
        beginp = endp+1;
    end
    GroundTruth.gnd = sel_gnd;
    GroundTruth.base_align = base_align;

end

