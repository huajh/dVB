function w = Metropolis_Weight(NodeNum,Neighbors)
    % update node i
    w = zeros(NodeNum,NodeNum);
    
    for i=1:NodeNum
        degree = length(Neighbors{i});                
        for j=1:length(Neighbors{i})
            nei = Neighbors{i}(j);
            degree2 = length(Neighbors{nei});                
            w(nei,i) = 1/(1+ max([degree,degree2]));
        end        
        w(i,i) = 1 - sum(w(:,i));
    end
end