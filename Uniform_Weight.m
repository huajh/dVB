
function w = Uniform_Weight(NodeNum,Neighbors)
% update node i            
    w = zeros(NodeNum,NodeNum);
  
    for i =1:NodeNum
        degree = length(Neighbors{i})+1;      
        for j=1:length(Neighbors{i})
            nei = Neighbors{i}(j);
            w(nei,i) = 1/degree;
        end
        w(i,i) = 1/degree;        
    end
end