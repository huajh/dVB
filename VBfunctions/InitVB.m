
function R0 = InitVB(data,K)
%
    [N,~] = size(data);    
    %rand('state',1);    
    [IDX,~] = kmeans(data,K,'start','uniform','emptyaction','drop');  %,
    R0 = zeros(N,K);
    for i = 1:K
        R0(:,i) = (IDX == i);
    end
end
