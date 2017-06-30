
function R0 = InitVB(data,K)
%
    [N,~] = size(data);                
    C = NaN;
%     while(isnan(C))
%         %[IDX,C] = kmeans(data,K,'start','uniform','emptyaction','drop');  %,
%         [IDX, C] = litekmeans(data, K,'Replicates',20);
%     end

   [~,IDX] = max(rand(N,K),[],2);
    
    R0 = zeros(N,K);
    for i = 1:K
        R0(:,i) = (IDX == i);
    end        
end
