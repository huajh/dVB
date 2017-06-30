function phi = Diff_NaturalParameter(idx, newphi, oldphiGroup, Neighbors, w)          
%        
    [D,K] = size(newphi.betaMu);
    phi = struct('alpha',zeros(1,K),'v',D*zeros(1,K),'invWBetaMuMu',zeros(D,D,K),...
         'betaMu',zeros(D,K),'beta',zeros(1,K));      
    
    phi.alpha = w(idx)*newphi.alpha;
    phi.v = w(idx)*newphi.v;
    phi.beta = w(idx)*newphi.beta;    
    phi.betaMu = w(idx)*newphi.betaMu;    
    phi.invWBetaMuMu = w(idx)*newphi.invWBetaMuMu;    
    
    for i=1:length(Neighbors)
        nei = Neighbors(i);
        phi.alpha = phi.alpha + w(nei)*oldphiGroup(nei).alpha;
        phi.v =  phi.v + w(nei)*oldphiGroup(nei).v;
        phi.beta = phi.beta + w(nei)*oldphiGroup(nei).beta;
        phi.betaMu = phi.betaMu + w(nei)*oldphiGroup(nei).betaMu;
        phi.invWBetaMuMu = phi.invWBetaMuMu + w(nei)*oldphiGroup(nei).invWBetaMuMu;
    end       
end


