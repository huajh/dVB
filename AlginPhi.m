function [perm_phi,perm_idx] = AlginPhi(basephi, phi)
   
    [D,K] = size(basephi.betaMu);
    perm_phi = struct('alpha',zeros(1,K),'v',D*ones(1,K),'invWBetaMuMu',zeros(D,D,K),...
                'betaMu',zeros(D,K),'beta',zeros(1,K));
        
    basemu = zeros(D,K);    
    mu = basemu;
    for i = 1:K
       basemu(:,i) = basephi.betaMu(:,i)/basephi.beta(i);   
       mu(:,i) = phi.betaMu(:,i)/phi.beta(i);   
    end 
    
     perm_idx = cluster_algin( mu', basemu' );
     
     perm_phi.alpha = phi.alpha(:,perm_idx);
     perm_phi.v = phi.v(:,perm_idx);
     perm_phi.invWBetaMuMu = phi.invWBetaMuMu(:,:,perm_idx);
     perm_phi.betaMu = phi.betaMu(:,perm_idx);
     perm_phi.beta = phi.beta(:,perm_idx);
end