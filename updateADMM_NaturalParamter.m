function newPhi = updateADMM_NaturalParamter(idx,phi_star,oldphiGroup,LambdaGroup,Neighbors,rhos)

    newPhi = phi_star;
    
    Ni = length(Neighbors);
    
        
    for i=1:Ni
        j = Neighbors(i);
        newPhi.alpha = newPhi.alpha -2*LambdaGroup(idx,j).alpha +...
            + rhos(1)*oldphiGroup(idx).alpha + rhos(1)*oldphiGroup(j).alpha;            
        newPhi.v = newPhi.v -2*LambdaGroup(idx,j).v + rhos(2)*oldphiGroup(idx).v ...
            + rhos(2)*oldphiGroup(j).v;
        newPhi.invWBetaMuMu = newPhi.invWBetaMuMu -2*LambdaGroup(idx,j).invWBetaMuMu ...
            + rhos(3)*oldphiGroup(idx).invWBetaMuMu +rhos(3)*oldphiGroup(j).invWBetaMuMu;
        newPhi.betaMu = newPhi.betaMu -2*LambdaGroup(idx,j).betaMu...
            + rhos(4)*oldphiGroup(idx).betaMu + rhos(4)*oldphiGroup(j).betaMu;
        newPhi.beta = newPhi.beta -2*LambdaGroup(idx,j).beta +...
            + rhos(5)*oldphiGroup(idx).beta + rhos(5)*oldphiGroup(j).beta;    
    end
    
    dens = 1 + 2*rhos*Ni; %Denominator
    %newPhi.den = den;
    newPhi.alpha = 1/dens(1)*(newPhi.alpha);
    newPhi.v = 1/dens(2)*(newPhi.v);
    newPhi.invWBetaMuMu = 1/dens(3)*(newPhi.invWBetaMuMu );
    newPhi.betaMu = 1/dens(4)*(newPhi.betaMu );
    newPhi.beta = 1/dens(5)*(newPhi.beta);    

end



