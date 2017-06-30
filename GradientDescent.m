function phi = GradientDescent(oldphi,phi_star,eta)
%
    phi = oldphi;
    
    phi.alpha = (1-eta)*phi.alpha + eta*phi_star.alpha;
    phi.v =  (1-eta)*phi.v + eta*phi_star.v;
    phi.beta = (1-eta)*phi.beta + eta*phi_star.beta;    
    phi.betaMu =(1-eta)*phi.betaMu + eta*phi_star.betaMu;
    phi.invWBetaMuMu =(1-eta)*phi.invWBetaMuMu + eta*phi_star.invWBetaMuMu;
    
       
end


