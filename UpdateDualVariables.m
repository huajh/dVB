function Dual = UpdateDualVariables(Dual,Phi1,Phi2,rhos)
               
    Dual.alpha = Dual.alpha + rhos(1)/2*(Phi1.alpha-Phi2.alpha);    
    Dual.v = Dual.v + rhos(2)/2*(Phi1.v-Phi2.v);
    Dual.invWBetaMuMu = Dual.invWBetaMuMu + rhos(3)/2*(Phi1.invWBetaMuMu-Phi2.invWBetaMuMu);
    Dual.betaMu = Dual.betaMu + rhos(4)/2*(Phi1.betaMu-Phi2.betaMu);
    Dual.beta = Dual.beta + rhos(5)/2*(Phi1.beta-Phi2.beta);
                
end