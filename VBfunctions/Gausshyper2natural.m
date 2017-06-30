function phi = Gausshyper2natural(hyper)
    [D,K] = size(hyper.Mu);
    phi = struct('alpha',zeros(1,K),'v',D*ones(1,K),'invWBetaMuMu',zeros(D,D,K),...
         'betaMu',zeros(D,K),'beta',zeros(1,K));   
    phi.alpha = hyper.alpha;
    phi.v = hyper.v;    
    phi.beta = hyper.beta;    
    
    for i = 1:K
        phi.invWBetaMuMu(:,:,i) = hyper.invW(:,:,i)+...
                        hyper.beta(i)*(hyper.Mu(:,i)*hyper.Mu(:,i)');
        phi.betaMu(:,i) = hyper.beta(i)*hyper.Mu(:,i);    
    end    
end