

function hyper = Gaussnatural2hyper(phi)
    [D,K] = size(phi.betaMu);
    hyper = struct('invW',zeros(D,D,K),'alpha',zeros(1,K),'v',D*ones(1,K),...
                'beta',zeros(1,K),'Mu',zeros(D,K)); 
    hyper.alpha = phi.alpha;
    %hyper.alpha = max(1e-5,phi.alpha);
    hyper.v = phi.v; 
    %hyper.v = max(D,phi.v); %=> to ensure > D 
    hyper.beta = phi.beta;   
    [~,K] = size(phi.beta);
    for i =1:K
        hyper.Mu(:,i) = phi.betaMu(:,i)/phi.beta(i);
        hyper.invW(:,:,i) = phi.invWBetaMuMu(:,:,i)-...
                        (phi.betaMu(:,i)*phi.betaMu(:,i)')/phi.beta(i);          
    end        
end