function Params = Hyper2Params(Hypers)

    Params.mix = Hypers.alpha./sum(Hypers.alpha(:));
    Params.Mu = Hypers.Mu;
    for k=1:length(Hypers.alpha)
        Params.Sigma(:,:,k) = 1/Hypers.v(k)*Hypers.invW(:,:,k);        
    end    
end