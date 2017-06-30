function [ kl_div] = kl_mixgaus( Hypers, GroundTruth )
%KL_MIXGAUS Summary of this function goes here
%   Detailed explanation goes here    
%
%   Estimated HyperParameters:  Hypers  
                                
%   True HyperParameters:       GroundTruth

    
    [D,K] = size(Hypers.Mu);    
    
    kl_div = zeros(K,1);    
    
    alpha = Hypers.alpha;
    beta = Hypers.beta;
    v = Hypers.v;
    Mu = Hypers.Mu;
    invW = Hypers.invW;         
    
    
    % KL divergence w.r.t Dirichlet Distribution
    E_logPi = psi(0,alpha) - psi(0,sum(alpha)); % 1 x K
    kl_div(1) =  (alpha - GroundTruth.alpha)*E_logPi' -...
        (sum(gammaln(alpha)) - gammaln(sum(alpha)))+...
        (sum(gammaln(GroundTruth.alpha)) - gammaln(sum(GroundTruth.alpha)));    
    
    %KL divergence w.r.t Normal-Wishart Distribution        
    logW = zeros(1,K);    
    UM = zeros(D,K);
    trMWM = zeros(1,K);
    U = zeros(D,D,K);
    for k = 1:K
		invW(:,:,k) = psd_mat(invW(:,:,k));
        U(:,:,k) = chol(invW(:,:,k));
        logW(k) = -2*sum(log(diag(U(:,:,k))));
        UM(:,k) = U(:,:,k)'\Mu(:,k);
        trMWM(k) = dot(UM(:,k),UM(:,k),1);
    end 
    E_logLambda = sum(psi(0,bsxfun(@minus,v+1,(1:D)')/2),1) + D*log(2) + logW; % 1 x K
    
    beta0 = GroundTruth.beta;
    v0 = GroundTruth.v;
    
    % with 0 is GroundTruth (be subtracted) , or second term
    logW0 = zeros(1,K);
    for k=1:K
        U0 = chol(GroundTruth.invW(:,:,k));
        logW0(k) = -2*sum(log(diag(U0)));
        U0U = U0/U(:,:,k);
        UM0 = U(:,:,k)'\GroundTruth.Mu(:,k); % U^T Mu0
        
        Phi0ESS= (v0(k)-D)/2*E_logLambda(k)-v(k)/2*dot(U0U(:),U0U(:),1)...
            -v(k)*beta0(k)/2*dot(UM0,UM0,1)+v(k)*beta0(k)*dot(UM(:,k),UM0,1)...
            -beta0(k)/2*(D/beta(k)+v(k)*trMWM(k));
        
        PhiESS = (v(k)-D)/2*E_logLambda(k)-D*v(k)/2-D/2;
        
        A0 = -D/2*log(abs(beta0(k))) + v0(k)/2*logW0(k) + v0(k)*D/2*log(2)...
            +sum(gammaln(bsxfun(@minus,v0(k)+1,(1:D)')/2),1);
        
        A = -D/2*log(abs(beta(k))) + v(k)/2*logW(k) + v(k)*D/2*log(2)...
            +sum(gammaln(bsxfun(@minus,v(k)+1,(1:D)')/2),1);
        
        kl_div(k+1) = PhiESS - Phi0ESS - A + A0;
    end
    kl_div = sum(kl_div(:));
end

