function [R,logR] = VBE_step(data,Hypers)   
%
    %	update the moments of parameters
    %	EQ          the expectation of Covariance matrix  N x K
    %	E_logLambda	the log expectation of precision	1 x K
    %	E_logPi 	the log expectation of the mixing proportion of the mixture components 1 x K
    [N,D] = size(data);
    [~,K] = size(Hypers.alpha);    
    beta = Hypers.beta;
    v = Hypers.v;
    alpha = Hypers.alpha;
    Mu = Hypers.Mu;
    invW = Hypers.invW;
    
    EQ = zeros(N,K);
    logW = zeros(1,K);
    for i=1:K                    
        U = chol(invW(:,:,i)+1e-8*eye(D));  % Cholesky  X=R'R
        logW(i) = -2*sum(log(diag(U)));
        Q = U'\bsxfun(@minus,data',Mu(:,i));        
        EQ(:,i) = D/beta(i) + v(i)*dot(Q,Q,1); % N x 1
    end
    E_logLambda = sum(psi(0,bsxfun(@minus,v+1,(1:D)')/2),1) + D*log(2) + logW;
    E_logPi = psi(0,alpha) - psi(0,sum(alpha)); % 1 x K
    
    %	update latent parameter: r
    logRho = bsxfun(@plus,-1/2*EQ,E_logPi + 1/2*E_logLambda -D/2*log(2*pi));
    %Normalization
    logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
    R = exp(logR);
    
end