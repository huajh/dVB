function Hypers = VBM_step(data,R,Prior,NODENUM)  
%
%   Latent parameters : the probability of each point in each component
%		R 		N x K
%
    %	update the statistics
    % 		avgN	1 x K
    % 		avgX	D x K
    % 		avgS	D x D x K
    %
    %	update the hyperparameters
    %   the parameter of weight (1 x K):
    %       alpha 	1 x K
    %   the parameters of preision (dim x dim x K):
    %       invW 	D x D x K inv(W)
    %       v 		1 x K
    %   the parameters of mean:
    %       Mu 		D x K
    %       beta 	1 x K         
    
    if nargin <4
        NODENUM = 1;
    end
    [~,K] = size(R);
    [~,D] = size(data);
    
    Hypers = struct('invW',zeros(D,D,K),'alpha',zeros(1,K),'v',D*zeros(1,K),...
                'beta',zeros(1,K),'Mu',zeros(D,K)); 
                
    avgN = NODENUM*sum(R);
    aXn = NODENUM*data'*R;  
    % =>changed
    %avgN = NodeNum*sum(R); 
    %aXn = NodeNum*data'*R;
    
    avgX = bsxfun(@times,aXn,1./avgN);
    
    Hypers.alpha = Prior.alpha0 + avgN;
    Hypers.v = Prior.v0 + avgN;   
    Hypers.beta = Prior.beta0 + avgN;               
    Hypers.Mu = bsxfun(@times,bsxfun(@plus,Prior.beta0.*Prior.mu0,aXn),1./Hypers.beta);
    sqrtR = sqrt(R);  
    for i=1:K
        avgNS = bsxfun(@times,data',sqrtR(:,i)');%=>changed
        Hypers.invW(:,:,i) = Prior.invW0 + NODENUM*(avgNS*avgNS') +...
            Prior.beta0*(Prior.mu0*Prior.mu0')-Hypers.beta(i)*(Hypers.Mu(:,i)*Hypers.Mu(:,i)');                
    end
    


end