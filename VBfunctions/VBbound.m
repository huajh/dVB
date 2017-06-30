function L = VBbound(data, Hypers,R,logR,Prior)
%%	stopping criterion
    alpha0 = Prior.alpha0;
    beta0 = Prior.beta0;    
    v0 = Prior.v0;
    mu0 = Prior.mu0;
    invW0 = Prior.invW0;
          
    % Dirichlet
    alpha = Hypers.alpha;
    % Gaussian
    beta = Hypers.beta;  
    Mu = Hypers.Mu;
    % Whishart
    v = Hypers.v;         
    invW = Hypers.invW;  %inv(W) = V'*V
    
    [D,K] = size(Mu);           
            
    nk = sum(R,1); 
%    nk = NodeNum*sum(R,1);  %-> changed
    
    % pattern recognition and machine learning page496
    Elogpi = psi(0,alpha)-psi(0,sum(alpha));
    
    E_pz = dot(nk,Elogpi);  %10.72
    E_qz = dot(R(:),logR(:)); %10.75        
    % ->changed
    %E_pz = NodeNum*dot(nk,Elogpi);  %10.72 
    %E_qz = NodeNum*dot(R(:),logR(:)); %10.75    
    
    logCoefDir0 = gammaln(K*alpha0)-K*gammaln(alpha0); % the coefficient of Dirichlet Distribution
    E_ppi = logCoefDir0+(alpha0-1)*sum(Elogpi); %10.73
    logCoefDir = gammaln(sum(alpha))-sum(gammaln(alpha));
    E_qpi = dot(alpha-1,Elogpi)+logCoefDir; %10.76
    
    U0 = chol(invW0);
    sqrtR = sqrt(R);
    
    %xbar = bsxfun(@times,data'*R,1./nk); % 10.52
    xbar = bsxfun(@times,data'*R,1./nk); % 10.52 =>changed
    
    logW = zeros(1,K);
    trSW = zeros(1,K);
    trM0W = zeros(1,K);
    xbarmWxbarm = zeros(1,K);
    mm0Wmm0 = zeros(1,K);
    for i = 1:K 
		invW(:,:,i) = psd_mat(invW(:,:,i));
        U = chol(invW(:,:,i));  % Cholesky  X=R'R
        logW(i) = -2*sum(log(diag(U)));      
        
        %Xs = bsxfun(@times,bsxfun(@minus,data',xbar(:,i)),sqrtR(:,i)');
        Xs = bsxfun(@times,bsxfun(@minus,data',xbar(:,i)),sqrtR(:,i)'); % =>changed
        
        V = chol( 1e-2*eye(D,D)+ Xs*Xs'/nk(i));
        Q = V/U;
        trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
        Q = U0/U;
        trM0W(i) = dot(Q(:),Q(:));
        q = U'\(xbar(:,i)-Mu(:,i));
        xbarmWxbarm(i) = dot(q,q);
        q = U'\(Mu(:,i)-mu0);
        mm0Wmm0(i) = dot(q,q);
    end
    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:D)')/2),1)+D*log(2)+logW; % 10.65
    Epmu = sum(D*log(beta0/(2*pi))+ElogLambda-D*beta0./beta-beta0*(v.*mm0Wmm0))/2;
    logB0 = v0*sum(log(diag(U0)))-0.5*v0*D*log(2)-logmvgamma(0.5*v0,D);
    EpLambda = K*logB0+0.5*(v0-D-1)*sum(ElogLambda)-0.5*dot(v,trM0W);
    E_logpMu_Lambda = Epmu + EpLambda; % 10.74

    Eqmu = 0.5*sum(ElogLambda+D*log(beta/(2*pi)))-0.5*D*K;
    logB =  -v.*(logW+D*log(2))/2-logmvgamma(0.5*v,D);
    HqLambda = -0.5*sum((v-D-1).*ElogLambda-v*D)-sum(logB);
    E_logqMu_Lambda = Eqmu - HqLambda;%10.77

    E_pX = 0.5*dot(nk,ElogLambda-D./beta-v.*trSW-v.*xbarmWxbarm-D*log(2*pi)); %10.71
    L = E_pX+E_pz+E_ppi+E_logpMu_Lambda-E_qz-E_qpi-E_logqMu_Lambda;
end