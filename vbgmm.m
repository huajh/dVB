%% Variational Bayesian Gaussian Mixture Model(VB-GMM)
%
%	@author         Junhao HUA
%	Create Time:	2012-12-5
%   Last Update:    2013/1/2
%   
%   @reference
%   M.bishop,pattern recognition and machine learning,2006
%
%%		

function [label,model,L] = vbgmm(Data,K)
%%	parameters Description:
% 		K		the number of mixing components
%		Data 	all observed data (dim*N)
%		N 		the number of data
%		dim 	the Dimension of data
%		Mix 	the weight vector of each Gaussian (1 x K)
%		Mu 		the mean vector of each Gaussian (dim x K)
%		Sigma 	the Covariance matrix of each Gaussian (dim x dim x K)		
%
%       the initialization of prior superparameters
%       They can be set to small positive numbers to give
%       broad prior distrbutions indicating ignorance about the prior
%       Dirichlet Distribution Parameters:
%           alpha0	1
%       Wishart Distribution Parameters:
%           invW0 	dim x dim
%    		v0 		1
%       Gaussian Distribution Parameters:
%           m0 		dim x 1
%           beta0 	1

    [dim,N] = size(Data);
   %prior = struct('alpha0',1e-5,'beta0',1e-5,'m0',1e-5*mean(Data,2),'v0',dim+1,'invW0',eye(dim,dim));
    prior = struct('alpha0',1e-5,'beta0',1e-5,'m0',zeros(dim,1),'v0',dim,'invW0',1e-5*eye(dim,dim));
    logL0 = -inf;
    esp = 1e-12;

	% latent parameters : the probability of each point in each component
	%		r 		N x K
    model.R = InitVB(Data',K);
    t = 0;
	maxtimes = 200;
    L = -inf(1,maxtimes);
	while t < maxtimes
        t = t +1;
        model = MaxStep(Data,model,prior); 
        model = ExpectStep(Data,model);
        logL = vbound(Data,model,prior)/N;
       
        %fprintf('%d %e %e \n',t, abs(logL-logL0),esp*abs(logL));
        L(t) = logL;
        if abs(logL-logL0) < esp*abs(logL)
            break;
        end  
        logL0 = logL;
    end
    model.mix = model.alpha./sum(model.alpha(:));
    for k=1:K
        model.Sigma(:,:,k) = (1/model.V(k)).*model.invW(:,:,k);
    end
    
    L = L(1:t);            
    [~,label] = max(model.R,[],2);   
    %disp(['Total iteratons:',num2str(t)]);


function model = MaxStep(data,model,prior)  
%%
    %	update the statistics
    % 		avgN	1 x K
    % 		avgX	dim x K
    % 		avgS	dim x dim x K
    %	update the superparameters
    %   the parameter of weight (1 x K):
    %       alpha 	1 x K
    %   the parameters of preision (dim x dim x K):
    %       invW 	dim x dim x K inv(W)
    %       V 		1 x K
    %   the parameters of mean:
    %       M 		dim x K
    %       beta 	1 x K         
    [~,K] = size(model.R);
    avgN = sum(model.R);
    aXn = data*model.R;
    avgX = bsxfun(@times,aXn,1./avgN);
    
    ws = (prior.beta0*avgN)./(prior.beta0+avgN);
    sqrtR = sqrt(model.R);
    for i = 1:K
        avgNS = bsxfun(@times,bsxfun(@minus,data,avgX(:,i)),sqrtR(:,i)');
        avgS(:,:,i) = avgNS*avgNS'/avgN(i);
        Xkm0 = avgX(:,i)-prior.m0;
        model.invW(:,:,i) = prior.invW0 +avgNS*avgNS'+ ws(i).*(Xkm0*Xkm0');
    end    
    
    model.alpha = prior.alpha0 + avgN;
    model.V = prior.v0 + avgN;   
    model.beta = prior.beta0 + avgN;       
    model.M = bsxfun(@times,bsxfun(@plus,prior.beta0.*prior.m0,aXn),1./model.beta);
    model.Sigma = avgS;
    

function model = ExpectStep(data,model)   
%%
    %	update the moments of parameters
    %	EQ          the expectation of Covariance matrix  N x K
    %	E_logLambda	the log expectation of precision	1 x K
    %	E_logPi 	the log expectation of the mixing proportion of the mixture components 1 x K
    [dim,N] = size(data);
    [~,K] = size(model.M);
    EQ = zeros(N,K);
    logW = zeros(1,K);
    for i=1:K
        U = chol(model.invW(:,:,i));  % Cholesky  X=R'R
        logW(i) = -2*sum(log(diag(U)));
        Q = U'\bsxfun(@minus,data,model.M(:,i));        
        EQ(:,i) = dim/model.beta(i) + model.V(i)*dot(Q,Q,1); % N x 1
    end      
    E_logLambda = sum(psi(0,bsxfun(@minus,model.V+1,(1:dim)')/2),1) + dim*log(2) + logW;
    E_logPi = psi(0,model.alpha) - psi(0,sum(model.alpha)); % 1 x K
    %	update latent parameter: r
    logRho = bsxfun(@plus,-1/2*EQ,E_logPi + 1/2*E_logLambda -dim/2*log(2*pi));
    %logRho = bsxfun(@minus,EQ,2*E_logPi + E_logLambda -dim*log(2*pi))/(-2);
    model.logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
    model.R = exp(model.logR);    
  

function L = vbound(X, model, prior)
%%	stopping criterion
    alpha0 = prior.alpha0;
    beta0 = prior.beta0;
    m0 = prior.m0;
    v0 = prior.v0;
    
    invW0 = prior.invW0;
    % Dirichlet
    alpha = model.alpha;
    % Gaussian
    beta = model.beta;  
    m = model.M;
    % Whishart
    v = model.V;         
    invW = model.invW;  %inv(W) = V'*V
    
    R = model.R;
    logR = model.logR;

    [dim,k] = size(m);
    nk = sum(R,1);
    % pattern recognition and machine learning page496
    Elogpi = psi(0,alpha)-psi(0,sum(alpha));
    E_pz = dot(nk,Elogpi);  %10.72
    E_qz = dot(R(:),logR(:)); %10.75    
    logCoefDir0 = gammaln(k*alpha0)-k*gammaln(alpha0); % the coefficient of Dirichlet Distribution
    E_ppi = logCoefDir0+(alpha0-1)*sum(Elogpi); %10.73
    logCoefDir = gammaln(sum(alpha))-sum(gammaln(alpha));
    E_qpi = dot(alpha-1,Elogpi)+logCoefDir; %10.76
    
    U0 = chol(invW0);
    sqrtR = sqrt(R);
    xbar = bsxfun(@times,X*R,1./nk); % 10.52
    logW = zeros(1,k);
    trSW = zeros(1,k);
    trM0W = zeros(1,k);
    xbarmWxbarm = zeros(1,k);
    mm0Wmm0 = zeros(1,k);
    for i = 1:k
        U = chol(invW(:,:,i));
        logW(i) = -2*sum(log(diag(U)));      
        Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
        V = chol(1e-12*eye(dim,dim)+ Xs*Xs'/nk(i));
        Q = V/U;
        trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
        Q = U0/U;
        trM0W(i) = dot(Q(:),Q(:));
        q = U'\(xbar(:,i)-m(:,i));
        xbarmWxbarm(i) = dot(q,q);
        q = U'\(m(:,i)-m0);
        mm0Wmm0(i) = dot(q,q);
    end
    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:dim)')/2),1)+dim*log(2)+logW; % 10.65
    Epmu = sum(dim*log(beta0/(2*pi))+ElogLambda-dim*beta0./beta-beta0*(v.*mm0Wmm0))/2;
    logB0 = v0*sum(log(diag(U0)))-0.5*v0*dim*log(2)-logmvgamma(0.5*v0,dim);
    EpLambda = k*logB0+0.5*(v0-dim-1)*sum(ElogLambda)-0.5*dot(v,trM0W);
    E_logpMu_Lambda = Epmu + EpLambda; % 10.74

    Eqmu = 0.5*sum(ElogLambda+dim*log(beta/(2*pi)))-0.5*dim*k;
    logB =  -v.*(logW+dim*log(2))/2-logmvgamma(0.5*v,dim);
    HqLambda = -0.5*sum((v-dim-1).*ElogLambda-v*dim)-sum(logB);
    E_logqMu_Lambda = Eqmu - HqLambda;%10.77

    E_pX = 0.5*dot(nk,ElogLambda-dim./beta-v.*trSW-v.*xbarmWxbarm-dim*log(2*pi)); %10.71
    L = E_pX+E_pz+E_ppi+E_logpMu_Lambda-E_qz-E_qpi-E_logqMu_Lambda;
	

