%
%	Algtiothm       Gaussian Mixture Model by Expectation Miximum (EM-GMM)
%	@author         Junhao HUA
%	Create Time     2012/12/4
%   Last Updated    2013/1/9
%   email:          huajh7 AT gmail.com
%

function [label,model,L] = emgmm(Data, K)  
%%	parameters Description:
%		M		the number of Gaussian function
%		Data 	all observed data (dim*N)
%		dim 	the Dimension of data
%		N 		the number of data
%		mix 	the weight vector of each Gaussian (1*K)
%		Mu 		the mean vector of each Gaussian (dim*K)
%		Sigma 	the Covariance matrix of each Gaussian (dim*dim*K)

    model = InitByKmeans(Data,K); 
    [dim,N] = size(Data);
    epsilon = 1e-12;
    t = 0;
    maxTimes = 2000;
    logold = -realmax;
    L = -inf(1,maxTimes);
    while(t<maxTimes)     
    	t = t+1;
        Pkx = E_step(Data,model);
        model = M_step(Data,model,Pkx);
        [logL,Pkx] = vbound(Data, model);
       % fprintf('%d. %e %e\n',t,abs(logL-logold),epsilon*abs(logL));
        L(t) = logL/N;
        if(abs(logL-logold)<epsilon*abs(logL))
            break;
        end
        logold = logL;        
    end 
    L = L(1:t);
    [~,label] = max(Pkx,[],2); 
    disp(['Total iteratons:',num2str(t)]);    

function Pkx = E_step(Data,model)
%%
    % E- Step
    [~,N] = size(Data);
    [~,K] = size(model.M);
    pxk = ones(N,K);
    for i = 1:K
        % probability p(x|i) N x M
        %pxk(:,i) = GaussPDF(Data,model.M(:,i),model.Sigma(:,:,i));
        pxk(:,i) = mvnpdf(Data',model.M(:,i)',model.Sigma(:,:,i));
    end
    % calc posterior P(i|x) N*M
    pxk = repmat(model.mix,[N 1]).*pxk;
    Pkx = pxk./(repmat(sum(pxk,2),[1 K])+realmin);
    
function model = M_step(Data,model,Pkx)
%% M-step
    [dim,N] = size(Data);
    [~,K] = size(model.M);
    PkX = sum(Pkx);
    for i=1:K
        model.mix(i) = PkX(i)/N;
        model.M(:,i) = Data*Pkx(:,i)/PkX(i);
        datatmp = Data-repmat(model.M(:,i),1,N);
        model.Sigma(:,:,i) = (repmat(Pkx(:,i)',dim,1).*datatmp*datatmp')/PkX(i);
        model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(dim)*(1e-6);
    end
    


function model = InitByKmeans(Data,K)
%%  initialization by k-means
    [dim,N] = size(Data);
	[IDX,M0] = kmeans(Data',K,'emptyaction','drop','start','uniform'); 
    [IDX, M0] = litekmeans(Data', K,'Replicates',1);
    M0 = M0';
    mix0 = zeros(1,K);
    Sigma0 = zeros(dim,dim,K);
    for i=1:K
        mix0(i)=sum(i==IDX)/N;
        idx_temp = find(IDX==i);
    	Data_tmp1 = Data(:,idx_temp)-repmat(M0(:,i),1,length(idx_temp));
    	Sigma0(:,:,i) = Data_tmp1*Data_tmp1'/sum(i==IDX) + eye(dim)*(1e-6);
    end
    model.M = M0;
    model.Sigma = Sigma0;
    model.mix = mix0;
    

function [logL,Pkx] = vbound(Data, model)
%% calc the log likelihood  
    [~,N] = size(Data);
    [~,K] = size(model.M);
    pxk = zeros(N,K);
    for i = 1:K
        % probability p(x|i) N x K
        %pxk(:,i) = GaussPDF(Data,model.M(:,i),model.Sigma(:,:,i));
        pxk(:,i) = mvnpdf(Data',model.M(:,i)',model.Sigma(:,:,i));
    end
    pxk = repmat(model.mix,[N 1]).*pxk;
    Pkx = pxk./(repmat(sum(pxk,2),[1 K])+realmin);
    loglik = pxk*model.mix';
    loglik(loglik<realmin) = realmin;
    logL = sum(log(loglik));


