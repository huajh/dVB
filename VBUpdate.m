function [Hypers,History,R,Label] = VBUpdate(data,K,GroundTruth)
    
    % data: N x D 
    [~,D] = size(data);
    Prior = struct('alpha0',1e-5,'beta0',1e-5,'mu0',1e-5*ones(D,1),'v0',D+1,'invW0',1e-5*eye(D,D));    
    
    L0 = -inf;        
    eps =1e-12;
    
    maxIters = 300;
    
    % history 
    History.kl_error = zeros(1,maxIters);
    History.old_Params = repmat(struct('mix',zeros(1,K),'Mu',zeros(D,K),'Sigma',zeros(D,D,K)),maxIters,1);
    
    
    R = InitVB(data,K);
    
    for t=1:maxIters           
        Hypers = VBM_step(data,R,Prior);
        [R,logR] = VBE_step(data,Hypers);
        [Hypers,R] = AlignVBResults(Hypers,R);  
        L = VBbound(data, Hypers,R,logR,Prior);
        
        History.Lseq(t) = L;
        History.kl_error(t) = kl_mixgaus(Hypers, GroundTruth);            
        History.old_Params(t) = Hyper2Params(Hypers);        
     
%         fprintf('%e %e \n',abs(L-L0),eps*abs(L));
%         if abs(L-L0) < eps*abs(L)
%             break;
%         end          
        L0 = L;
        mixx = History.old_Params(t).mix;
        fprintf('%d: %f %f %f %f\n',t,mixx(1),mixx(2),mixx(3),History.kl_error(t));
    end
    
    [Hypers,R] = AlignVBResults(Hypers,R);
    [~,Label] = max(R,[],2); 
 
    
   disp(['Total iteratons:',num2str(t)]);    
end





