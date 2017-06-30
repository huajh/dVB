function [ MixModel,flag ] = dvbgmm_admm( Network, NodeSample, K, GroundTruth,rho)
%DVBGMM Summary of this function goes here
%   Detailed explanation goes here
%   This function solves a gaussian mixture model using variational bayesian
%   algorithm and admm in a distributed fashion.
%% Paramters Setting

    MixModel = [];
    flag = 0;
    Neighbors = Network.Nodes.neighbors;
    NodeNum = Network.Conf.NodeNumber;        
    [DataNum,D] = size(NodeSample{1}.data);
     
    maxIters = 1000;

    Prior = struct('alpha0',1e-5,'beta0',1e-5,'mu0',1e-5*ones(D,1),'v0',D+1,'invW0',1e-5*eye(D,D));     
    %natural parametes vector  
    phiGroup = repmat(struct('alpha',zeros(1,K),'v',D*ones(1,K),'invWBetaMuMu',zeros(D,D,K),...
                'betaMu',zeros(D,K),'beta',zeros(1,K)),NodeNum,1);
            
    LambdaGroup = repmat(struct('alpha',zeros(1,K),'v',D*zeros(1,K),'invWBetaMuMu',zeros(D,D,K),...
                'betaMu',zeros(D,K),'beta',zeros(1,K)),NodeNum,NodeNum);
            
    Hypers = repmat(struct('invW',zeros(D,D,K),'alpha',zeros(1,K),'v',D*ones(1,K),...
                'beta',zeros(1,K),'Mu',zeros(D,K)),NodeNum,1);
    LatentVar = repmat(struct('R',zeros(DataNum,K)),NodeNum,1);
    
    base_align = GroundTruth.base_align;
    gnd = GroundTruth.gnd;
    nsample = length(gnd);

%% main part  

    %for the first running / initialization    
    for i=1:NodeNum        
        inter_R = InitVB(NodeSample{i}.data,K);
        inter_hypers = VBM_step(NodeSample{i}.data,inter_R,Prior,NodeNum);
        [Hypers(i),LatentVar(i).R] = AlignVBResults(inter_hypers,inter_R,base_align);
        phiGroup(i) = Gausshyper2natural(Hypers(i));
    end  
    
    AC_list = zeros(maxIters,1);
    NMI_list = zeros(maxIters,1);
    old_std_AC = 0;
    old_std_NMI = 0;
    
    for t = 1:maxIters
        
        % ADMM tuning parameters        
        % using different penalty parameters for individual parameters    
        rhos = [1,1,1,1,1]*rho;        
        eta = 1-1./(1+0.05*t).^2;
        %eta = 1;
        for i = 1:NodeNum          
            for j=1:length(Neighbors{i})
                nei = Neighbors{i}(j);
                LambdaGroup(i,nei) = UpdateDualVariables(LambdaGroup(i,nei),phiGroup(i),phiGroup(nei),eta*rhos);
            end
        end
        
        oldphiGroup = phiGroup;
        % Running ADMM          
        for i = 1:NodeNum         
            [Hypers(i),flag] = projection(Hypers(i));
            if flag == 1
                return;
            end
            % local estimation via trandtional variational Bayes                                    
            inter_R = VBE_step(NodeSample{i}.data,Hypers(i));
            inter_hypers = VBM_step(NodeSample{i}.data,inter_R,Prior,NodeNum);
            %[inter_hypers,LatentVar(i).R] = AlignVBResults(inter_hypers,inter_R,base_align);            
            LatentVar(i).R = inter_R;
            % update natural parameters vector            
            phi_star = Gausshyper2natural(inter_hypers);          
            %phiGroup(i) = phi_star;   
            
            % Cooperate with its neighbors          
            phiGroup(i) = updateADMM_NaturalParamter(i,phi_star,oldphiGroup,LambdaGroup,Neighbors{i},rhos);                  
            Hypers(i) = Gaussnatural2hyper(phiGroup(i)); 
            
        end
        
        % evaluation, performace
        label = [];
        for i=1:NodeNum
            [~,tmp] = max(LatentVar(i).R,[],2);
            label = [label;tmp];
        end
        label = label_map( label,gnd );
        AC = length(find(label-gnd == 0))/nsample;
        NMI = MutualInfo(gnd,label);        
        
        
        NMI_list(t) = NMI;
        AC_list(t) = AC;
        if t>100
            std_AC = std(AC_list(t-5:t));
            std_NMI = std(NMI_list(t-5:t));
            if (std_AC <1e-5 && std_NMI < 1e-5) ||...
                    (abs(old_std_AC - std_AC) < 1e-8 && abs(old_std_NMI-std_NMI)<1e-8)
                break;
            end     
            old_std_AC = std_AC;
            old_std_NMI = std_NMI;
        end
%         if t<=5
%             fprintf('admm %d AC: %f NMI: %f\n',t, AC, NMI);
%         else
%             fprintf('admm %d AC: %f NMI: %f stdAC: %f stdNMI: %f\n', ...
%             t, AC, NMI,std(AC_list(t-5:t)), std(NMI_list(t-5:t)));
%         end      
    end
   % fprintf('t=%d\n',t);
%% Output
    MixModel.Hypers = Hypers;
    Label = [];
    for i=1:NodeNum
        MixModel.Params(i) = Hyper2Params(Hypers(i));
        [~,tmp] = max(LatentVar(i).R,[],2);
        Label = [Label;tmp];        
    end  
    MixModel.Label = Label;
end


