function [ MixModel ] = dvbgmm_dSVB( Network, NodeSample, K, GroundTruth,tau)
%DVBGMM Summary of this function goes here
%   Detailed explanation goes here

%% Paramters Setting

    MixModel = [];
    Neighbors = Network.Nodes.neighbors;
    NodeNum = Network.Conf.NodeNumber;        
    [DataNum,D] = size(NodeSample{1}.data);
    
    % history saving
    maxIters = 300;
    
    Prior = struct('alpha0',1e-5,'beta0',1e-5,'mu0',1e-5*ones(D,1),'v0',D+1,'invW0',1e-5*eye(D,D));     
    %natural parametes vector  
    phiGroup = repmat(struct('alpha',zeros(1,K),'v',D*ones(1,K),'invWBetaMuMu',zeros(D,D,K),...
                'betaMu',zeros(D,K),'beta',zeros(1,K)),NodeNum,1);
            
    Hypers = repmat(struct('invW',zeros(D,D,K),'alpha',zeros(1,K),'v',D*ones(1,K),...
                'beta',zeros(1,K),'Mu',zeros(D,K)),NodeNum,1);
    LatentVar = repmat(struct('R',zeros(DataNum,K)),NodeNum,1);
    
    base_align = GroundTruth.base_align;
    gnd = GroundTruth.gnd;
    nsample = length(gnd);      
    
%     fea = [];    
%     for i=1:NodeNum
%         fea = [fea;NodeSample{i}.data];
%     end
%     all_R = InitVB(fea,K);
    
%% main part  

    % DVB-Diffusion: Adpate-Then-Combine
    IS_DIFF_UNIFORM = true;
    IS_DIFF_METROPOLIS = false;
    
    if IS_DIFF_UNIFORM
        w = Uniform_Weight(NodeNum,Neighbors);
    elseif IS_DIFF_METROPOLIS
        w = Metropolis_Weight(NodeNum,Neighbors);        
    end
            
    maxdegree = 0;
    for i=1:NodeNum
        maxdegree = max(maxdegree, length(Neighbors{i}));
    end
        
    AC_list = zeros(maxIters,1);
    NMI_list = zeros(maxIters,1);   
    old_std_AC = 0;
    old_std_NMI = 0;
    
    for t = 1:maxIters    
        % tuning parameters
%         kappafactor = 0.5; % forgetting rate
%         delayfactor = 1;               
       % Adpatation        
        for i = 1:NodeNum            
            % local estimation via trandtional variational Bayes
            if t == 1
                inter_R = InitVB(NodeSample{i}.data,K);
              %  inter_R = all_R((i-1)*DataNum+1:i*DataNum,:);
            else
                inter_R = VBE_step(NodeSample{i}.data,Hypers(i));
            end
            inter_hypers = VBM_step(NodeSample{i}.data,inter_R,Prior,NodeNum);
            %L = VBbound(NodeSample{i}.data, Hypers,R,logR,Prior);
            %[inter_hypers,LatentVar(i).R] = AlignVBResults(inter_hypers,inter_R,base_align);         
            LatentVar(i).R = inter_R;
            phi_star = Gausshyper2natural(inter_hypers);                       
            
            % Cooperate with its neighbors                                                
            %tau = 0.3;
            
            eta = 1/(2+tau*t);   
            %eta = 0.005;
            phiGroup(i) = GradientDescent(phiGroup(i),phi_star,eta);
            
        end
        % Combination        
        oldphiGroup = phiGroup;
  
        for i = 1:NodeNum
            phiGroup(i) = Diff_NaturalParameter(i,oldphiGroup(i), oldphiGroup, Neighbors{i}, w(:,i));
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
%             fprintf('svb %d AC: %f NMI: %f\n',t, AC, NMI);
%         else
%             fprintf('svb %d AC: %f NMI: %f stdAC: %f stdNMI: %f\n', ...
%             t, AC, NMI,std(AC_list(t-5:t)), std(NMI_list(t-5:t)));
%         end      
    end
    
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




