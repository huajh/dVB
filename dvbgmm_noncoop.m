function [ MixModel ] = dvbgmm_noncoop( Network, NodeSample, K, GroundTruth)
%DVBGMM Summary of this function goes here
%   Detailed explanation goes here

%% Paramters Setting

    MixModel = [];
    NodeNum = Network.Conf.NodeNumber;        
    [DataNum,D] = size(NodeSample{1}.data);
    
    % history saving
    maxIters = 100;
    
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
    
%% main part  

    old_std_AC = 0;
    old_std_NMI = 0;
    AC_list = zeros(maxIters,1);
    NMI_list = zeros(maxIters,1);
    for t = 1:maxIters  
        
        % Running ADMM, Duffusion or Consensus         
        for i = 1:NodeNum            
            % local estimation via trandtional variational Bayes  
            if t == 1
                inter_R = InitVB(NodeSample{i}.data,K);
            else
                inter_R = VBE_step(NodeSample{i}.data,Hypers(i));
            end
            inter_hypers = VBM_step(NodeSample{i}.data,inter_R,Prior,NodeNum);
            %L = VBbound(NodeSample{i}.data, Hypers,R,logR,Prior);
            %[Hypers(i),LatentVar(i).R] = AlignVBResults(inter_hypers,inter_R,base_align);                      
            Hypers(i) = inter_hypers;
            LatentVar(i).R = inter_R;
            phiGroup(i) = Gausshyper2natural(Hypers(i));                                                         
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
        if t>50
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
%             fprintf('%d AC: %f NMI: %f\n',t, AC, NMI);
%         else
%             fprintf('%d AC: %f NMI: %f stdAC: %f stdNMI: %f\n', ...
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


