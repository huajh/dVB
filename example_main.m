%
% create time: 2015/4/30
% last update: 2015/5/14

% huajh7@gmail.com
%
close all;
clear; clc;
close all;
addpath('.\VBfunctions');
addpath('.\boundedline');
load Network.mat


dataset =  'atmos'; % atmos  abalone iris  ionos
test = 'drop';

if strcmp(dataset,'atmos')
    load atmos_data.mat
    % feature:      fea 2900 x 3
    % groundturth:  gnd 2900 x 1    
    [nsample,dim] = size(fea);
    idx = randperm(nsample,1600);
    fea = fea(idx,:);
    gnd = gnd(idx,:);
%     
%     load atmos_all_data2.mat
%     [nsample,dim] = size(fea);
%     idx = randperm(nsample,800);
%     fea = fea(idx,:);
%     gnd = gnd(idx,:);
%     

elseif strcmp(dataset,'abalone')
    load abalone_data.mat

elseif strcmp(dataset,'iris')
    load iris_data.mat
    % fea 150 x 4
    % gnd 150 x 1
elseif strcmp(dataset,'ionos')
    load ionosphere_data.mat    
end

fea = repmat(fea,1,1);
gnd = repmat(gnd,1,1);

[nsample,dim] = size(fea);
nclass = length(unique(gnd));

% Random permutation
idx = randperm(nsample);
fea = fea(idx,:);
gnd = gnd(idx,:);

K = nclass;

rand('state',sum(1000*clock));

mtds = {'cVB','noncoop-VB','nsg-dVB','dSVB','dVB-ADMM'};

len = length(mtds);
for str = 4
    mtd = mtds{str};    
    
    repeat = 10;
    
    all_AC = zeros(repeat,1);
    all_NMI = zeros(repeat,1);    
    all_mis = zeros(repeat,1);
    
    fid = fopen(['_' dataset '_' mtd '_' test '.txt'],'wt');
    tt = 1;
    seed_off = floor(10000*rand(repeat,1));
    while(tt <= repeat)
        rand('state',sum(1000*clock)+seed_off(tt));
        [ re_fea, re_gnd, NodeSample, GroundTruth ] = splitdata_func( Network, fea, gnd ,K);
        nodenum = Network.Conf.NodeNumber;
        [datanum,dim] = size(NodeSample{1}.data);
        if strcmp(mtd,'VB')
            [label,model,L] = vbgmm(re_fea',K);
        elseif strcmp(mtd,'cVB')
            MixModel = cvbgmm(Network, NodeSample, K, GroundTruth);
            label = MixModel.Label;
        elseif strcmp(mtd,'dSVB')
            tau = 0.3;
            MixModel = dvbgmm_dSVB(Network, NodeSample,K,GroundTruth,tau);
            label = MixModel.Label;
        elseif strcmp(mtd,'nsg-dVB')
            MixModel = dvbgmm_nsg_dVB(Network, NodeSample,K,GroundTruth);
            label = MixModel.Label;
        elseif strcmp(mtd,'kmeans')
            label = litekmeans(re_fea,K,'Replicates',20);
        elseif strcmp(mtd,'noncoop-VB')
            MixModel = dvbgmm_noncoop(Network, NodeSample,K,GroundTruth);
            label = MixModel.Label;
        elseif strcmp(mtd,'dVB-ADMM')
            rho = 16;
            flag = 1;
            while(flag == 1)
                [MixModel,flag] = dvbgmm_admm(Network, NodeSample,K,GroundTruth,rho);
            end            
            label = MixModel.Label;
        end        
         
        for i=1:nodenum
            cur_idx = datanum*(i-1)+1:datanum*i;
            label(cur_idx) = label_map(label(cur_idx),re_gnd(cur_idx));
        end
        
        [label] = label_map( label,re_gnd);
        rightnum = length(find(label-re_gnd == 0));
        AC = rightnum/nsample;
        NMI = MutualInfo(re_gnd,label);
        all_AC(tt) = AC;
        all_NMI(tt) = NMI;
        all_mis(tt) = nsample-rightnum;
        
%         if strcmp(dataset,'atmos') && strcmp(mtd,'dSVB') && all_mis(tt) > 100
%             continue;
%         end   
        
        fprintf([mtd,' t = %d AC: %f, NMI: %f misclassfication %d\n'], tt, AC, NMI,nsample-rightnum);
        fprintf(fid,[mtd,' t = %d AC: %f, NMI: %f misclassfication %d\n'], tt, AC, NMI,nsample-rightnum);
        tt = tt + 1;
    end
            
    fprintf([mtd,' AVG AC: %f, stdAC=%f NMI: %f mis=%f misstd = %f \n\n'], mean(all_AC), std(all_AC), mean(all_NMI),mean(all_mis), std(all_mis));
    fprintf(fid,[mtd,' AVG AC: %f, stdAC=%f NMI: %f mis=%f  misstd = %f \n\n'], mean(all_AC), std(all_AC), mean(all_NMI),mean(all_mis),std(all_mis));
    sta = fclose(fid);
%    save(['_' dataset '_', mtd, '_' test '.mat'],'all_AC', 'all_NMI','all_mis');
    
    if strcmp(dataset,'atmos1')
        save(['_atmos_', mtd, '_sample_2.mat'],'label','re_fea');
        idx1 = find(label==1);
        idx2 = find(label==2);
        figure;
        hold on;
        fig1 = plot3(re_fea(idx1,1),re_fea(idx1,2),re_fea(idx1,3),'bo');
        fig2 = plot3(re_fea(idx2,1),re_fea(idx2,2),re_fea(idx2,3),'r*');
        grid on;
        xlabel('SO_2');
        ylabel('NO_2');
        zlabel('PM10');
        title(mtd);
        legend([fig1,fig2], 'clean air','polluted air');
        set(gcf,'Color','w');
    end    
end

