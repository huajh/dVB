%
% create time: 2015/4/30
% last update: 2015/4/30
% huajh7@gmail.com
%

close all;
clear; clc;
addpath('.\VBfunctions');
addpath('.\boundedline');
load ('COIL20_PCA.mat')
% feature:      fea 9298 x 256
% groundturth:  gnd 9298 x 1
%gnd = gnd - 1;

fea = repmat(fea,10,1);
gnd = repmat(gnd,10,1);

load Network2.mat

num_avg = 10; % 30
repeat = 10;  % 10

rand('state',sum(1000*clock));
mtd = 'admm';
AC_result = zeros(num_avg,repeat,9);
NMI_result = zeros(num_avg,repeat,9);
AVG_AC = zeros(9,1);
AVG_NMI = zeros(9,1);

fid = fopen(['_res_' mtd '_1.txt'],'wt');
for K=6:10
    tic;
    avgAC = 0;
    avgNMI = 0;
    for t=1:num_avg
        
        restart = 1;
        while(restart)
            restart = 0;
            clusts = randperm(20,K);
            fprintf('Digit number:');
            fprintf(fid,'Digit number:');
            for ppp = 1:K
                fprintf('%d ',clusts(ppp) - 1);
                fprintf(fid,'%d ',clusts(ppp) - 1);
            end
            fprintf('\n\n');
            fprintf(fid,'\n\n');

            oldAC = 0;
            oldNMI = 0;
            re_idx = [];
            re_gnd = [];
            for i=1:K
                idx = find(gnd==clusts(i));
                gnd2 = i*ones(length(idx),1);
                re_idx = [re_idx;idx];
                re_gnd = [re_gnd;gnd2];
            end

            [re_idx,ord] = sort(re_idx);
            re_gnd = re_gnd(ord);
            re_fea = fea(re_idx,:);
            nsample = length(re_gnd);

            [ re_fea, re_gnd, NodeSample, GroundTruth ] = splitdata_func( Network, re_fea, re_gnd ,K);

            seed_off = floor(10000*rand(repeat,1));
            for tt = 1:repeat  
                rand('state',sum(1000*clock)+seed_off(tt));
                rho = 16;
                flag = 1;
                cnt = 0;
                while(flag == 1)
                    cnt = cnt + 1;
                    if cnt > 10
                        fprintf('cnt = %d\n',cnt);
                        restart = 1;
                        break;
                    end
                    [MixModel,flag] = dvbgmm_admm(Network, NodeSample,K,GroundTruth,rho);
                end
                if restart == 1
                    break;
                end
                label = MixModel.Label;

                [ new_label] = label_map( label,re_gnd );

                AC = length(find(new_label-re_gnd == 0))/nsample;
                NMI = MutualInfo(re_gnd,new_label);

                if oldAC < AC
                    oldAC = AC;
                end
                if oldNMI < NMI
                    oldNMI = NMI;
                end
                fprintf([mtd,' %d AC: %f, MI: %f\n'],tt, oldAC, oldNMI);
                fprintf(fid,[mtd,' %d AC: %f, MI: %f\n'],tt, oldAC, oldNMI);
                AC_result(t,tt,K-1) = oldAC;
                NMI_result(t,tt,K-1) = oldNMI;
            end
        end
        
        fprintf([mtd, ': K = %d t = %d AC = %f, NMI = %f\n\n'], K,t, oldAC,oldNMI);
        fprintf(fid, [mtd, ': K = %d t = %d AC = %f, NMI = %f\n\n'], K,t, oldAC,oldNMI);
        avgAC = avgAC + oldAC;
        avgNMI = avgNMI + oldNMI;
    end
    AVG_AC(K-1) = avgAC/num_avg;
    AVG_NMI(K-1) = avgNMI/num_avg;
    fprintf([mtd, ' avgAC = %f, avgMI = %f\n\n'],AVG_AC(K-1),AVG_NMI(K-1));
    fprintf(fid,[mtd, ' avgAC = %f, avgMI = %f\n\n'],AVG_AC(K-1),AVG_NMI(K-1));
    toc;
end
for K=2:10
    fprintf([mtd, 'K = %d avgAC = %f, avgMI = %f\n'],K, AVG_AC(K-1),AVG_NMI(K-1));
    fprintf(fid,[mtd, 'K = %d avgAC = %f, avgMI = %f\n'],K, AVG_AC(K-1),AVG_NMI(K-1));
end
save(['_', mtd, '_result_1.mat'],'AC_result', 'NMI_result','AVG_AC','AVG_NMI')

fclose(fid);




