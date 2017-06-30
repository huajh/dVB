function [ hypers,flag ] = projection( hypers )
%PROJECTION Summary of this function goes here
%   Detailed explanation goes here
    [dum1,dum2,K] = size(hypers.invW);
    flag = 0;
    for i=1:K
        [hypers.invW(:,:,i),flag] = psd_mat(hypers.invW(:,:,i));
        if flag == 1
            break;
        end
    end
end

