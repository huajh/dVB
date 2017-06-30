function [ prob ] = GaussPDF(Data,M,Sigma)
%  Calculate Gaussian Probability Distribution Function
%   Data  dim x N
%   M     dim x 1
%   Sigma dim x dim

    [dim,N] = size(Data);
    Data = Data'-repmat(M',N,1);
    prob = sum((Data/Sigma).*Data,2); % Data*inv(Sigma)
    prob = exp(-0.5*prob)/sqrt((2*pi)^dim*(abs(det(Sigma))+realmin));
    
end