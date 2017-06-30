function [ X] = psd_ls( Y )
%PROX_OPT Summary of this function goes here
%   Detailed explanation goes here

%       Positive Definte Least Squares Problem 
%       
%       min_X  trace(X'*X) - trace(Y'*X)
%       subject to X is positive definite matrix
%       X,Y are symmetric n-by-n matrices
%     
%       @author: Junhao Hua
%       @email:  huajh7@gmail.com
%       Latest update: 2015/4/16
%     
%       using Accelerated Projected Gradient(APG) algorithm
    
    [~,S] = eig(Y);    
    if isempty(find(diag(S)<0))
        X = Y;
        return;
    end
    
    persistent psd_ls_count;
    if isempty(psd_ls_count)
        psd_ls_count = 0;
    end
    psd_ls_count = psd_ls_count + 1;
    fprintf('psd_ls_count: %d\n', psd_ls_count);
    n = size(Y,1);
    S = eye(n); % search points
    X0 = eye(n);
    maxiter = 100;
   tau = 0.1;
    for i=1:maxiter
        % linear search tau        
        %tau = backtrack_ls(tau,S,Y);
        mat = S - tau*(S - Y);
        [U,E] = eig(mat);
        if ~isempty(find(diag(E)<0))
            X = U*max(1e-8,E)*U';        
        else
            X = mat;
        end        
        f(i) = func(X,Y);
        % stopping criterion: ||X-X0||_F^2 < epson        
        if sum(sum(X-X0).^2)<1e-10
            break;
        end        
        
        S = X + (i-1)/(i+2)*(X - X0);
        X0 = X;
    end
    
   % plot(1:size(f,2),f,'b-');
    
end

function f = func(X,Y)
    f = trace(X'*X) - trace(Y'*X); 
end

function tau = backtrack_ls(tau,S,Y)          
    while(1)
        Delta = S - Y;        
        mat = S - tau*Delta;        
        [U,E] = eig(mat);
        if ~isempty(find(diag(E)<0))
            Z = U*max(0,E)*U';        
        else
            Z = mat;
        end                
        if func(S,Y) + Delta*(Z-S) + 1/(2*tau)*sum(sum(Z-S)) < func(Z,Y)
            tau = tau/2;
        else
            break;
        end
    end
        
    
end

