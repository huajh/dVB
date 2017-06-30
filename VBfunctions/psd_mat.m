function [ X,flag ] = psd_mat( mat )
%PSD Summary of this function goes here
%
%   to ensure the matrix being Positive Definite Matrix
%
%
    persistent psd_count;
    flag = 0;
    mat = (mat + mat')/2;
    [dum,p] = chol(mat);
    X = mat;
    if p == 0       
        return;
    end
    cnt = 0;
    while( p>0)
        cnt = cnt + 1;
        if cnt > 10
            flag = 1;
            break;
        end
        if isempty(psd_count)
            psd_count = 0;
        end
        psd_count = psd_count + 1;
        %fprintf('psd_count: %d\n', psd_count);
        [U,S] = eig(X);
        X = U*max(1e-5,S)*U';
        [dum,p] = chol(X);        
    end
end

