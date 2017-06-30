clear all;
load('COIL20.mat');	
% 
%
% feature:      fea 1440 x 1024
% groundturth:  gnd 1440 x 1
%
% It contains 20 objects. The images of each objects were taken
% 5 degrees apart as the object is rotated on a turntable and
% each object has 72 images. The size of each image is 32  32
% pixels, with 256 gray levels per pixel. Thus, each image is
% represented by a 1,024-dimensional vector.

data = NormalizeFea(fea); % nsample x dim
score = 0.90;

[nsample,dim] = size(data);

xbar = mean(data,1);
means = bsxfun(@minus, data, xbar);
cov = means'*means/nsample;
[V,D] = eig(cov);
eigval = diag(D);
[~,idx] = sort(eigval,'descend');
eigval = eigval(idx);
V = V(idx,:);
p = 0;
for i=1:dim
   perc = sum(eigval(1:i))/sum(eigval);
   if perc > score
       p = i
       break;       
   end 
end

E = V(1:p,:);

fea = means*E';


save COIL20_PCA.mat fea gnd



