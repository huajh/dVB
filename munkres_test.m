% Example 1: a 5 x 5 example

[assignment,cost] = munkres(magic(5));
[assignedrows,dum]=find(assignment);
disp(assignedrows'); % 3 2 1 5 4
disp(cost); %15

% Example 2: 400 x 400 random data

% n=400;
% A=rand(n);
% tic
% [a,b]=munkres(A);
% toc                 % about 6 seconds 
