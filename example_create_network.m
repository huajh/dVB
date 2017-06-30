clc;
close all;
clear;

Conf.Square = sqrt(5);
Conf.NodeNumber = 10;
Conf.CommDist = 0.8;

is_create_network = 1;
if is_create_network == 1
    Network = CreateNetworksFunc(Conf);
    save Network_10.mat Network
else
    load Network_10.mat
end

nodenum = size(Network.Nodes.loc,1);
lap_matrix = zeros(nodenum);
for i=1:nodenum
    idx = Network.Nodes.neighbors{i};
    lap_matrix(i,idx) = -1;
    lap_matrix(i,i) = length(idx);
end
eig_val = eig(lap_matrix);
eig_val = sort(eig_val,'ascend');
algeb_conn = eig_val(2) % algebraic connectivity
avg_deg = sum(diag(lap_matrix))/nodenum   % average values

DrawNetworks(Network);