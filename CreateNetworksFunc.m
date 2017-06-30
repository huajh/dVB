function [ Network ] = CreateNetworksFunc(Conf)
%CREATENETWORKS Summary of this function goes here
%   Detailed explanation goes here

    num = Conf.NodeNumber;
    square = Conf.Square;
    maxDist = Conf.CommDist;
    
    loc = square*rand(num,2) - square/2;        
     
    Dists = Euclid_Dist(loc(:,1),loc(:,2));
    
    % without self-loop
    Dists = Dists + 10*maxDist*eye(num);
    
    Neighbors = cell(num,1);
    maxDegree = 0;
    edges = 0;
    for i=1:num
        Neighbors{i} = find(Dists(i,:)<=maxDist);
        if length(Neighbors{i}) > maxDegree
            maxDegree = length(Neighbors{i});
        end
        edges = edges + length(Neighbors{i});
    end
   
    Nodes.loc = loc;
    Nodes.neighbors = Neighbors;
    
    Network.maxDegree = maxDegree;
    Network.edges = edges/2; %% undirected graph
    Network.Conf = Conf;
    Network.Nodes = Nodes;
end

function dist = Euclid_Dist(X,Y)
    len = length(X);
    xx = repmat(X,1,len);
    yy = repmat(Y,1,len);    
    dist = sqrt((xx-xx').^2+(yy-yy').^2);
end