function fig = DrawNetworks( Network )
%DRAWNETWORKS Summary of this function goes here
%   Detailed explanation goes here

num = Network.Conf.NodeNumber;
%maxDegree = Network.maxDegree;
loc = Network.Nodes.loc;
square = Network.Conf.Square;
Neighbors = Network.Nodes.neighbors;

fig = figure;
plot(loc(:,1),loc(:,2),'ro','MarkerSize',8,'LineWidth',2);
axis([-square/2,square/2,-square/2,square/2]);   
for i=1:num
    for k = 1:length(Neighbors{i})
        j = Neighbors{i}(k);
        %     c = num2str(Dists(i,j),'%.2f');
        %     text((loc(i,1) + loc(j,1))/2,(loc(i,2) + loc(j,2))/2,c,'Fontsize',10);
        %     hold on;
        line([loc(i,1),loc(j,1)],[loc(i,2),loc(j,2)],'LineWidth',0.8,'Color','b');
    end
end
set(gcf, 'Color', 'w');
end

