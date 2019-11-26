clear all; close all;


plotData = 1;
n = 2; Ntrain = 1000; Ntest = 10000; K = 10;
alpha = [0.33,0.34,0.33]; % must add to 1.0
meanVectors = [-18 0 18;-8 0 8];
covEvalues = [3.2^2 0;0 0.6^2];
covEvectors(:,:,1) = [1 -1;1 1]/sqrt(2);
covEvectors(:,:,2) = [1 0;0 1];
covEvectors(:,:,3) = [1 -1;1 1]/sqrt(2);

t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

Xlabel = zeros(1,Ntrain);
Xlabel(:,ind1) = 1;
Xlabel(:,ind2) = 2;
Xlabel(:,ind3) = 3;

t = rand(1,Ntest);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtest = zeros(n,Ntrain);
Xtest(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtest(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtest(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

x = Xtrain(1,:);
t = Xtrain(2,:);


% create the network with 1 layer
net = feedforwardnet(10);
net.performParam.regularization = 0.01;
net = train(net, x, t);
y = net(x);
perf = perform(net, t, y);

view(net);