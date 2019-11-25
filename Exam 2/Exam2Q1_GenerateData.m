clear all, close all,

plotData = 1;
n = 3; Ntrain = 1000; 
alpha = [0.15,0.25,0.30,0.30]; % must add to 1.0
meanVectors = [-10 -10 10 10;-10 10 -10 10;10 10 10 10];
covEvalues = [3.2^2 0 0;0 0.6^2 0;0 0 1.2^2];
covEvectors(:,:,1) =  0.7*[3 5 4;-2 10 9;1 2 4]/sqrt(2);
covEvectors(:,:,2) =  0.7*[3 5 4;-2 10 9;1 2 4]/sqrt(2);
covEvectors(:,:,3) =  0.7*[3 5 4;-2 10 9;1 2 4]/sqrt(2);
covEvectors(:,:,4) =  0.7*[3 5 4;-2 10 9;1 2 4]/sqrt(2);

t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= alpha(1)+alpha(2)+alpha(3));
ind4 = find(alpha(1)+alpha(2)+alpha(3) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Ltrain = zeros(1,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Ltrain(ind1) = 1;
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Ltrain(ind2) = 2;
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);
Ltrain(ind3) = 3;
Xtrain(:,ind4) = covEvectors(:,:,4)*covEvalues^(1/2)*randn(n,length(ind4))+meanVectors(:,4);
Ltrain(ind4) = 4;

% MAP classify
Ltrain_classified = MAP_Classifier(Xtrain,meanVectors,covEvectors,alpha,4);
% Calculate the misclassified samples
ind12 = find(Ltrain_classified==2 & Ltrain==1);
ind13 = find(Ltrain_classified==3 & Ltrain==1);
ind14 = find(Ltrain_classified==4 & Ltrain==1);
ind21 = find(Ltrain_classified==1 & Ltrain==2);
ind23 = find(Ltrain_classified==3 & Ltrain==2);
ind24 = find(Ltrain_classified==4 & Ltrain==2);
ind31 = find(Ltrain_classified==1 & Ltrain==3);
ind32 = find(Ltrain_classified==2 & Ltrain==3);
ind34 = find(Ltrain_classified==4 & Ltrain==3);
ind41 = find(Ltrain_classified==1 & Ltrain==4);
ind42 = find(Ltrain_classified==2 & Ltrain==4);
ind43 = find(Ltrain_classified==3 & Ltrain==4);

Pe1 = (size(ind12,2)+size(ind13,2)+size(ind14,2))/size(ind1,2)
Pe2 = (size(ind21,2)+size(ind23,2)+size(ind24,2))/size(ind2,2)
Pe3 = (size(ind31,2)+size(ind32,2)+size(ind34,2))/size(ind3,2)
Pe4 = (size(ind41,2)+size(ind42,2)+size(ind43,2))/size(ind4,2)
% % Save Xtrain dataset
% fid = fopen('Xtrain_1000_3.txt','wt');
% fid2 = fopen('Ltrain_1000_3.txt','wt');
% fprintf(fid,'%g %g %g\n',Xtrain(1,:),Xtrain(2,:),Xtrain(3,:)); 
% fprintf(fid2,'%g\n',Ltrain); 
% fclose(fid);
% fclose(fid2);

% Draw Xtrain and misclassified samples
figure(2), subplot(2,2,1),
plot3(Xtrain(1,:),Xtrain(2,:),Xtrain(3,:),'.g'),hold on,
plot3(Xtrain(1,ind12),Xtrain(2,ind12),Xtrain(3,ind12),'xb'),hold on,
plot3(Xtrain(1,ind13),Xtrain(2,ind13),Xtrain(3,ind13),'xr'),hold on,
plot3(Xtrain(1,ind14),Xtrain(2,ind14),Xtrain(3,ind14),'xm'),hold on,
legend('All samples','Classified Label = 2','Classified Label = 3','Classified Label = 4');
title('Wrong Classification of True Label = 1 ');
subplot(2,2,2),
plot3(Xtrain(1,:),Xtrain(2,:),Xtrain(3,:),'.g'),hold on,
plot3(Xtrain(1,ind21),Xtrain(2,ind21),Xtrain(3,ind21),'xb'),hold on,
plot3(Xtrain(1,ind23),Xtrain(2,ind23),Xtrain(3,ind23),'xr'),hold on,
plot3(Xtrain(1,ind24),Xtrain(2,ind24),Xtrain(3,ind24),'xm'),
legend('All samples','Classified Label = 1','Classified Label = 3','Classified Label = 4');
title('Wrong Classification of True Label = 2 ');
subplot(2,2,3),
plot3(Xtrain(1,:),Xtrain(2,:),Xtrain(3,:),'.g'),hold on,
plot3(Xtrain(1,ind31),Xtrain(2,ind31),Xtrain(3,ind31),'xb'),hold on,
plot3(Xtrain(1,ind32),Xtrain(2,ind32),Xtrain(3,ind32),'xr'),hold on,
plot3(Xtrain(1,ind34),Xtrain(2,ind34),Xtrain(3,ind34),'xm'),
legend('All samples','Classified Label = 1','Classified Label = 2','Classified Label = 4');
title('Wrong Classification of True Label = 3 ');
subplot(2,2,4),
plot3(Xtrain(1,:),Xtrain(2,:),Xtrain(3,:),'.g'),hold on,
plot3(Xtrain(1,ind41),Xtrain(2,ind41),Xtrain(3,ind41),'xb'),hold on,
plot3(Xtrain(1,ind42),Xtrain(2,ind42),Xtrain(3,ind42),'xr'),hold on,
plot3(Xtrain(1,ind43),Xtrain(2,ind43),Xtrain(3,ind43),'xm'),
legend('All samples','Classified Label = 1','Classified Label = 2','Classified Label = 3');
title('Wrong Classification of True Label = 4 ');


if plotData == 1
    figure(1), 
    plot3(Xtrain(1,ind1),Xtrain(2,ind1),Xtrain(3,ind1),'.r'),hold on,
    plot3(Xtrain(1,ind2),Xtrain(2,ind2),Xtrain(3,ind2),'.b'),hold on,
    plot3(Xtrain(1,ind3),Xtrain(2,ind3),Xtrain(3,ind3),'.g'),hold on,
    plot3(Xtrain(1,ind4),Xtrain(2,ind4),Xtrain(3,ind4),'.m'),
    title('Training Data'), axis equal,
end

% 
% 


