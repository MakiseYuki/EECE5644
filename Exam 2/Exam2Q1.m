% clear all, close all,
% 
% plotData = 1;
% n = 3; Nsample = 1000; Ntest = 10000; 
% alpha = [0.15,0.35,0.2,0.3]; % must add to 1.0
% meanVectors = [5 -5 -5 5;5 5 -5 -5;5 5 5 5];
% covEvalues = [1.3^2 0 0;0 1.2^2 0;0 0 1.4^2];
% covEvectors(:,:,1) =  0.8*[5 1 2;1 5 0;2 0 3]/sqrt(2);
% covEvectors(:,:,2) =  0.8*[5 1 2;1 5 0;2 0 3]/sqrt(2);
% covEvectors(:,:,3) =  0.8*[5 1 2;1 5 0;2 0 3]/sqrt(2);
% covEvectors(:,:,4) =  0.8*[5 1 2;1 5 0;2 0 3]/sqrt(2);
% 
% t = rand(1,Nsample);
% ind1 = find(0 <= t & t <= alpha(1));
% ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
% ind3 = find(alpha(1)+alpha(2) < t & t <= alpha(1)+alpha(2)+alpha(3));
% ind4 = find(alpha(1)+alpha(2)+alpha(3) < t & t <= 1);
% xSample = zeros(n,Nsample);
% xSample(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+ meanVectors(:,1);
% xSample(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+ meanVectors(:,2);
% xSample(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+ meanVectors(:,3);
% xSample(:,ind4) = covEvectors(:,:,4)*covEvalues^(1/2)*randn(n,length(ind4))+ meanVectors(:,4);
% 
% trueLabel = zeros(1,Nsample);
% trueLabel(:,ind1) = 1;
% trueLabel(:,ind2) = 2;
% trueLabel(:,ind3) = 3;
% trueLabel(:,ind4) = 4;
% 
% MAPLabel_classified = MAP_Classifier(xSample,meanVectors,covEvectors,alpha,4);
% 
% % Calculate the misclassified samples
% ind11 = find(MAPLabel_classified==1 & trueLabel==1);
% ind12 = find(MAPLabel_classified==2 & trueLabel==1);
% ind13 = find(MAPLabel_classified==3 & trueLabel==1);
% ind14 = find(MAPLabel_classified==4 & trueLabel==1);
% ind21 = find(MAPLabel_classified==1 & trueLabel==2);
% ind22 = find(MAPLabel_classified==2 & trueLabel==2);
% ind23 = find(MAPLabel_classified==3 & trueLabel==2);
% ind24 = find(MAPLabel_classified==4 & trueLabel==2);
% ind31 = find(MAPLabel_classified==1 & trueLabel==3);
% ind32 = find(MAPLabel_classified==2 & trueLabel==3);
% ind33 = find(MAPLabel_classified==3 & trueLabel==3);
% ind34 = find(MAPLabel_classified==4 & trueLabel==3);
% ind41 = find(MAPLabel_classified==1 & trueLabel==4);
% ind42 = find(MAPLabel_classified==2 & trueLabel==4);
% ind43 = find(MAPLabel_classified==3 & trueLabel==4);
% ind44 = find(MAPLabel_classified==4 & trueLabel==4);
% 
% confusion_matrix = [length(ind11) length(ind12) length(ind13) length(ind14);...
%                     length(ind21) length(ind22) length(ind23) length(ind24);...
%                     length(ind31) length(ind32) length(ind33) length(ind34);...
%                     length(ind41) length(ind42) length(ind43) length(ind44)];
% 
% pError1 = (size(ind12,2)+size(ind13,2)+size(ind14,2))/size(ind1,2);
% pError2 = (size(ind21,2)+size(ind23,2)+size(ind24,2))/size(ind2,2);
% pError3 = (size(ind31,2)+size(ind32,2)+size(ind34,2))/size(ind3,2);
% pError4 = (size(ind41,2)+size(ind42,2)+size(ind43,2))/size(ind4,2);
% accuracy_rate = (confusion_matrix(1,1)+confusion_matrix(2,2)+confusion_matrix(3,3)+confusion_matrix(4,4))/length(xSample(1,:));
% 
% 
% testData_10000 = xSample;
% testLabel_10000 = trueLabel;
% save('testData_10000.mat','testData_10000','testLabel_10000');


figure(2), subplot(2,2,1),
plot3(xSample(1,:),xSample(2,:),xSample(3,:),'.g'),hold on,
plot3(xSample(1,ind12),xSample(2,ind12),xSample(3,ind12),'*c'),hold on,
plot3(xSample(1,ind13),xSample(2,ind13),xSample(3,ind13),'*r'),hold on,
plot3(xSample(1,ind14),xSample(2,ind14),xSample(3,ind14),'*m'),hold on,
legend('Samples = 1000','Classified Label = 2','Classified Label = 3','Classified Label = 4');
title('Wrong Decision in True Label = 1 ');
subplot(2,2,2),
plot3(xSample(1,:),xSample(2,:),xSample(3,:),'.g'),hold on,
plot3(xSample(1,ind21),xSample(2,ind21),xSample(3,ind21),'*c'),hold on,
plot3(xSample(1,ind23),xSample(2,ind23),xSample(3,ind23),'*r'),hold on,
plot3(xSample(1,ind24),xSample(2,ind24),xSample(3,ind24),'*m'),
legend('Samples = 1000','Classified Label = 1','Classified Label = 3','Classified Label = 4');
title('Wrong Decision in True Label = 2 ');
subplot(2,2,3),
plot3(xSample(1,:),xSample(2,:),xSample(3,:),'.g'),hold on,
plot3(xSample(1,ind31),xSample(2,ind31),xSample(3,ind31),'*c'),hold on,
plot3(xSample(1,ind32),xSample(2,ind32),xSample(3,ind32),'*r'),hold on,
plot3(xSample(1,ind34),xSample(2,ind34),xSample(3,ind34),'*m'),
legend('Samples = 1000','Classified Label = 1','Classified Label = 2','Classified Label = 4');
title('Wrong Decision in True Label = 3 ');
subplot(2,2,4),
plot3(xSample(1,:),xSample(2,:),xSample(3,:),'.g'),hold on,
plot3(xSample(1,ind41),xSample(2,ind41),xSample(3,ind41),'*c'),hold on,
plot3(xSample(1,ind42),xSample(2,ind42),xSample(3,ind42),'*r'),hold on,
plot3(xSample(1,ind43),xSample(2,ind43),xSample(3,ind43),'*m'),
legend('Samples = 1000','Classified Label = 1','Classified Label = 2','Classified Label = 3');
title('Wrong Decision in True Label = 4 ');


if plotData == 1
    figure(1), 
    plot3(xSample(1,ind1),xSample(2,ind1),xSample(3,ind1),'.r'),hold on,
    plot3(xSample(1,ind2),xSample(2,ind2),xSample(3,ind2),'.b'),hold on,
    plot3(xSample(1,ind3),xSample(2,ind3),xSample(3,ind3),'.g'),hold on,
    plot3(xSample(1,ind4),xSample(2,ind4),xSample(3,ind4),'.m'),
    title('Setting Data Distribution and plotting Samples = 1000'), axis equal,
    legend('Class 1','Class 2','Class 3','Class 4');
end
