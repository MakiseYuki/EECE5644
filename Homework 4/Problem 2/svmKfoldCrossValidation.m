close all, clear all,
N = 1000; n = 2; K = 10; N2 = 1000;
mu(:,1) = [0;0]; %mu(:,2) = [1;0]; 
Sigma(:,:,1) = [1 0;0 1]; %Sigma(:,:,2) = [1 0;0 4];
p = [0.35,0.65]; % class priors for labels 0(negtive) and 1(positive) respectively
% Generate the first (negtive) samples
label = rand(1,N) >= p(1); l = 2*(label-0.5);
label_test = rand(1,N2) >= p(1); l2 = 2*(label_test-0.5);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
Nc2 = [length(find(label_test==0)),length(find(label_test==1))];

x = zeros(n,N); % reserve space
x_test = zeros(n,N2);

% Generate the second (positive) sameples
rad = 2 + rand(1,Nc(2));
ang = -pi + (2*pi)*rand(1,Nc(2));
x_pos = rad.*cos(ang);
y_pos = rad.*sin(ang);
x(1,find(label==1)) = x_pos;
x(2,find(label==1)) = y_pos;

% Generate the testing (positive) samples
rad = 2 + rand(1,Nc2(2));
ang = -pi + (2*pi)*rand(1,Nc2(2));
x_pos = rad.*cos(ang);
y_pos = rad.*sin(ang);
x_test(1,find(label_test==1)) = x_pos;
x_test(2,find(label_test==1)) = y_pos;

% Draw samples from each class pdf
x(:,label==0) = randGaussian(Nc(1),mu(:,1),Sigma(:,:,1));
x_test(:,label_test==0) = randGaussian(Nc2(1),mu(:,1),Sigma(:,:,1));

figure(1),
plot(x(1,find(label==0)),x(2,find(label==0)),'r.'); hold on,
plot(x(1,find(label==1)),x(2,find(label==1)),'b.');
title('Original data');
legend('Negative data','Positive data');
axis([-5 5 -5 5]);

% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; 
end
CList = 10.^linspace(-7,3,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
% 
% figure(1), subplot(1,2,1),
% plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
% xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
% title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
% [dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
% CBest= CList(indBestC); 
% SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
% d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
% indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
% indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
% figure(1), subplot(1,2,2), 
% plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
% plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
% title('Training Data (RED: Incorrectly Classified)'),
% pTrainingError_Linear = length(indINCORRECT)/N; % Empirical estimate of training error probability
% Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
% [h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
% figure(1), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,


% figure(2), subplot(1,2,1),
% plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
% xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
% title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
% [dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
% CBest= CList(indBestC); 
% SVMBest = fitcsvm(x_test',l2','BoxConstraint',CBest,'KernelFunction','linear');
% d = SVMBest.predict(x_test')'; % Labels of training data using the trained SVM
% indINCORRECT = find(l2.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
% indCORRECT = find(l2.*d == 1); % Find training samples that are correctly classified by the trained SVM
% figure(2), subplot(1,2,2), 
% plot(x_test(1,indCORRECT),x_test(2,indCORRECT),'g.'), hold on,
% plot(x_test(1,indINCORRECT),x_test(2,indINCORRECT),'r.'), axis equal,
% title('Training Data (RED: Incorrectly Classified)'),
% pTrainingError_testLinear = length(indINCORRECT)/N; % Empirical estimate of training error probability
% Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
% [h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
% figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
% 
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
% dummy = ceil(linspace(0,N,K+1));
% for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end
% CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
% for sigmaCounter = 1:length(sigmaList)
%     [sigmaCounter,length(sigmaList)],
%     sigma = sigmaList(sigmaCounter);
%     for CCounter = 1:length(CList)
%         C = CList(CCounter);
%         for k = 1:K
%             indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
%             xValidate = x(:,indValidate); % Using folk k as validation set
%             lValidate = l(indValidate);
%             if k == 1
%                 indTrain = [indPartitionLimits(k,2)+1:N];
%             elseif k == K
%                 indTrain = [1:indPartitionLimits(k,1)-1];
%             else
%                 indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
%             end
%             % using all other folds as training set
%             xTrain = x(:,indTrain); lTrain = l(indTrain);
%             SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
%             dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
%             indCORRECT = find(lValidate.*dValidate == 1); 
%             Ncorrect(k)=length(indCORRECT);
%         end 
%         PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
%     end 
% end
% figure(3), subplot(1,2,1),
% contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
% title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
% [dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
% CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
% SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
% d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
% indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
% indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
% figure(3), subplot(1,2,2), 
% plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
% plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
% title('Training Data (RED: Incorrectly Classified)'),
% pTrainingError_Gauss = length(indINCORRECT)/N; % Empirical estimate of training error probability
% Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
% [h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
% figure(3), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
% 
% figure(4), subplot(1,2,1),
% contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
% title('Gaussian-SVM Cross-Val Accuracy Estimate on Testing samples'), axis equal,
% [dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
% CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
% SVMBest = fitcsvm(x_test',l2','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
% d = SVMBest.predict(x_test')'; % Labels of training data using the trained SVM
% indINCORRECT = find(l2.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
% indCORRECT = find(l2.*d == 1); % Find training samples that are correctly classified by the trained SVM
% figure(4), subplot(1,2,2), 
% plot(x_test(1,indCORRECT),x_test(2,indCORRECT),'g.'), hold on,
% plot(x_test(1,indINCORRECT),x_test(2,indINCORRECT),'r.'), axis equal,
% title('Training Data (RED: Incorrectly Classified)'),
% pTrainingError_testGauss = length(indINCORRECT)/N; % Empirical estimate of training error probability
% Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
% [h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
% figure(4), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
% 
% 
% 
