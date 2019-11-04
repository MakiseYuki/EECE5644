clear all; close all;
% x in, y comes out, y is the value from 0 to 1
% w is vector, v is scaler, train the model from the given x
% two class --- positive and negitive
% x as data process to the classifier out put y
% y(x) = P(label = positive|x)

N = 999;
mu = [1 -1;1 -1];
Sigma(:,:,1) = 0.1*[15 2;3 2];
Sigma(:,:,2) = 0.1*[6 7;3 17]; % Sigma should be no diagonal with distinct eigenvalues
classPriors = [0.3,0.7]; thr = [0,cumsum(classPriors)];
d = size(mu,1);
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);

for m = 1:length(classPriors)
    ind = find(thr(m)<u & u<=thr(m+1));     
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(:,ind) = repmat(m,1,length(ind)); % label 1 is negtive 2 is positive
end

figure(1),
plot(x(1,find(labels==1)),x(2,labels==1),'b.'); hold on,
plot(x(1,find(labels==2)),x(2,find(labels==2)),'r.');
title('True Data of two classes')
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');

% The following is the process of LDA

x1_indices = find(labels==1);
x2_indices = find(labels==2);
x1 = x(:,x1_indices); 
x2 = x(:,x2_indices);

N1 = length(x1);
N2 = length(x2);

mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,indicesFDA] = sort(diag(D),'descend');
w = V(:,indicesFDA(1)); % Fisher LDA projection vector
b = 0;
dis_1 = w'*x1;
dis_2 = w'*x2;
% dis_1 = sum(dis_1);
% dis_2 = sum(dis_2);
% 
figure(2),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
title('True data of two classes');
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');
axis equal,

subplot(2,1,2), 
plot(dis_1(1,:),zeros(1,N1),'r*');
hold on;
plot(dis_2(1,:),zeros(1,N2),'bo');
title('Fisher LDA projection');
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');
axis equal,

% Train the parameter of y(x) = 1/(1+exp(dis_1,dis_2))

y1 = 1./(1 + exp(dis_1)); %negative
y2 = 1./(1 + exp(dis_2)); %positive : modeling 


fun = @(z)sum(sum(log(1./(1 + exp(z'*x2)))));
z0 = w;
z = fminsearch(fun,z0);

% 
% newW = [z(1,1,1);z(1,1,2)];
% newB = [z(1,2,1);z(1,2,2)];

dis_t1 = (z'*x1)/N1;
dis_t2 = (z'*x2)/N2;

% MAP classifier
Nc = [length(find(labels==1)),length(find(labels==2))];
discriminantScore = [evalGaussian(x,mu(:,1),Sigma(:,:,1));evalGaussian(x,mu(:,2),Sigma(:,:,2))]';

for i = 1:N
    [val,loc] = max(discriminantScore(i,:));
    decision(i) = loc;
end

confusion_matrix = zeros(2,2);
confusion_matrix(1,1) = length(find(decision==1&labels==1));
confusion_matrix(1,2) = length(find(decision==2&labels==1));
confusion_matrix(2,1) = length(find(decision==1&labels==2));
confusion_matrix(2,2) = length(find(decision==2&labels==2));
total_number_hit = confusion_matrix(1,1) + confusion_matrix(2,2);
total_number_miss = N - total_number_hit;

errorRate_MAP = total_number_miss/N;
% dis_t1 = sum(dis_t1);
% dis_t2 = sum(dis_t2);

% for i = 1:d
%     
%     t_FDA = find(dis_2(i,:) < 0);
%     t_traning = find(dis_t2(i,:)<0);
%     traning_error = length(t_traning)/N2;
%     errorFDA = length(t_FDA)/N2;
%     
%     figure(3),
%     semilogy(i,traning_error,'r+'), hold on,
%     semilogy(i,errorFDA,'b*'), hold on,
%     axis equal
% end

% Disply the FDA classifier
neg_corr_FDA = find(dis_t1 < 0);
pos_corr_FDA = find(dis_t2 > 0);
neg_pos_FDA = find(dis_1 > 0);
pos_neg_FDA = find(dis_2 < 0);
error_FDA = length(neg_pos_FDA) + length(pos_neg_FDA);
errorRate_FDA = error_FDA/N;

figure(3),
%plot(x(1,:),x(2,:),'g.'); hold on,
plot(x(1,x1_indices(neg_corr_FDA)),x(2,x1_indices(neg_corr_FDA)),'go'); hold on,
plot(x(1,x2_indices(pos_corr_FDA)),x(2,x2_indices(pos_corr_FDA)),'gx'); hold on,
plot(x(1,x1_indices(neg_pos_FDA)),x(2,x1_indices(neg_pos_FDA)),'ro'); hold on,
plot(x(1,x2_indices(pos_neg_FDA)),x(2,x2_indices(pos_neg_FDA)),'rx'); hold on,
title('Fisher FDA classifier'); text(2,-1,"Error rate " + errorRate_FDA); text(2,-2,"Error count: " + error_FDA);
legend('Class Negative Correct','Class Positive Correct','Negtive determined as Positive','Positive determined as Negtive');
axis equal;

% Display the MAP classifier
figure(4),
plot(x(1,find(decision==1&labels==1)),x(2,find(decision==1&labels==1)),'go'); hold on,
plot(x(1,find(decision==2&labels==2)),x(2,find(decision==2&labels==2)),'gx'); hold on,
plot(x(1,find(decision==2&labels==1)),x(2,find(decision==2&labels==1)),'ro'); hold on,
plot(x(1,find(decision==1&labels==2)),x(2,find(decision==1&labels==2)),'rx'); hold on,
title('MAP classifier'); text(2,-1,"Error rate " + errorRate_MAP); text(2,-2,"Error count: " + total_number_miss);
legend('Class Negative Correct','Class Positive Correct','Negtive determined as Positive','Positive determined as Negtive')
axis equal;


% Display the LOG classifier
neg_corr_LOG = find(dis_t1 < 0);
pos_corr_LOG = find(dis_t2 > 0);
neg_pos_LOG = find(dis_t1 > 0);
pos_neg_LOG = find(dis_t2 < 0);
error_LOG = length(neg_pos_LOG) + length(pos_neg_LOG);
errorRate_LOG= error_LOG/N;

figure(5),
%plot(x(1,:),x(2,:),'g.'); hold on,
plot(x(1,x1_indices(neg_corr_LOG)),x(2,x1_indices(neg_corr_LOG)),'go'); hold on,
plot(x(1,x2_indices(pos_corr_LOG)),x(2,x2_indices(pos_corr_LOG)),'gx'); hold on,
plot(x(1,x1_indices(neg_pos_LOG)),x(2,x1_indices(neg_pos_LOG)),'ro'); hold on,
plot(x(1,x2_indices(pos_neg_LOG)),x(2,x2_indices(pos_neg_LOG)),'rx'); hold on,
title('LOG classifier'); text(2,-1,"Error rate " + errorRate_LOG); text(2,-2,"Error count: " + error_LOG);
legend('Class Negative Correct','Class Positive Correct','Negtive determined as Positive','Positive determined as Negtive');
axis equal;

figure(6),
plot(dis_t1(1,:),zeros(1,N1),'r*');
hold on;
plot(dis_t2(1,:),zeros(1,N2),'bo');
title('LOG classifier projection');
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');
axis equal,



