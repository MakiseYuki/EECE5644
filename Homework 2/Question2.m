clear all;
close all;

%Sample 1
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [1 0;0 1];
sigma(:,:,2) = [1 0;0 1];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 1'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 2
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [3 1;1 0.8];
sigma(:,:,2) = [3 1;1 0.8];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 2'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(2), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 3
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [2;2];
sigma(:,:,1) = [2 0.5;0.5 1];
sigma(:,:,2) = [2 0.5;0.5 1];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(3), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 3'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(3), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(3), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 4
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [1 0;0 1];
sigma(:,:,2) = [1 0;0 1];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(4), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 4'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(4), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(4), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 5
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [3 1;1 0.8];
sigma(:,:,2) = [3 1;1 0.8];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(5), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 5'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(5), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(5), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 6
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [2;2];
sigma(:,:,1) = [2 0.5;0.5 1];
sigma(:,:,2) = [2 0.5;0.5 1];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(6), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 6'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

figure(6), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(6), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

function g = evalGaussian(x,mu,Sigma)
    %Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end