%% EECE5644 - Homework 2 - Question 2 and 3
clear all; close all; clc;
rng('default');

%% Dataset 1
nSamples = 400;
mu{1} = [0,0]; mu{2} = [3,3];
sigma{1} = eye(2); sigma{2} = eye(2);
prior = [0.5; 0.5];
xMin = -3; xMax = 6;
yMin = -3; yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,1);
titleString = 'Part 1';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,1);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,1);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

%% Dataset 2
nSamples = 400;
mu{1} = [0,0]; mu{2} = [3,3];
sigma{1} = [3, 1; 1, 0.8]; sigma{2} = [3, 1; 1, 0.8];
prior = [0.5; 0.5];
xMin = -5; xMax = 7;
yMin = -3; yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,2);
titleString = 'Part 2';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,2);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,2);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

%% Dataset 3
nSamples = 400;
mu{1} = [0,0]; mu{2} = [2,2];
sigma{1} = [2 0.5; 0.5 1]; sigma{2} = [2 -1.9; -1.9 5];
prior = [0.5; 0.5];
xMin = -5; xMax = 7;
yMin = -4; yMax = 7;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,3);
titleString = 'Part 3';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,3);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,3);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

%% Dataset 4
nSamples = 400;
mu{1} = [0,0]; mu{2} = [3,3];
sigma{1} = eye(2); sigma{2} = eye(2);
prior = [0.05; 0.95];
xMin = -3; xMax = 6;
yMin = -3; yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,4);
titleString = 'Part 4';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,4);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,4);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

%% Dataset 5
nSamples = 400;
mu{1} = [0,0]; mu{2} = [3,3];
sigma{1} = [3, 1; 1, 0.8]; sigma{2} = [3, 1; 1, 0.8];
prior = [0.05; 0.95];
xMin = -5; xMax = 7;
yMin = -3; yMax = 6;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,5);
titleString = 'Part 5';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,5);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,5);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

%% Dataset 6
nSamples = 400;
mu{1} = [0,0]; mu{2} = [2,2];
sigma{1} = [2 0.5; 0.5 1]; sigma{2} = [2 -1.9; -1.9 5];
prior = [0.05; 0.95];
xMin = -5; xMax = 7;
yMin = -4; yMax = 7;

nClass = numel(mu);
[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); subplot(2,3,6);
titleString = 'Part 6';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% MAP Classification and Visualization for Question 2
[ind01MAP,ind10MAP,ind00MAP,ind11MAP,pEminERM] = classifyMAP(data, classIndex, mu, sigma, nSamples, prior);
figure(2); subplot(2,3,6);
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('MAP Pe=%.4f',pEminERM), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(3); subplot(2,3,6);
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pEminLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);
