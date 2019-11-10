clear all; close all;

image_folder = '/Users/arsen/Documents/GitHub/EECE5644_Machine-Learning/Homework 4/Problem 1';
file_names = dir(fullfile(image_folder,'*.jpg'));
total_images = numel(file_names);

imagePlane = imread(fullfile(image_folder,file_names(1).name));
imageBird = imread(fullfile(image_folder,file_names(2).name));
Image = imagePlane;

K = 5;

maskKMean = kMeanCluster(Image,K);
maskGMM = kGaussian_color_EM(Image,K);

figure(1),
subplot(1,3,1); imshow(Image); title('original');
subplot(1,3,2); imshow(maskKMean); title('K mean segmentation');
subplot(1,3,3); imshow(maskGMM); title('GMM segamentation');
disp('number of segments ='); disp(K)
