clear all; close all;

image_folder = '/Users/arsen/Documents/GitHub/EECE5644_Machine-Learning/Homework 4/Problem 1';
file_names = dir(fullfile(image_folder,'*.jpg'));
total_images = numel(file_names);

imagePlane = imread(fullfile(image_folder,file_names(1).name));
imageBird = imread(fullfile(image_folder,file_names(2).name));
Image = imagePlane;

for i = 1:4
K = i+1;

maskKMean = kMeanCluster(Image,K);
maskGMM = kGaussian_color_EM(Image,K);
maskKMean = rgb2gray(maskKMean);
maskGMM = rgb2gray(maskGMM);

figure(1),
subplot(4,3,3*i-2); imshow(Image); title('original');
subplot(4,3,3*i-1); imshow(maskKMean); title("K mean segmentation with " + K + " Means");
subplot(4,3,3*i); imshow(maskGMM); title("GMM segamentation with " + K + " Gaussian");
disp('number of segments ='); disp(K), hold on,
end