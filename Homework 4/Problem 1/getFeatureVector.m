function featureVector = getFeatureVector(Image)
% clear all; close all;
% Image = imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg');
Image = im2double(Image);
feature = reshape(Image,size(Image,1)*size(Image,2),3); 
[ver,hor,page] = size(Image);
[pixels,colorDim] = size(feature);
featureVector = zeros(pixels,colorDim+2);

for i = 1:pixels
    featureVector(i,3) = feature(i,1);
    featureVector(i,4) = feature(i,2);
    featureVector(i,5) = feature(i,3);
    
    featureVector(i,1) = mod(i,size(Image,1));
    featureVector(i,2) = fix(i/size(Image,1))+1;
    
    if featureVector(i,1)==0
        featureVector(i,1) = ver;
        featureVector(i,2) = featureVector(i,2)-1;
    end
    
    featureVector(i,1) = (featureVector(i,1)-1)/ver;
    featureVector(i,2) = (featureVector(i,2)-1)/hor;
end