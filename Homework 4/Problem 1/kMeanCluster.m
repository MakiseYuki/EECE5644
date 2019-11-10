function mask = kMeanCluster(Image,K)
iteration = 10;
% K-means clustering
Image = im2double(Image);

featureVector_5 = getFeatureVector(Image);

featureVector = featureVector_5(:,3:5);    
center = featureVector( ceil(rand(K,1)*size(featureVector,1)) ,:);
disLabel = zeros(size(featureVector,1),K+2);

for n = 1:iteration
   for i = 1:size(featureVector,1)
      for j = 1:K  
        disLabel(i,j) = norm(featureVector(i,:) - center(j,:));      
      end
      [Distance, CN] = min(disLabel(i,1:K));               % 1:K are Distance from Cluster Centers 1:K 
      disLabel(i,K+1) = CN;                                % K+1 is Cluster Label
      disLabel(i,K+2) = Distance;                          % K+2 is Minimum Distance
   end
   for i = 1:K
      A = (disLabel(:,K+1) == i);                          % Cluster K Points
      center(i,:) = mean(featureVector(A,:));                      % New Cluster Centers
      if sum(isnan(center(:))) ~= 0                    % If CENTS(i,:) Is Nan Then Replace It With Random Point
         NC = find(isnan(center(:,1)) == 1);           % Find Nan Centers
         for Ind = 1:size(NC,1)
         center(NC(Ind),:) = reatureVector(randi(size(featureVector,1)),:);
         end
      end
   end
end
X = zeros(size(featureVector));
for i = 1:K
idx = find(disLabel(:,K+1) == i);
X(idx,:) = repmat(center(i,:),size(idx,1),1); 
end
T = reshape(X,size(Image,1),size(Image,2),3);
Trans = zeros(size(Image,1),size(Image,2),3);

Trans = T;
mask = Trans;

% figure()
% subplot(121); imshow(Image); title('original')
% subplot(122); imshow(Trans); title('segmented')
% disp('number of segments ='); disp(K)