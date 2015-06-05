function V = GetFilters(imgs,patchSize,patchStep,numFilters)


addpath('./Utils')

num = numel(imgs);
RandIdx=randperm(num);

Rx = zeros(patchSize(1)*patchSize(2),patchSize(1)*patchSize(2));
for i = RandIdx 
    im = im2col_general(imgs{i},patchSize,patchStep); % collect all the patches of the ith image in a matrix
    imgs{i}=[];
    im = bsxfun(@minus, im, mean(im,2)); % patch-mean removal 
    Rx = Rx + im*im'; % sum of all the input images' covariance matrix
end
clear imgs;
Rx = Rx/(num*size(im,2));
[E,D] = eig(Rx);
[~,ind] = sort(diag(D),'descend');
V = E(:,ind(1:numFilters));  % principal eigenvectors 
clear Rx;



 



