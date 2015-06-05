function [outImg,height,width,Idx]=PCNOutput(inImg,imgSize,unionType,patchSize,patchStep,numFilters,V,pooling,poolingSize,method)
% Version 1.000
%
% Code provided by Gan Yanhai, Liu Jun and Dong Junyu
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% =======INPUT=============
% this function can only process one sample a time
% inImg:input data,each column is a sample
% imgSize:size of input image
% unionType:organization of every layer
% patchSize:size of patches derived from input image
% patchStep:define the length to move when obtain patches from input
% numFilters:the number of filters of every group in a layer
% V:filters used to convolute
% pooling:declare whether to add a pooling layer after every
% convolution layer
% poolingSize:size of window to make pooling
% metho:define the method used to make pooling
% =======OUTPUT============
% outImg:output images after this layer
% height:height of output image
% widht:width of output image
% Idx:identify which filter this output image is from

addpath('./Utils')
[~,group]=size(unionType);
outImg = cell(numFilters*group,1); 
cnt = 1;
height=ceil((imgSize(1)-patchSize(1))/patchStep(1))+1;
width=ceil((imgSize(2)-patchSize(2))/patchStep(2))+1;
for j=1:group
    index=find(unionType(:,j));
    tempImg=zeros(imgSize);
    for k=1:length(index)
        tempImg=tempImg+inImg{index(k)};
    end
    tempImg=tempImg/length(index);
    im = im2col_general(tempImg,patchSize,patchStep);
    im = bsxfun(@minus, im, mean(im,2)); % patch-mean removal 
    for k=1:numFilters
        outImg{cnt}=reshape(V(:,cnt)'*im,height,width);
        cnt=cnt+1;
    end
end  
clear inImg;
if pooling
    %disp('start pooling');
    outImg=MaxPooling(outImg,[height,width],poolingSize,method);
    height=ceil(height/poolingSize(1));
    width=ceil(width/poolingSize(2));
end
Idx = 1:numFilters*group;