function [ftrain,model]=PCNTrain(trainData,Option)
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

% This function is used to train the network
% trainData is the input data to train the network,it consists of many
% different columns and each column is a vectorized input sample ex. an
% image.
% =======INPUT=============
% Option contains the parameters used to set up the network,following are the list of its components:
% Option.imgSize:size of input image
% Option.numStage:layers of the network
% Option.numFilters:the number of filters of every group in a layer
% Option.patchSize:size of patches derived from input image
% Option.patchStep:define the length to move when obtain patches from input
% image in two directions
% Option.pooling:declare whether to add a pooling layer after every
% convolution layer
% Option.poolingSize:size of window to make pooling
% Option.poolingMethos:define the method used to make pooling
% Option.unionType:organization of every layer
% =======OUTPUT============
% ftrain:the output features of trainData through the net
% mode:save the filters of ecah layer


sampleNum=numel(trainData);
orgnHeight=Option.imgSize(1);
orgnWidth=Option.imgSize(2);
patchSize=Option.patchSize;
patchStep=Option.patchStep;

disp('stage 1');
disp('calculate the filters');
V=GetFilters(trainData,Option.patchSize,Option.patchStep,Option.numFilters(1));
model.V{1}=V;
% deal with every layer
for i=2:Option.numStage
    disp(['stage ' num2str(i)]);
    num=Option.numFilters(i); %number of filters of every group in layer i
    model.V{i}=[];
    disp('calculate the filters');
    
    % first get the type of organization,the columns of
    % matrix type is the number of groups in this layer and each column
    % corresponds a group
    % for each group it produce numFilters filters based on the input
    % from the filters identified by one in the column
    type=Option.unionType{i};  
    group=size(type,2);    
    Rx = zeros(patchSize(1)*patchSize(2),patchSize(1)*patchSize(2),group);
    for k=1:sampleNum
        outImg=trainData(k);
        height=orgnHeight;
        width=orgnWidth;
        for l=1:i-1
            [outImg,height,width,Idx]=PCNOutput(outImg,[height width],Option.unionType{l},patchSize,patchStep,Option.numFilters(l),model.V{l},Option.pooling,Option.poolingSize,Option.poolingMethod);
        end
        for j=1:group
            tempImg=zeros(height,width);
            index=find(type(:,j));
            for m=1:length(index)
                tempImg=tempImg+outImg{Idx==index(m)};
            end
            tempImg=tempImg/length(index);
            im = im2col_general(tempImg,patchSize,patchStep);
            im = bsxfun(@minus, im, mean(im,2));
            Rx(:,:,j) = Rx(:,:,j) + im*im';
        end
    end
    Rx = Rx/(sampleNum*size(im,2));   
    for j=1:group
        [E,D] = eig(Rx(:,:,j));
        [~,ind] = sort(diag(D),'descend');
        V = E(:,ind(1:num));  % principal eigenvectors 
        model.V{i}{j}=V;
    end
    model.V{i}=[model.V{i}{:}];
end

disp('Extract features');
ftrain=cell(sampleNum,1);
for i=1:sampleNum
    ftrain{i}=PCNExtFea(trainData(i),model,Option);
    trainData{i}=[];
end
ftrain=[ftrain{:}];
end

