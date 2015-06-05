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

% This program is a demo of PCN

clear all;
clc;
addpath Liblinear;
addpath MNISTdata;
addpath Utils;


%% load data and labels,train data and test data is randomly chosen from the data
load mnist_basic;
TrnSize=100;
trainData=mnist_train(1:TrnSize,1:end-1)';
testData=mnist_test(:,1:end-1)';
trainLabels=mnist_train(1:TrnSize,end);
testLabels=mnist_test(:,end);
clear mnist_train;
clear mnist_test;

testData = testData(:,1:50:end);
testLabels = testLabels(1:50:end); 

%% set parameters for the network
Option.imgSize=[28 28];
imgcell_train=mat2imgcell(trainData,Option.imgSize(1),Option.imgSize(2),'gray');
clear trainData;
nTestImg=size(testData,2);
imgcell_test=mat2imgcell(testData,Option.imgSize(1),Option.imgSize(2),'gray');
clear testData;
fid=fopen('result.txt','at');
fprintf(fid,'A new rotate:\n');
for patchSize=7:1:7
    for patchStep=1%:patchSize
        for ii=6:6
            for jj=11:11
                for iter=1%0:floor((ii-3)/2)
                    Option.patchSize=[patchSize patchSize];
                    Option.patchStep=[patchStep patchStep];
                    Option.numStage=2;
                    Option.numFilters=[ii jj];
                    Option.histBlockSize=[7 7];
                    Option.blkOverLapRatio = 0.5;
                    Option.Hashing=true;
                    Option.Pyramid = [];
                    Option.pooling=false;
                    Option.poolingSize=[2 2];
                    Option.poolingMethod='maxpooling';
                    unionType=zeros(ii,ii);     %unionType describe the combination of every layer
                    for i=1:ii
                       % unionType(1,i)=1;
                        unionType(i,i)=1;
                       % unionType(mod(i+iter,ii)+1,i)=1;
                       % unionType(mod(i+2*iter,ii)+1,i)=1;
                    end
                    Option.unionType{1}=1;
                    Option.unionType{2}=unionType;
%                     unionType=zeros(ii*jj,ii*jj);
%                     for i=1:ii*jj
%                        % unionType(1,i)=1;
%                         unionType(i,i)=1;
%                        % unionType(mod(i+iter,ii)+1,i)=1;
%                        % unionType(mod(i+2*iter,ii)+1,i)=1;
%                     end
%                     Option.unionType{3}=unionType;
                    fprintf(fid,'patchSize(%d,%d),patchStep(%d,%d),numFilters(%d,%d),\nunionType:\n%s\n',patchSize,patchSize,patchStep,patchStep,ii,jj,num2str(unionType));
                    
                    clear unionType;

                    Option

                    %% train the network
                    tic;
                    disp('begin train PCN');
                    [ftrain,model]=PCNTrain(imgcell_train,Option);

                    ftrain=sparse(ftrain);
                    PCNTrainTime=toc;

                    %% train svm
                    tic;

                    disp('train SVM');
                    svmStruct=train(trainLabels,ftrain','-s 1 -q');

                    clear ftrain;
                    svmTrainTime=toc;


                 %% test period
                    tic;

                    nCorrRecog = 0;

                    disp('begin test');
                    for idx = 1:1:nTestImg

                        ftest=PCNExtFea(imgcell_test(idx),model,Option);   %extract features of test data

                        [xLabel_est, accuracy, decision_values] = predict(testLabels(idx),sparse(ftest'), svmStruct, '-q'); 

                        if xLabel_est == testLabels(idx)
                            nCorrRecog = nCorrRecog + 1;
                        end

                        if 0==mod(idx,nTestImg/100); 
                            fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',[idx 100*nCorrRecog/idx toc/idx]); 
                        end 

                    end
                    testTime=toc;


                  %% print the result
                    fprintf(fid,'average accuracy %.2f%%\n',nCorrRecog/nTestImg*100);
                    fprintf(fid,'PCNTrainTime:%.2f\n',PCNTrainTime);
                    fprintf(fid,'SVMTrainTime:%.2f\n',svmTrainTime);
                    fprintf(fid,'testTime:%.2f\n',testTime);
                    fprintf(fid,'dim of feature:%d\n\n',length(ftest));
                end
            end
        end
    end
end
fclose(fid);



