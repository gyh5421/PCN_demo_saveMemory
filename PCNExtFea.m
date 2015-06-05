function ftest=PCNExtFea(ftest,model,Option)
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

% This function is used to extract features through PCN,it can process one
% sample a time
% Input:
% ftest is the input image,model contains the structure of PCN trained
% before and option contains some hyper-parameters
% Output:the features

height=Option.imgSize(1);
width=Option.imgSize(2);
for i=1:Option.numStage
    [ftest,height,width,~]=PCNOutput(ftest,[height,width],Option.unionType{i},Option.patchSize,Option.patchStep,Option.numFilters(i),model.V{i},Option.pooling,Option.poolingSize,Option.poolingMethod);
%     height=ceil((height-Option.patchSize(1))/Option.patchStep(1))+1;
%     width=ceil((width-Option.patchSize(2))/Option.patchStep(2))+1;
end
if Option.Hashing
    [ftest,~] = HashingHist(Option,ones(1,numel(ftest)),ftest); 
else
    ftest=[ftest{:}];
    ftest=ftest(:);
end
end

