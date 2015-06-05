function [outImgs] = MaxPooling(imgs,imgSize,poolingSize,method)
num=numel(imgs);
height=ceil(imgSize(1)/poolingSize(1));
width=ceil(imgSize(2)/poolingSize(2));
outImgs=cell(num,1);
for i=1:num
    im=imgs{i};
    patches=im2col_general(im,poolingSize,poolingSize);
    imgs{i}=[];
    if strcmp(method,'maxpooling')
        patches=max(patches);
    else
        patches=mean(patches,1);
    end
    outImgs{i}=reshape(patches,height,width);
end
end

