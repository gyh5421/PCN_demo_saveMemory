function im = im2col_general(varargin)
% 

NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{2}; 

z = size(InImg,3);
im = cell(z,1);
if NumInput == 2
    for i = 1:z
        im{i} = im2colstep(InImg(:,:,i),patchsize12)';
    end
else
    hStep=varargin{3}(1);
    wStep=varargin{3}(2);
    patchHeight=patchsize12(1);
    patchWidth=patchsize12(2);
    [height,width,~]=size(InImg);
    hPadding=mod(height-patchHeight,hStep);
    hPadding=mod(hStep-hPadding,hStep);
    wPadding=mod(width-patchWidth,wStep);
    wPadding=mod(wStep-wPadding,wStep);
    tempImage=zeros(height+hPadding,width+wPadding,z);
    tempImage(1:height,1:width,:)=InImg(:,:,:);
%     for i=1:hPadding
%         for j=1:wPadding
%             tempImage(:,width+j,:)=tempImage(:,width,:);
%         end
%         tempImage(height+i,:,:)=tempImage(height,:,:);
%     end
    for i = 1:z
        im{i} = im2colstep(tempImage(:,:,i),patchsize12,varargin{3})';
    end 
end
im = [im{:}]';
    
    