%compute features over scalespace
function [feats,R,win_posw,win_posh,winw,winh] = ...
    compute_features_scale_space(im,border,scaleratio,nori,strideh,stridew,outfile)
  
  II = imread(im);
  if(size(II,3) > 1) II=rgb2gray(II);end;

  II = im2double(II);  
    
  IW = 64;
  IH = 128;
  blocks = [64 32 16 6;
            64 32 16 6];

  [h w nch] = size(II);
  [gw,gh] = get_sampling_grid(IW,IH,blocks);%%%%%%%%%%%%%%%%%%%%%%

  num_scales=min(floor(log(h/IH)/log(scaleratio)),floor(log(w/IW)/log(scaleratio)))+1;
  scales = scaleratio.^(0:num_scales-1);
  
  num_feats = 0;


  padw = 8; padh = 8;

  for s = 1:num_scales
    I = imresize(II,1/scales(s));
    I = padarray(I,[padh padw],'replicate');


    [h w nch] = size(I);
    offsetw = max(border,floor(mod(w,stridew)/2))+1;
    offseth = max(border,floor(mod(h,strideh)/2))+1;
    [loch,locw] = meshgrid(offseth:strideh:size(I,1)-IH-border+1,...
                           offsetw:stridew:size(I,2)-IW-border+1);

    R = compute_gradient(I,nori);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    level_feats = compute_gradient_features(R,IW,IH,locw,loch,gw,gh);%%%%%%%%%%%%%%%%%%%%%


    locw(:) = locw(:) - padw;
    loch(:) = loch(:) - padh;


    count = size(level_feats,1);
    feats(num_feats+1:num_feats+count,:)  = level_feats;
    win_posw(num_feats+1:num_feats+count) = round(locw(:)*scales(s)); 
    win_posh(num_feats+1:num_feats+count) = round(loch(:)*scales(s)); 
    winw(num_feats+1:num_feats+count) = round(IW*scales(s)); 
    winh(num_feats+1:num_feats+count) = round(IH*scales(s)); 
%     fprintf(1,'\tscale=%.3f [%dx%d], feats=%d\n',scales(s),round(IW* ...
%                                                       scales(s)),round(IH*scales(s)),count);
    num_feats = num_feats + count;
  end; 
  

  if(nargin == 7)
      dlmwrite(outfile,feats,'delimiter',' ');
  end
end
