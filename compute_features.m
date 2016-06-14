function feats=compute_features(im,border,nori,sampling,outfile)
    I=imread(im);
    [NH,NW,dim]=size(I);
    
    if(size(I,3) > 1) I=rgb2gray(I);end;
    I = im2double(I);

    R = compute_gradient(I,nori);%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %window parameters
    IW=64; IH=128;
    
    blocks = [64 32 16 6;
              64 32 16 6];
    
    [gw,gh] = get_sampling_grid(IW,IH,blocks);%%%%%%%%%%%%%%%%%%%%%%%%
    
    if(sampling == 1)
      %center clip 
      loch = floor(mod(NH,IH)/2)+1;
      locw = floor(mod(NW,IW)/2)+1;
    else 
      %ramdomly sample 128x64 images from the image
      locw=floor(rand(sampling,1)*(NW-IW - 2*border))+1+border;
      loch=floor(rand(sampling,1)*(NH-IH - 2*border))+1+border;
    end
    feats = compute_gradient_features(R,IW,IH,locw,loch,gw,gh);%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %dlmwrite(outfile,feats,'delimiter',' ');
end
