function feats=compute_gradient_features(R,W,H,locW,locH,gw,gh)
   nori = size(R,3);
   dim = compute_feature_dim(gw,gh,nori);
   feats=cell(length(dim),1);
   nl = length(dim);
   for i = 1:length(dim)
     feats{i} = zeros(size(locH,1)*size(locH,2),dim(i));
   end

   feat_indx = 1;

   for posj=1:size(locW,2)
     for posi=1:size(locW,1)
       posw = locW(posi,posj);
       posh = locH(posi,posj);
       
       R_=normalize_response(R(posh:posh+H-1,posw:posw+W-1,:));
       A=R_;
       C=cumsum2D(R_);
       level_sum = sum(C(end,end,:))+1e-10;

       for l=1:nl
         dim_indx = 1;

         ww = gw{l}+1;
         hh = gh{l}+1;
         for j = 2:size(ww,2)
           for i = 2:size(ww,1)
             wi=ww(i,j); wi_=ww(i-1,j-1);
             hi=hh(i,j); hi_=hh(i-1,j-1);
             oriR=C(hi,wi,:)-C(hi_,wi,:)-C(hi,wi_,:)+C(hi_,wi_,:);
             feats{l}(feat_indx,dim_indx:dim_indx+nori-1) = oriR/level_sum;
             dim_indx = dim_indx+nori;
           end
         end
       end
       feat_indx = feat_indx+1;
     end
   end
   feats = concat_features(feats);
end
