function []=train()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t1=clock; 
nmax_param.sw = 0.1;
nmax_param.sh = 0.1;
nmax_param.ss = 1.3;
nmax_param.th = 0.0;
scaleratio = 2^(1/8);
nori=9;
border=0;
block_sizes = [ 64 32 16 6;  
                        64 32 16 6];
full_360=0; 
window_size=[128 64]; 
stridew = 16;
strideh = 16; 
nlevels =size(block_sizes,2);
offh=0; offw=0;
methods='iksvm_phog';
%%%%%%%
% root_dir='\\Charles-laptop\e';
root_dir='E:';
if ismac
imgSetNeg=imageSet('/Volumes/Data/Dataset/INRIAPerson/train_64x128_H96/neg');
imgSetPos=imageSet('/Volumes/Data/Dataset/INRIAPerson/train_64x128_H96/pos');
imgSetTest=imageSet('/Volumes/Data/Dataset/INRIAPerson/70X134H96/Test/pos');
modeldir='/Volumes/Data/Matlab/svm_hog/model/';
elseif ispc
result_comment='_with_bootstrap_1';
result_dir=[root_dir '/Matlab/pedestrian_detection/result/' methods result_comment];

imgNegDir=[root_dir '/Dataset/INRIAPerson/train_64x128_H96/neg_cropped'];
imgPosDir=[root_dir '/Dataset/INRIAPerson/96X160H96/Train/pos'];
imgTestDir=[root_dir '/Dataset/INRIAPerson/Test/'];
modeldir=[root_dir '/Matlab/pedestrian_detection/models'];
hardfeatsdir=[root_dir '/Dataset/INRIAPerson/hardfeats'];

feats_dir=[root_dir '/Matlab/pedestrian_detection/features/'];
imgTestList = imageSet(imgTestDir,'recursive');
imgTestNegList = imgTestList(1).ImageLocation;
imgTestPosList = imgTestList(2).ImageLocation;
imgNegList=imageSet(imgNegDir).ImageLocation;
imgPosList=imageSet(imgPosDir).ImageLocation;
end
%%%%%%%%%%%%%%%% Compute Features %%%%%%%%%%%%%%%%%%%%%%
FeatsPos=computefeats(imgPosList,border,nori,1);
FeatsNeg=computefeats(imgNegList,border,nori,1);
save([feats_dir '/FeatsPos_' methods '.mat'], 'FeatsPos');
save([feats_dir '/FeatsNeg_' methods '.mat'], 'FeatsNeg');
%%%%%%%%%%%%%%%% Trainning Model %%%%%%%%%%%%%%%%%%%%%%
tic;
param = [];
model=train_save(FeatsPos, FeatsNeg,1,methods,modeldir);
t1=toc;
%%%%%%%%%%%%%%%%% Bootstraping Fast 1 %%%%%%%%%%%%%%%%%%%%
% tic;
% nmax_param.th=0.0;
% if(~exist('model','var'))
%     load([modeldir '/model_' methods '.mat'], 'model');
% end
% FeatsNegHard=bootstrap(imgNegList,nori,bor192.16der,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param,hardfeatsdir,methods);
% save([feats_dir '/FeatsNegHard_' methods '.mat'], 'FeatsNegHard');
% t2=toc;
%%%%%%%%%%%%%%%%% Bootstraping 1 %%%%%%%%%%%%%%%%%%%%
tic;
nmax_param.th=0.0;
if(~exist('model','var'))
    load([modeldir '/model_' methods '.mat'], 'model');
end
FeatsNegHard=bootstrap(imgNegList,nori,border,strideh,stridew,scaleratio,model,nmax_param,hardfeatsdir,methods);
save([feats_dir '/FeatsNegHard_1_' methods result_comment '.mat'], 'FeatsNegHard');
t2=toc;
%%%%%%%%%%%%%%%%%% Retrainning Model%%%%%%%%%%%%%%%%%%%%
if(~exist('FeatsNegHard','var'))
    load([feats_dir '/FeatsNegHard_1_' methods '.mat'], 'FeatsNegHard');
end
if(~exist('FeatsNeg','var'))
    load([feats_dir '/FeatsNeg_' methods '.mat'], 'FeatsNeg');
end
if(~exist('FeatsPos','var'))
    load([feats_dir '/FeatsPos_' methods '.mat'], 'FeatsPos');
end
tic;
FeatsNeg=[FeatsNeg;FeatsNegHard];
model=train_save(FeatsPos, FeatsNeg,1,[bootstrap_methods],modeldir); %1 intersection, 2 chisquared, and 3 JS kernels
t3=toc;
%%%%%%%%%% Compute Approx Model%%%%%%%%%%%%%%%%%%%%
if(~exist('model','var'))
    load([modeldir '/model_bootstrap_' methods '.mat'], 'model');
end
% if(~exist('model','var'))
%     load([modeldir '/model_' methods '.mat'], 'model');
% end
tic;
param.NSAMPLE = 100; 
param.BINARY = 1; 
model = compute_approx_model(model,param);
fprintf('[binary %i] %.2fs to train approx model\n',param.BINARY, toc);
save([modeldir '/model_approx_bootstrap_' methods '.mat'], 'model');
t4=toc;
%%%%%%%%%%%%%%% benchmark %%%%%%%%%%%%%%%%%%%%%%
if(~exist('model','var'))
    load([modeldir '/model_approx_bootstrap_' methods '.mat'], 'model');
end
tic;
nmax_param.sw = 0.1;                  %bandwidth along width
nmax_param.sh = 0.1;                  %bandwidth along height 
nmax_param.ss = 1.3;                  %bandwidth along scale 
nmax_param.th = -2.0;
%%%%%%%%%%%%%%%% Benchmark Exact %%%%%%%%%%%%%%%%%%%%
benchmark(imgTestPosList,[result_dir '/det+pos/'],nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param, 1);
benchmark(imgTestNegList,[result_dir '/det+neg/'],nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param, 1);
%%%%%%%%%%%%%%%% Benchmark Approx %%%%%%%%%%%%%%%%%%%%
% benchmark(imgTestPosList,[result_dir '/det+pos/'],nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param, 1);
% benchmark(imgTestNegList,[result_dir '/det+neg/'],nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param, 1);
%%%%%%%%%%%%%%%% Draw %%%%%%%%%%%%%%%%%%%%
draw_roc([result_dir '/']);
t5=toc;
%%%%%%%%%%%% Helper Function %%%%%%%%%%%%%%%%
function feat=bootstrap(imList,nori,border,strideh,stridew,scaleratio,model,nmax_param,hardfeatsdir,methods)
parfor_progress(length(imList));
parfor i=1:length(imList);
    tic;
    image=imread(imList{i});
   [feats,win_posw,win_posh,winw,winh] = ...
        compute_features_scale_space_nopad(im,border,scaleratio,nori,stridew,strideh);
    feature_time=toc;
    rawr=[win_posw;win_posh;winw;winh]';
    labels=ones(size(feats,1),1); 
    tic;
    [~, ~, raws]=svmpredict(labels,feats,model);
    classification_time=toc;
%     disp(sprintf('%.2fs to compute features, %.2fs to classify %i features..\n',feature_time, classification_time, size(rawr,1)));
    indx = raws > nmax_param.th;
    features{i} = feats(indx,:);
    parfor_progress;
%  draw_det(image, dr(:,1),dr(:,2),dr(:,3),dr(:,4),ds,nmax_param.th);
end
tic;
feat=zeros(60000,2268);
sum=0;
for i=1:length(features)
    f(1:size(features{i},1),:)=features{i};
    sum=sum+size(features{i},1);
    feat=[feat;f];
end
feat=feat(1:sum,:);
save([hardfeatsdir '/' methods '_hardfeatures.mat'], 'feat');
fprintf('%.2fs to save hardfeatures\n', toc);
end
%%%%%%%%%%%% Helper Function %%%%%%%%%%%%%%%%
function feat=bootstrap_fast(imList,nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param,hardfeatsdir,methods)
parfor_progress(length(imList));
parfor i=1:length(imList);
    tic;
    image=imread(imList{i});
    [feats,win_posw,win_posh,winw,winh]=compute_features_fast_nopad(image,nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw);
    feats=double(feats);
    feature_time=toc;
    rawr=[win_posw;win_posh;winw;winh]';
    labels=ones(size(feats,1),1); 
    tic;
    [~, ~, raws]=svmpredict(labels,feats,model);
    classification_time=toc;
%     disp(sprintf('%.2fs to compute features, %.2fs to classify %i features..\n',feature_time, classification_time, size(rawr,1)));
    indx = raws > nmax_param.th;
    features{i} = feats(indx,:);
    parfor_progress;
%  draw_det(image, dr(:,1),dr(:,2),dr(:,3),dr(:,4),ds,nmax_param.th);
end
tic;
feat=zeros(60000,2268);
for i=1:length(features)
    f(1:size(features{i},1),:)=features{i};
    feat=[feat;f];
end
save([hardfeatsdir '/' methods '_hardfeatures.mat'], 'feat');
fprintf('%.2fs to save hardfeatures\n', toc);
end
%%%%%%%%%%%% Helper Function %%%%%%%%%%%%%%%%
function [feats]=computefeats(imList,border,nori,sampling)
feats=zeros(length(imList),2268);
fprintf('computing features..\n');
parfor i=1:length(imList)
    feats(i,:)=compute_features(imList{i},border,nori,sampling);
end
fprintf('done. \n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=train_save(FeatsPos,FeatsNeg,kerneltype,methods,modeldir)
LabelsPos=ones(size(FeatsPos,1),1);
LabelsNeg=-ones(size(FeatsNeg,1),1);
fprintf('training model..\n');
tic;
KERNEL_TYPES = [5 6 7]; 
model = svmtrain([LabelsPos;LabelsNeg], [FeatsPos;FeatsNeg], sprintf('-t %i -b 1',KERNEL_TYPES(kerneltype)));
fprintf('%.2fs to train model\n',toc);
save([modeldir '/model_' methods '.mat'], 'model');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function benchmark(imgTestList, SaveFolder,nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw,model,nmax_param, approx)
parfor_progress(length(imgTestList));
parfor i=1:length(imgTestList);
% for i=1:10;
    tic;
   [pathstr, name, ext]=fileparts(imgTestList{i});
    im=imgTestList{i};
   [feats,win_posw,win_posh,winw,winh]=compute_features_fast_nopad(im,nori,border,window_size,block_sizes,strideh,stridew,scaleratio,offh,offw);
    feats=double(feats);
    rawr=[win_posw;win_posh;winw;winh]';
    feature_time =toc;
    labels=ones(size(feats,1),1); 
    tic;
if approx==0
     tic;
     [~, ~, raws]=svmpredict(labels,feats,model);
     classification_time=toc;
     fprintf('compute features in %.2fs, classify %i features using Libsvm in %.2fs \n',feature_time, length(raws), classification_time);
elseif approx==1
    tic;
    raws=svmpredict_approx(feats, model);
    classification_time=toc;
    fprintf('%fs to predict values using PWLApprox.\n',toc);
    fprintf('compute features in %.2fs, classify %i features using PWL approx in %.2fs \n',feature_time, length(raws), classification_time);
end
    indx = raws > nmax_param.th;
    raws = raws(indx,:);
    rawr = rawr(indx,:);
    parsave([SaveFolder name],rawr,raws);
%   draw_det(im, dr(:,1),dr(:,2),dr(:,3),dr(:,4),ds,nmax_param.th);
    parfor_progress;
end
parfor_progress(0);

end
end
