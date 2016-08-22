function draw_roc(det_dir)
if ismac
img_dir = '/Volumes/Data/Dataset/INRIAPerson/Test/pos/'; 
gt_dir  = '/Volumes/Data/Matlab/pedestrian_detection/benchmark/gt_bbox/'; 
                                               
elseif ispc
img_dir = 'E:/Dataset/INRIAPerson/Test/pos/';
gt_dir  = 'E:/Matlab/pedestrian_detection/benchmark/gt_bbox/';
end


nmax_param.sw = 0.1;                
nmax_param.sh = 0.1;                
nmax_param.ss = 1.3;               

clear det; clear fppi;        
if ismac
%Dalal and Triggs (CVPR'05, linear SVM + HOG)
dt_dir = '/Volumes/Data/Matlab/svm_hog/benchmark/dt_hog/stride_8_8_sqrt_sqrt_sqrt_2/';
elseif ispc
dt_dir = 'E:/Matlab/svm_hog/benchmark/dt_hog/stride_8_8_sqrt_sqrt_sqrt_2/';    
end
[det{1},fppi{1}]=benchmark_det_inria(dt_dir,img_dir,gt_dir);

% if ismac
% det_dir = '/Volumes/Data/Matlab/svm_hog/benchmark/dt_hog/stride_8_8_1.05/';
% elseif ispc
% det_dir = 'E:/Matlab/svm_hog/benchmark/dt_hog/stride_8_8_1.05/';    
% end
% [det{2},fppi{2}]=benchmark_det_inria(det_dir,img_dir,gt_dir);
% 
% 
% % model1 
% nmax_param.th =  0;
% if ismac
% det_dir = '/Volumes/Data/Matlab/svm_hog/benchmark/iksvm_p_hog/';
% elseif ispc
% det_dir = 'E:/Matlab/svm_hog/benchmark/iksvm_p_hog/';
% end
% [det{3},fppi{3}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
% 
% %nmax_param.th = -0.5;
% %det_dir = 'model1/';
% %[det{3},fppi{3}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);

%model2 

% if ismac
% det_dir = '/Volumes/Data/Matlab/svm_hog/benchmark/mod1/';
% elseif ispc
% det_dir = 'E:/Matlab/svm_hog/benchmark/mod1/';
% end
nmax_param.th = 0.0;
[det{2},fppi{2}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
% nmax_param.th = 0.1;
% [det{3},fppi{3}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
% nmax_param.th = 0.2;
% [det{4},fppi{4}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
% nmax_param.th = 0.3;
% [det{5},fppi{5}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
% nmax_param.th = 0.4;
% [det{6},fppi{6}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);

% nmax_param.th = 0;
% if ismac
% det_dir = '/Volumes/Data/Matlab/svm_hog/benchmark/linsvm_p_hog/';
% elseif ispc
% det_dir = 'E:/Matlab/svm_hog/benchmark/linsvm_p_hog/';
% end
% [det{3},fppi{3}]=benchmark_det(det_dir,img_dir,gt_dir,nmax_param);
legtxt={'Dalal-Triggs';
            'Hiksvm-Phog'};
%            '8x8, 1.0500, dalal-triggs';
%           '8x8, 1.0905, iksvm-phog-3';
%            '8x8, 1.0905, iksvm-phog-2.84';
%            '8x8, 1.0905, linsvm-phog'};

    
h=figure;
hold on;
% sty = {'--k';'-k';'-r';'-b';'-m'};
sty = {'b-*';'r-^';'k-';'m-';'c-';'r-';'g-';'b-';'y-'};
for i = 1:length(det)
%      semilogx(fppi{i},det{i},'LineWidth',0.7,'color',[rand rand rand]);
     plot(fppi{i},det{i},sty{i},'LineWidth',0.8);
end;
grid on; box on;
% axis([0 2 0 1]);
xlabel('FPPI');
ylabel('Recall');
legend(legtxt,'Location','SouthEast');
title('Pedestrian Detection Inria');
end

