% Jan.28 2019 Homework 1 for ECE 271B, Prblem 5
% Code written by Yudi Wang
% Using PCA and LDA to do face recognition

clc;
clear;
close all;
%%
% Implement of PCA


% Using the trainging data to generate the projection matrix
Sample_PCA = zeros(50*50,240);

% input all the training sample to consist a row matrix
for j = 0 : 5
%   for 6 person
    for i = 1 : 40
%   each one have 40 pictures for trainging
        str=strcat('trainset/subset',int2str(j),'/person_',int2str(j+1),'_',int2str(i),'.jpg');
        s_col=imread(str);
        s_col=im2double(s_col);
        s_col=reshape(s_col,50*50,1);
        Sample_PCA(:,j*40+i)=s_col;
    end
end
  
%   Obtain Covariance Matrix
    Sample_Cov=cov(Sample_PCA');
    
%   dim(E_Vec) = 2500 * 2500
    [E_Vec,E_Val]=eig(Sample_Cov);
    
%   dim(E_Vec_Pri) = 2500 * 16
    E_Vec_Pri = E_Vec(:,2485:2500);

% %   Display the first 16 eigen faces
% EigenFaces = zeros(50,50);
% figure();
% for i = 1 : 16
%     
%     Vec_temp = E_Vec_Pri(:,i);
%     EigenFaces = reshape(Vec_temp,50,50);
%     subplot(4,4,17-i);
%     imshow(EigenFaces,[]);
%     title(['Eigen Faces',int2str(17-i)]);
%     
% end
% saveas(gcf,'PCA - 16 Principal Components.tif');

%%
% Using PCA projection matrix to reduce the training sample dimension to 15

% Generate 15 dimensional projection result for the 240 training data
Sample_PCA_proj = E_Vec_Pri(:,2:16)'*Sample_PCA;

% Build up 6 Multi-dimensional Gaussian Distribution for 6 faces
Mu_PCA_proj = zeros(15,6);
Sigma_PCA_proj = zeros(15,15,6);
for j = 0:5
    Sample_PCA_proj_temp = Sample_PCA_proj(:,40*j+1:40*j+40);
    Mu_PCA_proj(:,j+1) = mean(Sample_PCA_proj_temp,2);
    Sigma_PCA_proj(:,:,j+1) = cov(Sample_PCA_proj_temp'); 
end

% Load the testing set to generate a test matrix
Test_PCA = zeros(50*50,60);
for j = 0 : 5
%   for 6 person
    for i = 1 : 10
%   each one have 10 pictures for testing
        str=strcat('testset/subset',int2str(j+6),'/person_',int2str(j+1),'_',int2str(i),'.jpg');
        s_col=imread(str);
        s_col=im2double(s_col);
        s_col=reshape(s_col,50*50,1);
        Test_PCA(:,j*10+i)=s_col;
    end
end
% Test Sample - do the PCA projection
    Test_PCA_proj =E_Vec_Pri(:,2:16)'* Test_PCA;
    
%Using BDR to decide the class for each point

result_PCA = zeros(1,60);
for i = 1 : 60
    Prob_i_temp = 0;
    idx = 0;
    Prob_Min = 1e9;
    for j = 1 : 6
        Prob_i_temp = transpose(Test_PCA_proj(:,i)-Mu_PCA_proj(:,j))*inv(squeeze(Sigma_PCA_proj(:,:,j)))*(Test_PCA_proj(:,i)-Mu_PCA_proj(:,j)) + log(det(squeeze(Sigma_PCA_proj(:,:,j)))); 
        if(Prob_i_temp < Prob_Min)
            idx = j;
            Prob_Min = Prob_i_temp;
        end
    end
    result_PCA(1,i) = idx;
end

% Calculate the error rate for each class and the average error rate
error_PCA = zeros(1,6);
for i = 1 : 60 
    c_temp=floor((i-1)/10)+1;
    if(result_PCA(1,i) ~= c_temp)
        error_PCA(1,c_temp) = error_PCA(c_temp) + 1;
    end
end

error_PCA = error_PCA/10;
error_aver_PCA = mean(error_PCA);
