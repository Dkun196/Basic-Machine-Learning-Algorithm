clc;
clear;
close all;
%%
% Implementation of the PCA + LDA

% First using PCA to reduce the dimension from 2500 to 30
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
    E_Vec_Pri = E_Vec(:,2471:2500);

% Generate 30 dimensional projection result for the 240 training data, the
% projection now is 30*240
Sample_PCA_proj = E_Vec_Pri'*Sample_PCA;

%%

% Then using LDA to reduce the dimension from 30 to 15
Sample_LDA_m = zeros(30,6);

for j = 0 : 5
    Sample_LDA_m(:,j+1) = mean(Sample_PCA_proj(:,40*j+1:40*j+40),2);
end

% Calculate the covariance matrix for each class
Sigma_LDA = zeros(30,30,6);
for j = 1 : 6
    Sample_LDA_temp = Sample_PCA_proj(:,40*j-39:40*j);
    SampleMean_LDA=Sample_LDA_m(:,j).*ones(30,40);
    Sample_LDA_temp = Sample_LDA_temp - SampleMean_LDA;%try to delete this part
    Sigma_LDA(:,:,j)=cov(Sample_LDA_temp');
end

% Apply LDA for 15 different pairs of classes

Proj_LDA = zeros(30,15);
count = 0;

for m = 1 : 6
    for n = (m + 1) : 6
        count = count + 1;
%       obtain the Sigma and Mu for this turn
        Sigma_0 = squeeze(Sigma_LDA(:,:,m));
        Sigma_1 = squeeze(Sigma_LDA(:,:,n));
        Mu_0 = squeeze(Sample_LDA_m(:,m));
        Mu_1 = squeeze(Sample_LDA_m(:,n));
        Proj_LDA(:,count) = inv(Sigma_0 + Sigma_1)*(Mu_1 - Mu_0);
    end
end

%%
% Generate 15 dimensional projection result for the 240 training data

Sample_LDA_proj = zeros(15,40,6);

for j = 1 : 6
    Sample_LDA_proj(:,:,j) = Proj_LDA'*Sample_PCA_proj(:,40*j-39:40*j);
end


% Build up 6 Multi-dimensional Gaussian Distribution for 6 faces
Mu_LDA_proj = zeros(15,6);
Sigma_LDA_proj = zeros(15,15,6);
for j = 0:5
    Sample_LDA_proj_temp = squeeze(Sample_LDA_proj(:,:,j+1));
    Mu_LDA_proj(:,j+1) = mean(Sample_LDA_proj_temp,2);
    Sigma_LDA_proj(:,:,j+1) = cov(Sample_LDA_proj_temp'); 
%     Sigma_LDA_proj(:,:,j+1) = Sigma_LDA_proj(:,:,j+1) * (39/40);
end

%%
% Load the testing set to generate a test matrix, 

% first do the PCA projection
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
Test_PCA_proj =E_Vec_Pri'* Test_PCA;
    
% Then do LDA projection
Test_LDA_proj =Proj_LDA'* Test_PCA_proj;
    
%Using BDR to decide the class for each point
result_PCA_LDA = zeros(1,60);

for i = 1 : 60
    Prob_i_temp = 0;
    idx = 0;
    Prob_Min = 1e9;
    for j = 1 : 6
        Prob_i_temp = transpose(Test_LDA_proj(:,i)-Mu_LDA_proj(:,j))*inv(squeeze(Sigma_LDA_proj(:,:,j)))*(Test_LDA_proj(:,i)-Mu_LDA_proj(:,j)) + log(det(squeeze(Sigma_LDA_proj(:,:,j)))); 
        if(Prob_i_temp < Prob_Min)
            idx = j;
            Prob_Min = Prob_i_temp;
        end
    end
    result_PCA_LDA(1,i) = idx;
end

% Calculate the error rate for each class and the average error rate
error_PCA_LDA = zeros(1,6);
for i = 1 : 60 
    c_temp=floor((i-1)/10)+1;
    if(result_PCA_LDA(1,i) ~= c_temp)
        error_PCA_LDA(1,c_temp) = error_PCA_LDA(c_temp) + 1;
    end
end

error_PCA_LDA = error_PCA_LDA/10;
error_aver_PCA_LDA = mean(error_PCA_LDA);
    
    
    


