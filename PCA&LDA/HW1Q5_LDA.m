clc;
clear;
close all;
%%
% Implement of LDA

% Using the training data to generate the projection matrix
Sample_LDA =zeros(50*50,40,6);

for j = 0 : 5
%   for 6 person
    for i = 1 : 40
%   each one have 40 pictures for trainging,save the data into 6 different
%   sample matrix
        str=strcat('trainset/subset',int2str(j),'/person_',int2str(j+1),'_',int2str(i),'.jpg');
        s_col=imread(str);
        s_col=im2double(s_col);
        s_col=reshape(s_col,50*50,1);
        Sample_LDA(:,i,j+1)=s_col;
    end
end

Sample_LDA_m = squeeze(mean(Sample_LDA,2));

% Calculate the covariance matrix for each class
Sigma_LDA = zeros(50*50,50*50,6);
for j = 1 : 6
    Sample_LDA_temp = squeeze(Sample_LDA(:,:,j));
    Sigma_LDA(:,:,j)=cov(Sample_LDA_temp');
end

% Apply LDA for 15 different pairs of classes

Proj_LDA = zeros(50*50,15);
count = 0;
gamma_M = eye(50*50);% To avoid singular matrix, using RDA with regularization parameter

for m = 1 : 6
    for n = (m + 1) : 6
        count = count + 1;
%       obtain the Sigma and Mu for this turn
        Sigma_0 = squeeze(Sigma_LDA(:,:,m));
        Sigma_1 = squeeze(Sigma_LDA(:,:,n));
        Mu_0 = squeeze(Sample_LDA_m(:,m));
        Mu_1 = squeeze(Sample_LDA_m(:,n));
        Proj_LDA(:,count) = inv(Sigma_0 + Sigma_1 + gamma_M)*(Mu_1 - Mu_0);
    end
end
% Display the 15 LDA Linear Discriminants
figure();
for i = 1 : 15
    
    LDA_temp = Proj_LDA(:,i);
    LDAFaces = reshape(LDA_temp,50,50);
    subplot(4,4,16 - i);
    imshow(LDAFaces,[]);
    title(['LDA ',int2str(16 - i)]);
    
end
saveas(gcf,'LDA - 15 Linear Discriminants.tif');
%%
% Using LDA projection matrix to reduce the training sample dimension to 15


% Generate 15 dimensional projection result for the 240 training data

Sample_LDA_proj = zeros(15,40,6);

for j = 1 : 6
    Sample_LDA_proj(:,:,j) = Proj_LDA'*squeeze(Sample_LDA(:,:,j));
end


% Build up 6 Multi-dimensional Gaussian Distribution for 6 faces
Mu_LDA_proj = zeros(15,6);
Sigma_LDA_proj = zeros(15,15,6);

for j = 0:5
    Sample_LDA_proj_temp = squeeze(Sample_LDA_proj(:,:,j+1));
    Mu_LDA_proj(:,j+1) = mean(Sample_LDA_proj_temp,2);
    Sigma_LDA_proj(:,:,j+1) = cov(Sample_LDA_proj_temp'); 
end

% Load the testing set to generate a test matrix
Test_LDA = zeros(50*50,60);
for j = 0 : 5
%   for 6 person
    for i = 1 : 10
%   each one have 10 pictures for testing
        str=strcat('testset/subset',int2str(j+6),'/person_',int2str(j+1),'_',int2str(i),'.jpg');
        s_col=imread(str);
        s_col=im2double(s_col);
        s_col=reshape(s_col,50*50,1);
        Test_LDA(:,j*10+i)=s_col;
    end
end
% Test Sample - meanFace, do the projection
    
    Test_LDA_proj =Proj_LDA'* Test_LDA;
    
%Using BDR to decide the class for each point

result_LDA = zeros(1,60);
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
    result_LDA(1,i) = idx;
end

% Calculate the error rate for each class and the average error rate
error_LDA = zeros(1,6);
for i = 1 : 60 
    c_temp=floor((i-1)/10)+1;
    if(result_LDA(1,i) ~= c_temp)
        error_LDA(1,c_temp) = error_LDA(c_temp) + 1;
    end
end

error_LDA = error_LDA/10;
error_aver_LDA = mean(error_LDA);
    
    
    


