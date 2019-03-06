clc;
clear;
close all;
% Jan.29 2019 Homework 1 for ECE 271B, Prblem 5
% Code written by Yudi Wang
% Using PCA and LDA to do classification

%%
% Set up the variable for the Gaussian Distribution
alpha = 2;
sigma = 10;

Mu_1 = [alpha; 0];
Mu_2 = [-alpha; 0];

Sigma_1 = [1,0;0,sigma];
Sigma_2 = Sigma_1;

Sample_1 = mvnrnd(Mu_1,Sigma_1,500);
Sample_2 = mvnrnd(Mu_2,Sigma_2,500);

figure();
plot(Sample_1(:,1),Sample_1(:,2),'+');
hold on;
plot(Sample_2(:,1),Sample_2(:,2),'o');
% title('alpha = 10, sigma = 2');
% saveas(gcf,'alpha = 10, sigma = 2.tif');
title('alpha = 2, sigma = 10');
saveas(gcf,'alpha = 2, sigma = 10.tif');

    
%%
Sample = zeros(2,2000);

% Doing the PCA analysis for the 2000 * 2 sample point
Sample(:,1:500) = Sample_1';
Sample(:,501:1000) = Sample_2';

%   Obtain Covariance Matrix
    Sample_Cov=cov(Sample');
%   dim(E_Vec) = 2500 * 2500
    [E_Vec,E_Val]=eig(Sample_Cov);
    
%   dim(E_Vec_Pri) = 2500 * 16
    E_Vec_Pri = E_Vec(:,2);
    
%   Two point
    Point_a = 15* [E_Vec_Pri(1) , -E_Vec_Pri(1)];
    Point_b = 15* [E_Vec_Pri(2) , -E_Vec_Pri(2)];
    
    figure();
    plot(Sample_1(:,1),Sample_1(:,2),'+');
    hold on;
    plot(Sample_2(:,1),Sample_2(:,2),'o');
    line(Point_a,Point_b);
%     title('alpha = 10, sigma = 2 with PCA');
%     saveas(gcf,'alpha = 10, sigma = 2 with PCA.tif');
    
    title('alpha = 2, sigma = 10 with PCA');
    saveas(gcf,'alpha = 2, sigma = 10 with PCA.tif');


%%
% Doing the LDA analysis for the 2000 * 2 sample point

    Proj_LDA = inv(Sigma_1 + Sigma_2)*(Mu_1 - Mu_2);

    %   Two point
    Point_a = [Proj_LDA(1),-Proj_LDA(1)];
    Point_b = [Proj_LDA(2),-Proj_LDA(2)];
    
    figure();
    plot(Sample_1(:,1),Sample_1(:,2),'+');
    hold on;
    plot(Sample_2(:,1),Sample_2(:,2),'o');
    line(Point_a,Point_b);
    title('alpha = 2, sigma = 10 with LDA');
    saveas(gcf,'alpha = 2, sigma = 10 with LDA.tif');

    title('alpha = 10, sigma = 2 with LDA');
    saveas(gcf,'alpha = 10, sigma = 2 with LDA.tif');