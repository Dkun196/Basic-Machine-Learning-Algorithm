%Nov.21 2018 Homework 3 for ECE 271A
%Code written by Yudi Wang
%The goal is to segment the cheetah image into  
%two components:cheetah (foreground) and grass(background).

close all;
clear;
clc;

load('Alpha.mat');
%calculate the error rate with DataSet 1~4 by strategy 1
%BDR means the Bayesian Parameter Estimation, MLE means the maxium likelihood, MAP means
%Bayesian Model 
%%
%calculate the error rate for 4 dataset changed with alpha strategy 1 by three methods

ErrorBDR_D1_1=BDR(1,1);
ErrorMLE_D1_1=MLE(1,1);
ErrorMAP_D1_1=MAP(1,1);

ErrorBDR_D2_1=BDR(2,1);
ErrorMLE_D2_1=MLE(2,1);
ErrorMAP_D2_1=MAP(2,1);

ErrorBDR_D3_1=BDR(3,1);
ErrorMLE_D3_1=MLE(3,1);
ErrorMAP_D3_1=MAP(3,1);

ErrorBDR_D4_1=BDR(4,1);
ErrorMLE_D4_1=MLE(4,1);
ErrorMAP_D4_1=MAP(4,1);


ErrorMLE_D1_1=ones(1,9)*ErrorMLE_D1_1;
ErrorMLE_D2_1=ones(1,9)*ErrorMLE_D2_1;
ErrorMLE_D3_1=ones(1,9)*ErrorMLE_D3_1;
ErrorMLE_D4_1=ones(1,9)*ErrorMLE_D4_1;
%%

%plot the comparsion picture for strategy 1
close all;
figure(1);
hold;
plot(alpha,ErrorBDR_D1_1,'b');
plot(alpha,ErrorMLE_D1_1,'r');
% plot(alpha,ErrorMAP_D1_1,'k');

legend('Bayesian','ML');
set(gca,'XScale','log');
title('Dataset 1-Strategy 1');
xlabel('alpha');
ylabel('Error Rate');

    
figure(2);
hold;
plot(alpha,ErrorBDR_D2_1,'b');
plot(alpha,ErrorMLE_D2_1,'r');
plot(alpha,ErrorMAP_D2_1,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 2-Strategy 1');
xlabel('alpha');
ylabel('Error Rate');

figure(3);
hold;
plot(alpha,ErrorBDR_D3_1,'b');
plot(alpha,ErrorMLE_D3_1,'r');
plot(alpha,ErrorMAP_D3_1,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 3-Strategy 1');
xlabel('alpha');
ylabel('Error Rate');

figure(4);
hold;
plot(alpha,ErrorBDR_D4_1,'b');
plot(alpha,ErrorMLE_D4_1,'r');
plot(alpha,ErrorMAP_D4_1,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 4-Strategy 1');
xlabel('alpha');
ylabel('Error Rate');

%%
%calculate the error rate for 4 dataset changed with alpha strategy 2 by three methods
ErrorBDR_D1_2=BDR(1,2);
ErrorMLE_D1_2=MLE(1,2);
ErrorMAP_D1_2=MAP(1,2);

ErrorBDR_D2_2=BDR(2,2);
ErrorMLE_D2_2=MLE(2,2);
ErrorMAP_D2_2=MAP(2,2);

ErrorBDR_D3_2=BDR(3,2);
ErrorMLE_D3_2=MLE(3,2);
ErrorMAP_D3_2=MAP(3,2);

ErrorBDR_D4_2=BDR(4,2);
ErrorMLE_D4_2=MLE(4,2);
ErrorMAP_D4_2=MAP(4,2);



ErrorMLE_D1_2=ones(1,9)*ErrorMLE_D1_2;
ErrorMLE_D2_2=ones(1,9)*ErrorMLE_D2_2;
ErrorMLE_D3_2=ones(1,9)*ErrorMLE_D3_2;
ErrorMLE_D4_2=ones(1,9)*ErrorMLE_D4_2;

%%
%plot the comparsion picture for strategy 2
close all;
figure(5);
hold;
plot(alpha,ErrorBDR_D1_2,'b');
plot(alpha,ErrorMLE_D1_2,'r');
plot(alpha,ErrorMAP_D1_2,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 1-Strategy 2');
xlabel('alpha');
ylabel('Error Rate');

figure(6);
hold;
plot(alpha,ErrorBDR_D2_2,'b');
plot(alpha,ErrorMLE_D2_2,'r');
plot(alpha,ErrorMAP_D2_2,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 2-Strategy 2');
xlabel('alpha');
ylabel('Error Rate');

figure(7);
hold;
plot(alpha,ErrorBDR_D3_2,'b');
plot(alpha,ErrorMLE_D3_2,'r');
plot(alpha,ErrorMAP_D3_2,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 3-Strategy 2');
xlabel('alpha');
ylabel('Error Rate');

figure(8);
hold;
plot(alpha,ErrorBDR_D4_2,'b');
plot(alpha,ErrorMLE_D4_2,'r');
plot(alpha,ErrorMAP_D4_2,'k');
legend('Bayesian','MLE','MAP');
set(gca,'XScale','log');
title('Dataset 4-Strategy 2');
xlabel('alpha');
ylabel('Error Rate');
