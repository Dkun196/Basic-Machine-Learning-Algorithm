%Dec.1 2018 Homework 5 for ECE 271A
%Code written by Yudi Wang
%The goal is to segment the cheetah image into  
%two components:cheetah (foreground) and grass(background) using the mixture Gaussian Model.
%%
%Set up
close all;
clear;
clc;

SetUp();

global  BG FG Ch_row Ch_col;

Dim=[1,2,4,8,16,24,32,40,48,56,64];
C=[1,2,4,8,16,32];


% %%
% 
% %For a) part
% 
% 
% %Using the EM process to generate 5 BG and 5 FG
% 
% Mu_bg=zeros(8,64,5);
% Pi_bg=zeros(1,8,5);
% Sigma_bg=zeros(64,64,8,5);
% 
% Mu_fg=zeros(8,64,5);
% Pi_fg=zeros(1,8,5);
% Sigma_fg=zeros(64,64,8,5);
% 
% for r=1:5  
%     [Mu_bg(:,:,r),Pi_bg(1,:,r),Sigma_bg(:,:,:,r)] = EMProcess(BG,8); 
%     [Mu_fg(:,:,r),Pi_fg(1,:,r),Sigma_fg(:,:,:,r)] = EMProcess(FG,8);
% end
% 
% %Using the ML method to generate the probability map
% 
% Pb=zeros(5,11,Ch_row,Ch_col);
% Pf=zeros(5,11,Ch_row,Ch_col);
% 
% %%
% for r=1
%     for d=1:11
%         Pb(r,d,:,:)=GP(Dim(d),8,Mu_bg(:,:,r),Sigma_bg(:,:,:,r),Pi_bg(1,:,r));
%         Pf(r,d,:,:)=GP(Dim(d),8,Mu_fg(:,:,r),Sigma_fg(:,:,:,r),Pi_fg(1,:,r));
%     end
% end
% 
% %Compare the probability map and generate the result
% 
% %%
% Error=zeros(1,11);
% for r=1
%     b=Pb(r,:,:,:);
%     for s=1
%         f=Pf(s,:,:,:);
%         for d=11
%             Error(1,d)=Compare(squeeze(b(d,:,:)),squeeze(f(d,:,:)));
%         end
%         figure(r);
%         plot(Error);
%         hold;
%     end
% end


%%
%For the b) part

Pb=zeros(11,Ch_row,Ch_col);
Pf=zeros(11,Ch_row,Ch_col);

for c=1
    mix=C(c);
    [Mu_bg,Pi_bg,Sigma_bg] = EMProcess(BG,mix); 
    [Mu_fg,Pi_fg,Sigma_fg] = EMProcess(FG,mix);
    
    Error=zeros(1,11);
    
    for d=11
        
        dim=Dim(d);
        Pb(d,:,:)=GP(dim,mix,Mu_bg(:,1:dim),Sigma_bg(1:dim,1:dim,:),Pi_bg);
        Pf(d,:,:)=GP(dim,mix,Mu_fg(:,1:dim),Sigma_fg(1:dim,1:dim,:),Pi_fg);
        
        Error(1,d)=Compare(squeeze(Pb(d,:,:)),squeeze(Pf(d,:,:)));
    end
        figure(c);
        plot(Error);
        hold;
end







