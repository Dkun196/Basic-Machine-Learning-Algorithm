% %Oct.23 2018 Homework 2 for ECE 271A
% %Coded written by Yudi Wang
% %The goal is to segment the ¡°cheetah¡± image into  
% %two components:cheetah (foreground) and grass(background).

clear;
clc;

load('TrainingSamplesDCT_8_new.mat');
Pi_BG=1053/(1053+250);
Pi_FG=250/(1053+250);

% Calculate for the 64-dimensional gaussian distribution
BG=TrainsampleDCT_BG;
FG=TrainsampleDCT_FG;

Sigma64_BG=cov(BG);
Sigma64_FG=cov(FG);

alpha64_BG=log((2*pi)^64*det(Sigma64_BG))-2*log(Pi_BG);
alpha64_FG=log((2*pi)^64*det(Sigma64_FG))-2*log(Pi_FG);

mu64_BG=mean(BG);
mu64_FG=mean(FG);


% Draw 64*2 independent gaussian distributions for comparsion 
%  Pick up the best eight features

sigma_BG=sqrt(var(BG));
sigma_FG=sqrt(var(FG));

% for i=1:64
%     x_FG=(mu64_FG(i)-(3*sigma_FG(i))):sigma_FG(i)/12:(mu64_FG(i)+(3*sigma_FG(i)));
%     Px1_FG=normpdf(x_FG,mu64_FG(i),sigma_FG(i));
%     x_BG=(mu64_BG(i)-(3*sigma_BG(i))):sigma_BG(i)/12:(mu64_BG(i)+(3*sigma_BG(i)));
%     Px0_BG=normpdf(x_BG,mu64_BG(i),sigma_BG(i));
%     switch((i+15)/16)
%         case 1
%             j=0;
%             figure(1);
%         case 2
%             j=1;
%             figure(2);
%         case 3
%             j=2;
%             figure(3);
%         case 4
%             j=3;
%             figure(4);
%     end
%     subplot(4,4,i-16*j);
%     plot(x_FG,Px1_FG);
%     hold;
%     plot(x_BG,Px0_BG);
%     legend('Cheetah','Grass');
%     number=int2str(i);
%     title(number);
%     xlabel('Feature');
%     ylabel('Probability');
% end

best=[1,18,19,25,27,32,40,41];
% best=[1,14,20,25,27,32,40,41];
worst=[2,3,4,5,59,62,64,63];

BG8=TrainsampleDCT_BG(:,best);
FG8=TrainsampleDCT_FG(:,best);

mu8_BG=mean(BG8);
mu8_FG=mean(FG8);
sigma8_BG=sqrt(var(BG8));
sigma8_FG=sqrt(var(FG8));

BG8_worst=TrainsampleDCT_BG(:,worst);
FG8_worst=TrainsampleDCT_FG(:,worst);

mu8_BG_worst=mean(BG8_worst);
mu8_FG_worst=mean(FG8_worst);
sigma8_BG_worst=sqrt(var(BG8_worst));
sigma8_FG_worst=sqrt(var(FG8_worst));

% for i=1:8
%     x_FG=(mu8_FG(i)-(3*sigma8_FG(i))):sigma8_FG(i)/12:(mu8_FG(i)+(3*sigma8_FG(i)));
%     Px1_FG=normpdf(x_FG,mu8_FG(i),sigma8_FG(i));
%     x_BG=(mu8_BG(i)-(3*sigma8_BG(i))):sigma8_BG(i)/12:(mu8_BG(i)+(3*sigma8_BG(i)));
%     Px0_BG=normpdf(x_BG,mu8_BG(i),sigma8_BG(i));
%     
%     x_FG_worst=(mu8_FG_worst(i)-(3*sigma8_FG_worst(i))):sigma8_FG_worst(i)/12:(mu8_FG_worst(i)+(3*sigma8_FG_worst(i)));
%     Px1_FG_worst=normpdf(x_FG_worst,mu8_FG_worst(i),sigma8_FG_worst(i));
%     x_BG_worst=(mu8_BG_worst(i)-(3*sigma8_BG_worst(i))):sigma8_BG_worst(i)/12:(mu8_BG_worst(i)+(3*sigma8_BG_worst(i)));
%     Px0_BG_worst=normpdf(x_BG_worst,mu8_BG_worst(i),sigma8_BG_worst(i));
%     
%     figure(5);
%     subplot(4,2,i);
%     plot(x_FG,Px1_FG);
%     hold;
%     plot(x_BG,Px0_BG);
%     number=int2str(best(i));
%     title(number);
%     ylabel('Probability');
%     xlabel('Best-8');
%     
%     figure(6);
%     subplot(4,2,i);
%     plot(x_FG_worst,Px1_FG_worst);
%     hold;
%     plot(x_BG_worst,Px0_BG_worst);
%     number=int2str(worst(i));
%     title(number);
%     ylabel('Probability');
%     xlabel('Worst-8');
% end


Sigma8_BG=cov(BG8);
Sigma8_FG=cov(FG8);

alpha8_BG=log((2*pi)^8*det(Sigma8_BG))-2*log(Pi_BG);
alpha8_FG=log((2*pi)^8*det(Sigma8_FG))-2*log(Pi_FG);


%----------------Q3----------------%

Cheetah=imread('cheetah.bmp');
Cheetah=im2double(Cheetah);
[Ch_row,Ch_col]=size(Cheetah);

Cheetah=padarray(Cheetah,[7,7],'replicate');
 zigzag=textread('Zig-Zag Pattern.txt');
 zigzag=zigzag+1;
  
 
 Feature64X=zeros(Ch_row,Ch_col);
 Feature8X=zeros(Ch_row,Ch_col);
 block_coe64=zeros(1,64);
 block_coe64=zeros(1,8);
 
 for i = 8:1:(Ch_row+7)
    for j = 8:1:(Ch_col+7)
        
    block_window=zeros(8,8);
    block_window(1:8,1:8)=Cheetah(i:(i+7),j:j+7);                                
    blockDCT=dct2(block_window);
    block_coe64(zigzag)=blockDCT;
    
    block_coe8=block_coe64(best);
    
    Px0_BG64 = (block_coe64-mu64_BG)*inv(Sigma64_BG)*(block_coe64-mu64_BG)'+alpha64_BG;
    Px1_FG64 = (block_coe64-mu64_FG)*inv(Sigma64_FG)*(block_coe64-mu64_FG)'+alpha64_FG;
 
    if(Px0_BG64>Px1_FG64)
          Feature64X(i-7,j-7)=1;
    end
    
    Px0_BG8 = (block_coe8-mu8_BG)*inv(Sigma8_BG)*(block_coe8-mu8_BG)'+alpha8_BG;
    Px1_FG8 = (block_coe8-mu8_FG)*inv(Sigma8_FG)*(block_coe8-mu8_FG)'+alpha8_FG;
 
    if(Px0_BG8>Px1_FG8)
          Feature8X(i-7,j-7)=1;
    end
      
    end
 end
 
figure(7);
imshow(Feature64X,[]);

figure(8);
imshow(Feature8X,[]);

%Using mask to calculate the error rate
mask=imread('cheetah_mask.bmp');
mask=im2double(mask);

error64_0=0;
error64_1=0;
total_0=0;
total_1=0;
error8_0 =0;
error8_1=0;


for i=1:Ch_row
    for j=1:Ch_col
       if(mask(i,j)==1)
            total_1=total_1+1;
       else
           total_0=total_0+1;
       end
    end
end

for i=1:Ch_row
    for j=1:Ch_col
        if(mask(i,j)==1&&Feature64X(i,j)~=1)
            error64_1=error64_1+1;
        end
        if(mask(i,j)==0&&Feature64X(i,j)~=0)
            error64_0=error64_0+1;
        end
       if(mask(i,j)==1&&Feature8X(i,j)~=1)
            error8_1=error8_1+1;
        end
        if(mask(i,j)==0&&Feature8X(i,j)~=0)
            error8_0=error8_0+1;
        end
    end
end

errorRate64=(error64_0/total_0)*Pi_BG+(error64_1/total_1)*Pi_FG;

errorRate8=(error8_0/total_0)*Pi_BG+(error8_1/total_1)*Pi_FG;   


