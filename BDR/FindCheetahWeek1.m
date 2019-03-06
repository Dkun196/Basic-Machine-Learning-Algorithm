%Oct.12 2018 Homework 1 for ECE 271A
%Coded by Yudi Wang
%The goal is to segment the ¡°cheetah¡± image into  
%two components:cheetah (foreground) and grass(background).

clear;
%-------------------Q1-------------------%
load('TrainingSamplesDCT_8.mat');
Pi_BG=1053/(1053+250);
Pi_FG=250/(1053+250);

%-------------------Q2-------------------%
FG_abs=abs(TrainsampleDCT_FG);
BG_abs=abs(TrainsampleDCT_BG);
[FG_Neworder,FG_position]=sort(FG_abs,2,'descend');
[BG_Neworder,BG_position]=sort(BG_abs,2,'descend');
FG=FG_position(:,2);
BG=BG_position(:,2);
 
Px1_FG=zeros(1,64);
Px0_BG=zeros(1,64);
 
 for n=1:250
      Px1_FG(FG(n))= Px1_FG(FG(n))+1;
 end
 for m=1:1053 
      Px0_BG(BG(m))= Px0_BG(BG(m))+1;
 end

 Px1_FG= Px1_FG/250;
 Px0_BG= Px0_BG/1053;
 
 figure(3);
 FG_hist=bar(Px1_FG);
 figure(4);
 BG_hist=bar(Px0_BG);
 
 Binary=zeros(1,64); 
for k=1:64
    if (Pi_BG*Px0_BG(k))<(Pi_FG*Px1_FG(k))
        Binary(k)=1;
    end
end

%-------------------Q3-------------------%
%Using the offered pic to obtain feature X
Cheetah=imread('cheetah.bmp');
Cheetah=im2double(Cheetah);
[Ch_row,Ch_col]=size(Cheetah);

Cheetah=padarray(Cheetah,[7,7]);
 zigzag=textread('Zig-Zag Pattern.txt');
 zigzag=zigzag+1;
 r=8;  
 
 FeatureX=ones(Ch_row,Ch_col);
 block_reordered=zeros(1,64);
 
 for i = 8:1:(Ch_row+7)
    for j = 8:1:(Ch_col+7)
        
    block_window=zeros(8,8);
    block_window(1:8,1:8)=Cheetah(i:(i+7),j:j+7);                
                      
    blockDCT=dct2(block_window);
    block_reordered(zigzag)=blockDCT;
    
    block_reordered=abs(block_reordered);
    [NewOrder,position]=sort(block_reordered,'descend');
    
    FeatureX(i-7,j-7)=position(2);    
    end
    
 end
figure(1);
FrequencyPic=FeatureX/255;
imshow(FrequencyPic,[]);

final=zeros(Ch_row,Ch_col);

for x=1:1:Ch_row
  for y=1:1:Ch_col
    final(x,y)=Binary(FeatureX(x,y));
  end
end

figure(2);
imshow(final);
%-------------------Q4-------------------%
%Using mask to calculate the error rate
mask=imread('cheetah_mask.bmp');
mask=im2double(mask);

error_0  =0;
error_1=0;
total_0=0;
total_1=0;


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
        if(mask(i,j)==1&&final(i,j)~=1)
            error_1=error_1+1;
        end
        if(mask(i,j)==0&&final(i,j)~=0)
            error_0=error_0+1;
        end
     
    end
end

errorRate=(error_0/total_0)*Pi_BG+(error_1/total_1)*Pi_FG;

    



  




