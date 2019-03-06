function [] = SetUp()



global Cheetah zigzag CheetahPad BG FG Pi_BG Pi_FG mask Ch_row Ch_col total_0 total_1  block_coe64;

%%
%Load the test set
load('TrainingSamplesDCT_8_new.mat');

BG=TrainsampleDCT_BG;
FG=TrainsampleDCT_FG;

[row_BG,~]=size(BG);
[row_FG,~]=size(FG);

Pi_BG=row_BG/(row_BG+row_FG);
Pi_FG=row_FG/(row_BG+row_FG);

%%
%Calculate the mask once

mask=imread('cheetah_mask.bmp');
mask=im2double(mask);

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

%%
%Calculate the DCT values for each silding windows


Cheetah=imread('cheetah.bmp');
Cheetah=im2double(Cheetah);
[Ch_row,Ch_col]=size(Cheetah);

CheetahPad=padarray(Cheetah,[7,7],'replicate','post');

zigzag=textread('Zig-Zag Pattern.txt')+1;

block_coe64=zeros(Ch_row,Ch_col,64);

blockDCT=zeros(64,1);
block_window=zeros(8,8);

for i = 1:1:Ch_row
    for j = 1:1:Ch_col
        
    
    block_window(1:8,1:8)=CheetahPad(i:(i+7),j:j+7);                                
    blockDCT(zigzag)=dct2(block_window);
    block_coe64(i,j,:)=blockDCT;
    
    end
end



  
end

