function [ErrorBDR]=BDR(index,strategy)
ErrorBDR=zeros(1,9);

[Pi_BG,Pi_FG,BG,FG,W0,mu0_BG,mu0_FG,alpha,n_BG,n_FG,Sigma_BG,Sigma_FG]=SetUp(index,strategy);
%---- ----%
% Sigma_BG=cov(BG);
% Sigma_FG=cov(FG);

mu_hat_BG=mean(BG);
mu_hat_FG=mean(FG);

Sigma_0=zeros(64,64);

for a=1:9
    
for i=1:64
    Sigma_0(i,i)=alpha(a)*W0(i);
end



mu_n_BG=Sigma_0*inv(Sigma_0+(1/n_BG)*Sigma_BG)*mu_hat_BG'+(1/n_BG)*Sigma_BG*inv(Sigma_0+(1/n_BG)*Sigma_BG)*mu0_BG';
mu_n_FG=Sigma_0*inv(Sigma_0+(1/n_FG)*Sigma_FG)*mu_hat_FG'+(1/n_FG)*Sigma_FG*inv(Sigma_0+(1/n_FG)*Sigma_FG)*mu0_FG';

Sigma_n_BG=Sigma_0*inv(Sigma_0+(1/n_BG)*Sigma_BG)*(1/n_BG)*Sigma_BG;
Sigma_n_FG=Sigma_0*inv(Sigma_0+(1/n_FG)*Sigma_FG)*(1/n_FG)*Sigma_FG;


alpha64_BG=log((2*pi)^64*det(Sigma_n_BG+Sigma_BG))-2*log(Pi_BG);
alpha64_FG=log((2*pi)^64*det(Sigma_n_FG+Sigma_FG))-2*log(Pi_FG);

beta_BG=inv(Sigma_n_BG+Sigma_BG);
beta_FG=inv(Sigma_n_FG+Sigma_FG);


%--------------------------------------%
Cheetah=imread('cheetah.bmp');
Cheetah=im2double(Cheetah);
[Ch_row,Ch_col]=size(Cheetah);

Cheetah=padarray(Cheetah,[7,7],'replicate');
 zigzag=textread('Zig-Zag Pattern.txt');
 zigzag=zigzag+1;
  
 
 Feature64X=zeros(Ch_row,Ch_col);

 block_coe64=zeros(1,64);
 
 for i = 8:1:(Ch_row+7)
    for j = 8:1:(Ch_col+7)
        
    block_window=zeros(8,8);
    block_window(1:8,1:8)=Cheetah(i:(i+7),j:j+7);                                
    blockDCT=dct2(block_window);
    block_coe64(zigzag)=blockDCT;
    
    Px0_BG64 = (block_coe64-mu_n_BG')*beta_BG*(block_coe64-mu_n_BG')'+alpha64_BG;
    Px1_FG64 = (block_coe64-mu_n_FG')*beta_FG*(block_coe64-mu_n_FG')'+alpha64_FG;
 
    if(Px0_BG64>Px1_FG64)
          Feature64X(i-7,j-7)=1;
    end
    

    end
 end
 

 for b=260:270
Feature64X(:,b)=zeros(255,1);
 end
 
figure(1);
subplot(3,3,a);
imshow(Feature64X,[]);

%---- ----%

ErrorBDR(1,a)=ErrorCheck(Feature64X,Ch_row,Ch_col,Pi_BG,Pi_FG);
% disp(ErrorBDR);
end
end