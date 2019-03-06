function [Mu_EM,Pi_EM,Sigma_EM] = EMProcess(D,C)
%initialize the parameter for the iteration

    [row,~]=size(D);
    
    %Generate Sigma, Mu, Pi
    Sigma_K=zeros(64,64,C);
    Sigma1=cov(D);
    Mu_K=zeros(C,64);
    
    for j=1:C
        Sigma_K(:,:,j)=diag((diag(Sigma1)));
        m=mean(D);
        Mu_K(j,:)=random('Normal',m,abs(m)*0.1);
    end
    
    Pi_K=ones(1,C)/C;
    
    %for the first iteration
    Sigma_EM=Sigma_K;
    Mu_EM=Mu_K;
    Pi_EM=Pi_K;

    h=zeros(C,row);
    
for a=1:3 

    % Q function - h_ij 
    for i=1:row
        for j=1:C
            h(j,i)=mvnpdf(D(i,:),Mu_EM(j,:),Sigma_EM(:,:,j));
        end
    end
    
    h=h.*Pi_EM';
    h=h./sum(h);
    h(h<1e-9)=1e-9;

    %%
    %Update Sigma, Mu and Pi for the next iteration
    
    Mu_EM_new=zeros(C,64);
    
    
    for j=1:C
        
        h_row=h(j,:);
        Mu_EM_new(j,:)=h_row*D/sum(h_row); %
        Pi_EM(j)=mean(h_row);
        D_diag=sum(h_row'.*(D-Mu_EM(j,:)).^2)/sum(h_row);
        D_diag(D_diag<1e-5)=1e-5;
        Sigma_EM(:,:,j)=diag(D_diag);
        
    end
        
        if((norm(Mu_EM_new-Mu_EM)/norm(Mu_EM))<1e-5)
            break;
        end
        
        Mu_EM=Mu_EM_new;
  
end
    
    
end

