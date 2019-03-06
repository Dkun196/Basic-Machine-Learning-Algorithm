function [ErrorRate,FeatureX] = MLprocess(Diml,C,Mu_EM_BG,Mu_EM_FG,Sigma_EM_BG,Sigma_EM_FG,Pi_EM_BG,Pi_EM_FG)

% 
% global Ch_row Ch_col Pi_BG Pi_FG;
%  
% FeatureX=zeros(Ch_row,Ch_col);
% 
% Px0_BG=0;
% Px1_FG=0;
% 
% for i=1:Ch_row
%  for j=1:Ch_col
%     block_coe=block_coe64(i,j,1:Diml);
%    
% %     %Calculate the probability of mixture gaussian distribution of
% %     %background   
% %     for J=1:C
% %         Pxz_BG=mvnpdf(block_coe,Mu_EM_BG(J,:),squeeze(Sigma_EM_BG(:,:,J)));
% %         Px0_BG=Px0_BG+Pxz_BG*Pi_EM_BG(1,J);
% %         Pxz_FG=mvnpdf(block_coe,Mu_EM_FG(J,:),squeeze(Sigma_EM_FG(:,:,J)));
% %         Px1_FG=Px1_FG+Pxz_FG*Pi_EM_FG(1,J);   
% %     end
% 
%     if(Px0_BG*Pi_BG>Px1_FG*Pi_FG)
%           FeatureX(i,j)=1;
%     end
% 
%  end
% end
% 
% 
%  for b=260:270
%     FeatureX(:,b)=zeros(255,1);
%  end
%  
% ErrorRate=ErrorCheck(FeatureX);
% 


end

