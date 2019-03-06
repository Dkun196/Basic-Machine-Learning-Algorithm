function [errorRate64]=ErrorCheck(Feature64X,Ch_row,Ch_col,Pi_BG,Pi_FG)

%Using mask to calculate the error rate
mask=imread('cheetah_mask.bmp');
mask=im2double(mask);

error64_0=0;
error64_1=0;
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
        if(mask(i,j)==1&&Feature64X(i,j)~=1)
            error64_1=error64_1+1;
        end
        if(mask(i,j)==0&&Feature64X(i,j)~=0)
            error64_0=error64_0+1;
        end
  
    end
end

errorRate64=(error64_0/total_0)*Pi_BG+(error64_1/total_1)*Pi_FG;


end