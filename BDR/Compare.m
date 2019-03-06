function [e] = Compare(Pb,Pf)

   global Ch_row Ch_col Pi_BG Pi_FG;

    FeatureX=zeros(Ch_row,Ch_col);

        for i=1:Ch_row
        for j=1:Ch_col
            
            if(Pb(i,j)*Pi_BG<Pf(i,j)*Pi_FG)
                  FeatureX(i,j)=1;
            end
            
        end
        end
            
        FeatureX(:,260:270)=zeros(Ch_row,11);
        
        e=ErrorCheck(FeatureX);
        
    
end

