function [d] = backtracking_d(d, z, alpha, d_old)
    len3 = 110*110*100;
    len2 = 110*110+1;
    len1 = 110;
    %d = abs(d);
    z_sum = sum(z,4);
    z_temp = reshape(z_sum, len3, 1);   
    %d_temp = reshape(d, len3, 1);
    
    bt_value = mean(d,3);
    
    [B,indb] = sort(z_temp,1);
    for i = 1:(uint8(alpha*len3))
        index_temp = indb(len3-i);
        d3 = floor((index_temp-1)/len2)+1;
        d2 = floor((mod(index_temp-1,len2))/(len1)) + 1;
        d1 = mod((mod(index_temp,len2))-1,len1) + 1;
        d(d1,d2,d3) = d(d1,d2,d3) * 0.01  + d_old(d1,d2,d3) * 0.99;%0.5;%bt_value(d1,d2); %0.5;%;
    end
end

