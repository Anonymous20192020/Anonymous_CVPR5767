function [d] = backtracking_btb(d, z, alpha, d_old, z_old, btb)
    len3 = 110*110*100;
    len2 = 110*110+1;
    len1 = 110;

    z_sum = sum(z,4);
    z_temp = reshape(z_sum, len3, 1);   
    
    bt_value = mean(d,3);
    
    ax = ones(1,100) * 0.05;
    btb = sum(btb,3);
    [B,indb] = sort(z_temp,1);
    for i = 1:(uint8(alpha*len3))
        btbi = sum(btb',1);
        
        index_temp = indb(len3-i);
        d3 = floor((index_temp-1)/len2)+1;
        d2 = floor((mod(index_temp-1,len2))/(len1)) + 1;
        d1 = mod((mod(index_temp,len2))-1,len1) + 1;
        %norm(d)/norm(z)
        zz = z -z_old;
        c = sum(btbi .* ax);
        p = min(norm(reshape(zz(d1,d2,d3,:),1,10))/abs((d(d1,d2,d3) - d_old(d1,d2,d3))+0.00001),1);
        [c,p]
        %d(d1,d2,d3) = d(d1,d2,d3) * 0.01  + d_old(d1,d2,d3) * 0.99 * c;%0.5;%bt_value(d1,d2); %0.5;%;
        d(d1,d2,d3) = d(d1,d2,d3) + d_old(d1,d2,d3) * 0.1 * c * p;%0.5;%bt_value(d1,d2); %0.5;%;
    end
end

