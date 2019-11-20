function [z] = backtracking_3d(d, z, alpha, size, value)
    %%%%%%%%%
    %%%
    %%%
    %%%
    %%%
    %%%
    len3 = size(0) * size(1) * size(2);
    len2 = size(0) * size(1) + 1;
    len1 = size(0);
    
    d_temp = reshape(d, len3, 1);
    [B,indb] = sort(d_temp,1);
    
    for i = 1:(uint8(alpha * len3))
        index_temp = indb(len3 - i);
        d3 = floor((index_temp - 1) / len2) + 1;
        d2 = floor((mod(index_temp - 1,len2)) / (len1)) + 1;
        d1 = mod((mod(index_temp,len2)) - 1,len1) + 1;
        z(d1,d2,d3,:) = value;%bt_value;
    end
end

