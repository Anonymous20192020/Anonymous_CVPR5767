function [ssim] = SSIM_bt(x, z, k1, k2, l)
    num_pic = size(x,3);
    z = z(6:105,6:105,:);
    c1 = (k1*l)^2;
    c2 = (k2*l)^2;
    ssim = zeros(num_pic,1);
    for i = 1:num_pic
        x(:,:,i) = (x(:,:,i)-min(min(x(:,:,i))))/(max(max(x(:,:,i)))-min(min(x(:,:,i))));
        z(:,:,i) = (z(:,:,i)-min(min(z(:,:,i))))/(max(max(z(:,:,i)))-min(min(z(:,:,i))));
        temp_x = x(:,:,i);
        temp_y = z(:,:,i);
        mean_x = mean(mean(temp_x));
        mean_y = mean(mean(temp_y));
        std_x = std2(temp_x);
        std_y = std2(temp_y);
        var_xy = cov(temp_x,temp_y);
        var_xy = var_xy(1,2);
        ssim(i) = [(2*mean_x*mean_y+c1)*(2*var_xy+c2)]/[(mean_x^2+mean_y^2+c1)*(std_x^2+std_y^2+c2)];
    end
    %ssim = 20 * log10(1) - 10 * log10(mse);
end