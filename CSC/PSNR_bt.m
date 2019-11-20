function [mse, psnr] = PSNR_bt(x, z)
    num_pic = size(x,3);
    z = z(6:105,6:105,:);
    mse = zeros(num_pic,1);
    for i = 1:num_pic
        x(:,:,i) = (x(:,:,i)-min(min(x(:,:,i))))/(max(max(x(:,:,i)))-min(min(x(:,:,i))));
        z(:,:,i) = (z(:,:,i)-min(min(z(:,:,i))))/(max(max(z(:,:,i)))-min(min(z(:,:,i))));       
        mse(i) = mse(i) + mean(mean((z(:,:,i)-x(:,:,i)).^2));
    end
    psnr = 20 * log10(1) - 10 * log10(mse);
end

