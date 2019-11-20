figure;
plot(out_psnr)
figure;
plot(out_ssim)
figure;
out_final_z = zeros(21,1);
out_final_d = zeros(21,1);
for i=1:21
    temp_z = out_z((i-1)*110+1:i*110,:,:,:);
    out_final_z(i) = sum(sum(sum(sum(temp_z))));
    temp_d = out_d((i-1)*110+1:i*110,:,:);
    out_final_d(i) = sum(sum(sum(temp_d)));
end
plot(out_final_z);
figure;
plot(out_final_d)
    
    
    