function [ d_res, z_res, Dz, obj_val ] = admm_learn_conv2D_weighted_sparse(b, kernel_size, mask, ...
                    lambda_residual, lambda_prior, ...
                    max_it, tol, ...
                    verbose)
           
    %Kernel size contains kernel_size = [psf_s, psf_s, k]
    psf_s = kernel_size(1);
    k = kernel_size(3);
    n = size(b,3);
                
    %PSF estimation
    psf_radius = floor( psf_s/2 );
    size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, n];
    size_z = [size_x(1), size_x(2), k, n];
    size_k = [2*psf_radius + 1, 2*psf_radius + 1, k]; 
    size_k_full = [size_x(1), size_x(2), k]; 
    
    %{
    %Edgetaper b
    edgetaper_filter = fspecial( 'gaussian', size_k(1:2), size_k(1)/3 );
    edgetaper_filter = edgetaper_filter ./ sum(edgetaper_filter(:));
    for i = 1:size(b, 3)
        b(:,:,i) = edgetaper(b(:,:,i), edgetaper_filter);
    end
    %}
    % Objective
    objective = @(z, dh) objectiveFunction( z, dh, b, mask, lambda_residual, lambda_prior, psf_radius, size_z, size_x );
    objective_bt = @(z, dh) objectiveFunction_bt( z, dh, b, mask, lambda_residual, lambda_prior, psf_radius, size_z, size_x );
    %Prox for masked data
    [M, Mtb] = precompute_MProx(b,  mask, psf_radius); %M is MtM
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); 
    
    %Prox for sparsity
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
    
    %alpha = 2/3;
    %ProxSparse = @(u, theta) solve_image(u, 1/theta, alpha);
    
    %Prox for kernel constraints
    ProxKernelConstraint = @(u) KernelConstraintProj( u, size_k_full, psf_radius);
    
    %% Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
    gamma_heuristic = 60 * lambda_prior * 1/max(b(:));
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic]; %[gamma_heuristic / 2000, gamma_heuristic];
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic]; %[gamma_heuristic / 2, gamma_heuristic];
    
    %% Initialize variables for K
    varsize_D = {size_x, size_k_full};
    xi_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    xi_D_hat = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    
    u_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    v_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    
    %Load from ref
    %d_load = load('filters_bristow_obj5.4e+04_11:06:19:29.mat');
    %d = padarray( d_load.d, [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');
    
    %Initial iterates
    d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');
    d = circshift(d , -[psf_radius, psf_radius, 0] );
    d_hat = fft2(d);
    
    %% Initialize variables for Z
    varsize_Z = {size_x, size_z};
    xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    %Initial iterates
    z = randn(size_z);
    z_hat = reshape(fft2(reshape(z, size_z(1), size_z(2), [])), size_z);
    
    %% Display it.
    if strcmp(verbose, 'all')
        iterate_fig = figure();
        filter_fig = figure();
        display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, 0);
    end
    
    obj_val = objective(z, d_hat);
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
    end
    
    %Iteration for local back and forth
    max_it_d = 10;
    max_it_z = 10;
    
    obj_val_filter = obj_val;
    obj_val_z = obj_val;
    
    %Iterate
    max_it = 20;
    out_psnr = zeros(max_it+1, 1);
    out_ssim = zeros(max_it+1, 1);
    
    
    out_d = d;
    out_z = z;
    z_old = z;
    for i = 1:max_it
        
        %% Update kernels
        
        %Recompute what is necessary for kernel convterm later
        rho = gammas_D(2)/gammas_D(1);
        [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho);
        
        obj_val_min = min(obj_val_filter, obj_val_z);
        
        d_old = d;
        d_hat_old = d_hat;
        d_D_old = d_D;
        %d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
        obj_val_old = obj_val;
        
        for i_d = 1:max_it_d

            %Compute v_i = H_i * z
            v_D{1} = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
            v_D{2} = d;

            %Compute proximal updates
            u_D{1} = ProxDataMasked( v_D{1} - d_D{1}, lambda(1)/gammas_D(1) );
            u_D{2} = ProxKernelConstraint( v_D{2} - d_D{2});

            for c = 1:2
                %Update running errors
                d_D{c} = d_D{c} - (v_D{c} - u_D{c});

                %Compute new xi and transform to fft
                xi_D{c} = u_D{c} + d_D{c};
                xi_D_hat{c} = fft2(xi_D{c});
            end

            %Solve convolutional inverse
            % d = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z);
            d = real(ifft2( d_hat ));
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                fprintf('--> Obj %3.3g \n', obj_val )
            end
        end
        
        obj_val_filter_old = obj_val_old;
        obj_val_filter = obj_val;
        
        %Debug progress
        d_diff = d - d_old;
        d_comp = d;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            obj_val = objective(z, d_hat);
            fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i, obj_val, norm(d_diff(:),2)/ norm(d_comp(:),2))
        end
        if(i<100)
            i;
            bt_b = objective_bt(z, d_hat);
            d = backtracking_btb(d, z, (max_it - i)/(3*max_it), d_old, z_old, bt_b);        
            d_hat = fft2(d);
        end
        %% Update sparsity term
        out_d = [out_d;d];
        out_z = [out_z;z];        
        %Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);
        dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] ); %Same for all images
        
        z_old = z;
        z_hat_old = z_hat;
        %d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
        d_Z_old = d_Z;
        obj_val_old = obj_val;
        for i_z = 1:max_it_z

            %Compute v_i = H_i * z
            v_Z{1} = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
            v_Z{2} = z;

            %Compute proximal updates
            u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
            u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );

            for c = 1:2
                %Update running errors
                d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});

                %Compute new xi and transform to fft
                xi_Z{c} = u_Z{c} + d_Z{c};
                xi_Z_hat{c} = reshape( fft2( reshape( xi_Z{c}, size_x(1), size_x(2), [] ) ), size(xi_Z{c}) );
            end

            %Solve convolutional inverse
            % z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z);
            z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2),[]) )), size_z );
            %
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                fprintf('--> Obj %3.3g \n', obj_val )
            end
        
        end
        
        %z = backtracking(d, z, (max_it - i)/(2*max_it));
        
        obj_val_z_old = obj_val_old;
        obj_val_z = obj_val;
        
        if obj_val_min <= obj_val_filter && obj_val_min <= obj_val_z
            z_hat = z_hat_old;
            z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2),[]) )), size_z );
            
            d_hat = d_hat_old;
            d = real(ifft2( d_hat ));
            
            obj_val = objective(z, d_hat);
            %break;
        end
        %}
        
        %Display it.
        if strcmp(verbose, 'all')
            display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, i);
        end

        %Debug progress
        z_diff = z - z_old;
        z_comp = z;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('Iter Z %d, Obj %3.3g, Diff %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2))
        end
        Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
        
        [mse, psnr] = PSNR_bt(b, Dz);
        ssim = SSIM_bt(b, Dz, 0.01, 0.03, 1);
        reshape(psnr,1,10)
        reshape(ssim,1,10)
        sum(ssim)/10;
        sum(psnr)/10;
        out_psnr(i) = sum(psnr)/10;
        out_ssim(i) = sum(ssim)/10;

        
        %Termination
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol && norm(d_diff(:),2)/ norm(d_comp(:),2) < tol
            i
            %break;
        end
    end
    
    %Final estimate
    z_res = z;
    
    d_res = circshift( d, [psf_radius, psf_radius, 0] ); 
    d_res = d_res(1:psf_radius*2+1, 1:psf_radius*2+1,:);
    
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));

    [mse, psnr] = PSNR_bt(b, Dz);
    ssim = SSIM_bt(b, Dz, 0.01, 0.03, 1);
    reshape(psnr,1,10);
    reshape(ssim,1,10);
    sum(ssim)/10;
    sum(psnr)/10;
    out_psnr(max_it+1) = sum(psnr)/10;
    out_ssim(max_it+1) = sum(ssim)/10;
    %save
 
%     figure;
%     plot(out_psnr)
%     figure;
%     plot(out_ssim)
%     figure;
%     out_final_z = zeros(21,1);
%     out_final_d = zeros(21,1);
%     for i=1:21
%         temp_z = out_z((i-1)*110+1:i*110,:,:,:);
%         out_final_z(i) = sum(sum(sum(sum(temp_z))));
%         temp_d = out_d((i-1)*110+1:i*110,:,:);
%         out_final_d(i) = sum(sum(sum(temp_d)));
%     end
%     %figure;
%     plot(out_final_z);
%     figure;
%     plot(out_final_d)
%     figure;   
    %save('dat100.mat',out_psnr,out_ssim,'out_final_d','out_final_z')
return;

function [u_proj] = KernelConstraintProj( u, size_k_full, psf_radius)
    
    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius,0] ); 
    u_proj = u_proj(1:psf_radius*2+1, 1:psf_radius*2+1,:);
    
    %Normalize
    u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1] );
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_k_full(1) - (2*psf_radius+1), size_k_full(2) - (2*psf_radius+1), 0], 0, 'post');
    u_proj = circshift(u_proj, -[psf_radius, psf_radius, 0] );

return;

function [M, Mtb] = precompute_MProx(b,  mask, psf_radius)
    
    M = padarray(mask, [psf_radius, psf_radius, 0], 0, 'both');
    Mtb = padarray(b, [psf_radius, psf_radius, 0], 0, 'both') .* M;
    
return;

function [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
% Computes the spectra for the inversion of all H_i

%Size
sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

%Precompute spectra for H
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [sy*sx, k, n] ), [3,2,1] ), [1 2] ), [1 sy*sx]); %n * k * s

%Precompute the inverse matrices for each frequency
zhat_inv_mat = reshape( cellfun(@(A)(1/rho * eye(k) - 1/rho * A'*pinv(rho * eye(n) + A * A')*A), zhat_mat, 'UniformOutput', false'), [1 sy*sx]);

return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;

%Rho
%rho = gammas(2)/gammas(1);
function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_hat, rho, size_z )

    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

    %Reshape to cell per frequency
    xi_hat_1_cell = num2cell( permute( reshape(xi_hat{1}, sx * sy, n), [2,1] ), 1);
    xi_hat_2_cell = num2cell( permute( reshape(xi_hat{2}, sx * sy, k), [2,1] ), 1);
    
    %Invert
    x = cellfun(@(Sinv, A, b, c)(Sinv * (A' * b + rho * c)), zhat_inv_mat, zhat_mat, xi_hat_1_cell, xi_hat_2_cell, 'UniformOutput', false);
    
    %Reshape to get back the new Dhat
    d_hat = reshape( permute(cell2mat(x), [2,1]), [sy,sx,k] );

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z )


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    
    %Rho
    rho = gammas(2)/gammas(1);
    
    %Compute b
    b = dhatT .* permute( repmat( reshape(xi_hat{1}, sy*sx, 1, n), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(xi_hat{2}, sy*sx, k, n), [2,1,3] );
    
    %Invert
    scInverse = repmat( ones([1,sx*sy]) ./ ( rho * ones([1,sx*sy]) + dhatTdhat.' ), [k,1,n] );
    x = 1/rho *b - 1/rho * scInverse .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    
    %Final transpose gives z_hat
    z_hat = reshape(permute(x, [2,1,3]), size_z);

return;

function f_val = objectiveFunction( z, d_hat, b, mask, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

    %Dataterm and regularizer
    zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* zhat, 3), size_x) ));
    f_z = lambda_residual * 1/2 * norm( reshape( mask .* Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - mask .* b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;

function temp = objectiveFunction_bt( z, d_hat, b, mask, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

    %Dataterm and regularizer
    zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* zhat, 3), size_x) ));
    temp = mask .* Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - mask .* b;
return;


function [] = display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, iter)

    %Params
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

    figure(iterate_fig);
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
    Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);

    subplot(3,2,1), imagesc(b(:,:,1));  axis image, colormap gray, title('Orig');
    subplot(3,2,2), imagesc(Dz(:,:,1)); axis image, colormap gray; title(sprintf('Local iterate %d',iter));
    subplot(3,2,3), imagesc(b(:,:,2));  axis image, colormap gray;
    subplot(3,2,4), imagesc(Dz(:,:,2)); axis image, colormap gray;
    subplot(3,2,5), imagesc(b(:,:,3));  axis image, colormap gray;
    subplot(3,2,6), imagesc(Dz(:,:,3)); axis image, colormap gray;

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = circshift( d(:,:,j + 1), [psf_radius, psf_radius] ); 
        d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title(sprintf('Local filter iterate %d',iter));
        
return;
