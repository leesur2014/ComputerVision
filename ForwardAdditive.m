function M = ForwardAdditive(It, It1, n_iters)
% input - image at time t, image at t+1 
% output - M affine transformation matrix
p = zeros(6, 1);
M = eye(3);
%n_iters = 10;
It = im2double(It);
It1 = im2double(It1);

% 3a) Compute image gradients - will warp these images in step 3b)
[It1x, It1y] = gradient(It1);

% 4) Evaluate Jacobian - constant for affine warps, but not in general
[xrange,yrange] = meshgrid(1:size(It1,2),1:size(It1,1));
x = transpose(yrange);
x = reshape(x, [1 size(x,1)*size(x,2)]);%1 by N
y = transpose(xrange);
y = reshape(y, [1 size(y,1)*size(y,2)]);
J = [x;y;ones(1,length(x))];
J = repmat(J, [2 1]);%6 by N

j = 0;
while j<n_iters
    % 1) Compute warped image with current parameters
    warped_q = M*[reshape(xrange,[1 numel(xrange)]);reshape(yrange, [1 numel(yrange)]);ones(1,numel(xrange))];
    xq = reshape(transpose(warped_q(1,:)),[size(xrange,1) size(xrange,2)]);
    yq = reshape(transpose(warped_q(2,:)),[size(yrange,1) size(yrange,2)]);
    t1 = interp2(It1, xq, yq);
    t1(isnan(t1)) = 0;
    
    % 2) Compute error image
    err_im = It - t1;
    b = reshape(err_im, [size(err_im,1)*size(err_im,2) 1]);
    
    % 3b) Evaluate gradient
    It1x_ = interp2(It1x, xq, yq);
    It1y_ = interp2(It1y, xq, yq);
    It1x_(isnan(It1x_)) = 0;
    It1y_(isnan(It1y_)) = 0;
        
    % 4) Evaluate Jacobian - constant for affine warps. Precomputed above
    
    % 5) Compute steepest descent images
    It1x_ = reshape(It1x_, [numel(It1x_) 1]);
    It1y_ = reshape(It1y_, [numel(It1y_) 1]);
    dIt1 = repmat([It1x_ It1y_], [1 3]);%6 by N
    A = dIt1 .* transpose(J);%N by 6  
    
    % 6) Compute Hessian and inverse
    H = transpose(A)*A;
    H_inv = inv(H);
   
    % 7) Compute steepest descent parameter updates
    tmp = transpose(A)*b;
    
    % 8) Compute gradient descent parameter updates
    dp = H_inv*tmp;
    %disp(['dist' num2str(norm(dp,2))]);
    % 9) Update warp parmaters
    p = p + dp;
    M = [1+p(1) p(2) p(3);p(4) 1+p(5) p(6);0 0 1];
    
    j = j + 1;
end


