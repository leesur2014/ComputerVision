function M = InverseCompositional(It, It1,n_iters)
% input - image at time t, image at t+1, rectangle (top left, bot right coordinates)
% output - movement vector, [u,v] in the x- and y-directions.

It = im2double(It);
It1 = im2double(It1);
%n_iters = 10;
M = eye(3);% [1+p(1) p(2)  p(3);
           %  p(4)  1+p(5) p(6);
           %  0      0      1];

% 3) Evaluate gradient of T
[Itx, Ity] = gradient(It);

% 4) Evaluate Jacobian - constant for affine warps
[xrange,yrange] = meshgrid(1:size(It,2),1:size(It,1));
x = transpose(yrange);
x = reshape(x, [1 size(x,1)*size(x,2)]);%1 by N
y = transpose(xrange);
y = reshape(y, [1 size(y,1)*size(y,2)]);
J = [x;y;ones(1,length(x))];
J = repmat(J, [2 1]);%6 by N

% 5) Compute steepest descent images, VT_dW_dp
Itx_ = reshape(Itx, [numel(Itx) 1]);
Ity_ = reshape(Ity, [numel(Ity) 1]);
dIt = repmat([Itx_ Ity_], [1 3]);%6 by N
A = dIt .* transpose(J);%N by 6 

% 6) Compute Hessian and inverse
H = transpose(A)*A;
H_inv = inv(H);

j = 0;
while j < n_iters
    % 1) Compute warped image with current parameters
    warped_q = M*[reshape(xrange,[1 numel(xrange)]);reshape(yrange, [1 numel(yrange)]);ones(1,numel(xrange))];
    xq= reshape(transpose(warped_q(1,:)),[size(xrange,1) size(xrange,2)]);
    yq = reshape(transpose(warped_q(2,:)),[size(yrange,1) size(yrange,2)]);
    I_new = interp2(It1, xq, yq);
    I_new(isnan(I_new)) = 0;
    
    % 2) Compute error image - NB reversed
    err_im = I_new - It;
    b = reshape(err_im, [size(err_im,1)*size(err_im,2) 1]);
    
    % 7) Compute steepest descent parameter updates
    tmp = transpose(A)*b;
    
    % 8) Compute gradient descent parameter updates
    dp = H_inv*tmp;
    
    % 9) Update warp parmaters
    dM = [1+dp(1) dp(2) dp(3);dp(4) 1+dp(5) dp(6);0 0 1];
    dM = inv(dM);
    M = M*dM;
    
    j = j + 1;
   
end

end 