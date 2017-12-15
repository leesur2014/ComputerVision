function M = InverseAdditive(It, It1,n_iters)
% input - image at time t, image at t+1 
% output - M affine transformation matrix
p = zeros(6, 1);
M = eye(3);
%n_iters = 50;
It = im2double(It);
It1 = im2double(It1);

% 3) Evaluate gradient of T
[Itx, Ity] = gradient(It);

% 4) Evaluate Gamma_x
[xrange,yrange] = meshgrid(1:size(It1,2),1:size(It1,1));
x = transpose(yrange);
x = reshape(x, [1 size(x,1)*size(x,2)]);%1 by N
y = transpose(xrange);
y = reshape(y, [1 size(y,1)*size(y,2)]);
G = [x;y;ones(1,length(x))];
G = repmat(G, [2 1]);%6 by N

% 5) Compute modified steepest descent images
Itx_ = reshape(Itx, [numel(Itx) 1]);
Ity_ = reshape(Ity, [numel(Ity) 1]);
dIt = repmat([Itx_ Ity_], [1 3]);%6 by N
A_star = dIt .* transpose(G);%N by 6 

% 6) Compute modified Hessian and inverse
H_star = transpose(A_star)*A_star;
H_star_inv = inv(H_star);

j = 0;
while j<n_iters
    % 1) Compute warped image with current parameters
    warped_q = M*[reshape(xrange,[1 numel(xrange)]);reshape(yrange, [1 numel(yrange)]);ones(1,numel(xrange))];
    xq = reshape(transpose(warped_q(1,:)),[size(xrange,1) size(xrange,2)]);
    yq = reshape(transpose(warped_q(2,:)),[size(yrange,1) size(yrange,2)]);
    t1 = interp2(It1, xq, yq);
    t1(isnan(t1)) = 0;
    
    % 2) Compute error image
    err_im = t1 - It;
    b = reshape(err_im, [size(err_im,1)*size(err_im,2) 1]);
   
   
    % 7) Compute steepest descent parameter updates
    tmp = transpose(A_star)*b; 
    
    % 8) Compute gradient descent parameter updates
    dp_star = H_star_inv*tmp;
    %disp(['dist' num2str(norm(dp,2))]);
    
    % 9) Update warp parmaters
    ss = M(1:2,1:2);
    s_p_inv = kron(diag(ones(3,1)), ss);
    dp = s_p_inv*dp_star;
    p = p - dp;
    M = [1+p(1) p(2) p(3);p(4) 1+p(5) p(6);0 0 1];
    
    j = j + 1;
end


