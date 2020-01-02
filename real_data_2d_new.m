clc;
clear;
dat = load("cifar-10-batches-mat/data_batch_1.mat");
data = dat.data;

for i = 1 : 100
    img = reshape(data(i,:),[32,32,3]);
    image_gray{i} = rgb2gray(img); % 32 * 32 
    
    % TODO: randomly pick some 8*8 piece
    Y(:,i) = double(reshape(image_gray{i}(11:18,11:18), [2^6, 1]))./255;
end

y = Y(:);
N = 64;
p = 100;
lambda = 0.3;
lambda2 = 0.05;
num_of_matrix = log2(N);
iterations = 100;
TOL = 1e-03;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 0: random initial Bhat and Phat and solve Xhat %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("Generating Random Phat...\n");
for idx = 1:num_of_matrix
    dim = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    Pi = eye(dim);
    permutation = randperm(dim);
    Pi = Pi(permutation, :);
    Phat{idx} = Pi;   
end

fprintf("Generating Random Bhat...\n");
for idx = 1:num_of_matrix
    n = 2 ^ idx;
    half = n / 2;
    Bi = zeros(n);
    for i = 1 : half
        Bi(i,i) = randn(1) + randn(1) * j;
        Bi(half + i, i) = randn(1)+ randn(1) * j;
        Bi(half + i, half + i) = randn(1)+ randn(1) * j;
        Bi(i, half + i) = randn(1)+ randn(1) * j;
    end
    Bhat{idx} = Bi;
end
Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, .5);
B = Bhat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Step 1: Iteratively solve Phat and Bhat and Xhat   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1;
lambda_target=0.1;
lambda=0.05;
while iter < iterations
    %%%%%%%%%
    lambda=min(1.1*lambda, lambda_target);
    lambda
    %%%%%%%%%%
    
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % from right to left, iteratively solve B: 2*2, 4*4, 8*8, .....
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            Solve P                                   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate L and R
    Bfix = multiplicationB(Bhat);
    for idx = num_of_matrix:-1:1
        % calculate the fixed part for B
        n = 2^idx;
        % calculate the left fix part for P
        PLfix = eye(N);
        for i = (idx - 1):-1:1
            Pi = Phat{i};
            Ii = eye(N / 2^i);
            Pi = kron(Ii, Pi);
            PLfix = Pi * PLfix;
        end
        Lfix = Bfix * PLfix;
        % calculate the right fixed part for P
        Rfix = Xhat;
        for i = num_of_matrix: -1 :idx + 1
            Pi = Phat{i};
            Ii = eye(N/2^i);
            Pi = kron(Ii, Pi);
            Rfix = Pi * Rfix;
        end
        %TODO:update P{idx} Find a better way for doing this
        % Y = AX = Lfix * (kron(Ii,Phat{idx})) * Rfix
        % Y(:) = kron(Rfix, Lfix) * (kron(Ii,Phat{idx}))(:)
        fprintf("\n");
        fprintf("    Updating Phat{%d}\n", idx);
        
        RL = zeros(N*p, n^2);
        for j = 1:N/2^idx
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        
        % min \| RL * p - y\|_2^2 such that sum_i
        Aeq = [kron(eye(n), ones(n,1).'); kron(ones(n,1).', eye(n))];
        beq = [ones(2*n,1)];
        tic
        phat = lsqlin(abs(RL), y, [],[], Aeq, beq, zeros(n^2,1), ones(n^2,1))
        toc
        
        tic
        cvx_begin quiet
            variable Pidx(n, n)
            minimize norm(Lfix * kron(eye(N/2^idx), Pidx) * Rfix - Y, 'fro');
            subject to 
                Pidx(:)>=0;
                Pidx(:)<=1;
                for k = 1:n
                    sum(Pidx(k, :)) == 1;
                    sum(Pidx(:, k)) == 1;
                end
        cvx_end
        toc

        Phat{idx} = Pidx;
        fprintf("    target is %f\n", norm(get_A(Bhat,Phat) * Xhat - Y, 'fro'));
    end
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            Solve B                                   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('\n');
    % calculate L and R
    for idx = 1: num_of_matrix
        Pfix = multiplicationP(Phat);
        Rfix = Pfix * Xhat;
        n = 2^idx;
        for i = 1:idx - 1
            Bi = Bhat{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;
        end 
        Lfix = eye(N);
        for i = (idx + 1) : num_of_matrix
            Bi = Bhat{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;
        end
        fprintf("\n");
        fprintf("    Updating Bhat{%d}\n", idx); 
        % Let B be butterfly, i.e. restrict the zero position
        [ridx, cidx] = find(B{idx});
        RL = zeros(N*p, n^2);
        for j = 1:N/2^idx
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end

        non_zero_b_idx = find(B{idx}(:));
        RLs = RL(:,non_zero_b_idx);
        if idx == 1
            bhat = inv(RLs.' * RLs + lambda2 * eye(4)) * RLs.' * y;
        else
            bhat = RLs\y;
        end
        norm(bhat)
        Bidx = sparse(ridx, cidx, bhat);
        fprintf("    target ls  is %f\n", norm(Lfix * kron(eye(N/(2^idx)), Bidx) * Rfix - Y, 'fro'))
        Bhat{idx} = Bidx;
    end
    
    Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda);
    target = norm(get_A(Bhat, Phat) * Xhat - Y, 'fro');
    sparsity = mean((norms(Xhat,1)./norms(Xhat,2)).^2);
    fprintf("    target  is %f ", target)
    fprintf(" target < TOL: %d\n", target < TOL)
    fprintf("    sparsity is %f ", sparsity)
    fprintf("sparse: %d\n", sparsity <= 0.5 * N)
    
    if target < TOL && sparsity <= 0.1 * N
        break
    end
    iter = iter + 1;
end

fprintf("Optimization finished");
mean((norms(Xhat,1)./norms(Xhat,2)).^2) % sparsity of Xhat

Aest = get_A(Bhat, Phat);
svd(abs(Aest))
Xest = Aest\Y;
mean((norms(Xest,1)./norms(Xest,2)).^2)
histfit(sort(real(Xest(:))),50)


F = kron(dftmtx(2),dftmtx(4));
XFourier = F * Y;
mean((norms(XFourier,1)./norms(XFourier,2)).^2)
figure
histfit(sort(real(XFourier(:))),50)


