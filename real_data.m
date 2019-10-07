clc;
clear;
data = load("cifar-10-batches-mat/data_batch_1.mat");
data = data.data;

% 10000 * ?32 * 32 * 3) images, sparse in 2 dimensional fourier transform
for i = 1 : 100
    img = reshape(data(i,:),[32,32,3]);
    image_gray{i} = rgb2gray(img);
    Y(:,i) = double(reshape(image_gray{i}(16:17,15:18),[8,1]));
end

lambda = 5.0000e-02;
N = 8;
p = 100;
num_of_matrix = log2(N);
iterations = 1000;

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
        Bi(i,i) = randn(1);
        Bi(half + i, i) = randn(1);
        Bi(half + i, half + i) = randn(1);
        Bi(i, half + i) = randn(1);
    end
    Bhat{idx} = Bi;
end


Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda);

iter = 1;
lambda_target=5e-2;
lambda=5e-3;
while iter < iterations
    %%%%%%%%%%
    lambda=min(1.1*lambda,lambda_target);
    lambda
    %%%%%%%%%%
    
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % from right to left, iteratively solve B 2*2, 4*4, 8*8, .....
    
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
        %TODO:update P{idx}
        % Y = AX = Lfix * (kron(Ii,Phat{idx})) * Rfix
        % Y(:) = kron(Rfix, Lfix) * (kron(Ii,Phat{idx}))(:)
        fprintf("    Updating Phat{%d}\n", idx);
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
        fprintf("    target is %f\n", norm(Lfix * kron(eye(N/2^idx), Pidx) * Rfix - Y, 'fro'))
        Phat{idx} = Pidx;
        % update X
    end
    
 
%     Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda);


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
        cvx_begin quiet
            variable Bidx(n,n) complex
            minimize norm(Lfix * kron(eye(N/2^idx), Bidx) * Rfix - Y, 'fro')
            subject to
            norm(Bidx, 'fro') <= 2 * sqrt(2^(idx-1))
            for i = 1 : n
                for j = 1 : n  
                    if i ~= j && (i - n/2) ~= j && (j - n/2) ~= i
                        Bidx(i,j) == 0
                    end
                end
            end
        cvx_end
        Bhat{idx} = Bidx;
        fprintf("    target is %f\n", norm(Lfix * kron(eye(N/2^idx), Bidx) * Rfix - Y, 'fro'))
    end
%     B_est{iter} = Bhat;
    Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda);
    iter = iter + 1;
end