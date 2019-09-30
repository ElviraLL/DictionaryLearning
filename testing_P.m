clc;
clear; 
lambda = 5.0000e-04;
N = 2^3; %dimension for signal N = d = m
p = 100; %number of observations 
num_of_matrix = log2(N);
iterations = 20;

fprintf("Generating X...\n");
for i = 1:p
    X(:,i) = full(sprand(N,1,0.2));
end

fprintf("Generating P...\n");
for idx = 1:num_of_matrix
    n = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    half = n / 2;
    Pi = zeros(n);
    for i = 1 : half
        Pi(i,2*i - 1) = 1;
        Pi(i + half, 2 * i) = 1;
    end
    P{idx} = Pi;
end

fprintf("Generating Butterfly B...\n");
for idx = 1:num_of_matrix
    n = 2 ^ idx;
    half = n / 2;
    Bi = zeros(n);
    for i = 1 : half
        Bi(i,i) = 1;
        Bi(half + i, i) = 1;
        w = exp(2 * pi * j / n); % here j is the complex unit
        Bi(half + i, half + i) = - w^(-(i-1));
        Bi(i, half + i) = w^(-(i-1));
    end
    B{idx} = Bi;
end

A = get_A(B,P);
% A = dftmtx(N);

Y = A * X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Test 1: fix X and B than solve P                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Random Phat
fprintf("Generating Random Phat...\n");
for idx = 1:num_of_matrix
    dim = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    Pi = eye(dim);
    permutation = randperm(dim);
    Pi = Pi(permutation, :);
    Phat{idx} = Pi;   
end


Xhat = X;
Bfix = multiplicationB(B);
iter = 1;
while iter < iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % from right to left, iteratively solve B 2*2, 4*4, 8*8, .....
    
    % calculate L and R
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
        fprintf("    Difference between updating is %f \n", norm(Phat{idx} - Pidx, 'fro'))
        fprintf("    Relative Error is %f\n", norm(Phat{idx} - P{idx}, 'fro')/norm(P{idx},'fro'))
       
        Phat{idx} = Pidx;
        % update X
    end
    iter = iter + 1;
end