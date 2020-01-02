clc;
clear; 
lambda = 5.0000e-04;
N = 2^5; %dimension for signal N = d = m
p = 100; %number of observations 
num_of_matrix = log2(N);
iterations = 1000;

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
fprintf("A is a dft matrix: %d\n", norm(A - dftmtx(N)) < 0.0001);
Y = A * X;


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

Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda, X);




iter = 1;
while iter < iterations
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
        fprintf("    Difference between updating is %f \n", norm(Phat{idx} - Pidx, 'fro'))
        fprintf("    Relative Error is %f\n", norm(Phat{idx} - P{idx}, 'fro')/norm(P{idx},'fro'))
        fprintf("    target is %f\n", norm(Lfix * kron(eye(N/2^idx), Pidx) * Rfix - Y, 'fro'))
        Phat{idx} = Pidx;
        % update X
    end
    
    
    
    Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda, X);

    
    
    
    % calculate L and R
    for idx = 1: num_of_matrix
        Pfix = multiplicationP(P);
        Rfix = Pfix * X;
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
        error_B = norm(Bidx - B{idx} ,'fro') / norm(B{idx},'fro');
        fprintf("    Relative Error for B is %f\n", error_B);
        fprintf("    target is %f\n", norm(Lfix * kron(eye(N/2^idx), Bidx) * Rfix - Y, 'fro'))
    end
    B_est{iter} = Bhat;
    Xhat = updateX(Pbat, Bhat, Y, N , p, num_of_matrix, lambda, X);

    fprintf("    Error in A is %f\n", norm(A - get_A(Bhat,Phat), 'fro'));
    fprintf("    Error in Y is %f\n", norm(Y - get_A(Bhat,Phat) * Xhat, 'fro'));
    iter = iter + 1;
end



















