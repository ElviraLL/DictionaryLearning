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
%                 Test 1: fix X and P than solve B                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Random Bhat
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


% Pfix = multiplicationP(P);
% Rfix = Pfix * X;
iter = 1;
while iter < iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % from right to left, iteratively solve B 2*2, 4*4, 8*8, .....
    
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
    iter = iter + 1;
    fprintf("\n");
end


% make sure norm(get_A(Bhat, P)*X - A*X) matches the target
% ls version of optimization
