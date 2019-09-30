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
%                 Test 1: fix P and B than solve X                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1;
while iter < iterations
    Xhat = updateX(P, B, Y, N , p, num_of_matrix, lambda, X);
    iter = iter + 1;
end
    