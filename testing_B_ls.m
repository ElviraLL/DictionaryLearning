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
fprintf("A is a dft matrix: %d\n", norm(A - dftmtx(N)) < 0.0001);
Y = A * X;
y = Y(:);



idx = 3;
Xhat = X;
Phat = P;
Bhat = B;
fprintf('\n');
% calculate Lfix and Rfix
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

AA = Lfix;
BB = kron(eye(N/(2^idx)), Bhat{idx});
CC = Rfix;
norm(Y - AA * BB * CC)


% ???????
fprintf("\n");
fprintf("    Updating Bhat{%d}\n", idx); 
% Let B be butterfly, i.e. restrict the zero position


tic;
Bidx = Bhat{idx};
[ridx, cidx] = find(Bidx);

I = eye(N/2^idx);
Bb = kron(I,Bidx);
norm(kron(Rfix.',Lfix) * Bb(:) - y)
RL = zeros(N*p, n^2);
for j = 1:N/2^idx
    Rj = Rfix(((j-1) * n + 1): j * n, :);
    Lj = Lfix(:, (j-1) * n + 1: j * n);
    RL = RL  + kron(Rj.', Lj);
end
b = Bhat{idx}(:);
norm(RL * Bhat{idx}(:) - y)

non_zero_b_idx = find(b);
RLs = RL(:,non_zero_b_idx);
b = b(non_zero_b_idx);
norm(RLs  * b - y)
% solve b
bhat = RLs\y;
Bidx = sparse(ridx, cidx, bhat);
toc
tic
cvx_begin quiet
    variable Bidxc(n,n) complex
    minimize norm(Lfix * kron(eye(N/2^idx), Bidxc) * Rfix - Y, 'fro')
    subject to
    norm(Bidxc, 'fro') <= 2 * sqrt(2^(idx-1))
    for i = 1 : n
        for j = 1 : n  
            if i ~= j && (i - n/2) ~= j && (j - n/2) ~= i
                Bidxc(i,j) == 0
            end
        end
    end
cvx_end
toc
fprintf("    difference between ls and cvx in Frobenius norm is %f \n", norm(Bidxc - Bidx, 'fro'))


