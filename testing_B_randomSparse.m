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

fprintf("Generating P and B...\n");
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
    B{idx} = full(sprand(n, n, 1));

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
    Bhat{idx} = full(sprand(n, n, 1));
end


Xhat = X;
Pfix = multiplicationP(P);
Rfix = Pfix * Xhat;
iter = 1;
while iter < iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % from right to left, iteratively solve B 2*2, 4*4, 8*8, .....
    
    % calculate L and R
    for idx = 1: num_of_matrix
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
        
 
        % Let B be butterfly, i.e. restrict the zero position
        lambda_2 = 1.0e-9;
        fprintf("\n");
        fprintf("    Updating Bhat{%d}\n", idx);
        cvx_begin quiet
            variable Bidx(n,n)
            minimize norm(Lfix * kron(eye(N/2^idx),Bidx)* Rfix - Y, 'fro') + lambda_2 * sum(sum(abs(Bidx)))  
            subject to
            norm(Bidx,'fro')<=n
        cvx_end
        Bhat{idx} = Bidx;
        error_B = norm(Bidx - B{idx} ,'fro') / norm(B{idx},'fro');
        fprintf("    Relative Error for B is %f\n", error_B);
    end
    B_est{iter} = Bhat;
    iter = iter + 1;
    fprintf("\n");
end

