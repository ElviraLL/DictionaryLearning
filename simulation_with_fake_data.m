clc;
clear; 
lambda = 5.0000e-04;
N = 2^2; %dimension for signal N = d = m
p = 100; %number of observations 
num_of_matrix = log2(N);

iterations = 20; 
tol = 0.0001;

fprintf("Generating A...\n");
% generate a linear transformation A
A = dctmtx(N);
%A = dftmtx(N); % generate DFT matrix

% generate a sparse X
% m = d = N
fprintf("Generating X...\n");
for i = 1:p
    X(:,i) = full(sprand(N,1,0.2));
end

% generate Y = A*X
% Y is going to be our images, (4*4) * 100
fprintf("Generating Y...\n");
Y = A * X;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% algorithm
% step 0: random initial all P and B
fprintf("Random initial Phat and Bhat...\n")
for idx = 1:num_of_matrix
    dim = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    Pi = eye(dim);
    permutation = randperm(dim);
    Pi = Pi(permutation, :);
    Phat{idx} = Pi;
    Bhat{idx} = full(sprand(dim, dim, 1));
    % R = sprand(m,n,density)
    % idx represent the size of permitation and sparse matrix
    % generate random sparse matrices
    % S{num_of_matrix + 1 - idx} = full(sprand(i,i,2*i/i^2));
end

fprintf("Algorithm start...\n")
% step 0.1: calculate Ahat for random initials
Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda, X);

% optimization 

iter = 1;
while iter <= iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)
    % step 2: permutations
    % go through from right to left and fix everything else, optimize
    % over the permutation matrices, then optimize over X
    % update from right to left
    Bfix = multiplicationB(Bhat);
    for idx = num_of_matrix:-1:1
        
        % calculate the fixed part for B
        n = size(Phat{idx}) * [1; 0];
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
        Phat{idx} = Pidx;
        % update X
        Xhat = updateX(Phat, Bhat, Y, N ,p, num_of_matrix, lambda, X);
    end
    
    
    % step 3: sparse matrix (non structured now)
    % go through from right to left and fix everything else, optimize
    % over the sparse matrix, then optimize over X
    Pfix = multiplicationP(Phat);
    for idx = 1: num_of_matrix
        
        % size of Bhat{idx} the matrix you want to update
        n = 2^idx;
        % calculate the fixed part for P
        
        % calculate the fixed left part of B
        Rfix = Pfix * Xhat;
        for i = 1 : idx - 1
            Bi = Bhat{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;  
        end
        % calculate the left fixed part of B
        Lfix = eye(N);
        for i = idx + 1 : num_of_matrix
            Bi = Bhat{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;  
        end
        % update Bhat{i}
        % Y = Lfix * (kron(Ii,Bhat{idx})) * Rfix
        
%         RLprod = zeros(size(Y(:)) * [1;0], n^2);
%         for i = 1: (N/n)
%             ri = Rfix((i - 1) * n + 1: i * n,:); % row block of Rfix with size n * N
%             li = Lfix(:, (i - 1) * n + 1: i * n); % col block of Lfix with size N *n
%             RLprod = RLprod + kron(ri', li);
%             % RLprod
%         end
%         yflat = Y(:);

        % solve bi as a sparse vector
        % ??????Bidx??0
        lambda_2 = 1.0e-9;
        fprintf("\n");
        fprintf("    Updating Bhat{%d}\n", idx);
        cvx_begin quiet
            variable Bidx(n,n)
            minimize norm(Lfix * kron(eye(N/2^idx),Bidx)* Rfix - Y, 'fro') + lambda_2 * sum(sum(abs(Bidx)))  
            subject to
            norm(Bidx,'fro')<=n
        cvx_end
%         Bidx = reshape(bidx,[n,n]);
        fprintf("    Difference between updating is %f \n", norm(Bhat{idx} - Bidx, 'fro'))
        Bhat{idx} = Bidx;
        % update X
        Xhat = updateX(Phat, Bhat, Y, N ,p, num_of_matrix, lambda, X);
    end 
    X_est{iter} = Xhat;
    err_x = norm(Xhat - X, 'fro') / norm(X, 'fro');  
    fprintf("    Finished iteration %d, total error is %f\n", iter, err_x)
    fprintf("\n");
    iter = iter + 1;
    if err_x < tol
        break
    end
end
