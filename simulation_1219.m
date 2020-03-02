% x_i individually? not working
% initial Phat in solver

% real matrix A, real X, real y, 
% linear constrained ls to sove permutation
% ls to solve butterfly
clc;
clear;

N = 2^3;
p = 100;
lambda = 0.3;
lambda2 = 0.5;
num_of_matrix = log2(N);
iterations = 2;
TOL = 1e-03;

fprintf("Generating X...\n");
for i = 1:p
    X(:,i) = full(sprand(N,1,0.2));
end
A = dftmtx(N);
% A = hadamard(N);
Y = A*X;
y = Y(:);
%%
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
    P0{idx} = Pi;   
end
Phat = P0;

fprintf("Generating Random Bhat...\n");
for idx = 1:num_of_matrix
    n = 2 ^ idx;
    half = n / 2;
    Bi = zeros(n);
    for i = 1 : half
%         Bi(i,i) = randn(1) + randn(1) * j;
%         Bi(half + i, i) = randn(1)+ randn(1) * j;
%         Bi(half + i, half + i) = randn(1)+ randn(1) * j;
%         Bi(i, half + i) = randn(1)+ randn(1) * j;
        Bi(i,i) = randn(1);
        Bi(half + i, i) = randn(1);
        Bi(half + i, half + i) = randn(1);
        Bi(i, half + i) = randn(1);
    end
    B0{idx} = Bi;
end
Bhat = B0;

Ahat = get_A(Bhat,Phat);
% Xhat = updateX(false, Ahat, Y, N , p, 0.005); %for real matrices
Xhat = inv(Ahat)*Y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Step 1: Iteratively solve Phat and Bhat and Xhat   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1;
lambda_target=0.1;
lambda=0.05;

k_target = 0.4 * N;
k = N;
while iter < iterations
%%%%%%%%%%
%     lambda=min(1.1*lambda, lambda_target);
%     lambda
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
        fprintf("\n");
        fprintf("    Updating Phat{%d}\n", idx);
        RL = zeros(N*p, n^2);
        for j = 1:N/2^idx
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        
        RL_real = [real(RL);imag(RL)];
        y_real = [real(y); imag(y)];
        Aeq = [kron(eye(n), ones(n,1).'); kron(ones(n,1).', eye(n))];
        beq = [ones(2*n,1)];
        
        options = optimoptions('lsqlin','Display',"off");
        ph = lsqlin(RL_real, y_real, [],[], Aeq, beq, zeros(n^2,1), ones(n^2,1),[],options);
        pidx = reshape(ph, [n,n]);
        
        Phat{idx} = pidx;
        
        
        %%%%%%DANGER%%%%%%
        [vv, ii]=max(Phat{idx});
        Temp=zeros(n);
        for jj=1:n
            Temp(ii(jj),jj)=1;
        end
        %%%%%%%%%%%%%%%%%%
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
        [ridx, cidx] = find(B0{idx}); % non-zero index of the matrix B{idx}
        RL = zeros(N*p, n^2);
        for j = 1:N/n
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        non_zero_b_idx = find(B0{idx}(:));
        RLs = RL(:,non_zero_b_idx);
        

        if idx == 1 
            bhat = inv(RLs.' * RLs + lambda2 * eye(4)) * RLs.' * y;
        else
            bhat = RLs\y;
        end
                
%         [mm,nn] = size(RLs);
%         cvx_begin
%         variable bhat(nn,1)
%         minimize (norm(y - RLs * bhat, 2))
%         subject to 
%             norm(bhat) <= 5*sqrt(n)
%         cvx_end 
        
        Bidx = sparse(ridx, cidx, bhat);
%         Bhat{idx} = Bidx./norms(Bidx);        
        Bhat{idx} = Bidx;
    end
    
    Ahat = get_A(Bhat, Phat);
    Ahat = Ahat./norms(Ahat+eps);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            Solve X                                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Xhat = updateXspgl1(Ahat, Y, Xhat, norm(Y,'fro')*.95^(iter), []);
     sigma(iter) = 1*.98^iter;
     Xhat = updateXsep(Ahat, Y, 1*.995^iter, p); 
     diagnostic(iter,:)=[cond(full(Bhat{1})) cond(full(Bhat{2})) cond(full(Bhat{3})) norm(Xhat) mean((norms(Xhat,1).^2./norms(Xhat,2).^2))]
     rel_target(iter) = norm(Ahat * Xhat - Y, 'fro')/norm(Y,'fro')

    target = norm(Ahat * Xhat - Y, 'fro');
    sparsity = mean((norms(Xhat,1)./norms(Xhat,2)).^2);
    sparse_rec(iter) = sparsity;
    Xrec{iter} = Xhat;
    Brec{iter} = Bhat;
    Prec{iter} = Phat;
    target_rec{iter} = target;
    
    fprintf("    target  is %f ", target)
    fprintf(" target < TOL: %d\n", target < TOL)
    fprintf("    sparsity is %f ", sparsity)
    fprintf("sparse: %d\n", sparsity <= 0.5 * N)
    
    if target < TOL && sparsity <= 0.5 * N
        break
    end
    iter = iter + 1
end
%%
fprintf("Optimization finished");
svd((Ahat))
Xest = Ahat\Y;


subplot(2,2,1)
histfit(sort(real(Xest(:))),50)
title("sparsity of Xest")

subplot(2,2,2)
histfit(sort(real(Xhat(:))),50)
title("sparsity of Xhat")

subplot(2,2,3)
imagesc(abs(Ahat'*A))
title("Ahat' * A")

subplot(2,2,4)
imagesc(real(Ahat'*A))
title("real(Ahat' * A)")

norm(multiplicationB(Bhat))
absXhat = abs(Xhat);
[mean((norms(Xhat,1)./norms(Xhat,2)).^2), mean((norms(Xest,1)./norms(Xest,2)).^2), mean((norms(X,1)./norms(X,2)).^2)]

figure
plot(rel_target)
% F = kron(dftmtx(2),dftmtx(4));
% XFourier = F * Y;
% mean((norms(XFourier b ,1)./no        67y9rms(XFourier,2)).^2)
% figure
% histfit(sort(real(XFourier(:))),50)

% todo: learn for image it self