%TODO:
% Now Yi = inv(D) * Xi * (inv(D)).', where Yi is the ith 'image'
% Thus we have Y = kron(inv(D), inv(D)) * X
clc;
clear;
dat = load("cifar-10-batches-mat/data_batch_1.mat");
data = dat.data;
p = 50;
iterations = 200;
TOL = 2;

N = 8;
num_of_matrix = log2(N);
N0=32;
im=zeros(N0,N0,3);
for cpt=1:p
    im = reshape(data(cpt,:),[N0, N0, 3]);
    im = permute(im, [2 1 3]);
    image_gray{cpt} = double(rgb2gray(im(22-(N-1):22,22-(N-1):22,:)))./255;
%     image_gray{cpt} = double(rgb2gray(im))./255;
    if cpt == 1
        Ycat = image_gray{cpt};
        Ycat2 = image_gray{cpt}.';
    else
        Ycat = cat(2, Ycat, image_gray{cpt});
        Ycat2 = cat(2, Ycat2, (image_gray{cpt}.'));   
    end
    Y(:,cpt) = reshape(image_gray{cpt}, [N^2,1]); % this is the 1024*1 version
end

% fprintf("Generating X...\n");
% D1 = dftmtx(N);
% D2 = dftmtx(N);
% for i = 1:p
%     X(:,i) = full(sprand(N^2,1,0.2));
%     image_gray{i} = inv(D1) * reshape(X(:,i), N, N) * inv(D2).';
%     if i == 1
%         Ycat = image_gray{i};
%         Ycat2 = image_gray{i}.';
%     else
%         Ycat = cat(2, Ycat, image_gray{i});
%         Ycat2 = cat(2, Ycat2, (image_gray{i}.'));   
%     end
%     Y(:,i) = reshape(image_gray{i}, [N^2,1]); % this is the 1024*1 version    
% end


%since the fourier transfer is doing wrt the column of images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Step 0: random initial Bhat and Phat and solve the initial Xhat  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("Generating Random Phat...\n");
for idx = 1:num_of_matrix
    dim = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    Pi = eye(dim);
    permutation = randperm(dim);
    Pi = Pi(permutation, :);
    P0{idx} = Pi; 
end
Pinv1 = P0;
Pinv2 = P0;

fprintf("Generating Random Bhat...\n");
for idx = 1:num_of_matrix
    n = 2 ^ idx;
    half = n / 2;
    Bi = zeros(n);
    for i = 1 : half
        Bi(i,i) = randn(1) + randn(1) * j;
        Bi(half + i, i) = randn(1) + randn(1) * j;
        Bi(half + i, half + i) = randn(1) + randn(1) * j;
        Bi(i, half + i) = randn(1) + randn(1) * j;
    end
    B0{idx} = Bi./norms(Bi);
end
Binv1 = B0;
Binv2 = B0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Iterations: alternatively solve A and X               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1;
%%
while iter < iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)
    
    %%%%%%%% Xhat %%%%%%%%
    Dinv1 = getDinv(Binv1, Pinv1);
    Dinv2 = getDinv(Binv2, Pinv2);
    Ahat = kron(Dinv1, Dinv2);
    if iter <= 5
        sigma = 0.001;
    else
        sigma = 1*0.96^iter;
    end
    Xhat = updateX(Ahat, Y, sigma, p);
    

    %%%%%%%% Ahat %%%%%%%%
    % Y = AX = Dinv2 kron Dinv1 X  
    % [Yi]_32*32 = Dinv1 [Xi] Dinv2^T
    % [Yi]_32*3200 = Dinv1 [Xi * Dinv2^T]_32*3200
    % [Ycat] = Dinv1 * [Zcat] (stack)

    % rewrite the problem into Y = Dinv1 Z form
    Zcat = (reshape(Xhat(:,1),[N,N])) * Dinv2.';
    for i = 2:p
        Zcat = cat(2, Zcat, ((reshape(Xhat(:,i),[N,N])) * Dinv2.'));
    end

    y = Ycat(:);
    Pfix = multiplicationPinv(Pinv1);
    fprintf("\n    updating Binv1's");
    for idx = num_of_matrix:-1:1
        n = 2^idx;
        [Lfix,Rfix] = getLRfix(Binv1, idx, num_of_matrix, N, true);
        Lfix = Pfix * Lfix;
        Rfix = Rfix * Zcat;
        [ridx, cidx] = find(B0{idx}); % non-zero index of the matrix B{idx}
        RL = zeros(N * N * p, n^2);
        for j = 1 : N/n
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        non_zero_b_idx = find(B0{idx}(:));
        RLs = RL(:,non_zero_b_idx);
        if idx == 1
            bhat = inv(RLs.' * RLs + 0.5 * eye(4)) * RLs.' * y;
        else
            bhat = RLs\y;
        end
        Bidx = sparse(ridx, cidx, bhat);
        Dinv1 = getDinv(Binv1, Pinv1);
        target = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro');
        fprintf("  Binv1{%d}: %.1f", idx, target);
        Binv1{idx} = Bidx./norms(Bidx);
    end


    %solve Pinv1's
    fprintf("\n    updating Pinv1's");
    Bfix = multiplicationBinv(Binv1);
    for idx = 1:num_of_matrix
        n = 2^idx;
        [Lfix, Rfix] = getLRfix(Pinv1, idx, num_of_matrix, N, false);
        Rfix = Rfix * Bfix * Zcat;
        RL = zeros(N* N * p, n^2);
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
%         %%%%%%DANGER%%%%%%
%         [vv, ii]=max(pidx);
%         Temp=zeros(n);
%         for jj=1:n
%             Temp(ii(jj),jj)=1;
%         end
%         pidx=Temp;
%         %%%%%%%%%%%%%%%%%%
        Dinv1 = getDinv(Binv1, Pinv1);
        target = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro');
        fprintf("  Pinv1{%d}: %.1f", idx, target);
        Pinv1{idx} = pidx;
    end

    
    % Y = AX = Dinv2 kron Dinv1 X  
    % [Yi]_32*32 = Dinv1 [Xi]  Dinv2^T
    % [Yi]_32*32 =(Dinv1 [Xi]) Dinv2^T
    % [Yi]^T  = Dinv2  (Dinv1 [Xi])^T
    % [Ycat2] = Dinv2 * [Zcat2] (stack)
    Dinv1 = getDinv(Binv1, Pinv1);
    Zcat2 = (Dinv1 * reshape(Xhat(:,1),[N,N])).';
    for i = 2:p
        Zcat2 = cat(2, Zcat2, (Dinv1 * reshape(Xhat(:,i),[N,N])).');    
    end
    
    y = Ycat2(:);
    Pfix = multiplicationPinv(Pinv2);
    fprintf("\n    updating Binv2's");
    for idx = num_of_matrix:-1:1
        n = 2^idx;
        [Lfix,Rfix] = getLRfix(Binv2, idx, num_of_matrix, N, true);
        Lfix = Pfix * Lfix;
        Rfix = Rfix * Zcat2;
        [ridx, cidx] = find(B0{idx}); % non-zero index of the matrix B{idx}
        RL = zeros(N * N * p, n^2);
        for j = 1 : N/n
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        non_zero_b_idx = find(B0{idx}(:));
        RLs = RL(:,non_zero_b_idx)+eps;
        if idx == 1
            bhat = inv(RLs.' * RLs + 0.5 * eye(4)) * RLs.' * y;
        else
            bhat = RLs\y;
        end
        Bidx = sparse(ridx, cidx, bhat);
        Dinv2 = getDinv(Binv2, Pinv2);
        target = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro');
        fprintf("  Binv2{%d}: %.1f", idx, target);
        Binv2{idx} = Bidx./norms(Bidx);
    end


    %solve Pinv's
    fprintf("\n    updating Pinv2's");
    Bfix = multiplicationBinv(Binv2);
    for idx = 1:num_of_matrix
        n = 2^idx;
        [Lfix, Rfix] = getLRfix(Pinv2, idx, num_of_matrix, N, false);
        Rfix = Rfix * Bfix * Zcat2;
        RL = zeros(N* N * p, n^2);
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
%         %%%%%%DANGER%%%%%%
%         [vv, ii]=max(pidx);
%         Temp=zeros(n);
%         for jj=1:n
%             Temp(ii(jj),jj)=1;
%         end
%         pidx=Temp;
%         %%%%%%%%%%%%%%%%%%
        Dinv2 = getDinv(Binv2, Pinv2);
        target = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro');
        fprintf("  Pinv2{%d}: %.1f", idx, target);
        Pinv2{idx} = pidx;
    end
   
%     Xhat = updateXsep(Ahat, Y, 1*.98^iter, p); 
    sparsity = mean((norms(Xhat,1)./norms(Xhat,2)).^2);
    sparse_rec(iter) = sparsity;
    X_rec{iter} = Xhat;
    B1_rec{iter} = Binv1;
    B2_rec{iter} = Binv2;
    P1_rec{iter} = Pinv1;
    P2_rec{iter} = Pinv2;
    target_rec{iter} = target;
    
    fprintf('\n');
    fprintf("    target  is %f ", target)
    fprintf(" target < TOL: %d\n", target < TOL)
    fprintf("    sparsity is %f ", sparsity)
    fprintf("sparse: %d\n", sparsity <= 0.5 * N)
    
    if target < TOL && sparsity <= 0.5 * N
        break
    end
    iter = iter + 1;
end
%%
fprintf("Optimization finished");
svd((Ahat));

for i = 1:p
    Yhat{i} = Dinv1 * reshape(Xhat(:,i), N, N) * Dinv2.';
end


subplot(2,2,1)
histfit(sort(real(Xhat(:))),50)
title("sparsity of Xest")

subplot(2,2,3)
imagesc(abs(Dinv1') * abs(dftmtx(N)));
title("correlation between Dinv1 and fft");

subplot(2,2,4)
imagesc(abs(Dinv2') * abs(dftmtx(N)));
title("correlation between Dinv2 and fft");

figure
idd = 3
subplot(1,2,1)
imagesc(abs(Yhat{idd}))
title("recoveried image")

subplot(1,2,2)
imagesc(abs(image_gray{idd}))
title("original image")
