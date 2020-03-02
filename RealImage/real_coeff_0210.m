clc;
clear;
dat = load("cifar-10-b0atches-mat/data_batch_1.mat");
data = dat.data;
p = 200;
iterations = 8000;
TOL = 0.05;
N = 2^5; 
num_of_matrix = log2(N);
N0=32;
lambda2=0.5;
complex = true;

D = dftmtx(N);
% Dinv2 = inv(D);
im=zeros(N0,N0,3);
for cpt=1:p
    im = reshape(data(cpt,:),[N0, N0, 3]);
    im = permute(im, [2 1 3]);
    image_gray{cpt} = double(rgb2gray(im(32-(N-1):32,32-(N-1):32,:)))./255;
    real_sparse{cpt} = inv(D) * image_gray{cpt} * inv(D).';
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

tic
%since the fourier transfer is doing wrt the column of images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Step 0: random initial Binv1 and Pinv1 and solve the initial Xhat  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("Generating Random Pinv1...\n");
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

fprintf("Generating Random Binv1...\n");
if complex
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
else
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
        B0{idx} = Bi./norms(Bi);
    end
end
    
Binv1 = B0;
Binv2 = B0;
Dinv1 = get_A(Binv1, Pinv1);
Dinv2 = get_A(Binv2, Pinv2);
Ahat = kron(Dinv2, Dinv1);
Xhat = getX(Ahat, Y, 0.999, p);
% A = kron(D, D);
% Xhat = inv(A) * Y;



iter = 1
%%
while iter < iterations
    fprintf("----------------------iteration %d--------------------------\n", iter)

    Zcat = (reshape(Xhat(:,1),[N,N])) * Dinv2.';
    for i = 2:p
        Zcat = cat(2, Zcat, ((reshape(Xhat(:,i),[N,N])) * Dinv2.'));
    end
    y = Ycat(:);
    Bfix = multiplicationB(Binv1);
    for idx = num_of_matrix:-1:1
        % calculate the fixed part for B
        n = 2^idx;
        % calculate the left fix part for P
        PLfix = eye(N);
        for i = (idx - 1):-1:1
            Pi = Pinv1{i};
            Ii = eye(N / 2^i);
            Pi = kron(Ii, Pi);
            PLfix = Pi * PLfix;
        end
        Lfix = Bfix * PLfix;
        % calculate the right fixed part for P
        Rfix = Zcat;
        for i = num_of_matrix: -1 :idx + 1
            Pi = Pinv1{i};
            Ii = eye(N/2^i);
            Pi = kron(Ii, Pi);
            Rfix = Pi * Rfix;
        end
        fprintf("\n");
        fprintf("    Updating Pinv1{%d}", idx);
        RL = zeros(N*N*p, n^2);
        for j = 1:N/2^idx
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        
        RL_real = [real(RL); imag(RL)];
        y_real = [real(y); imag(y)];
        Aeq = [kron(eye(n), ones(n,1).'); kron(ones(n,1).', eye(n))];
        beq = [ones(2*n,1)];
        
        options = optimoptions('lsqlin','Display',"off");
        ph = lsqlin(RL_real, y_real, [],[], Aeq, beq, zeros(n^2,1), ones(n^2,1),[],options);
        pidx = reshape(ph, [n,n]);

        %%%%%%DANGER%%%%%%
        [vv, ii]=max(Pinv1{idx});
        Temp=zeros(n);
        for jj=1:n
            Temp(ii(jj),jj)=1;
        end
        pidx=Temp;
        %%%%%%%%%%%%%%%%%%
        Pinv1{idx} = pidx;
        Dinv1 = get_A(Binv1, Pinv1);
        fprintf("   target is %.2f\n", norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'))
    end
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            Solve B                                   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('\n');
    % calculate L and R
    Pfix = multiplicationP(Pinv1);
    for idx = 1: num_of_matrix
        Rfix = Pfix * Zcat;
        n = 2^idx;
        for i = 1:idx - 1
            Bi = Binv1{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;
        end
        
        Lfix = eye(N);
        for i = (idx + 1) : num_of_matrix
            Bi = Binv1{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;
        end
        
        fprintf("\n");
        fprintf("    Updating Binv1{%d}", idx); 
        % Let B be butterfly, i.e. restrict the zero position
        [ridx, cidx] = find(B0{idx}); % non-zero index of the matrix B{idx}
        RL = zeros(N*N*p, n^2);
        for j = 1:N/n
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        non_zero_b_idx = find(B0{idx}(:));
        RLs = RL(:,non_zero_b_idx);
        Bidx = (RLs + randn(size(RLs))/(sqrt(size(RLs,1))+sqrt(size(RLs,2)))*1e-2 )\y;
        Bidx = full(sparse(ridx, cidx, Bidx));
        Binv1{idx} = Bidx./(norms(Bidx)+eps);         
        Dinv1 = get_A(Binv1, Pinv1);
        fprintf("   target is %.2f\n", norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'))
    end

    Dinv1 = get_A(Binv1, Pinv1);
    Zcat2 = (Dinv1 * reshape(Xhat(:,1),[N,N])).';
    for i = 2:p
        Zcat2 = cat(2, Zcat2, (Dinv1 * reshape(Xhat(:,i),[N,N])).');    
    end
    y = Ycat2(:);
    Zcat = Zcat2;
    Bfix = multiplicationB(Binv2);
    for idx = num_of_matrix:-1:1
        % calculate the fixed part for B
        n = 2^idx;
        % calculate the left fix part for P
        PLfix = eye(N);
        for i = (idx - 1):-1:1
            Pi = Pinv2{i};
            Ii = eye(N / 2^i);
            Pi = kron(Ii, Pi);
            PLfix = Pi * PLfix;
        end
        Lfix = Bfix * PLfix;
        % calculate the right fixed part for P
        Rfix = Zcat;
        for i = num_of_matrix: -1 :idx + 1
            Pi = Pinv2{i};
            Ii = eye(N/2^i);
            Pi = kron(Ii, Pi);
            Rfix = Pi * Rfix;
        end
        fprintf("\n");
        fprintf("    Updating Pinv2{%d}", idx);
        RL = zeros(N*N*p, n^2);
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
        %%%%%%DANGER%%%%%%
        [vv, ii]=max(Pinv2{idx});
        Temp=zeros(n);
        for jj=1:n
            Temp(ii(jj),jj)=1;
        end
        pidx=Temp;
        %%%%%%%%%%%%%%%%%%
        Pinv2{idx} = pidx;
        Dinv1 = get_A(Binv1, Pinv2);
        fprintf("   target is %.2f\n", norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'))
    end
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                            Solve B                                   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('\n');
    % calculate L and R
    Pfix = multiplicationP(Pinv2);
    for idx = 1: num_of_matrix
        Rfix = Pfix * Zcat;
        n = 2^idx;
        for i = 1:idx - 1
            Bi = Binv2{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;
        end
        
        Lfix = eye(N);
        for i = (idx + 1) : num_of_matrix
            Bi = Binv2{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;
        end
        
        fprintf("\n");
        fprintf("    Updating Binv2{%d}", idx); 
        [ridx, cidx] = find(B0{idx}); 
        RL = zeros(N*N*p, n^2);
        for j = 1:N/n
            Rj = Rfix(((j-1) * n + 1): j * n, :);
            Lj = Lfix(:, (j-1) * n + 1: j * n);
            RL = RL  + kron(Rj.', Lj);
        end
        non_zero_b_idx = find(B0{idx}(:));
        RLs = RL(:,non_zero_b_idx);
        Bidx = (RLs + randn(size(RLs))/(sqrt(size(RLs,1))+sqrt(size(RLs,2)))*1e-2 )\y;
        
        Bidx = full(sparse(ridx, cidx, Bidx));
        Binv2{idx} = Bidx./(norms(Bidx)+eps);         
        %Binv1{idx} = Bidx.;
        Dinv2 = get_A(Binv2, Pinv2);
        fprintf("   target is %.2f\n", norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'))
    end

    Dinv1 = get_A(Binv1, Pinv1);
    Dinv2 = get_A(Binv2, Pinv2);
    Ahat = kron(Dinv2, Dinv1);
    sigma = 1 * .99^iter;
    Xhat = getX(Ahat, Y, sigma, p);
    fprintf("          target is %.2f;\n",norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'))

    diagnostic1(iter,:)=[iter cond(full(Binv1{1})) cond(full(Binv1{2})) cond(full(Binv1{3})) norm(Xhat) mean((norms(Xhat,1).^2./norms(Xhat,2).^2))]
    diagnostic2(iter,:)=[iter cond(full(Binv2{1})) cond(full(Binv2{2})) cond(full(Binv2{3})) norm(Xhat) mean((norms(Xhat,1).^2./norms(Xhat,2).^2))]

    target = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro'); 
    rel_target(iter) = norm(Y - kron(Dinv2, Dinv1) * Xhat, 'fro')/norm(Y,'fro')
    Zcat_rec{iter} = Zcat;
    X_rec{iter} = Xhat;
    B1_rec{iter} = Binv1;
    P1_rec{iter} = Pinv1;
    target_rec{iter} = target;
    if rel_target(iter) < TOL
        break
    end
    iter = iter + 1;
end

toc
%%

for i = 1:p
    Xhat_sparse{i} = reshape(Xhat(:,i), N, N);
    Yhat{i} = Dinv1 * Xhat_sparse{i} * Dinv2.';
end

idd = randi([1 p],1,1)

subplot(3,2,1)
histfit(sort(real(Xhat(:))),50)
title("sparsity of Xest")

subplot(3,2,2)
plot(rel_target)
title("relavent error")

subplot(3,2,3)
imagesc(abs(inv(Dinv1)'*D))
title("correlation between Dinv1 and fft");

subplot(3,2,4)
imagesc(abs(inv(Dinv2)'*D))
title("correlation between Dinv2 and fft");



subplot(3,2,5)
imagesc(abs(Yhat{idd}))
title("recovered image")

subplot(3,2,6)
imagesc(abs(image_gray{idd}))
title("original image")

D = dftmtx(N);
A = kron(D,D);
Xdft = A * Y;

dct = dctmtx(N);
Adct = kron(dct, dct);
Xdct = Adct * Y;
[mean((norms(Xhat,1).^2) ./ (norms(Xhat,2).^2)), mean((norms(Xdft,1).^2) ./ (norms(Xdft,2).^2)), mean((norms(Xdct,1).^2) ./ (norms(Xdct,2).^2))]



