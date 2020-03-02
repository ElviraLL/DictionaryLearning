clc;
clear;
dat = load("cifar-10-batches-mat/data_batch_1.mat");
data = dat.data;
n_data = size(data,1);
N = 2^3;
num_of_matrix = log2(N);
N0 = 32;


batch_size = 45;
num_of_batch = 20;
p = batch_size * num_of_batch; % size of training data
epoches = 20;
TOL = 0.002;
complex = false;



%%%%%%%%%% pre-processing and random sampling %%%%%%%%%%
data_idx = randsample(n_data, p); %random shuffle
for i = 1:p
    cpt = data_idx(i);
    im = reshape(data(cpt,:),[N0, N0, 3]);
    im = permute(im, [2 1 3]);
    image_gray{i} = double(rgb2gray(im(32-(N-1):32,32-(N-1):32,:)))./255;
    Y(:,i) = reshape(image_gray{i}, N^2, 1);
end

%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%
fprintf("Generating Random Pinv...\n");
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

fprintf("Generating Random Binv...\n");
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


%%%%%%%%%%%%%%%%%%%%%% training %%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:epoches
    epoch_indices  = randperm(p);
    for batch = 1:num_of_batch
        % get Y and Ycat
        for i = 1:batch_size
            idx = epoch_indices(i + (batch - 1) * batch_size);
            if i == 1
                Ycat = image_gray{idx};
                Ycat2 = image_gray{idx}.';
            else
                Ycat = cat(2, Ycat, image_gray{idx});
                Ycat2 = cat(2, Ycat2, image_gray{idx}.');
            end
            Y_batch(:,i) = reshape(image_gray{idx}, N^2, 1);
        end
        
        Xhat_batch = getX(Ahat, Y_batch, 0.99, batch_size);
        rel_target = norm(Y_batch - Ahat * Xhat_batch, 'fro')/norm(Y_batch,'fro');
        iter = 1;
        while rel_target > TOL
            fprintf("-----------------------------------------------------------\n")
            fprintf("               Epoch %d, Batch %d, Iteration %d              \n",epoch, batch, iter)
            fprintf("-----------------------------------------------------------\n")
            Zcat = (reshape(Xhat_batch(:,1),[N,N])) * Dinv2.';
            for i = 2:batch_size
                Zcat = cat(2, Zcat, ((reshape(Xhat_batch(:,i),[N,N])) * Dinv2.'));
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
                fprintf("    Updating Pinv1{%d}", idx);
                RL = zeros(N * N * batch_size, n^2);
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
                [vv, ii]=max(Pinv1{idx});
                Temp=zeros(n);
                for jj=1:n
                    Temp(ii(jj),jj)=1;
                end
                pidx=Temp;
                %%%%%%%%%%%%%%%%%%
                Pinv1{idx} = pidx;
                Dinv1 = get_A(Binv1, Pinv1);
                fprintf("   target is %.2f\n", norm(Y_batch - kron(Dinv2, Dinv1) * Xhat_batch, 'fro'))
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

                fprintf("    Updating Binv1{%d}", idx); 
                % Let B be butterfly, i.e. restrict the zero position
                [ridx, cidx] = find(B0{idx}); % non-zero index of the matrix B{idx}
                RL = zeros(N*N*batch_size, n^2);
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
                fprintf("   target is %.2f\n", norm(Y_batch - kron(Dinv2, Dinv1) * Xhat_batch, 'fro'))
            end
            
            Dinv1 = get_A(Binv1, Pinv1);
            Zcat2 = (Dinv1 * reshape(Xhat_batch(:,1),[N,N])).';
            for i = 2:batch_size
                Zcat2 = cat(2, Zcat2, (Dinv1 * reshape(Xhat_batch(:,i),[N,N])).');    
            end
            y = Ycat2(:);
            Zcat = Zcat2;
            fprintf('\n');
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
                fprintf("    Updating Pinv2{%d}", idx);
                RL = zeros(N*N*batch_size, n^2);
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
                fprintf("   target is %.2f\n", norm(Y_batch - kron(Dinv2, Dinv1) * Xhat_batch, 'fro'))
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

                fprintf("    Updating Binv2{%d}", idx); 
                [ridx, cidx] = find(B0{idx}); 
                RL = zeros(N*N*batch_size, n^2);
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
                fprintf("   target is %.2f\n", norm(Y_batch - kron(Dinv2, Dinv1) * Xhat_batch, 'fro'))
            end

            Dinv1 = get_A(Binv1, Pinv1);
            Dinv2 = get_A(Binv2, Pinv2);
            Ahat = kron(Dinv2, Dinv1);
            fprintf("    Updating Xhat");
            sigma = 1 * .99^iter;
            Xhat_batch = getX(Ahat, Y_batch, sigma, batch_size);
            target = norm(Y_batch - Ahat * Xhat_batch, 'fro');
            rel_target = norm(Y_batch - Ahat * Xhat_batch, 'fro')/norm(Y_batch,'fro');
            sparsity = get_ave_sparsity(Xhat_batch);
            fprintf("       target is %.2f, sparsity is %.2f;\n", target, sparsity );
            iter = iter + 1; 
        end
        % put Xhat back
        for i = 1 : batch_size
            idx = epoch_indices(i + (batch - 1) * batch_size);
            Xhat(:, idx) = Xhat_batch(:,i);
        end
    end
    epoch_error(epoch) = norm(Y - Ahat * Xhat, 'fro');
end



            
            
        
        


