function [Pinv, Binv] = solvePB(Ycat, Zcat, Pinv, Binv, num_of_matrix, N, p, B0)
%iteratively solve P's and B's from [Ycat] = (PPPP * BBBB) [Zcat]
%solve Binv's
y = Ycat(:);
Pfix = multiplicationPinv(Pinv);
fprintf("\n    updating Binv's");
for idx = num_of_matrix:-1:1
    fprintf(" ")
    fprintf("%d", idx)
    n = 2^idx;
    [Lfix,Rfix] = getLRfix(Binv, idx, num_of_matrix, N, true);
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
    RLs = RL(:,non_zero_b_idx)+eps;
    if idx == 1
        bhat = inv(RLs.' * RLs + 0.5 * eye(4)) * RLs.' * y;
    else
        bhat = RLs\y;
    end
    Bidx = sparse(ridx, cidx, bhat);
    Binv{idx} = Bidx./norms(Bidx);
end


%solve Pinv's
fprintf("\n    updating Pinv's");
Bfix = multiplicationBinv(Binv);
for idx = 1:num_of_matrix
    fprintf(" ")
    fprintf("%d", idx)
    n = 2^idx;
    [Lfix, Rfix] = getLRfix(Pinv, idx, num_of_matrix, N, false);
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
%     %%%%DANGER%%%%%%
%     [vv, ii]=max(pidx);
%     Temp=zeros(n);
%     for jj=1:n
%         Temp(ii(jj),jj)=1;
%     end
%     pidx=Temp;
%     %%%%%%%%%%%%%%%%
    Pinv{idx} = pidx;
end
end