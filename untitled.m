i = 1;
lambdas = [0:0.00002:0.0001,0.0002:0.0002: 0.001, 0.002:0.002:0.01, 0.02:0.02: 0.1, 0.2:0.2:1];
for lambda = lambdas
    lambda
    cvx_begin quiet
        variable Bidx(n,n)
        minimize norm(Lfix * kron(eye(N/2^idx),Bidx)* Rfix - Y, 'fro') + lambda * norm(Bidx, 1)  
    cvx_end
    B_test{i} = Bidx;
    fprintf("Difference between updating is %f \n", norm(Bhat{idx} - Bidx, 'fro'))
    i = i + 1
end


cvx_begin
    variable Xt(N,p)
    minimize norm(A * Xt - Y, 'fro')
cvx_end