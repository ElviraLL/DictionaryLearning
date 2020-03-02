function Xhat = getX(A, Y, sigma, p)
    %Y = AX
    %y_i = A x_i
    [m,n] = size(A);
    options.verbosity = 0;
    options.iterations = 50 * m;
    for i = 1:p
        Xhat(:,i) = spg_bpdn(A, Y(:,i), sigma*norm(Y(:,i)), options);
    end
end