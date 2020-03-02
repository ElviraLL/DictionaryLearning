function Xhat = updateXsep(A, Y, sigma, p)
    %Y = AX
    %y_i = A x_i
    options.verbosity=0;
    for i = 1:p
        Xhat(:,i) = spg_bpdn(A, Y(:,i),sigma*norm(Y(:,i)), options);
    end
end