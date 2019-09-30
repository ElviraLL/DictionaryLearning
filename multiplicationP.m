function result = multiplicationP(P)
    num_of_matrix = size(P) * [0;1];
    N = size(P{num_of_matrix}) * [1; 0]; % find the first element of size
    result = eye(N);
    for idx = num_of_matrix:-1:1
        Pi = P{idx};
        Ii = eye(N/(2^idx));
        Pi = kron(Ii, Pi);
        result = Pi * result;
    end
end