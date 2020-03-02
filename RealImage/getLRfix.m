function [Lfix,Rfix] = getLRfix(cellinput, idx, num_of_matrix, N, reverse)
    %IF reverse is true, calculate the product as b1*b2*b3*---*b(idx-1) from right to left  
    %else, calculate the production as b(idx-1)*---b3*b2*b1 from left
    n = 2^idx; %size of the small block
    Lfix = eye(N); 
    Rfix = eye(N);
    if reverse==true
        for i = (idx - 1) : -1 : 1
            Bi = cellinput{i};
            Ii = eye(N / 2^i);
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;
        end
        for i = num_of_matrix: -1 :idx + 1
            Bi = cellinput{i};
            Ii = eye(N / 2^i);
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;
        end
    else
        for i = 1:idx - 1
            Bi = cellinput{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Rfix = Bi * Rfix;
        end        
        for i = (idx + 1) : num_of_matrix
            Bi = cellinput{i};
            Ii = eye(N/(2^i));
            Bi = kron(Ii, Bi);
            Lfix = Bi * Lfix;
        end
    end
end