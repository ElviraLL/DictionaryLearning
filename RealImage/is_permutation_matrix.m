function result = is_permutation_matrix(X)
dim = size(X,1);
colsum = sum(X);
rowsum = sum(X,2);
result = true;
for i = 1:dim
    if colsum(i) ~= 1 || rowsum(i) ~= 1
        result = false;
    end
end
for i = 1:dim
    for j = 1:dim
        if X(i,j) ~= 1 && X(i,j) ~= 0
            result = false;
        end
    end
end
end

