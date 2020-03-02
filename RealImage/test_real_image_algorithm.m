clc;
clear; 
lambda = 5.0000e-04;
N = 32; %dimension for signal N = d = m
p = 100; %number of observations 
num_of_matrix = log2(N);
iterations = 1000;
dat = load("cifar-10-batches-mat/data_batch_1.mat");
data = dat.data;

fprintf("Generating P...\n");
for idx = 1:num_of_matrix
    n = 2^idx; %dimension of the basic submatrix block  
    % generate random permutation
    half = n / 2;
    Pi = zeros(n);
    for i = 1 : half
        Pi(i,2*i - 1) = 1;
        Pi(i + half, 2 * i) = 1;
    end
    P{idx} = Pi;
end

fprintf("Generating Butterfly B...\n");
for idx = 1:num_of_matrix
    n = 2 ^ idx;
    half = n / 2;
    Bi = zeros(n);
    for i = 1 : half
        Bi(i,i) = 1;
        Bi(half + i, i) = 1;
        w = exp(2 * pi * j / n); % here j is the complex unit
        Bi(half + i, half + i) = - w^(-(i-1));
        Bi(i, half + i) = w^(-(i-1));
    end
    B{idx} = Bi;
end

D = dftmtx(N);
for i = 1:num_of_matrix
    Pinv{i} = inv(P{i});
    Binv{i} = inv(B{i});
end
Dinv = getDinv(Binv, Pinv);
fprintf("check the correctness of Dinv: %.2f\n", norm(Dinv - inv(D)))


N0=32;
im=zeros(N0,N0,3);
for cpt=1:p
    im = reshape(data(cpt,:),[N0, N0, 3]);
    im = permute(im, [2 1 3]);
    image_gray{cpt} = double(rgb2gray(im(32-(N-1):32,32-(N-1):32,:)));
    Xsparse{cpt} = Dinv * image_gray{cpt} * Dinv.'; % generate the sparse representation
    Xfft2{cpt} = fft2(image_gray{cpt});
    if cpt == 1
        Ycat = image_gray{cpt};
        Ycat2 = image_gray{cpt}.';
    else
        Ycat = cat(2, Ycat, image_gray{cpt});
        Ycat2 = cat(2, Ycat2, (image_gray{cpt}.'));   
    end
    Y(:,cpt) = image_gray{cpt}(:); % this is the 1024*1 version
    X(:,cpt) = Xsparse{cpt}(:);
end

y = Y(:,1);
x = X(:,1);
A = kron(D, D);
Ainv = getADouble(Binv,Binv,Pinv,Pinv);
norm(x - Ainv * y)
norm(x - inv(A) * y)
% calculation logic correct.


