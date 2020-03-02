clc
clear;
load("real_data_0205_p1000.mat")



p = 50
im=zeros(N0,N0,3);
for i=1:p
    cpt = i + 200
    im = reshape(data(cpt,:),[N0, N0, 3]);
    im = permute(im, [2 1 3]);
    image_test{i} = double(rgb2gray(im(32-(N-1):32,32-(N-1):32,:)))./255;
    Y_test(:,i) = reshape(image_test{i}, [N^2,1]);
end
%%
alpha = 8;
% how many times of "sparsity" we want to keep


X_test = inv(Ahat) * Y_test;
figure(4)
idx = 1; 
n = 5
for c = 1:n
    i = 29+c;
    
    x = X_test(:,i);
    Xsparse = reshape(x, N, N);
    Yrecover{i} = Dinv1 * Xsparse * Dinv2.';

    xsort = sort(abs(x), "descend");
    sparsity = alpha * floor((norm(x,1)/norm(x,2))^2) + 1
%     sparsity = floor(N^2 * 0.1);
    threshold = xsort(sparsity);
    xcut = x;
    xcut(abs(x) < threshold) = 0;
    Xcomp = reshape(xcut, N,N);


    subplot(3,n,idx + 2 * n)
    imagesc(abs(Dinv1 * Xsparse * Dinv2.'));
    if c == 1
        title("direct recover",'position',[-12 15])
    end
    set(gca,'xtick',[],'ytick',[])
    
    
    subplot(3,n,idx+n)
    imagesc(abs(Dinv1 * Xcomp * Dinv2.'));
    str_title = sprintf("compressed recover",  sparsity*100/N^2);
    if c == 1
        title(str_title,'position',[-12 15])   
    end
    set(gca,'xtick',[],'ytick',[])
    
    subplot(3,n,idx)
    imagesc(image_test{i});
    if c == 1
        title("original image",'position',[-13 15])
    end
    set(gca,'xtick',[],'ytick',[])

    idx = idx + 1;
    
end
colormap('gray')