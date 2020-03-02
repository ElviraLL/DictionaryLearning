clc;
clear;
load("results\sparsity28_32by32_p200_real.mat")
D1 = inv(Dinv1);
D2 = inv(Dinv2);

% Dinv2 = inv(D);
% im=zeros(N0,N0,3);
% for cpt=1:p
%     im = reshape(data(cpt,:),[N0, N0, 3]);
%     im = permute(im, [2 1 3]);
%     image_gray{cpt} = double(rgb2gray(im(32-(N-1):32,32-(N-1):32,:)))./255;
% end
%testing which combination of columns or rows gives basis of matrix.
X1 = reshape(Xhat(:,1),N,N);
X1hat = D1 * image_gray{1} * D2.';
X1hattemp = zeros(N,N)
for i = 1:N
    for j = 1:N
        X1hattemp = X1hattemp + image_gray{1}(i,j) * D1(:,i) * D2(:,j).';
    end
end
norm(X1hat - X1hattemp)

%plot(some combinations of D1 D2)
figure(1)
idx = 1
for j = [4,10,14,16]
    for i = 1:2:N
        i
        subplot(4,N/2,idx)
        imagesc(D1(:,i) * D2(:,j).');
        s = sprintf("j=%d,i=%d",j, i);
        title(s);
        idx = idx + 1;
    end
end

%plot D1 D2
figure(2)
idx = 1
for i = 1:N
    for j = 1:N
%         set(gcf, 'Position',  [0, 0, 8000, 8000])
%         subplot(N,N,idx)
        imagesc(real((D1(:,i) * D2(:,j).')));
        idx = idx + 1;
        pause(0.1)
    end
end

% plot dft basis
D = dftmtx(N);
figure(3)
idx = 1
for i = 1:4:N
    for j = 1:4:N
        subplot(N,N,idx)
        imagesc(real((D(:,i)* D(:,j).')));
        set(gca,'xtick',[],'ytick',[])
        idx = idx + 1;
        pause(0.5)
    end
end


%plot D1 D2 basis
figure(4)
idx = 1
for i = 1:2:N
    for j = 1:2:N
        subplot(N/2,N/2,idx)
        imagesc(real((D1(:,i)* D2(:,j).')));
        set(gca,'xtick',[],'ytick',[])
        idx = idx + 1;
    end
end



% plot image and x sparse
figure(5)
idx = 1;
for c = 1:5
    i = 65 + c;
    subplot(2,5,idx);
    imagesc(image_gray{i});
    set(gca,'xtick',[],'ytick',[])
    subplot(2,5,idx + 5);
    imagesc(reshape(Xhat(:,i), N,N))
    idx = idx + 1; 
    set(gca,'xtick',[],'ytick',[])
end
hp4 = get(subplot(2,5,10),'Position')
colorbar('Position', [hp4(1)+hp4(3)+0.02  hp4(2)  0.012  hp4(2)+hp4(3)*5.68])
colormap('gray')

