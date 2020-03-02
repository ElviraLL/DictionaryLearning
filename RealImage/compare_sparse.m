clc
clear;
load("real_data_0405_p5000.mat")

D = dftmtx(N);
A = kron(D,D);
Xdft = A * Y;

dct = dctmtx(N);
Adct = kron(dct, dct);
Xdct = Adct * Y;
[mean((norms(Xhat,1).^2) ./ (norms(Xhat,2).^2)), mean((norms(Xdft,1).^2) ./ (norms(Xdft,2).^2)), mean((norms(Xdct,1).^2) ./ (norms(Xdct,2).^2))]


for i = 1:p
    s_dft(i) = (norm(Xdft(:,i),1) / norm(Xdft(:,i),2))^2;
    s_jw(i) = (norm(Xhat(:,i),1) / norm(Xhat(:,i),2))^2;
    s_dct(i) = (norm(Xdct(:,i),1) / norm(Xdct(:,i),2))^2;
end

figure(5)
plot(s_dft, s_jw, 'x', s_dft, s_dft)
legend('dft vs our result','dft vs dft')

figure(6)
plot(s_dct, s_jw, 'o', s_dct, s_dct)
legend('dct vs our result', 'dct vs dct')

figure(7)
plot(s_dct, s_jw, 'o', s_dct, s_dct, 'x', s_dft, s_jw, '*', s_dft, s_dft)
legend('dct vs our result', 'dct vs dct', 'dft vs our result', 'dft vs dft')


figure(10)
subplot(1,2,1)
imagesc(inv(Dinv1))
colorbar

subplot(1,2,2)
imagesc(inv(Dinv2))
colorbar


for i = 1:N
plot(dct(:,i));
pause(0.5);
end


