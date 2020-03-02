% simulation --- convex hull of 2-dim unit sphere in R3
i=1
for n = -1:0.01:1
    m = sqrt(1-n^2); %x (-1,1), y (0,1)
    x(i) = n;
    y(i) = m;
    z(i) = 0;
    i = i + 1;
end

for n = -1:0.01:1   %x (-1,1), y(-1, 0)
    m = -sqrt(1-n^2);
    x = [x n];
    y = [y m];
    z = [z 0];
end
    


for n = -1:0.01:1   % (y (-1,1), z(0,1))
    m = sqrt(1-n^2);
    x = [x 0];
    y = [y n];
    z = [z m];
end


for n = 0:0.01:1     %z (0,1) x(0,1)
    m = sqrt(1-n^2);
    x = [x m];
    y = [y 0];
    z = [z n];
end

for n = 0:0.01:1  % z(0,1), x(-1,0)
    m = -sqrt(1-n^2);
    x = [x m];
    y = [y 0];
    z = [z n];
end


for n = -1:0.01:1   % (y (-1,1), z(-1,0))
    m = -sqrt(1-n^2);
    x = [x 0];
    y = [y n];
    z = [z m];
end

for n = -1:0.01:0     %z (0,1) x(0,1)
    m = sqrt(1-n^2);
    x = [x m];
    y = [y 0];
    z = [z n];
end

for n = -1:0.01:0  % z(0,1), x(-1,0)
    m = -sqrt(1-n^2);
    x = [x m];
    y = [y 0];
    z = [z n];
end

[k1,av1] = convhull(x,y,z);

trisurf(k1,x,y,z,'FaceColor','y')
xlabel('x')
ylabel('y')
zlabel('z')
axis equal
    
    