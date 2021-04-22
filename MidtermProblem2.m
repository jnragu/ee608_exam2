cv = [.5;.5];
f = (cv(1)-1)^(2) + 2*(cv(2) - 2)^(2);
h = [1 - cv(1)^2-cv(2)^2, cv(1)+cv(2)];
h = h.';
x = -5:.05:5;
y = -5:.05:5;
[X,Y] = meshgrid(x,y);
F = (X-1)^(2) + 2*(Y - 2)^(2);


% plot the whole thing

% but subdue it with an overlay
hf=fill([1 1 -1 -1]*5,[-1 1 1 -1]*5,'w','facealpha',0.8);

H1 =  X+Y;
H2 = 1 - X^2-Y^2;
feasible = (H1>=0 & H2 > 0);
F(~feasible) = NaN;
contour(X,Y,F,'linewidth',2,'ShowText','on');



