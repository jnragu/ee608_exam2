clear
clc
format long
ipp = 1;
syms X Y;
barrier = -ipp*(log(1 - X^2-Y^2) + log(X+Y));
minf = (X-1)^(2) + 2*(Y - 2)^(2);
f = minf + barrier;
% Initial Guess (Choose Initial Guesses):
x(1) = .5;
y(1) = .5;
h1(1) = 1 - x(1)^2-y(1)^2;
h2(1) = x(1)+y(1);
values(1) =(x(1)-1)^(2) + 2*(y(1) - 2)^(2);
e = .002; % Convergence Criteria
i = 1; % Iteration Counter
% Gradient and Hessian Computation:
df_dx = diff(f, X);
df_dy = diff(f, Y);
J = [subs(df_dx,[X,Y], [x(1),y(1)]) subs(df_dy, [X,Y], [x(1),y(1)])]; % Gradient
ddf_ddx = diff(df_dx,X);
ddf_ddy = diff(df_dy,Y);
ddf_dxdy = diff(df_dx,Y);
ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(1),y(1)]);
ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(1),y(1)]);
ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(1),y(1)]);
H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1]; % Hessian
S = inv(H); % Search Direction
% Optimization Condition:
ipp = ipp*.8;
barrier = -ipp*(log(1 - X^2-Y^2) + log(X+Y));
f = minf + barrier;
df_dx = diff(f, X);
df_dy = diff(f, Y);
lastVal = 0;
while abs(values(i)-lastVal) > e && i<100
    I = [x(i),y(i)]';
    x(i+1) = I(1)-S(1,:)*J';
    y(i+1) = I(2)-S(2,:)*J';
    h1(i+1) = 1 - x(i+1)^2-y(i+1)^2;
    h2(i+1) = x(i+1)+y(i+1);
    values(i+1) =(x(i+1)-1)^(2) + 2*(y(i+1) - 2)^(2);
    lastVal = values(i);
    i = i+1;
    J = [subs(df_dx,[X,Y], [x(i),y(i)]) subs(df_dy, [X,Y], [x(i),y(i)])]; % Updated Jacobian
    ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(i),y(i)]);
    ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(i),y(i)]);
    ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(i),y(i)]);
    H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1]; % Updated Hessian
    S = inv(H); % New Search Direction
    
    ipp = ipp*.8;
    barrier = -ipp*(log(1 - X^2-Y^2) + log(X+Y));
    f = minf + barrier;
    df_dx = diff(f, X);
    df_dy = diff(f, Y);
    
    
end
% Result Table:`
Iter = 1:i;
X_coordinate = x';
Y_coordinate = y';
Iterations = Iter';
Values = values';
H1 = h1';
H2 = h2';
T = table(Iterations,X_coordinate,Y_coordinate,H1,H2,Values);
xEnd = X_coordinate(i);
yEnd = Y_coordinate(i);
value =(xEnd-1)^(2) + 2*(yEnd - 2)^(2);

figure('Color','w');
[X,Y] = meshgrid(x,y);
F = (X-1)^(2) + 2*(Y - 2)^(2);


% plot the whole thing

% but subdue it with an overlay

scatter(X_coordinate,Y_coordinate);
hold on
for i=1:length(X_coordinate) - 1
    p1 = [X_coordinate(i) Y_coordinate(i)];
    p2 = [X_coordinate(i+1) Y_coordinate(i+1)];
    dp = p2 - p1;
    quiver(p1(1),p1(2),dp(1),dp(2),0);
    hold on
end

tab = T;
