clear; 
rng default;

n = 1001; 

A = randn(n, n);
B = randn(n, n);

C = zeros(n, n);

tic
for i = 1:50
    C = C + (A*B'/n .* A) + B - A;
end
toc