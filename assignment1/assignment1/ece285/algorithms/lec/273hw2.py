import numpy as np
import matplotlib.pyplot as plt

def f(x, A, m, n):
    k1 = 0
    k2 = 0
    for i in range(m):
        t1 = np.log(1 - A[i, :] @ x.T)
        k1 += t1

    for i in range(n):
        t2 = np.log(1 - x[i]**2)
        k2 += t2

    return -k1 - k2

# Gradient
def grad(x, A, m, n):
    k1 = np.zeros(n)
    for i in range(m):

        m1 = A[i, :] * (1 / (1 - A[i, :] @ x.T))
        k1 += m1

    k2 = (2 * x) / (1 - x * x)


    return k1 + k2



# Parameters.
alpha = 0.2
beta = 0.8
count = 1
m = 100
n = 100
A = np.random.normal(0, 1, size=(m, n))
x = np.zeros(n)
g = grad(x, A, m, n)
err = 1
count = 1
y = f(x, A, m, n)
curve2 = [y]
step = []
while err > 1e-3:
    count +=1
    t = 1
    g = grad(x, A, m, n)
 # Algorithm implementation.
    while (max(A @ (x - (t * g))) >= 1) | (max(abs(x - (t * g))) >= 1):
        t *= beta

    while f((x - (t * g)), A, m, n) > (f(x, A, m, n) - (alpha * t * np.linalg.norm(g) ** 2)):
        t *= beta
        # step.append(t)

    x = x - (t * g)
    err = np.linalg.norm(g)
    new_y = f(x, A, m, n)
    curve2.append(new_y)
    step.append(t)


plt.plot(curve2, 'bo-')
plt.title('Backtracking Line Search')
plt.xlabel('iterations'); plt.ylabel('objective function value')
plt.figure()


plt.scatter(range(len(step)), step)
plt.title('Stepsize Length Changes')
plt.xlabel('iterations'); plt.ylabel('Step Length')
plt.show()
