
import numpy as np

def f1(x):
    return x[0]**2 + 3*x[1]**2 + np.cos(x[0] + x[1])

def coordinate_descent(f, x0, lr=0.1, max_iter=100):
    x = np.array(x0)
    for _ in range(max_iter):
        for i in range(len(x)):
            grad = np.zeros_like(x)
            grad[i] = 1
            x[i] -= lr * grad[i] * (f(x + grad) - f(x - grad)) / 2
    return x

x0 = [0.5, 0.5]
solution = coordinate_descent(f1, x0)
print("Решение методом покоординатного спуска:", solution)
