import time
import numpy as np
# from scipy.stats import bernoulli


# calculate gradient
def cal_gradient(x, i):
    if i == 1:
        gradient = 2 * x / 2
    else:
        gradient = 2 * (x - 2) / 2
    return gradient


# iteration for SGD
def SGD(x, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        j = np.random.binomial(1, 0.5, 1)  # the type of bernoulli(j) is rv_frozen
        gradient = cal_gradient(x, j)
        x = x - learning_rate * gradient
        if x-1 < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x, "and time cost is:", time_cost)
    return x

x0 = 10
lr = 0.01
step = 10000
thres = 0.001
SGD(10, lr, step, thres)


# iteration for NGD
def Nesterov_sgd(x0, y0, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        j = np.random.binomial(1, 0.5, 1)
        gradient = cal_gradient(y0, j)
        x1 = y0 - learning_rate * gradient
        y1 = x1 + (i-1)/(i+2) * (x1-x0)
        x0 = x1
        y0 = y1
        if abs(x1-1) < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x1, "and time cost is:", time_cost)
    return x1, y1


y0 = 10
Nesterov_sgd(x0, y0, lr, step, thres)

