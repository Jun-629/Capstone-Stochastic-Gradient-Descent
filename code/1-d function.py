import time


# calculate gradient
def cal_gradient(x):
    gradient = 2*x
    return gradient


# iteration for GD
def gradient_descent(x, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        gradient = cal_gradient(x)
        x = x - learning_rate * gradient
        if x-0 < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x, "and time cost is:", time_cost)
    return x


x0 = 1
lr = 0.001
step = 1000000
thres = 1e-20  # too small
gradient_descent(10, lr, step, thres)


# iteration for NGD
def Nesterov_gd(x0, y0, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        gradient = cal_gradient(y0)
        x1 = y0 - learning_rate * gradient
        y1 = x1 + (i-1)/(i+2) * (x1-x0)
        x0 = x1
        y0 = y1
        if abs(x1-0) < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x1, "and time cost is:", time_cost)
    return x1, y1


y0 = 1
Nesterov_gd(x0, y0, lr, step, thres)
