import numpy as np
import time


def rosenbrock_2d(theta):
    deriv0 = 2 * theta[0] - 2 + 400 * theta[0] * (theta[0] * theta[0] - theta[1])
    deriv1 = 200 * (theta[1] - theta[0] * theta[0])
    return np.array([deriv0, deriv1])


# iteration for GD
def gradient_descent(theta, learning_rate, steps, threshold):
    start = time.time()
    mini = [1, 1]
    for i in range(steps):
        grad_cur = rosenbrock_2d(theta)
        theta -= learning_rate * grad_cur
        print(theta)
        if np.linalg.norm([a - b for a, b in zip(theta, mini)], ord=2) < threshold:
            break
    print("times need:", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", theta, "and time cost is:", time_cost)
    return theta


theta_ori = [0, 0]
step = 10
thres = 1e-5
for i in range(1, 11):
    lr = i/1000
    print("learning rate is:", lr)
    gradient_descent(theta_ori, lr, step, thres)

# gradient_descent(theta_ori, 0.000001, step, thres)

print('XXXXXXXXXXXXXXXXXXXXXXXXXX')


# iteration for NGD
def Nesterov_gd(x0, y0, learning_rate, steps, threshold):
    start = time.time()
    mini = [1, 1]
    for i in range(steps):
        grad_cur = rosenbrock_2d(y0)
        x1 = y0 - learning_rate * grad_cur
        y1 = x1 + (i-1)/(i+2) * (x1-x0)
        x0 = x1
        y0 = y1
        if np.linalg.norm([a - b for a, b in zip(x0, mini)], ord=2) < threshold:
            break
    print("times need:", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x1, "and time cost is:", time_cost)
    return x1, y1


# y0 = [0, 0]
# for i in range(1, 11):
#     lr = i/1000
#     print("learning rate is:", lr)
#     Nesterov_gd(theta_ori, y0, lr, step, thres)

# import numpy as np
#
#
# def cal_rosenbrock(x1, x2):
#     """
#     计算rosenbrock函数的值
#     :param x1:
#     :param x2:
#     :return:
#     """
#     return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
#
#
# def cal_rosenbrock_prax(x1, x2):
#     """
#     对x1求偏导
#     """
#     return -2 + 2 * x1 - 400 * (x2 - x1 ** 2) * x1
#
#
# def cal_rosenbrock_pray(x1, x2):
#     """
#     对x2求偏导
#     """
#     return 200 * (x2 - x1 ** 2)
#
#
# def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
#     pre_x = np.zeros((2,), dtype=np.float32)
#     loss = 10
#     iter_count = 0
#     while loss > 0.001 and iter_count < max_iter_count:
#         error = np.zeros((2,), dtype=np.float32)
#         error[0] = cal_rosenbrock_prax(pre_x[0], pre_x[1])
#         error[1] = cal_rosenbrock_pray(pre_x[0], pre_x[1])
#
#         for j in range(2):
#             pre_x[j] -= step_size * error[j]
#
#         loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0
#
#         iter_count += 1
#     print("iter_count: ", iter_count, "the loss:", loss)
#
#     return pre_x
#
#
# if __name__ == '__main__':
#     w = for_rosenbrock_func()
#     print(w)

