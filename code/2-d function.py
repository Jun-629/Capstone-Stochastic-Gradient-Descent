import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#定义图像和三维格式坐标轴
fig = plt.figure()
ax2 = Axes3D(fig)
fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5, 5, 0.5)
yy = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(xx, yy)
Z = X**2 + Y**2


#作图
ax3.plot_surface(X, Y, Z, cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# plt.show()


# Theta1**2 + Theta2**2
# calculate gradient
def grad_2d(x):
    deriv0 = 2 * x[0]
    deriv1 = 2 * x[1]
    return np.array([deriv0, deriv1])


# iteration for GD
def gradient_descent(x, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        grad_cur = grad_2d(x)
        x = x - learning_rate * grad_cur
        if np.linalg.norm(x, ord=2) < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x, "and time cost is:", time_cost)
    return x


x0 = [10, 10]
lr = 0.001
step = 100000
thres = 1e-20
gradient_descent(x0, lr, step, thres)


# iteration for NGD
def Nesterov_gd(x0, y0, learning_rate, steps, threshold):
    start = time.time()
    for i in range(steps):
        grad_cur = grad_2d(y0)
        x1 = y0 - learning_rate * grad_cur
        y1 = x1 + (i-1)/(i+2) * (x1-x0)
        x0 = x1
        y0 = y1
        if np.linalg.norm(x0, ord=2) < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x1, "and time cost is:", time_cost)
    return x1, y1


y0 = [10, 10]
Nesterov_gd(x0, y0, lr, step, thres)
