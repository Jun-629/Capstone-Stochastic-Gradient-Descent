# import numpy as np
# import scipy.stats as ss
# from scipy.optimize import fsolve
#
#
# class VanillaOption:
#     def __init__(
#             self,
#             otype=1,  # 1: 'call'
#             # -1: 'put'
#             strike=110.,
#             maturity=1.,
#             market_price=10.):
#         self.otype = otype
#         self.strike = strike
#         self.maturity = maturity
#         self.market_price = market_price  # this will be used for calibration
#
#     def payoff(self, s):  # s: excercise price
#         otype = self.otype
#         k = self.strike
#         maturity = self.maturity
#         return max([0, (s - k) * otype])
#
#
# class Gbm:
#     def __init__(self,
#                  init_state=100.,
#                  drift_ratio=.0475,
#                  vol_ratio=.2
#                  ):
#         self.init_state = init_state
#         self.drift_ratio = drift_ratio
#         self.vol_ratio = vol_ratio
#
#
# def bsm_price(self, vanilla_option):
#     s0 = self.init_state
#     sigma = self.vol_ratio
#     r = self.drift_ratio
#
#     otype = vanilla_option.otype
#     k = vanilla_option.strike
#     maturity = vanilla_option.maturity
#
#     d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2)
#           * maturity) / (sigma * np.sqrt(maturity))
#     d2 = d1 - sigma * np.sqrt(maturity)
#
#     return (otype * s0 * ss.norm.cdf(otype * d1)  # line break needs parenthesis
#             - otype * np.exp(-r * maturity) * k * ss.norm.cdf(otype * d2))
#
#
# Gbm.bsm_price = bsm_price
#
#
# def f(x):
#     gbm2 = Gbm(vol_ratio=x)
#     option2 = VanillaOption(otype=1)
#     outcome = gbm2.bsm_price(option2)-10
#     return outcome
#
#
# ans_sig_2 = fsolve(f,0.1)
# print('>>> The implied volatility is ' + str(ans_sig_2))

import numpy as np
import scipy.stats as ss
import random
import time

class VanillaOption:
    def __init__(
        self,
        otype = 1, # 1: 'call'
                  # -1: 'put'
        strike = 110.,
        maturity = 1.,
        market_price = 10.):
      self.otype = otype
      self.strike = strike
      self.maturity = maturity
      self.market_price = market_price

class Gbm:
    def __init__(self,
                 init_state = 100.,
                 drift_ratio = .0475,
                 vol_ratio = .2
                ):
        self.init_state = init_state
        self.drift_ratio = drift_ratio
        self.vol_ratio = vol_ratio


def bsm_price(self, vanilla_option):
    s0 = self.init_state
    sigma = self.vol_ratio
    r = self.drift_ratio

    otype = vanilla_option.otype
    k = vanilla_option.strike
    maturity = vanilla_option.maturity

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    option_price = otype * s0 * ss.norm.cdf(otype * d1) - otype * np.exp(-r * maturity) * k * ss.norm.cdf(otype * d2)

    return option_price


def gradient(self, vanilla_option):
    s0 = self.init_state
    sigma = self.vol_ratio
    r = self.drift_ratio

    otype = vanilla_option.otype
    k = vanilla_option.strike
    maturity = vanilla_option.maturity
    M = vanilla_option.market_price

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    Nd1 = ss.norm.cdf(otype * d1)
    Nd2 = ss.norm.cdf(otype * d2)
    Phid1 = ss.norm.pdf(d1)

    gradient = s0 * maturity * ((Nd1 * s0 - Nd2 * k * np.exp(-r*maturity) - M) * Phid1)
    return gradient


# Gbm.bsm_price = bsm_price
Gbm.gradient = gradient

# gbm = Gbm()
# option = VanillaOption()
# print(gbm.bsm_price(option))

n = 2


def cal_gradient(volatility):
    gbm = Gbm(vol_ratio=volatility)
    random.seed(0)
    T = 1
    gradient_process = 0
    for i in range(n):
        KK = random.randint(105, 115)
        MM = random.randint(7, 10)
        option = VanillaOption(maturity=T, strike=KK, market_price=MM)
        T += 1
        gradient_process += gbm.gradient(vanilla_option=option)
    gradient_final = 2 * gradient_process / n  # this is the calculated gradient
    return gradient_final


def gradient_descent(x, learning_rate, steps, threshold):
    start = time.time()
    # print(cal_gradient(x))
    for i in range(steps):
        grad = cal_gradient(x)
        x = x - learning_rate * grad
        if abs(x-0.2) < threshold:
            break
    print("times need: ", i)
    end = time.time()
    time_cost = round(end - start, 4)
    print("minimum is:", x, "and time cost is:", time_cost)
    # print(x)
    return x


x0 = 0.2
lr = 0.001
step = 10000
thres = 1e-10
gradient_descent(x0, lr, step, thres)
