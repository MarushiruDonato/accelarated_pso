# -*- coding: utf-8 -*-
# @Time : 2024/11/18 15:25
# @Author : Robert Guo
# @Email : gsc1997@foxmail.com
# @File : example.py
# @Desc :
import time

import numpy as np

from PSO import ParticleSwarmOptimization
from matplotlib import pyplot as plt


# this is a simple example, optimize the Schubert function,
# Schubert function is a multi-peak function, with multiple local optimal solutions,
# the upper and lower bounds are [-10, -10], [10, 10], the minimum value is -186.7309
def schubert(x):
    x1 = x[0]
    x2 = x[1]
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 = sum1 + (i * np.cos(((i + 1) * x1) + i))
        sum2 = sum2 + (i * np.cos(((i + 1) * x2) + i))
    return sum1 * sum2


# init PSO
def demo_func(x):
    x1, x2, x3 = x
    time.sleep(0.05)
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


pso = ParticleSwarmOptimization(target_func=demo_func,
                                dim=3,
                                max_iter=200, pop=200,
                                lower_bound=[-10] * 3,
                                upper_bound=[10] * 3,
                                w_max=1.0, w_min=0.4, c1=2.05, c2=2.05,
                                min_velocity_step=[0.0001] * 3,
                                verbose=True,
                                ensure_decimal_digits=False,
                                n_processes=10
                                )

# run PSO
pso.run()
# print result
print("best position: ", pso.global_best_position)
print("best value: ", pso.global_best_y)

plt.plot(pso.global_best_y_history)

plt.show()
