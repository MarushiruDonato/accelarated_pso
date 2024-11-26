# -*- coding: utf-8 -*-
# @Time : 2024/10/28 16:15
# @Author : Robert Guo
# @Email : gsc1997@foxmail.com
# @File : PSO.py
# @Desc :
import os
import sys
import time
from multiprocessing import Pool
from typing import Union

import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from loguru import logger
from tqdm import tqdm


class ParticleSwarmOptimization:
    def __init__(self, target_func, dim: int, max_iter: int, pop: int, upper_bound: Union[list, np.array], lower_bound: Union[list, np.array],
                 w_max: float = 1.0, w_min: float = 0.4, c1: float = 2.05, c2: float = 2.05,
                 min_velocity_step=None,
                 verbose: bool = True,
                 exit_error: float = -np.inf,
                 ensure_decimal_digits: bool = True,
                 min_pop_for_multiprocessing: int = 1,
                 n_processes: int = os.cpu_count() // 2
                 ):
        """
        init PSO
        :param target_func: target function, input is a list with dim dimensions, return a number
        :param dim: dimension of the input parameters
        :param max_iter: max iteration
        :param pop: population of the particles
        :param upper_bound: upper bound of the parameters, a list or np.array of dim length
        :param lower_bound: lower bound of the parameters, a list or np.array of dim length
        :param w_max: max inertia weight
        :param w_min: min inertia weight
        :param c1: learning factor, personal
        :param c2: learning factor, global
        :param min_velocity_step: if None, auto calculated; or a list or np.array of dim length, represents the min velocity step of each dimension
        :param verbose: whether print log, default is True
        :param exit_error: exit when the global best value less than or equal to this value; default set to negative infinity
        :param ensure_decimal_digits: whether ensure the decimal digits, default is True; if False, min_velocity_step will not work
        :param min_pop_for_multiprocessing: minimum population for multiprocessing, default set to 1
        :param n_processes: number of processes, default set to cpu core count / 2

        Example:
        ```python
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
        pso = ParticleSwarmOptimization(target_func=schubert, dim=2, max_iter=100, pop=100, lower_bound=[-10, -10], upper_bound=[10, 10],
                                        w_max=0.9, w_min=0.4, c1=2.05, c2=2.05,
                                        min_velocity_step=[0.01, 0.01],
                                        verbose=True, n_processes=1)
        # run PSO
        pso.run()
        # print result
        print("best position: ", pso.global_best_position)
        print("best value: ", pso.global_best_y)
        ```
        """
        self.func = target_func
        self.dim = dim
        self.max_iter = max_iter
        self.pop = pop
        self.w_max = w_max
        self.w_min = w_min
        self.w = np.linspace(w_max, w_min, max_iter)
        self.c_personal = c1
        self.c_global = c2
        self.upper_bound = np.array(upper_bound)
        self.lower_bound = np.array(lower_bound)
        self.stop_sign = False
        self.exit_error = exit_error

        # current iteration
        self.cur_iter = 0
        self.running = False
        self.ensure_decimal_digits = ensure_decimal_digits
        self.verbose = verbose
        # self.positions_history = np.zeros((self.max_iter + 1, self.pop, self.dim))

        self.n_processes = int(min(n_processes, os.cpu_count()))
        if n_processes > os.cpu_count():
            logger.info("n_processes is greater than cpu core count, set n_processes to cpu core count")

        # minimum population for multiprocessing
        self.min_pop_for_multiprocessing = min_pop_for_multiprocessing
        if self.n_processes > 1 and self.pop >= self.min_pop_for_multiprocessing and self.verbose:
            logger.info("Population is more than {}, using multiprocessing with {} processes".format(self.min_pop_for_multiprocessing, self.n_processes))

        # minimum velocity step
        self.min_velocity_step = self.get_minimal_velocity_step() if min_velocity_step is None else min_velocity_step

        # init positions and velocities
        self.positions = self.init_positions()
        # self.positions_history.append(self.positions)
        # self.positions_history[0] = self.positions
        self.v = self.init_velocities()

        # personal best positions and values
        self.p_best_positions = np.copy(self.positions)
        self.p_best_y = np.array([np.inf] * self.pop)

        # history of the best position and value of each iteration
        # self.iter_best_position_history = []
        self.iter_best_y_history = np.array([], dtype=np.float64)
        self.iter_best_position = np.zeros(dim)
        self.iter_best_y = np.inf

        # global best position and value
        self.global_best_y = np.inf
        self.global_best_position = np.zeros(dim)
        self.global_best_y_history = np.array([], dtype=np.float64)

        self.iter_y_history = []

        # update bests when init finished
        self.update_bests()

        if self.verbose:
            logger.info("init best position: " + ",".join([np.format_float_positional(i, 6, trim="-") for i in self.global_best_position]))
            logger.info("init best value: {}".format(np.round(self.global_best_y, 6)))

        assert w_max >= w_min, "w_max must be greater than w_min"
        assert len(upper_bound) == dim, "upper_bound dimension must be equal to dim"
        assert len(lower_bound) == dim, "lower_bound dimension must be equal to dim"
        assert len(self.min_velocity_step) == dim, "velocity_step dimension must be equal to dim"
        assert np.all(np.array(upper_bound) >= np.array(lower_bound)), "upper_bound must be greater than lower_bound"
        assert self.pop > 0, "pop must be greater than 0"
        assert self.max_iter >= 0, "max_iter must be greater than 0"
        assert self.n_processes >= 1, "n_processes must be greater than 1"

    def init_positions(self):
        """
        init particles' positions
        :return: initial positions
        """
        X = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop, self.dim))
        if self.ensure_decimal_digits:
            # decimal count of each dimension of the upper bound
            # 真实的小数点位数, 如果最小速度步长为0, 则取上限的小数位数
            decimal_count = [self.get_decimal_digit(i) for i in self.min_velocity_step]
            real_decimal_count = [self.get_decimal_digit(i) if i != 0 else self.get_decimal_digit(up)
                                  for i, up in
                                  zip(self.min_velocity_step, self.upper_bound)]
            X = np.round(X, max(real_decimal_count))

            # ensure the decimal digits, vectorized method
            decimal_count_for_all = np.array([decimal_count for _ in range(self.pop)])
            mask = decimal_count_for_all != -1
            vect_round = np.vectorize(np.round)
            if np.any(mask):
                X[mask] = vect_round(X[mask], decimal_count_for_all[mask])

            # round the decimal digits of each dimension according to the minimum velocity step
            # for i, x in enumerate(X):
            #     X[i] = np.vectorize(lambda x, y: np.round(x, y) if y != -1 else x)(x, decimal_count)
        return X

    def init_velocities(self):
        """
        init particles' velocities
        :return: initial velocities
        """
        velocity_upper_bound = self.upper_bound - self.positions
        velocity_lower_bound = self.lower_bound - self.positions

        # random velocity
        V = np.random.uniform(velocity_lower_bound, velocity_upper_bound, (self.pop, self.dim))
        if self.ensure_decimal_digits:
            V = self.ensure_velocity(V)
        return V

    def update_positions(self):
        """
        update particles' positions
        :return:
        """
        # update positions
        self.positions += self.v

        # clip the positions to the upper and lower bounds
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
        # self.positions_history.append(self.positions)
        # self.positions_history[self.cur_iter + 1] = self.positions

    def update_velocities(self):
        """
        update particles' velocities
        :return:
        """
        # update velocities
        temp = self.w[self.cur_iter] * self.v
        temp += self.c_personal * np.random.rand(self.pop, self.dim) * (self.p_best_positions - self.positions)
        temp += self.c_global * np.random.rand(self.pop, self.dim) * (self.global_best_position - self.positions)
        if self.c_personal + self.c_global > 4:
            # when c1 + c2 > 4, add a compression factor to constrain the velocity and enhance the local best search ability
            self.v = temp * (2 / abs(2 - self.c_personal - self.c_global - np.sqrt((self.c_personal + self.c_global - 4) * (self.c_personal + self.c_global))))
        else:
            self.v = temp
        if self.ensure_decimal_digits:
            self.v = self.ensure_velocity(self.v)
        # print(self.v)

    def ensure_velocity(self, V):
        """
        ensure velocities were greater than the minimum velocity step, and ensure the decimal digits;
        clip the velocities to the upper and lower bounds
        :param V:
        :return:
        """
        V = np.copy(V)
        # get decimal digits
        decimal_count = [self.get_decimal_digit(i) for i in self.min_velocity_step]

        # ensure velocity greater than the minimum velocity step, vectorized method
        mask = np.abs(V) < self.min_velocity_step
        min_velocity_step = np.array([self.min_velocity_step for _ in range(self.pop)])
        V[mask] = np.sign(V[mask]) * min_velocity_step[mask]

        # keep the decimal digits
        for i, count in enumerate(decimal_count):
            V[:, i] = np.round(V[:, i], count)

        # clip the velocities to ensure the positions were in the bounds
        V = np.clip(V, self.lower_bound - self.positions, self.upper_bound - self.positions)
        real_decimal_count = [self.get_decimal_digit(i) if i != 0 else self.get_decimal_digit(up)
                              for i, up in
                              zip(self.min_velocity_step, self.upper_bound)]

        V = np.round(V, max(real_decimal_count))

        return V

    def update_bests(self):
        self.running = True
        conditions = [self.n_processes is not None, self.n_processes > 1, self.pop >= self.min_pop_for_multiprocessing]
        if all(conditions):
            self.update_bests_multiprocessing()
        else:
            self.update_bests_single_process()

    def update_bests_single_process(self):
        """
        get global best, single process
        :return: global best position, global best value
        """
        iter_best_position = np.zeros(self.dim)
        iter_best_y = np.inf
        y_history = []
        for i in range(self.pop):
            try:
                y = self.func(self.positions[i])
            except Exception as e:
                # if error occurred, set y to inf
                y = np.inf
                logger.error(e)
                logger.error("Error in function, particle not updated")
                logger.error("Particle Position: {}".format(self.positions[i]))
            y_history.append(y)
            if y < iter_best_y:
                iter_best_position = self.positions[i].copy()
                iter_best_y = y
            # update personal best
            if y < self.p_best_y[i]:
                self.p_best_y[i] = y
                self.p_best_positions[i] = self.positions[i].copy()
            # update global best
            if y < self.global_best_y:
                self.global_best_y = y
                self.global_best_position = self.positions[i].copy()
        self.iter_y_history.append(np.array(y_history))

        self.iter_best_position = iter_best_position
        self.iter_best_y = iter_best_y
        self.iter_best_y_history = np.append(self.iter_best_y_history, iter_best_y)
        self.global_best_y_history = np.append(self.global_best_y_history, self.global_best_y)

    def update_bests_multiprocessing(self):
        """
        update bests using multiprocessing
        :return: global best position, global best value
        """
        results = []
        joblib_backend = "loky"

        # loky backend will throw errors under pyinstaller packaging,
        # if this program is packaged by some kind of packaging tool, set the backend to multiprocessing
        if hasattr(sys, 'frozen'):
            joblib_backend = "multiprocessing"
        if joblib_backend == "loky":
            results = Parallel(n_jobs=self.n_processes, backend=joblib_backend)(delayed(self.func)(pos) for pos in self.positions)

        if joblib_backend == "multiprocessing":
            pool = Pool(processes=self.n_processes)
            results = pool.map(self.func, self.positions)
            pool.close()
            pool.join()

        iter_best_y = min(results)
        iter_best_position = self.positions[np.argmin(results)].copy()
        self.iter_y_history.append(np.array(results))
        for i in range(self.pop):
            y = results[i]
            if y < self.p_best_y[i]:
                self.p_best_y[i] = y
                self.p_best_positions[i] = self.positions[i].copy()
        if iter_best_y <= self.global_best_y:
            self.global_best_y = iter_best_y
            self.global_best_position = iter_best_position

        self.iter_best_position = iter_best_position
        self.iter_best_y = iter_best_y
        self.iter_best_y_history = np.append(self.iter_best_y_history, iter_best_y)
        self.global_best_y_history = np.append(self.global_best_y_history, self.global_best_y)

    def get_cur_particles_bounds(self):
        """
        get the upper and lower bounds of the current particles
        :return: upper bound, lower bound
        """
        if self.cur_iter == 0:
            return self.upper_bound, self.lower_bound
        else:
            # 获取当前粒子的上下边界
            lower_bound = np.min(self.positions, axis=0)
            upper_bound = np.max(self.positions, axis=0)
            return upper_bound, lower_bound

    def get_minimal_velocity_step(self):
        """
        get the minimal velocity step of each dimension
        :return: minimal velocity step
        """
        def max_decimal_count(x: np.number, y: np.number):
            """
            return the num with more decimal digits
            :param x:
            :param y:
            :return:
            """
            return x if self.get_decimal_digit(x) > self.get_decimal_digit(y) else y

        min_bound = [max_decimal_count(i, j) if i != j else 0 for i, j in zip(self.upper_bound, self.lower_bound)]

        min_velocity_step = [self.get_decimal_min_step(i) if i != 0 else 0 for i in min_bound]
        return np.array(min_velocity_step)

    def get_total_solution_count(self):
        """
        get the total solution count of the solution space
        :return: total solution count
        """
        total_count = 1
        for i in range(self.dim):
            if self.upper_bound[i] == self.lower_bound[i]:
                total_count *= 1
                continue
            total_count *= int((self.upper_bound[i] - self.lower_bound[i]) / self.min_velocity_step[i] + 1)
        return total_count

    @staticmethod
    def get_decimal_min_step(num):
        """
        the minimal step of the number
        example:
            123, return 1
            0.0123, return 0.0001
        :param num: number
        :return: minimal step
        """
        if num % 1 == 0:
            return 1
        else:
            decimal = len(np.format_float_positional(num).split('.')[1])
            k = 1.0
            k /= 10 ** decimal
            return k

    @staticmethod
    def get_decimal_digit(num):
        """
        get the decimal digits of the number
        例如:
            0, return -1
            123, return 0
            0.0123, return 4
        :param num: number
        :return: decimal digits, if integer return 0, if zero return -1 , otherwise return the actual decimal digits
        """
        if num == 0:
            return -1
        if num % 1 == 0:
            return 0
        else:
            decimal = len(np.format_float_positional(num).split('.')[1])
            return decimal

    def run(self):
        time1 = time.perf_counter()
        p = tqdm(range(self.max_iter), desc="PSO Iteration", disable=not self.verbose)
        for cur_iter in range(self.cur_iter, self.max_iter):
            self.cur_iter = cur_iter

            # break when the global best value less than or equal to the exit error
            if np.less_equal(self.global_best_y, self.exit_error):
                break

            # break when the stop sign is True
            if self.stop_sign:
                break

            self.update_positions()
            self.update_velocities()
            self.update_bests()

            # log the best value of each iteration
            if self.verbose:
                logger.info("iter: {}, g_best: {}, iter_best: {}".format(cur_iter, np.round(self.global_best_y, 4), np.round(self.iter_best_y, 4)))
            p.update(1)
        if self.verbose:
            logger.info("PSO finis  hed, time: {:.6f}".format(time.perf_counter() - time1))
        self.running = False
        self.stop_sign = False

        # joblib won't close the pool automatically, close manually
        get_reusable_executor().shutdown(wait=True)

        return self.global_best_position, self.global_best_y
