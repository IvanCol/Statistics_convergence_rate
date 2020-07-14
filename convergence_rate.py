"""
@author: Иван

Критерий проверки на симметричность, модификация D/'Agostino проверки на
симметричность
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skewtest
from scipy.stats import skew
from scipy.stats import norm
from abc import ABCMeta, abstractmethod, abstractstaticmethod


plt.rc('figure', figsize=(9, 7))


# Декоратор
class IOperator:
    """
    Интерфейс
    """
    @abstractmethod
    def operator(self):
        pass


class Normsmp(IOperator):
    """
    Класс возвращает нормальное распределение с заданными параметрами
    """
    def operator(self, loc, scale, smpsize):
        return np.random.normal(loc, scale, smpsize)


class Montecarlo(IOperator):  # Декоратор
    """
    Класс возвращает определенное количество выборок с заданным распределением
    """
    def __init__(self, distr):
        self.distr = distr  # распределение

    def operator(self, loc, scale, smpsize, nmbtrials):
        return np.array([self.distr.operator(loc, scale, smpsize)
                         for i in range(nmbtrials)])


# Фабрика
class IStatdist(metaclass=ABCMeta):
    """
    Интерфейс
    """
    @abstractstaticmethod
    def dist():
        """метод интерфейса"""


class DAdist(IStatdist):
    """
    Класс статистики модификации D/'Agostino проверки на симметрию
    """

    def statlist(self, smp):
        return np.array(sorted([skewtest(smp[i])[0]
                                for i in range(len(smp))]))

    def statdistr(self, smp):
        statsmp = self.statlist(smp)
        return np.array([norm.cdf(i) for i in statsmp])

    def dist(self, smp):
        statdist = self.statdistr(smp)
        N = len(statdist)
        b = np.array([i / N for i in range(N)])
        return max([np.abs(statdist[i] - b[i]) for i in range(N)])


class Skewdist(IStatdist):
    """
    Класс распределения статистики критерия проверки на симметрию
    """

    def statlist(self, smp):
        return np.array(sorted([skew(smp[i]) for i in range(len(smp))]))

    def statdistr(self, smp):
        statsmp = self.statlist(smp)
        n = len(smp[0])
        scale_n = (6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2)
        return np.array([norm.cdf(i, scale=scale_n) for i in statsmp])

    def dist(self, smp):
        statdist = self.statdistr(smp)
        N = len(statdist)
        b = np.array([i / N for i in range(N)])
        return max([np.abs(statdist[i] - b[i]) for i in range(N)])


class StatFactory:
    """
    Фабрика Распределений статистик
    """

    @staticmethod
    def get_statdist(stat):
        try:
            if stat == "DAgostino":
                return DAdist()
            if stat == "Skew":
                return Skewdist()
            raise AssertionError("stat not found")
        except AssertionError as _e:
            print(_e)
        return None


class Approx:
    """
    Класс аппроксимации
    """
    def approx_func(self, x, a, b):
        return a * x ** (-b)

    def summary(self, n_bnd, DnN, eps):
        popt, pcov = curve_fit(self.approx_func, n_bnd, DnN)
        n_lim = np.ceil((np.power(popt[0] / eps, 1. / popt[1])))
        print('a = {0}\nb = {1}\n'.format(*tuple(popt)))
        print('n при eps =', n_lim)

        residuals = DnN - self.approx_func(n_bnd, popt[0], popt[1])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((DnN - np.mean(DnN)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print('R^2: ', r_squared)

        plt.plot(n_bnd, self.approx_func(n_bnd, popt[0], popt[1]),
                 label='trend line')
        plt.scatter(n_bnd, DnN, label='distances', color='red')
        plt.title('Trend')
        plt.xlabel('n')
        plt.ylabel('distances')
        plt.grid(True)
        plt.minorticks_on()
        plt.legend()
        plt.show()


# Фасад
class Research:
    def __init__(self):
        self._normsmp = Normsmp()
        self._montecarlo = Montecarlo(self._normsmp)
        self._statfactory = StatFactory()
        self._approx = Approx()

    def summary(self, loc, scale, nmbtrials, n_bnd, eps, stat):
        distlist = []
        for n in n_bnd:
            smp = self._montecarlo.operator(loc, scale, n, nmbtrials)
            statfactoty = self._statfactory.get_statdist(stat)
            distlist.append(statfactoty.dist(smp))
        #            print('DnN: ', distlist[-1])
        #            print('n: 'n)
        print('\n', stat)
        self._approx.summary(n_bnd, distlist, eps)


if __name__ == "__main__":
    loc = 0
    scale = 1
    nmbtrials = 1000
    n_bnd = range(8, 150, 5)
    eps = 0.01
    stat = ["Skew", "DAgostino"]
    for s in stat:
        Research().summary(loc, scale, nmbtrials, n_bnd, eps, s)
