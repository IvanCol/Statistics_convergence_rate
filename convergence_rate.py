import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.stats import skewtest
from scipy.stats import skew
from scipy.stats import kstest
from scipy.stats import kstwobign


################################
#         SIMULATION           #
################################

class Simulation:
    """
    Класс осуществляет моделирование одной выборки с заданным распределением,
    метод Монте карло, округление до определенного знака после запятой, 
    вычисление значений статистик. 
    """
    def __init__(self, number_trials, sample_size, distr='norm',
                 loc=0, scale=1):

        self.number_trials = number_trials  # N - Количество испытаний  
        self.sample_size = sample_size  # n - Объем выборки 
        self.distr = distr  # Распределение
        self.loc = loc  # Сдвиг
        self.scale = scale  # Масштаб

    def sample_simulation(self):
        """
        Возвращыет сгенерированный массив определенного распределения с заданными 
        параметрами сдвига, масштаба и возможно с параметром формы
        """
        if self.distr not in ['norm', 'expon', 'cauchy']:
            raise ValueError("Invalid distribution; dist must be 'norm', \
                                 'expon', 'cauchy'")
        if self.distr == 'norm':
            return np.random.normal(self.loc, self.scale, self.sample_size)
                            
        elif self.distr == 'expon':
            return np.random.exponential(self.scale, self.sample_size)
                             
        elif self.distr == 'cauchy':
            return np.random.standard_cauchy(self.sample_size)

    def montecarlo(self):
        """
        Возвращает массив из неупорядоченных выборок, состоящих из случайных чисел.
        """ 
        return np.array([self.sample_simulation() for i in range(self.number_trials)])


##############################
#         DISTANCE           #
##############################

# Плотность вероятности в точках  
def function_normal(scale, end, begin=-10):
    return integrate.quad(lambda x: 1/(scale * (2*np.pi) ** 0.5) * np.exp(-x**2/(2 * scale**2)), begin, end)[0]

# D_n_N
def dist(a, b):
    """
    Возвращает значения D_n_N = D_n + бельта
    """
    return max([np.abs(a[i] - b[i]) for i in range(len(a))]) 


#############################
#         GRAPHS            #
#############################

plt.rc('figure', figsize=(9, 7))


# График функции распределения
def Graph_F_x_teta(sample: np.ndarray, F_x_teta: np.array):
    """
    Возвращает график F(x,θ)
    """
    sample = np.sort(sample)
    plt.plot(sample, F_x_teta)
    plt.xlabel("x")
    plt.ylabel ("F(x,θ)") 
    plt.title("Функция распределения")
    plt.grid(True)
    plt.minorticks_on()
#    plt.grid(which = 'minor',
#             color = 'grey',
#             linestyle = ':')
    pylab.yticks(np.linspace(0, 1, 11))
#    return plt.show()


####################################
#         APPROXIMATION            #
####################################

class Approx:
    """
    Класс аппроксимации
    """
    def __init__(self, n_bnd, DnN, eps):
        """
        Конструктор принимает:
            Список объёмов выборок,
            Список расстояний,
            эпсилон.
        """
        self.n_bnd = n_bnd  # Список объёмов выборок
        self.DnN = DnN  # Список расстояний
        self.eps = eps  # Список расстояний

    def approx_func(self, x, a, b):
        return a * x**(-b)

    def summary(self, n_bnd, DnN, eps):
        popt, pcov = curve_fit(self.approx_func, n_bnd, DnN) 
        n_threshold = np.ceil((np.power(popt[0]/eps, 1./popt[1])))
        print('a = {0}\nb = {1}\n'.format(*tuple(popt)))
        print('n при eps =', n_threshold)
        
        residuals = DnN- self.approx_func(n_bnd, popt[0], popt[1])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((DnN-np.mean(DnN))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print('R^2: ', r_squared)
        
        plt.plot(n_bnd, self.approx_func(n_bnd, popt[0], popt[1]), label='trend line')
        plt.scatter(n_bnd, DnN, label='distances', color='red')
        plt.title('Trend')
        plt.xlabel('n')
        plt.ylabel('distances')
        plt.grid(True)
        plt.minorticks_on()
        plt.legend()
        plt.show()


###############################
#         RESEARCH            #
###############################

DA_list_Dn_N = []
S_list_Dn_N = []
Kol_list_Dn_N = []
Kolw_list_Dn_N = []
eps = 0.01

## Критерий D'Agostino
#
#N = 110
#n_bnd = range(50, 160, 10)
#distr = 'norm'
#loc = 0
#scale = 1
#F_x = np.array([i/N for i in range(N)])
#
#print('\n--------------------------------------------------------------\n')
#print('D\'Agostino')
##print('N: ', N)
#print('\n--------------------------------------------------------------\n')
#
#
#for n in n_bnd:
#    DA = Simulation(N, n, distr, loc, scale)
#    samples = DA.montecarlo()
#    stat = np.array(sorted([skewtest(samples[i])[0] for i in range(N)]))
#    scale_n = 1    
#    normal = np.array([norm.cdf(i) for i in stat])
#    DA_list_Dn_N.append(dist(normal, F_x))
#    print('Объём выборки: ', n)
#    print('DnN: ', DA_list_Dn_N[-1])
#    Graph_F_x_teta(stat, F_x)
#    Graph_F_x_teta(stat, normal)
#    pylab.legend(("D\'Agostino",
#                  "СНЗ"))
#    plt.show()
#    print('--------------------------------------------------------------')
#
#print('\n--------------------------------------------------------------\n')
#print('Аппроксимация для критерия D\'Agostino')
##print('N: ', N)
#print('\n--------------------------------------------------------------\n')
#
##print('Список расстояний по критерию D\'Agostino: ', DA_list_Dn_N)
#MyFile = open('DAgostinoDnbigN.txt', 'w')
#for item in DA_list_Dn_N:
#    MyFile.write("%s\n" % item)
#MyFile.close()
#DA_conc = Approx(n_bnd, DA_list_Dn_N, eps)
#DA_conc.summary(n_bnd, DA_list_Dn_N, eps)
##print('N: ', N)
##print('\nНормальный закон распределения со сдвигом {0} и масшатабом {1}'.format(loc, scale))



#
## Критерий проверки на симметричность
#
#N = 110
#n_bnd = range(50, 160, 10)
#distr = 'norm'
#loc = 0
#scale = 1
#F_x = np.array([i/N for i in range(N)])
#
#print('\n--------------------------------------------------------------\n')
#print('Критерий проверки на симметричность\n')
##print('N: ', N)
#print('\n--------------------------------------------------------------\n')
#
#
#for n in n_bnd: 
#    S = Simulation(N, n, distr, loc, scale)
#    samples = S.montecarlo()
#    stat = np.array(sorted([skew(samples[i]) for i in range(N)]))
#    scale_n = ( 6 * (n - 2) / ((n + 1) * (n + 3)) ) ** (1 / 2)
#    normal = np.array([norm.cdf(i, scale = scale_n) for i in stat])
#    S_list_Dn_N.append(dist(normal, F_x))
#    print('Объём выборки: ', n)
#    print(' DnN: ', S_list_Dn_N[-1])
#    Graph_F_x_teta(stat, F_x)
#    Graph_F_x_teta(stat, normal)
#    pylab.legend(("Критерий проверки на симметричность", 
#                  "Норм. распр. loc = 0, scale = sqrt(6*(n-2)/((n+1)(n+3)))"))
#    plt.show()
#    print('--------------------------------------------------------------')
#
#
#print('\n--------------------------------------------------------------\n')
#print('Аппроксимация для критерия проверки на симметричность\n')
#print('\n--------------------------------------------------------------\n')
#
##print('Список расстояний по критерию проверки на симметрию: ', S_list_Dn_N)
#MyFile = open('SkewDnNbigN.txt', 'w')
#for item in S_list_Dn_N:
#    MyFile.write("%s\n" % item)
#MyFile.close()
#S_conc = Approx(n_bnd, S_list_Dn_N, eps)
#S_conc.summary(n_bnd, S_list_Dn_N, eps)


# Критерий Колмогорова
N = 1000
n_bnd = range(5, 20, 5)
distr = ['norm']
loc = 0
scale = 1
F_x = np.array([i/N for i in range(N)])

print('\n--------------------------------------------------------------\n')
print('Критерий Колмогорова\n')
print('N: ', N)
print('\n--------------------------------------------------------------\n')

for distribution in distr:
    for n in n_bnd: 
        S = Simulation(N, n, distribution, loc, scale)
        samples = S.montecarlo()
        stat = np.array(sorted([kstest(samples[i], distribution)[0] for i in range(N)]))
        normal = np.array([kstwobign.cdf(i * n **(1/2)) for i in stat])
        Kol_list_Dn_N.append(dist(normal, F_x))
        print('Объём выборки: ', n, '\n')
        print('DnN: ', Kol_list_Dn_N[-1], '\n')
        #Graph_F_x_teta(stat, F_x)
        #Graph_F_x_teta(stat, normal)
        #pylab.legend(("Критерий Колмогорова",
        #              "Распределение Колмогорова"))
        #plt.show()
        print('\n--------------------------------------------------------------\n')
    print('Аппроксимация для критерия Колмогорова\n')
    print('для {} распределения\n'.format(distribution))
    print('\n--------------------------------------------------------------\n')

Kol_conc = Approx(n_bnd, Kol_list_Dn_N, eps)
Kol_conc.summary(n_bnd, Kol_list_Dn_N, eps)


# Критерий Колмогорова с поправкой
N = 1000
n_bnd = range(5, 20, 5)
distr = ['norm']
loc = 0
scale = 1
F_x = np.array([i/N for i in range(N)])

print('\n--------------------------------------------------------------\n')
print('Критерий Колмогорова\n')
print('N: ', N)
print('\n--------------------------------------------------------------\n')

for distribution in distr:
    for n in n_bnd: 
        S = Simulation(N, n, distribution, loc, scale)
        samples = S.montecarlo()
        stat = np.array(sorted([(kstest(samples[i], distribution)[0] * n ** (1/2) + 1 / (6 * n **(1/2)))
                                for i in range(N)]))
        normal = np.array([kstwobign.cdf(i) for i in stat])
        Kolw_list_Dn_N.append(dist(normal, F_x))
        print('Объём выборки: ', n, '\n')
        print('DnN: ', Kolw_list_Dn_N[-1], '\n')
        #Graph_F_x_teta(stat, F_x)
        #Graph_F_x_teta(stat, normal)
        #pylab.legend(("Критерий Колмогорова с поправкой",
        #              "Распределение Колмогорова"))
        #plt.show()
        print('\n--------------------------------------------------------------\n')
    print('Аппроксимация для критерия Колмогорова с поправкой Большева\n')
    print('для {} распределения\n'.format(distribution))
    print('--------------------------------------------------------------\n')

Kol_conc = Approx(n_bnd, Kolw_list_Dn_N, eps)
Kol_conc.summary(n_bnd, Kolw_list_Dn_N, eps)


########################
#         QA           #
########################

def qa_stat_tests():
    print("Testing Statistical Test Values\n")
    passed = True

    passed &= qa_skew()
    passed &= qa_D_Agostino()

    print("\nSUMMARY: ", "Ok\n" if passed else "Fail\n")


def qa_skew():
    passed = True
    passed &= (round(skew(test_sample), 6) == 0.265055)
    print("Критерий проверки на симметричность\n", "Ok" if passed else "Fail")
    return passed


def qa_D_Agostino():
    passed = True
    passed &= (round(skewtest(test_sample)[0], 6) == 0.446264)
    print("Модификация D'Agostino критерия проверки на симметричность\n", "Ok" if passed else "Fail")
    return passed


test_sample = [2, 8, 0, 4, 1, 9, 9, 0]
print("Let's do test\n")
qa_stat_tests()
