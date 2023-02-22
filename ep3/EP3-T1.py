from TrapezoidalMethod import *
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from os import path

# número de Euler
E = np.e

# epsilon para MAS
epsilon = 10E-6

# nome do diretório
dirname = path.dirname(__file__)

class NumericalConvergenceTableTwoVariables:
  outputFile = 'tables/tabela1.txt'
  chartFile = 'Gráfico1.pdf'

  def __init__(self, t_0: float, T: float, x_0: float, y_0: float):
    self.t_0 = t_0
    self.T = T
    self.x_0 = x_0
    self.y_0 = y_0

  def f_x(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_x(t, x(t), y(t)) = y(t) '''
    return y_t

  def f_y(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_y(t, x(t), y(t)) = -x(t) '''
    return -x_t
  
  def f(self, t: float, y: np.ndarray) -> np.ndarray:
    return np.array([self.f_x(t, y[0], y[1]), self.f_y(t, y[0], y[1])])

  def x(self, t: float) -> float:
    ''' x(t) = sin(t) + cos(t) '''
    return np.sin(t) + np.cos(t)

  def y(self, t: float) -> float:
    ''' y(t) = cos(t) - sin(t) '''
    return np.cos(t) - np.sin(t)

  def calculateNumericalConvergenceTable(self) -> str:
    result = ''

    errorNorm_n_minus_1 = 0
    h_n_minus_1 = 0

    fig, ax = plt.subplots()

    begin = 5
    for i in range(begin, 15):
      log_2_n = i
      n = 2 ** log_2_n
      h_n = (self.T - self.t_0) / n

      x_approximation, y_approximation = trapezoidalMethod(
        y_0=np.array([self.x_0, self.y_0]),
        t_0=self.t_0,
        T=self.T,
        n=n,
        f=self.f,
        fixedPointEpsilon=epsilon,
      )

      if i == begin:
        ax.plot(x_approximation, y_approximation,
                'k.', label='par ordenado x, y')
      else:
        ax.plot(x_approximation, y_approximation, 'k.', )

      x_errorModulus = np.absolute(
          globalDiscretizationError(self.T, self.x, x_approximation))
      y_errorModulus = np.absolute(
          globalDiscretizationError(self.T, self.y, y_approximation))

      errorNorm = float(np.linalg.norm(np.array([x_errorModulus, y_errorModulus])))

      p = '-----' if i == begin else convergenceOrderExponent(
          e_n=errorNorm_n_minus_1,
          e_n_plus_1=errorNorm,
          h_n=h_n_minus_1,
          h_n_plus_1=h_n
      )

      result += f'{n} & {h_n} & {errorNorm} & {p} \\\\\n'

      errorNorm_n_minus_1 = errorNorm
      h_n_minus_1 = h_n

    ax.set(xlabel='Aproximação de x', ylabel='Aproximação de y',
           title='Convergência da aproximação de duas variáveis')

    plt.grid(color='grey', linestyle='-', linewidth=0.5)

    plt.legend(loc='best')
    plt.savefig(f'{dirname}/{self.chartFile}')

    return result

  def generateTables(self):
    convergenceTable = self.calculateNumericalConvergenceTable()

    with open(f'{dirname}/{self.outputFile}', 'w') as file:
      file.write(convergenceTable)
      file.close()


class NumericalConvergenceTableTwoVariables2(NumericalConvergenceTableTwoVariables):
  outputFile = 'tables/tabela2.txt'
  chartFile = 'Gráfico2.pdf'

  def norm(self, x: float, y: float) -> float:
    ''' Euclidean norm '''
    return np.sqrt(x ** 2 + y ** 2)

  def f_x(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_x(t) = 3 * x(t) - 4 * y(t) '''
    return 3 * x_t - 4 * y_t

  def f_y(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_y(t) = x(t) - y(t) '''
    return x_t - y_t

  def x(self, t: float) -> float:
    return 2 * t * E ** t + E ** t

  def y(self, t: float) -> float:
    return t * E ** t

class LotkaVolterraEquationsSolutionApproximation:
  chartFile = 'Gráfico3.pdf'

  def __init__(self, alpha: float, beta: float, gamma: float, delta: float):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.delta = delta

  def f_x(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_x(t) = alpha * x(t) - beta * x(t) * y(t) '''
    return self.alpha * x_t - self.beta * x_t * y_t

  def f_y(self, t: float, x_t: float, y_t: float) -> float:
    ''' f_y(t) = delta * x(t) * y(t) - gamma * y(t) '''
    return self.delta * x_t * y_t - self.gamma * y_t

  def f(self, t: float, y: np.ndarray) -> np.ndarray:
    return np.array([self.f_x(t, y[0], y[1]), self.f_y(t, y[0], y[1])])

  def calculateApproximations(self, t_0: float, T: float, x_0: float, y_0: float) -> None:
    
    fig, ax = plt.subplots()
    begin = 5
    for i in range(begin, 15):
      log_2_n = i
      n = 2 ** log_2_n

      x_approximation, y_approximation = trapezoidalMethod(
        y_0=np.array([x_0, y_0]),
        t_0=t_0,
        T=T,
        n=n,
        f=self.f,
        fixedPointEpsilon=epsilon,
      )

      if begin == 8:
        ax.plot(x_approximation, y_approximation, 'k.', label = 'par ordenado x, y')
      else:
        ax.plot(x_approximation, y_approximation, 'k.' )

      print((x_approximation, y_approximation))
    
    ax.set(xlabel='Aproximação de x', ylabel='Aproximação de y',
      title='Convergência da aproximação de duas variáveis - Lotka Volterra')

    plt.grid(color='grey', linestyle='-', linewidth=0.5)

    plt.legend(loc='best')
    plt.savefig(f'{dirname}/{self.chartFile}')


ex1 = NumericalConvergenceTableTwoVariables(t_0=0, T=1, x_0=1, y_0=1)
ex1.generateTables()

ex1 = NumericalConvergenceTableTwoVariables2(t_0=0, T=1, x_0=1, y_0=0)
ex1.generateTables()

lotkaVolterra = LotkaVolterraEquationsSolutionApproximation(1, 2, 3, 4)
lotkaVolterra.calculateApproximations(t_0=0, T=1, x_0=1, y_0=1)

plt.show()
