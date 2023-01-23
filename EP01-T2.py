from EulerMethod import oneVariableEuler, twoVariableEuler, globalDiscretizationError, convergenceOrderExponent
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

E = np.e

class NumericalConvergenceTableOneVariable:
  outputFile = 'tables/T2-Tabela1.txt'

  def f(self, t: float, y_t: float) -> float:
    ''' f(t, y(t)) = (e^(2t)) * y(t) '''
    return (E ** (2.0 * t)) * y_t


  def y(self, t: float) -> float:
    ''' y(t) = e^((e^(2t) - 1) / 2), solution of y\'(t) = f(t, y(t)); y(0) = 1 '''
    return E ** ((E ** (2.0 * t) - 1) / 2.0)

  def calculateNumericalConvergenceTable(self, t_0: float, T: float, y_0: float) -> str:
    result = ''
    errorModulus_n_minus_1 = 0
    h_n_minus_1 = 0

    fig, ax = plt.subplots()
    
    for i in range(8, 20):
      log_2_n = i
      n = 2 ** log_2_n
      h_n = (T - t_0) / n

      approximation = oneVariableEuler(y_0, t_0, T, n, self.f)

      ax.plot( approximation, n,  'k.')

      errorModulus = np.absolute(globalDiscretizationError(T, self.y, approximation))

      p = '-----' if i == 8 else convergenceOrderExponent(
        e_n=errorModulus_n_minus_1, 
        e_n_plus_1=errorModulus, 
        h_n=h_n_minus_1, 
        h_n_plus_1=h_n
      )

      result += f'{n} & {h_n} & {errorModulus} & {p} \\\\\n'

      errorModulus_n_minus_1 = errorModulus
      h_n_minus_1 = h_n

    ax.set(xlabel='Aproximação', ylabel='nº de passos',
       title='Convergência da aproximação de uma variável')
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.ylim(0, 550000)
    plt.xlim(16, 28)
    plt.savefig("Gráfico2.pdf")
    plt.show()

    return result

  def generateTable(self):
    convergenceTable1 = self.calculateNumericalConvergenceTable(
      t_0=0, 
      T=1, 
      y_0=self.y(0)
    )

    with open(self.outputFile, 'w') as file:
      file.write(convergenceTable1)
      file.close()

class NumericalConvergenceTableTwoVariables:
  outputFile = 'tables/T2-Tabela2.txt'
  x_min = -1
  x_max = 2
  Savename = 'Gráfico3.pdf'

  def norm(self, x: float, y: float) -> float:
    ''' Maximum norm '''
    return max(x, y)

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

    for i in range(8, 20):
      log_2_n = i
      n = 2 ** log_2_n
      h_n = (self.T - self.t_0) / n

      x_approximation, y_approximation = twoVariableEuler(
        self.x_0, 
        self.y_0, 
        self.t_0, 
        self.T, 
        n, 
        self.f_x, 
        self.f_y
      )

      if i == 8:
        ax_x = ax.plot( x_approximation, n, 'k.', label = 'Aproximação de x')
        ax_y = ax.plot( y_approximation, n, '.', color = 'grey', label = 'Aproximação de y')
      else:
        ax_x = ax.plot( x_approximation, n, 'k.')
        ax_y = ax.plot( y_approximation, n, '.', color = 'grey',)
         
      x_errorModulus = np.absolute(globalDiscretizationError(self.T, self.x, x_approximation))
      y_errorModulus = np.absolute(globalDiscretizationError(self.T, self.y, y_approximation))

      errorNorm = self.norm(x_errorModulus, y_errorModulus)

      p = '-----' if i == 8 else convergenceOrderExponent(
        e_n=errorNorm_n_minus_1,
        e_n_plus_1=errorNorm,
        h_n=h_n_minus_1,
        h_n_plus_1=h_n
      )

      result += f'{n} & {h_n} & {errorNorm} & {p} \\\\\n'

      errorNorm_n_minus_1 = errorNorm
      h_n_minus_1 = h_n

    ax.set(xlabel='Aproximação', ylabel='nº de passos',
       title='Convergência da aproximação de duas variáveis')
    plt.grid(color='grey', linestyle='-', linewidth=0.5)

    plt.legend(loc='best')
    plt.ylim(0, 550000)
    plt.xlim(self.x_min , self.x_max)
    plt.savefig(f'{self.Savename}')
    plt.show()

    return result

  def generateTables(self):
    convergenceTable = self.calculateNumericalConvergenceTable()

    with open(self.outputFile, 'w') as file:
      file.write(convergenceTable)
      file.close()

class NumericalConvergenceTableTwoVariables2(NumericalConvergenceTableTwoVariables):
  outputFile = 'tables/T2-Tabela3.txt'
  x_min = 0
  x_max = 9
  Savename = 'Gráfico4.pdf'

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



ex1 = NumericalConvergenceTableOneVariable()
ex1.generateTable()

ex2 = NumericalConvergenceTableTwoVariables(t_0=0, T=1, x_0=1, y_0=1)
ex2.generateTables()

ex3 = NumericalConvergenceTableTwoVariables2(t_0=0, T=1, x_0=1, y_0=0)
ex3.generateTables()
