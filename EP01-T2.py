from EulerMethod import oneVariableEuler, twoVariableEuler, globalDiscretizationError, convergenceOrderExponent
import numpy as np
from typing import Tuple

# E = 2.7182818285
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
    for i in range(8, 20):
      log_2_n = i
      n = 2 ** log_2_n
      h_n = (T - t_0) / n

      approximation = oneVariableEuler(y_0, t_0, T, n, self.f)

      # print(approximation, n)

      errorModulus = np.absolute(globalDiscretizationError(T, self.y, approximation))

      p = '-----' if i == 0 else convergenceOrderExponent(
        e_n=errorModulus_n_minus_1, 
        e_n_plus_1=errorModulus, 
        h_n=h_n_minus_1, 
        h_n_plus_1=h_n
      )

      result += f'{n} & {h_n} & {errorModulus} & {p} \\\\\n'

      errorModulus_n_minus_1 = errorModulus
      h_n_minus_1 = h_n

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
  outputFileX = 'tables/T2-Tabela2X.txt'
  outputFileY = 'tables/T2-Tabela2Y.txt'

  x_0 = 1
  y_0 = 1

  def __init__(self, t_0: float, T: float):
    self.t_0 = t_0
    self.T = T

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


  def calculateNumericalConvergenceTable(self) -> Tuple[str, str]:
    x_result = ''
    y_result = ''
    x_errorModulus_n_minus_1 = 0
    y_errorModulus_n_minus_1 = 0
    h_n_minus_1 = 0
    for i in range(20):
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

      # print('x:', x_approximation, self.x(1))
      # print('y:', y_approximation, self.y(1))

      x_errorModulus = np.absolute(globalDiscretizationError(self.T, self.x, x_approximation))
      y_errorModulus = np.absolute(globalDiscretizationError(self.T, self.y, y_approximation))

      p_x = '-----' if i == 0 else convergenceOrderExponent(
        e_n=x_errorModulus_n_minus_1, 
        e_n_plus_1=x_errorModulus, 
        h_n=h_n_minus_1, 
        h_n_plus_1=h_n
      )

      p_y = '-----' if i == 0 else convergenceOrderExponent(
        e_n=y_errorModulus_n_minus_1, 
        e_n_plus_1=y_errorModulus, 
        h_n=h_n_minus_1, 
        h_n_plus_1=h_n
      )

      x_result += f'{n} & {h_n} & {x_errorModulus} & {p_x} \\\\\n'
      y_result += f'{n} & {h_n} & {y_errorModulus} & {p_y} \\\\\n'

      x_errorModulus_n_minus_1 = x_errorModulus
      y_errorModulus_n_minus_1 = y_errorModulus
      h_n_minus_1 = h_n

    return (x_result, y_result)

  def generateTables(self):
    convergenceTableX, convergenceTableY = self.calculateNumericalConvergenceTable()

    with open(self.outputFileX, 'w') as file:
      file.write(convergenceTableX)
      file.close()

    with open(self.outputFileY, 'w') as file:
      file.write(convergenceTableY)
      file.close()

ex1 = NumericalConvergenceTableOneVariable()
ex1.generateTable()

ex2 = NumericalConvergenceTableTwoVariables(0, 1)
ex2.generateTables()