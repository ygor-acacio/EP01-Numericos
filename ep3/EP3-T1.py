from TrapezeMethod import twoVariableEuler, globalDiscretizationError, convergenceOrderExponent
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

E = np.e

class LotkaVolterraEquationsSolutionApproximation:
  Savename = 'Gráfico5.pdf'

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

  def calculateApproximations(self, t_0: float, T: float, x_0: float, y_0: float) -> None:
    
    fig, ax = plt.subplots()
    for i in range(8, 20):
      log_2_n = i
      n = 2 ** log_2_n

      approximationX, approximationY = twoVariableEuler(
        x_0,
        y_0,
        t_0,
        T,
        n,
        self.f_x,
        self.f_y,
        # x_k_plus_1_supposed
        # y_k_plus_1_supposed
      )

      if i == 8:
        ax.plot(approximationX, approximationY, 'k.', label = 'par ordenado x, y')
      else:
        ax.plot(approximationX, approximationY, 'k.' )

      print((approximationX, approximationY))
    
    ax.set(xlabel='Aproximação de x', ylabel='Aproximação de y',
      title='Convergência da aproximação de duas variáveis - Lotka Volterra')

    plt.grid(color='grey', linestyle='-', linewidth=0.5)

    plt.legend(loc='best')
    plt.savefig(f'{self.Savename}')

lotkaVolterra = LotkaVolterraEquationsSolutionApproximation(1, 2, 3, 4)
lotkaVolterra.calculateApproximations(t_0=0, T=1, x_0=1, y_0=1)

plt.show()