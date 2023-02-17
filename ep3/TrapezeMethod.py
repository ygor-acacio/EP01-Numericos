from typing import Tuple, Callable
import numpy as np
  

def twoVariableEuler(
  x_0: float,
  y_0: float,
  t_0: float,
  T: float,
  n: int,
  f_x: Callable[[float, float, float], float],
  f_y: Callable[[float, float, float], float]
) -> Tuple[float, float]:
  h = (T - t_0) / float(n)

  x_k = x_0
  y_k = y_0
  t_k = t_0
  for i in range(n):
    x_k_plus_1 = x_k + h * f_x(t_k, x_k, y_k)
    y_k_plus_1 = y_k + h * f_y(t_k, x_k, y_k)
    x_k = x_k_plus_1
    y_k = y_k_plus_1
    t_k = t_k + h

  return (x_k, y_k)


def globalDiscretizationError(
  T: float, 
  y: Callable[[float], float], 
  approximation: float
) -> float:
  return y(T) - approximation

def convergenceOrderExponent(e_n: float, e_n_plus_1: float, h_n: float, h_n_plus_1: float) -> float:
  return np.log2(np.absolute(e_n / e_n_plus_1)) / np.log2(np.absolute(h_n / h_n_plus_1))