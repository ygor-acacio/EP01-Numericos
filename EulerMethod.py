from collections.abc import Callable
import numpy as np

def y_k_plus_1(
  y_k: float, 
  h: float, 
  t_k: float, 
  phi: Callable[[float, float, float], float]
) -> float:
  return y_k + h * phi(t_k, y_k, h)
  

def oneVariableEuler(
  y_0: float, 
  t_0: float,
  T: float,
  n: int,
  f: Callable[[float, float], float]
) -> float:
  h = (T - t_0) / float(n)
  
  y_k = y_0
  t_k = t_0
  for i in range(n):
    current_y = y_k_plus_1(y_k, h, t_k, lambda t_k, y_k, h : f(t_k, y_k))
    y_k = current_y
    t_k = t_k + h

  return y_k


def globalDiscretizationError(
  T: float, 
  y: Callable[[float], float], 
  approximation: float
) -> float:
  return y(T) - approximation

def convergenceOrderExponent(e_n: float, e_n_plus_1: float, h_n: float, h_n_plus_1: float) -> float:
  return np.log2(np.absolute(e_n / e_n_plus_1)) / np.log2(np.absolute(h_n / h_n_plus_1))