from typing import Tuple, Callable
import numpy as np
  
def trapezoidalMethod(
  y_0: np.ndarray,
  t_0: float,
  T: float,
  n: int,
  f: Callable[[float, np.ndarray], np.ndarray],
  fixedPointIterations: int,
) -> np.ndarray:
  '''
  Implementa o Método do Trapézio para um vetor `y`
  '''
  h = (T - t_0) / float(n)
  y_k = y_0
  t_k = t_0
  for _ in range(n):
    y_k_plus_1 = fixedPointIteration(
      phi=lambda x : y_k + (h/2) * (f(t_k, y_k) + f(t_k + h, x)),
      x_0=y_k,
      iterations=fixedPointIterations
    )

    y_k = y_k_plus_1
    t_k = t_k + h
  
  return y_k


def fixedPointIteration(
  phi: Callable[[np.ndarray], np.ndarray],
  x_0: np.ndarray,
  iterations: int,
) -> np.ndarray:
  ''' 
  Implementa o MAS para a função `phi`, com valor inicial `x_0`
  e critério número de iterações `iterations`
  '''
  x_n = phi(x_0)
  for _ in range(iterations): # type: ignore
    x_n = phi(x_n)

  return x_n

def globalDiscretizationError(
  T: float, 
  y: Callable[[float], float], 
  approximation: float
) -> float:
  return y(T) - approximation

def convergenceOrderExponent(
  e_n: float, 
  e_n_plus_1: float, 
  h_n: float, 
  h_n_plus_1: float
) -> float:
  return np.log2(np.absolute(e_n / e_n_plus_1)) / np.log2(np.absolute(h_n / h_n_plus_1))