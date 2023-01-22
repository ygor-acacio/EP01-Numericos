from EulerMethod import oneVariableEuler, globalDiscretizationError, convergenceOrderExponent
import numpy as np

# E = 2.7182818285
E = np.e

def f(t: float, y_t: float) -> float:
  ''' f(t, y(t)) = (e^(2t)) * y(t) '''
  return (E ** (2.0 * t)) * y_t

def y(t: float) -> float:
  ''' y(t) = e^((e^(2t) - 1) / 2), solution of y\'(t) = f(t, y(t)); y(0) = 1 '''
  return E ** ((E ** (2.0 * t) - 1) / 2.0)

def calculateNumericalConvergenceTable(t_0: float, T: float, y_0: float) -> None:
  errorModulus_n_minus_1 = 0
  h_n_minus_1 = 0
  for i in range(20):
    log_2_n = i
    n = 2 ** log_2_n
    h_n = (T - t_0) / n

    approximation = oneVariableEuler(y_0, t_0, T, n, f)

    errorModulus = np.absolute(globalDiscretizationError(T, y, approximation))

    p = '-----' if i == 0 else convergenceOrderExponent(e_n=errorModulus_n_minus_1, e_n_plus_1=errorModulus, h_n=h_n_minus_1, h_n_plus_1=h_n)

    print(f'{n} & {h_n} & {errorModulus} & {p} \\\\')

    errorModulus_n_minus_1 = errorModulus
    h_n_minus_1 = h_n


calculateNumericalConvergenceTable(t_0=0, T=1, y_0=y(0))