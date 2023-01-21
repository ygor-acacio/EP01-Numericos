# Nome: Daniel
# Nome: Ygor Acacio Maria
# Disciplina: MAP3122 - Métodos Numéricos e Aplicações
# Prof: Alexandre Roma

# Exercicio  - plotagem de grafico

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-np.pi, np.pi, 0.01)
y1 = np.cos(1 * t)
y2 = np.cos(2 * t)
y3 = np.cos(3 * t)

fig, ax = plt.subplots()
ax.plot(t, y1, color = 'black', linestyle='dotted', lw= 0.5, label = 'cos(t)')
ax.plot(t, y2, color = 'black', ls='--', lw= 0.5, label = 'cos(2t)')
ax.plot(t, y3, color = 'black', ls='-.', lw= 0.5, label = 'cos(3t)')

ax.set(xlabel='t', ylabel='y(t)',
       title='y(t)=cos(m.t), m = {1, 2, 3}')
ax.grid(color='grey', linestyle='-', linewidth=0.5)

plt.legend(loc='best')
plt.xlim(-4, 4)
plt.ylim(-1.5, 1.5) 
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.savefig("Gráfico.pdf")
plt.show()